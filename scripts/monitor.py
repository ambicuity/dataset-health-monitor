"""Main orchestration module for dataset health monitoring.

This module coordinates all health checks and reporting.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import yaml

from scripts.badges import generate_all_badges
from scripts.check_files import check_archive_contents, check_github_repo_files
from scripts.check_links import check_link
from scripts.checksum import verify_checksum
from scripts.huggingface import check_huggingface_dataset, is_huggingface_source
from scripts.open_issue import (
    close_issue,
    create_issue,
    find_open_issue,
    format_issue_body,
    get_labels_for_errors,
)
from scripts.schema_check import check_schema
from scripts.schema_drift import SchemaHistoryStore
from scripts.state_store import StateStore

logger = logging.getLogger(__name__)


def is_github_repo_url(url: str) -> bool:
    """Check if a URL is a GitHub repository URL.

    Uses proper URL parsing to avoid substring matching vulnerabilities.

    Args:
        url: URL to check.

    Returns:
        True if the URL is a valid GitHub repository URL.
    """
    try:
        parsed = urlparse(url)
        # Check for exact github.com hostname
        return parsed.netloc == "github.com" or parsed.netloc == "www.github.com"
    except Exception:
        return False


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    owner: str
    source_url: str
    expected_files: list[str] = field(default_factory=list)
    checksum: Optional[str] = None
    schema: Optional[list[str]] = None
    frequency: str = "daily"
    branch: str = "main"
    split: Optional[str] = None  # For HuggingFace datasets

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetConfig":
        """Create DatasetConfig from dictionary."""
        # Support both 'source_url' and 'source' for HuggingFace compatibility
        source = data.get("source_url") or data.get("source", "")

        return cls(
            name=data["name"],
            owner=data.get("owner", "unknown"),
            source_url=source,
            expected_files=data.get("expected_files", []),
            checksum=data.get("checksum"),
            schema=data.get("schema"),
            frequency=data.get("frequency", "daily"),
            branch=data.get("branch", "main"),
            split=data.get("split"),
        )


@dataclass
class CheckResult:
    """Result of all checks for a single dataset."""

    dataset_name: str
    is_healthy: bool
    errors: list[str] = field(default_factory=list)
    checksum: Optional[str] = None
    file_sizes: dict[str, int] = field(default_factory=dict)
    schema_hash: Optional[str] = None
    schema: Optional[list[str]] = None
    column_types: dict[str, str] = field(default_factory=dict)
    schema_drift_report: Optional[str] = None


def load_datasets_config(config_path: Path) -> list[DatasetConfig]:
    """Load dataset configurations from YAML file.

    Args:
        config_path: Path to the datasets YAML file.

    Returns:
        List of DatasetConfig objects.
    """
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "datasets" not in data:
        logger.warning("No datasets found in config file")
        return []

    datasets = []
    for item in data["datasets"]:
        try:
            config = DatasetConfig.from_dict(item)
            datasets.append(config)
        except (KeyError, TypeError) as e:
            logger.error("Invalid dataset config: %s - %s", item, e)

    logger.info("Loaded %d dataset configurations", len(datasets))
    return datasets


def check_huggingface(
    config: DatasetConfig,
    schema_history: SchemaHistoryStore,
) -> CheckResult:
    """Run health checks for a HuggingFace dataset.

    Args:
        config: Dataset configuration.
        schema_history: Schema history store for drift detection.

    Returns:
        CheckResult with check outcomes.
    """
    result = CheckResult(
        dataset_name=config.name,
        is_healthy=True,
    )

    logger.info("=" * 60)
    logger.info("Checking HuggingFace dataset: %s", config.name)
    logger.info("Source: %s", config.source_url)
    logger.info("=" * 60)

    # Check the HuggingFace dataset
    hf_result = check_huggingface_dataset(
        source=config.source_url,
        dataset_name=config.name,
        expected_schema=config.schema,
        expected_split=config.split,
    )

    if not hf_result.is_available:
        result.is_healthy = False
        result.errors.append(f"HuggingFace dataset not available: {hf_result.error_message}")
        return result

    if not hf_result.is_valid:
        result.is_healthy = False
        result.errors.append(hf_result.error_message or "Unknown error")
        return result

    # Store results
    result.schema = hf_result.schema
    result.schema_hash = hf_result.schema_hash
    result.column_types = hf_result.column_types
    result.checksum = hf_result.checksum

    # Track schema drift for HuggingFace datasets
    if hf_result.schema:
        snapshot, drift = schema_history.add_snapshot(
            config.name,
            hf_result.schema,
            hf_result.column_types,
        )

        if drift and drift.has_changes:
            logger.info("Schema drift detected for %s", config.name)
            result.schema_drift_report = drift.to_markdown()

    logger.info("✓ HuggingFace dataset %s is HEALTHY", config.name)
    logger.info("  Splits: %s", ", ".join(hf_result.splits))
    logger.info("  Features: %d", len(hf_result.schema or []))
    if hf_result.num_rows:
        logger.info("  Rows: %d", hf_result.num_rows)

    return result


def check_dataset(
    config: DatasetConfig,
    state_store: StateStore,
    schema_history: SchemaHistoryStore,
) -> CheckResult:
    """Run all health checks for a single dataset.

    Args:
        config: Dataset configuration.
        state_store: State store for comparison.
        schema_history: Schema history store for drift detection.

    Returns:
        CheckResult with all check outcomes.
    """
    # Check if this is a HuggingFace dataset
    if is_huggingface_source(config.source_url):
        return check_huggingface(config, schema_history)

    result = CheckResult(
        dataset_name=config.name,
        is_healthy=True,
    )

    logger.info("=" * 60)
    logger.info("Checking dataset: %s", config.name)
    logger.info("URL: %s", config.source_url)
    logger.info("=" * 60)

    # Get previous state for comparison
    prev_state = state_store.get_dataset_state(config.name)

    # 1. Link check
    logger.info("Running link check...")
    link_result = check_link(config.source_url)
    if not link_result.is_valid:
        result.is_healthy = False
        error_msg = f"Link broken: {link_result.error_message or 'Unknown error'}"
        if link_result.status_code:
            error_msg = f"Link broken: HTTP {link_result.status_code}"
        result.errors.append(error_msg)

    # 2. File existence check (for archives or GitHub repos)
    if config.expected_files and link_result.is_valid:
        logger.info("Running file existence check...")
        is_github = is_github_repo_url(config.source_url)
        is_archive = config.source_url.lower().endswith((".zip", ".tar.gz"))

        if is_github and not is_archive:
            file_result = check_github_repo_files(
                config.source_url,
                config.expected_files,
                config.name,
                branch=config.branch,
            )
        else:
            file_result = check_archive_contents(
                config.source_url,
                config.expected_files,
                config.name,
            )

        if not file_result.is_valid:
            result.is_healthy = False
            if file_result.missing_files:
                result.errors.append(
                    f"Missing files: {', '.join(file_result.missing_files)}"
                )
            if file_result.error_message:
                result.errors.append(f"File check error: {file_result.error_message}")

        result.file_sizes = file_result.file_sizes

        # Check for file size changes
        if prev_state and prev_state.file_sizes:
            for filename, size in file_result.file_sizes.items():
                prev_size = prev_state.file_sizes.get(filename)
                if prev_size is not None and size != prev_size:
                    change_pct = abs(size - prev_size) / prev_size * 100
                    if change_pct > 10:  # Alert on >10% change
                        logger.warning(
                            "File size changed: %s (%d -> %d bytes, %.1f%%)",
                            filename,
                            prev_size,
                            size,
                            change_pct,
                        )

    # 3. Checksum verification
    if link_result.is_valid:
        logger.info("Running checksum verification...")
        checksum_result = verify_checksum(
            config.source_url,
            config.checksum,
            config.name,
        )

        result.checksum = checksum_result.computed_checksum

        if not checksum_result.is_valid:
            result.is_healthy = False
            if checksum_result.checksum_changed:
                result.errors.append(
                    f"Checksum changed from {config.checksum} to {checksum_result.computed_checksum}"
                )
            elif checksum_result.error_message:
                result.errors.append(
                    f"Checksum verification error: {checksum_result.error_message}"
                )

    # 4. Schema check (for CSV/JSON files)
    if link_result.is_valid:
        logger.info("Running schema check...")
        schema_result = check_schema(
            config.source_url,
            config.name,
            expected_schema=config.schema,
        )

        result.schema_hash = schema_result.schema_hash
        result.schema = schema_result.current_schema
        result.column_types = schema_result.column_types

        # Track schema drift
        if schema_result.current_schema:
            snapshot, drift = schema_history.add_snapshot(
                config.name,
                schema_result.current_schema,
                schema_result.column_types,
            )

            if drift and drift.has_changes:
                logger.info("Schema drift detected for %s", config.name)
                result.schema_drift_report = drift.to_markdown()

        if not schema_result.is_valid:
            result.is_healthy = False
            if schema_result.schema_changed:
                result.errors.append(
                    f"Schema changed: expected {config.schema}, got {schema_result.current_schema}"
                )
            elif schema_result.error_message:
                result.errors.append(
                    f"Schema check error: {schema_result.error_message}"
                )

    # Log result summary
    if result.is_healthy:
        logger.info("✓ Dataset %s is HEALTHY", config.name)
    else:
        logger.error("✗ Dataset %s has ISSUES:", config.name)
        for error in result.errors:
            logger.error("  - %s", error)

    return result


def run_monitor(
    config_path: Path,
    state_path: Path,
    badges_path: Path,
    schema_history_path: Path,
    dry_run: bool = False,
) -> int:
    """Run the complete monitoring workflow.

    Args:
        config_path: Path to datasets configuration file.
        state_path: Path to state storage file.
        badges_path: Path to badges output directory.
        schema_history_path: Path to schema history directory.
        dry_run: If True, don't create/close issues.

    Returns:
        Exit code (0 = all healthy, 1 = issues found).
    """
    # Initialize state store
    state_store = StateStore(state_path)

    # Initialize schema history store
    schema_history = SchemaHistoryStore(schema_history_path)

    # Load dataset configurations
    datasets = load_datasets_config(config_path)
    if not datasets:
        logger.error("No datasets to check")
        return 1

    # Track overall health
    all_healthy = True
    results: list[CheckResult] = []
    schema_drifts: list[tuple[str, str]] = []  # (dataset_name, drift_report)

    # Check each dataset
    for config in datasets:
        result = check_dataset(config, state_store, schema_history)
        results.append(result)

        # Track schema drift for reporting
        if result.schema_drift_report:
            schema_drifts.append((config.name, result.schema_drift_report))

        if not result.is_healthy:
            all_healthy = False

            # Determine if we should create an issue
            if state_store.should_create_issue(config.name):
                if not dry_run:
                    # Get last known good state
                    prev_state = state_store.get_dataset_state(config.name)
                    last_good = None
                    if prev_state:
                        last_good = {
                            "last_success_timestamp": prev_state.last_success_timestamp,
                            "checksum": prev_state.checksum,
                            "schema": prev_state.schema,
                        }

                    # Create issue body with schema drift if available
                    issue_body = format_issue_body(
                        config.name,
                        config.source_url,
                        result.errors,
                        last_good,
                    )

                    # Append schema drift report if available
                    if result.schema_drift_report:
                        issue_body += "\n\n---\n\n" + result.schema_drift_report

                    labels = get_labels_for_errors(result.errors)

                    issue_result = create_issue(
                        title=f"[Dataset Health] {config.name} - Health Check Failed",
                        body=issue_body,
                        labels=labels,
                    )

                    if issue_result.success:
                        state_store.update_dataset_state(
                            config.name,
                            is_healthy=False,
                            error=" | ".join(result.errors),
                            issue_number=issue_result.issue_number,
                        )
                    else:
                        logger.error(
                            "Failed to create issue: %s",
                            issue_result.error_message,
                        )
                else:
                    logger.info("[DRY RUN] Would create issue for %s", config.name)
            else:
                logger.info(
                    "Skipping issue creation for %s (issue already exists)",
                    config.name,
                )

            # Update state with failure
            state_store.update_dataset_state(
                config.name,
                is_healthy=False,
                checksum=result.checksum,
                file_sizes=result.file_sizes,
                schema_hash=result.schema_hash,
                schema=result.schema,
                error=" | ".join(result.errors),
            )

        else:
            # Dataset is healthy
            prev_state = state_store.get_dataset_state(config.name)

            # Check if it was previously unhealthy (recovery)
            if prev_state and not prev_state.is_healthy:
                issue_number = prev_state.issue_number
                if not dry_run:
                    # Try to find and close the issue
                    if issue_number is None:
                        issue_number = find_open_issue(config.name)

                    if issue_number:
                        close_result = close_issue(
                            issue_number,
                            comment=f"✅ Dataset `{config.name}` has recovered and is now healthy.",
                        )
                        if close_result.success:
                            state_store.clear_issue(config.name)
                elif issue_number:
                    logger.info(
                        "[DRY RUN] Would close issue #%d for %s",
                        issue_number,
                        config.name,
                        )

            # Update state with success
            state_store.update_dataset_state(
                config.name,
                is_healthy=True,
                checksum=result.checksum,
                file_sizes=result.file_sizes,
                schema_hash=result.schema_hash,
                schema=result.schema,
            )

    # Save state
    state_store.save()

    # Generate badges
    logger.info("")
    logger.info("Generating badges...")
    generate_all_badges(state_store, badges_path)

    # Generate schema drift reports
    if schema_drifts:
        logger.info("")
        logger.info("Generating schema drift reports...")
        reports_dir = schema_history_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name, drift_report in schema_drifts:
            report_path = reports_dir / f"{dataset_name}_drift_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(drift_report)
            logger.info("Generated drift report: %s", report_path)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MONITORING SUMMARY")
    logger.info("=" * 60)
    logger.info("Total datasets checked: %d", len(results))
    healthy_count = sum(1 for r in results if r.is_healthy)
    unhealthy_count = len(results) - healthy_count
    logger.info("Healthy: %d", healthy_count)
    logger.info("Unhealthy: %d", unhealthy_count)

    if schema_drifts:
        logger.info("Schema drifts detected: %d", len(schema_drifts))

    if unhealthy_count > 0:
        logger.info("")
        logger.info("Failed datasets:")
        for r in results:
            if not r.is_healthy:
                logger.info("  - %s: %s", r.dataset_name, "; ".join(r.errors))

    return 0 if all_healthy else 1


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the monitor.

    Args:
        verbose: Enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dataset Health Monitor - CI for your ML datasets",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("datasets/datasets.yaml"),
        help="Path to datasets configuration file",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("state/dataset_state.json"),
        help="Path to state storage file",
    )
    parser.add_argument(
        "--badges",
        type=Path,
        default=Path("badges"),
        help="Path to badges output directory",
    )
    parser.add_argument(
        "--schema-history",
        type=Path,
        default=Path("state/schema_history"),
        help="Path to schema history directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks without creating/closing issues",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("Dataset Health Monitor starting...")
    logger.info("Config: %s", args.config)
    logger.info("State: %s", args.state)
    logger.info("Badges: %s", args.badges)
    logger.info("Schema History: %s", args.schema_history)

    return run_monitor(
        config_path=args.config,
        state_path=args.state,
        badges_path=args.badges,
        schema_history_path=args.schema_history,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
