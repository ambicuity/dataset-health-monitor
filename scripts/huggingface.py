"""HuggingFace dataset support module.

This module provides functionality to monitor HuggingFace datasets,
including fetching dataset info, schema, checksums, and integrating
with the existing monitoring infrastructure.
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# HuggingFace cache directory
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

# Maximum size to download for checksum (100MB)
MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024


@dataclass
class HuggingFaceDatasetInfo:
    """Information about a HuggingFace dataset."""

    dataset_id: str
    config: Optional[str] = None
    split: Optional[str] = None
    is_available: bool = False
    num_rows: Optional[int] = None
    size_bytes: Optional[int] = None
    features: dict[str, str] = field(default_factory=dict)
    splits: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class HuggingFaceCheckResult:
    """Result of a HuggingFace dataset check."""

    dataset_name: str
    dataset_id: str
    is_valid: bool = False
    is_available: bool = False
    schema: Optional[list[str]] = None
    schema_hash: Optional[str] = None
    column_types: dict[str, str] = field(default_factory=dict)
    checksum: Optional[str] = None
    num_rows: Optional[int] = None
    size_bytes: Optional[int] = None
    splits: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


def parse_huggingface_url(source: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse a HuggingFace dataset URL.

    Supports formats:
    - huggingface://owner/dataset_name
    - huggingface://owner/dataset_name/config
    - huggingface://owner/dataset_name/config/split

    Args:
        source: HuggingFace dataset URL.

    Returns:
        Tuple of (dataset_id, config, split).
    """
    if not source.startswith("huggingface://"):
        raise ValueError(f"Invalid HuggingFace URL: {source}")

    # Remove the protocol prefix
    path = source[len("huggingface://"):]
    parts = path.strip("/").split("/")

    if len(parts) < 2:
        raise ValueError(f"Invalid HuggingFace URL: {source}. Expected format: huggingface://owner/dataset")

    # First two parts are always owner/dataset_name
    dataset_id = f"{parts[0]}/{parts[1]}"
    config = parts[2] if len(parts) > 2 else None
    split = parts[3] if len(parts) > 3 else None

    return dataset_id, config, split


def is_huggingface_source(source: str) -> bool:
    """Check if a source URL is a HuggingFace dataset.

    Args:
        source: Source URL or identifier.

    Returns:
        True if the source is a HuggingFace dataset URL.
    """
    return source.startswith("huggingface://")


def get_dataset_info(
    dataset_id: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> HuggingFaceDatasetInfo:
    """Get information about a HuggingFace dataset.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "stanfordnlp/imdb").
        config: Optional dataset configuration name.
        split: Optional split name (e.g., "train", "test").

    Returns:
        HuggingFaceDatasetInfo with dataset details.
    """
    info = HuggingFaceDatasetInfo(
        dataset_id=dataset_id,
        config=config,
        split=split,
    )

    try:
        from datasets import load_dataset_builder, get_dataset_config_names

        logger.debug("Fetching info for HuggingFace dataset: %s", dataset_id)

        # Get available configs if not specified
        try:
            available_configs = get_dataset_config_names(dataset_id)
            if available_configs:
                logger.debug("Available configs: %s", available_configs)
        except Exception:
            available_configs = []

        # Use the specified config or the first available one
        effective_config = config
        if effective_config is None and available_configs:
            effective_config = available_configs[0]

        # Load dataset builder to get metadata
        builder = load_dataset_builder(dataset_id, name=effective_config)

        # Get dataset info
        ds_info = builder.info

        # Extract features (schema)
        if ds_info.features:
            for feature_name, feature_type in ds_info.features.items():
                info.features[feature_name] = _feature_type_to_string(feature_type)

        # Get available splits
        if ds_info.splits:
            info.splits = list(ds_info.splits.keys())

            # Get row count for the specified split or total
            if split and split in ds_info.splits:
                info.num_rows = ds_info.splits[split].num_examples
            else:
                info.num_rows = sum(
                    s.num_examples for s in ds_info.splits.values()
                    if s.num_examples is not None
                )

        # Get size
        if ds_info.download_size:
            info.size_bytes = ds_info.download_size

        info.is_available = True
        logger.info("✓ HuggingFace dataset info retrieved: %s", dataset_id)

    except ImportError:
        info.error_message = "HuggingFace datasets library not installed. Install with: pip install datasets"
        logger.error("✗ %s", info.error_message)
    except Exception as e:
        info.error_message = f"Failed to fetch dataset info: {e}"
        logger.error("✗ Failed to fetch HuggingFace dataset info: %s", e)

    return info


def _feature_type_to_string(feature_type: Any) -> str:
    """Convert HuggingFace feature type to a string representation.

    Args:
        feature_type: HuggingFace feature type object.

    Returns:
        String representation of the type.
    """
    type_name = type(feature_type).__name__

    # Handle common types
    if type_name == "Value":
        return str(feature_type.dtype)
    elif type_name == "ClassLabel":
        return f"ClassLabel(num_classes={feature_type.num_classes})"
    elif type_name == "Sequence":
        inner = _feature_type_to_string(feature_type.feature)
        return f"Sequence({inner})"
    elif isinstance(feature_type, dict):
        return "dict"
    elif hasattr(feature_type, "dtype"):
        return str(feature_type.dtype)

    return type_name


def check_huggingface_dataset(
    source: str,
    dataset_name: str,
    expected_schema: Optional[list[str]] = None,
    expected_split: Optional[str] = None,
) -> HuggingFaceCheckResult:
    """Check a HuggingFace dataset for health.

    Args:
        source: HuggingFace dataset URL (huggingface://owner/dataset).
        dataset_name: Name for logging and reporting.
        expected_schema: Optional expected column/feature names.
        expected_split: Optional expected split name.

    Returns:
        HuggingFaceCheckResult with check results.
    """
    result = HuggingFaceCheckResult(
        dataset_name=dataset_name,
        dataset_id="",
    )

    try:
        # Parse the URL
        dataset_id, config, split = parse_huggingface_url(source)
        result.dataset_id = dataset_id

        # Use specified split from URL or parameter
        effective_split = split or expected_split

        # Get dataset info
        info = get_dataset_info(dataset_id, config, effective_split)

        if not info.is_available:
            result.error_message = info.error_message or "Dataset not available"
            logger.error("✗ HuggingFace dataset not available: %s", dataset_name)
            return result

        result.is_available = True
        result.splits = info.splits
        result.num_rows = info.num_rows
        result.size_bytes = info.size_bytes

        # Extract schema from features
        if info.features:
            result.schema = list(info.features.keys())
            result.column_types = info.features
            result.schema_hash = _compute_schema_hash(result.schema)

        # Validate schema if expected
        if expected_schema:
            expected_set = set(expected_schema)
            current_set = set(result.schema or [])
            missing = expected_set - current_set

            if missing:
                result.error_message = f"Missing expected features: {', '.join(missing)}"
                logger.error("✗ Schema mismatch for %s: missing %s", dataset_name, missing)
                return result

        # Compute checksum from dataset metadata
        result.checksum = _compute_dataset_checksum(info)

        result.is_valid = True
        logger.info("✓ HuggingFace dataset check passed: %s", dataset_name)

    except ValueError as e:
        result.error_message = str(e)
        logger.error("✗ Invalid HuggingFace URL: %s", e)
    except Exception as e:
        result.error_message = f"Check failed: {e}"
        logger.error("✗ HuggingFace dataset check failed: %s", e)

    return result


def _compute_schema_hash(schema: list[str]) -> str:
    """Compute a hash of the schema for comparison.

    Args:
        schema: List of feature/column names.

    Returns:
        SHA256 hash of the schema.
    """
    schema_str = "|".join(sorted(schema))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def _compute_dataset_checksum(info: HuggingFaceDatasetInfo) -> str:
    """Compute a checksum for a HuggingFace dataset based on metadata.

    Since downloading full datasets can be expensive, we compute a
    checksum based on the dataset metadata (schema, size, rows).

    Args:
        info: HuggingFace dataset info.

    Returns:
        SHA256 checksum string.
    """
    # Create a deterministic string from metadata
    metadata_parts = [
        f"id:{info.dataset_id}",
        f"config:{info.config or 'default'}",
        f"rows:{info.num_rows or 0}",
        f"size:{info.size_bytes or 0}",
        f"features:{','.join(sorted(info.features.keys()))}",
        f"splits:{','.join(sorted(info.splits))}",
    ]
    metadata_str = "|".join(metadata_parts)

    checksum = hashlib.sha256(metadata_str.encode()).hexdigest()
    return f"sha256:{checksum}"


def download_dataset_sample(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    num_samples: int = 100,
) -> Optional[list[dict]]:
    """Download a sample of a HuggingFace dataset.

    Useful for schema validation and data quality checks.

    Args:
        dataset_id: HuggingFace dataset ID.
        config: Optional dataset configuration.
        split: Split to sample from.
        num_samples: Number of samples to download.

    Returns:
        List of sample records or None if download fails.
    """
    try:
        from itertools import islice

        from datasets import load_dataset

        logger.debug("Downloading %d samples from %s/%s", num_samples, dataset_id, split)

        # Load with streaming to avoid downloading entire dataset
        dataset = load_dataset(
            dataset_id,
            name=config,
            split=split,
            streaming=True,
        )

        # Use islice for efficient sampling
        samples = list(islice(dataset, num_samples))

        logger.debug("Downloaded %d samples", len(samples))
        return samples

    except ImportError:
        logger.error("HuggingFace datasets library not installed")
        return None
    except Exception as e:
        logger.error("Failed to download samples: %s", e)
        return None


def clear_huggingface_cache() -> None:
    """Clear the HuggingFace datasets cache.

    Useful for managing disk space in CI environments.
    """
    import shutil

    cache_dir = HF_CACHE_DIR / "datasets"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            logger.info("Cleared HuggingFace cache: %s", cache_dir)
        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)
