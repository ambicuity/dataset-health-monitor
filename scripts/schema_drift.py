"""Schema drift detection and visualization module.

This module provides functionality to track schema changes over time,
generate visual diff reports, and create markdown tables showing
added/removed columns and type changes.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum number of schema history entries to keep
MAX_SCHEMA_HISTORY = 50


@dataclass
class SchemaSnapshot:
    """A snapshot of a dataset's schema at a point in time."""

    timestamp: str
    schema_hash: str
    columns: list[str]
    column_types: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "schema_hash": self.schema_hash,
            "columns": self.columns,
            "column_types": self.column_types,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaSnapshot":
        """Create SchemaSnapshot from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            schema_hash=data["schema_hash"],
            columns=data["columns"],
            column_types=data.get("column_types", {}),
        )


@dataclass
class SchemaDiff:
    """Represents the differences between two schemas."""

    dataset_name: str
    old_schema: Optional[SchemaSnapshot]
    new_schema: SchemaSnapshot
    added_columns: list[str] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    type_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    has_changes: bool = False

    def to_markdown(self) -> str:
        """Generate a markdown representation of the schema diff.

        Returns:
            Markdown formatted string showing schema changes.
        """
        if not self.has_changes:
            return f"✅ No schema changes detected for `{self.dataset_name}`"

        lines = [
            f"## Schema Drift Report: `{self.dataset_name}`",
            "",
            f"**Detected:** {self.new_schema.timestamp}",
            "",
        ]

        if self.old_schema:
            lines.extend([
                "### Summary",
                "",
                f"- **Previous Schema Hash:** `{self.old_schema.schema_hash}`",
                f"- **Current Schema Hash:** `{self.new_schema.schema_hash}`",
                "",
            ])

        # Added columns
        if self.added_columns:
            lines.extend([
                "### ➕ Added Columns",
                "",
                "| Column Name | Type |",
                "|-------------|------|",
            ])
            for col in self.added_columns:
                col_type = self.new_schema.column_types.get(col, "unknown")
                lines.append(f"| `{col}` | {col_type} |")
            lines.append("")

        # Removed columns
        if self.removed_columns:
            lines.extend([
                "### ➖ Removed Columns",
                "",
                "| Column Name | Previous Type |",
                "|-------------|---------------|",
            ])
            for col in self.removed_columns:
                if self.old_schema:
                    col_type = self.old_schema.column_types.get(col, "unknown")
                else:
                    col_type = "unknown"
                lines.append(f"| `{col}` | {col_type} |")
            lines.append("")

        # Type changes
        if self.type_changes:
            lines.extend([
                "### 🔄 Type Changes",
                "",
                "| Column Name | Old Type | New Type |",
                "|-------------|----------|----------|",
            ])
            for col, (old_type, new_type) in self.type_changes.items():
                lines.append(f"| `{col}` | {old_type} | {new_type} |")
            lines.append("")

        # Full schema comparison table
        lines.extend([
            "### Full Schema Comparison",
            "",
            "| Column | Status | Type |",
            "|--------|--------|------|",
        ])

        all_columns = set(self.new_schema.columns)
        if self.old_schema:
            all_columns.update(self.old_schema.columns)

        for col in sorted(all_columns):
            if col in self.added_columns:
                status = "🟢 Added"
                col_type = self.new_schema.column_types.get(col, "unknown")
            elif col in self.removed_columns:
                status = "🔴 Removed"
                col_type = self.old_schema.column_types.get(col, "unknown") if self.old_schema else "unknown"
            elif col in self.type_changes:
                status = "🟡 Changed"
                col_type = f"{self.type_changes[col][0]} → {self.type_changes[col][1]}"
            else:
                status = "⚪ Unchanged"
                col_type = self.new_schema.column_types.get(col, "unknown")
            lines.append(f"| `{col}` | {status} | {col_type} |")

        lines.extend([
            "",
            "---",
            "*Report generated by Dataset Health Monitor*",
        ])

        return "\n".join(lines)


class SchemaHistoryStore:
    """Manages schema history storage for drift detection."""

    def __init__(self, history_dir: Path):
        """Initialize schema history store.

        Args:
            history_dir: Directory to store schema history files.
        """
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[SchemaSnapshot]] = {}

    def _get_history_file(self, dataset_name: str) -> Path:
        """Get the history file path for a dataset."""
        # Sanitize dataset name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
        return self.history_dir / f"{safe_name}_schema_history.json"

    def load_history(self, dataset_name: str) -> list[SchemaSnapshot]:
        """Load schema history for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            List of SchemaSnapshot objects, newest first.
        """
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        history_file = self._get_history_file(dataset_name)
        if not history_file.exists():
            self._cache[dataset_name] = []
            return []

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = [SchemaSnapshot.from_dict(item) for item in data.get("history", [])]
            self._cache[dataset_name] = history
            return history
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load schema history for %s: %s", dataset_name, e)
            self._cache[dataset_name] = []
            return []

    def save_history(self, dataset_name: str) -> None:
        """Save schema history for a dataset.

        Args:
            dataset_name: Name of the dataset.
        """
        if dataset_name not in self._cache:
            return

        history = self._cache[dataset_name]
        history_file = self._get_history_file(dataset_name)

        data = {
            "dataset_name": dataset_name,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "history": [snapshot.to_dict() for snapshot in history],
        }

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved schema history for %s", dataset_name)

    def add_snapshot(
        self,
        dataset_name: str,
        columns: list[str],
        column_types: Optional[dict[str, str]] = None,
    ) -> tuple[SchemaSnapshot, Optional[SchemaDiff]]:
        """Add a new schema snapshot and detect drift.

        Args:
            dataset_name: Name of the dataset.
            columns: List of column names.
            column_types: Optional dictionary of column types.

        Returns:
            Tuple of (new snapshot, schema diff if changes detected).
        """
        history = self.load_history(dataset_name)
        column_types = column_types or {}

        # Compute schema hash
        schema_str = "|".join(sorted(columns))
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        # Create new snapshot
        new_snapshot = SchemaSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            schema_hash=schema_hash,
            columns=columns,
            column_types=column_types,
        )

        # Check for drift
        diff = None
        if history:
            latest = history[0]
            if latest.schema_hash != schema_hash:
                diff = self._compute_diff(dataset_name, latest, new_snapshot)
        else:
            # First snapshot - check if there are any columns (initial state)
            if columns:
                diff = SchemaDiff(
                    dataset_name=dataset_name,
                    old_schema=None,
                    new_schema=new_snapshot,
                    added_columns=columns,
                    has_changes=True,
                )

        # Only add to history if schema changed or it's the first snapshot
        if not history or history[0].schema_hash != schema_hash:
            history.insert(0, new_snapshot)
            # Prune old entries
            if len(history) > MAX_SCHEMA_HISTORY:
                history = history[:MAX_SCHEMA_HISTORY]
            self._cache[dataset_name] = history
            self.save_history(dataset_name)

        return new_snapshot, diff

    def _compute_diff(
        self,
        dataset_name: str,
        old_schema: SchemaSnapshot,
        new_schema: SchemaSnapshot,
    ) -> SchemaDiff:
        """Compute the difference between two schemas.

        Args:
            dataset_name: Name of the dataset.
            old_schema: Previous schema snapshot.
            new_schema: Current schema snapshot.

        Returns:
            SchemaDiff object with detected changes.
        """
        old_cols = set(old_schema.columns)
        new_cols = set(new_schema.columns)

        added = list(new_cols - old_cols)
        removed = list(old_cols - new_cols)

        # Detect type changes for columns that exist in both
        type_changes = {}
        common_cols = old_cols & new_cols
        for col in common_cols:
            old_type = old_schema.column_types.get(col)
            new_type = new_schema.column_types.get(col)
            if old_type and new_type and old_type != new_type:
                type_changes[col] = (old_type, new_type)

        has_changes = bool(added or removed or type_changes)

        return SchemaDiff(
            dataset_name=dataset_name,
            old_schema=old_schema,
            new_schema=new_schema,
            added_columns=sorted(added),
            removed_columns=sorted(removed),
            type_changes=type_changes,
            has_changes=has_changes,
        )

    def get_latest_snapshot(self, dataset_name: str) -> Optional[SchemaSnapshot]:
        """Get the latest schema snapshot for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Latest SchemaSnapshot or None if no history.
        """
        history = self.load_history(dataset_name)
        return history[0] if history else None

    def get_drift_summary(self, dataset_name: str, days: int = 30) -> dict[str, Any]:
        """Get a summary of schema drift over a period.

        Args:
            dataset_name: Name of the dataset.
            days: Number of days to analyze.

        Returns:
            Dictionary with drift statistics.
        """
        history = self.load_history(dataset_name)
        if not history:
            return {
                "total_changes": 0,
                "columns_added": 0,
                "columns_removed": 0,
                "type_changes": 0,
            }

        # Filter to recent history
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        recent_history = []
        for snapshot in history:
            try:
                ts = datetime.fromisoformat(snapshot.timestamp.replace("Z", "+00:00"))
                if ts > cutoff:
                    recent_history.append(snapshot)
            except ValueError:
                continue

        if len(recent_history) < 2:
            return {
                "total_changes": 0,
                "columns_added": 0,
                "columns_removed": 0,
                "type_changes": 0,
            }

        total_added = 0
        total_removed = 0
        total_type_changes = 0

        for i in range(len(recent_history) - 1):
            newer = recent_history[i]
            older = recent_history[i + 1]
            diff = self._compute_diff(dataset_name, older, newer)
            total_added += len(diff.added_columns)
            total_removed += len(diff.removed_columns)
            total_type_changes += len(diff.type_changes)

        return {
            "total_changes": len(recent_history) - 1,
            "columns_added": total_added,
            "columns_removed": total_removed,
            "type_changes": total_type_changes,
        }


def generate_drift_report(
    diff: SchemaDiff,
    output_dir: Path,
) -> Path:
    """Generate a schema drift report file.

    Args:
        diff: Schema difference to report.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize dataset name for filename
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in diff.dataset_name
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"schema_drift_{safe_name}_{timestamp}.md"

    markdown_content = diff.to_markdown()

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info("Generated schema drift report: %s", report_file)
    return report_file
