"""State persistence module for dataset health monitoring.

This module manages the persistent state of dataset health checks,
tracking checksums, file sizes, schema hashes, timestamps, and uptime history.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum number of days to keep in uptime history
UPTIME_HISTORY_DAYS = 30


def _parse_timestamp(timestamp: str) -> datetime:
    """Parse an ISO timestamp string to a datetime object.

    Handles both 'Z' suffix and '+00:00' timezone formats.

    Args:
        timestamp: ISO format timestamp string.

    Returns:
        datetime object with timezone info.
    """
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@dataclass
class UptimeRecord:
    """Record of a single health check for uptime tracking."""

    timestamp: str
    is_healthy: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"timestamp": self.timestamp, "is_healthy": self.is_healthy}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UptimeRecord":
        """Create UptimeRecord from dictionary."""
        return cls(timestamp=data["timestamp"], is_healthy=data["is_healthy"])


@dataclass
class DatasetState:
    """State information for a single dataset."""

    name: str
    last_check_timestamp: str
    last_success_timestamp: Optional[str] = None
    checksum: Optional[str] = None
    file_sizes: dict[str, int] = field(default_factory=dict)
    schema_hash: Optional[str] = None
    schema: Optional[list[str]] = None
    is_healthy: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    issue_number: Optional[int] = None
    uptime_history: list[UptimeRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert UptimeRecord objects to dicts
        data["uptime_history"] = [r.to_dict() if isinstance(r, UptimeRecord) else r for r in self.uptime_history]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetState":
        """Create DatasetState from dictionary."""
        # Handle uptime_history separately
        uptime_history_data = data.pop("uptime_history", [])
        uptime_history = [
            UptimeRecord.from_dict(r) if isinstance(r, dict) else r
            for r in uptime_history_data
        ]
        return cls(**data, uptime_history=uptime_history)

    def add_uptime_record(self, is_healthy: bool) -> None:
        """Add a new uptime record and prune old entries.

        Args:
            is_healthy: Whether the check was successful.
        """
        now = datetime.now(timezone.utc)
        self.uptime_history.append(
            UptimeRecord(timestamp=now.isoformat(), is_healthy=is_healthy)
        )

        # Prune records older than UPTIME_HISTORY_DAYS
        cutoff = now - timedelta(days=UPTIME_HISTORY_DAYS)
        self.uptime_history = [
            r for r in self.uptime_history
            if _parse_timestamp(r.timestamp) > cutoff
        ]

    def get_uptime_percentage(self, days: int = 30) -> float:
        """Calculate uptime percentage over the specified period.

        Args:
            days: Number of days to calculate uptime for.

        Returns:
            Uptime percentage (0-100).
        """
        if not self.uptime_history:
            return 100.0 if self.is_healthy else 0.0

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_records = [
            r for r in self.uptime_history
            if _parse_timestamp(r.timestamp) > cutoff
        ]

        if not recent_records:
            return 100.0 if self.is_healthy else 0.0

        healthy_count = sum(1 for r in recent_records if r.is_healthy)
        return (healthy_count / len(recent_records)) * 100

    def get_health_status(self) -> str:
        """Get the current health status string.

        Returns:
            'healthy', 'degraded', or 'broken'.
        """
        if self.is_healthy:
            return "healthy"

        uptime = self.get_uptime_percentage(days=7)
        if uptime >= 50:
            return "degraded"
        return "broken"


@dataclass
class MonitorState:
    """Complete state for all monitored datasets."""

    version: str = "1.0"
    last_run_timestamp: str = ""
    datasets: dict[str, DatasetState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "last_run_timestamp": self.last_run_timestamp,
            "datasets": {
                name: state.to_dict() for name, state in self.datasets.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MonitorState":
        """Create MonitorState from dictionary."""
        state = cls(
            version=data.get("version", "1.0"),
            last_run_timestamp=data.get("last_run_timestamp", ""),
        )
        for name, dataset_data in data.get("datasets", {}).items():
            state.datasets[name] = DatasetState.from_dict(dataset_data)
        return state


class StateStore:
    """Manages persistent state storage for dataset monitoring."""

    def __init__(self, state_file: Path):
        """Initialize state store.

        Args:
            state_file: Path to the JSON state file.
        """
        self.state_file = state_file
        self._state: Optional[MonitorState] = None

    def load(self) -> MonitorState:
        """Load state from file.

        Returns:
            MonitorState object (creates new if file doesn't exist).
        """
        if self._state is not None:
            return self._state

        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = MonitorState.from_dict(data)
                logger.info("Loaded state from %s", self.state_file)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to load state file, creating new: %s", e
                )
                self._state = MonitorState()
        else:
            logger.info("No state file found, creating new state")
            self._state = MonitorState()

        return self._state

    def save(self) -> None:
        """Save state to file."""
        if self._state is None:
            return

        self._state.last_run_timestamp = datetime.now(timezone.utc).isoformat()

        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2)

        logger.info("Saved state to %s", self.state_file)

    def get_dataset_state(self, name: str) -> Optional[DatasetState]:
        """Get state for a specific dataset.

        Args:
            name: Dataset name.

        Returns:
            DatasetState or None if not found.
        """
        state = self.load()
        return state.datasets.get(name)

    def update_dataset_state(
        self,
        name: str,
        is_healthy: bool,
        checksum: Optional[str] = None,
        file_sizes: Optional[dict[str, int]] = None,
        schema_hash: Optional[str] = None,
        schema: Optional[list[str]] = None,
        error: Optional[str] = None,
        issue_number: Optional[int] = None,
    ) -> DatasetState:
        """Update state for a specific dataset.

        Args:
            name: Dataset name.
            is_healthy: Whether the dataset passed all checks.
            checksum: Current checksum value.
            file_sizes: Dictionary of file sizes.
            schema_hash: Current schema hash.
            schema: Current schema (column names).
            error: Error message if unhealthy.
            issue_number: GitHub issue number if one was created.

        Returns:
            Updated DatasetState.
        """
        state = self.load()
        now = datetime.now(timezone.utc).isoformat()

        if name in state.datasets:
            dataset_state = state.datasets[name]
        else:
            dataset_state = DatasetState(name=name, last_check_timestamp=now)
            state.datasets[name] = dataset_state

        dataset_state.last_check_timestamp = now

        # Add uptime record for tracking
        dataset_state.add_uptime_record(is_healthy)

        if is_healthy:
            dataset_state.last_success_timestamp = now
            dataset_state.is_healthy = True
            dataset_state.last_error = None
            dataset_state.consecutive_failures = 0
        else:
            dataset_state.is_healthy = False
            dataset_state.last_error = error
            dataset_state.consecutive_failures += 1

        if checksum is not None:
            dataset_state.checksum = checksum
        if file_sizes is not None:
            dataset_state.file_sizes = file_sizes
        if schema_hash is not None:
            dataset_state.schema_hash = schema_hash
        if schema is not None:
            dataset_state.schema = schema
        if issue_number is not None:
            dataset_state.issue_number = issue_number

        return dataset_state

    def should_create_issue(self, name: str) -> bool:
        """Check if we should create a new issue for this dataset.

        Prevents spam by not creating issues for consecutive failures
        if an issue already exists.

        Args:
            name: Dataset name.

        Returns:
            True if a new issue should be created.
        """
        dataset_state = self.get_dataset_state(name)

        if dataset_state is None:
            # First time checking this dataset
            return True

        if dataset_state.is_healthy:
            # Was healthy before, now failing - create issue
            return True

        if dataset_state.issue_number is not None:
            # Already have an open issue
            return False

        # Failed before but no issue - create one
        return True

    def clear_issue(self, name: str) -> None:
        """Clear the issue number for a dataset (when issue is closed).

        Args:
            name: Dataset name.
        """
        state = self.load()
        if name in state.datasets:
            state.datasets[name].issue_number = None
