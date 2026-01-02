"""State persistence module for dataset health monitoring.

This module manages the persistent state of dataset health checks,
tracking checksums, file sizes, schema hashes, and timestamps.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetState":
        """Create DatasetState from dictionary."""
        return cls(**data)


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
