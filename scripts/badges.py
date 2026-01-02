"""Badge generation module for dataset health monitoring.

This module generates shields.io-compatible badge JSON files
for displaying dataset uptime and health status.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.state_store import DatasetState, StateStore

logger = logging.getLogger(__name__)


@dataclass
class BadgeConfig:
    """Configuration for a shields.io badge."""

    schemaVersion: int
    label: str
    message: str
    color: str
    namedLogo: Optional[str] = None
    logoColor: Optional[str] = None


def get_uptime_color(uptime_percentage: float) -> str:
    """Get the appropriate color for an uptime percentage.

    Args:
        uptime_percentage: Uptime percentage (0-100).

    Returns:
        Color string for shields.io.
    """
    if uptime_percentage >= 99:
        return "brightgreen"
    if uptime_percentage >= 95:
        return "green"
    if uptime_percentage >= 90:
        return "yellowgreen"
    if uptime_percentage >= 80:
        return "yellow"
    if uptime_percentage >= 70:
        return "orange"
    return "red"


def get_status_color(status: str) -> str:
    """Get the appropriate color for a health status.

    Args:
        status: Health status ('healthy', 'degraded', 'broken').

    Returns:
        Color string for shields.io.
    """
    colors = {
        "healthy": "brightgreen",
        "degraded": "yellow",
        "broken": "red",
    }
    return colors.get(status, "lightgrey")


def generate_uptime_badge(dataset_state: DatasetState, dataset_name: str, days: int = 30) -> BadgeConfig:
    """Generate an uptime badge for a dataset.

    Args:
        dataset_state: The dataset state containing uptime history.
        dataset_name: Name of the dataset (used as fallback).
        days: Number of days to calculate uptime for.

    Returns:
        BadgeConfig for shields.io endpoint badge.
    """
    uptime = dataset_state.get_uptime_percentage(days=days)
    color = get_uptime_color(uptime)
    name = dataset_state.name or dataset_name

    return BadgeConfig(
        schemaVersion=1,
        label=f"{name} uptime",
        message=f"{uptime:.1f}%",
        color=color,
        namedLogo="databricks",
        logoColor="white",
    )


def generate_status_badge(dataset_state: DatasetState, dataset_name: str) -> BadgeConfig:
    """Generate a health status badge for a dataset.

    Args:
        dataset_state: The dataset state.
        dataset_name: Name of the dataset (used as fallback).

    Returns:
        BadgeConfig for shields.io endpoint badge.
    """
    status = dataset_state.get_health_status()
    color = get_status_color(status)
    name = dataset_state.name or dataset_name

    return BadgeConfig(
        schemaVersion=1,
        label=name,
        message=status,
        color=color,
        namedLogo="databricks",
        logoColor="white",
    )


def save_badge_json(badge: BadgeConfig, output_path: Path) -> None:
    """Save a badge configuration to a JSON file.

    Args:
        badge: Badge configuration.
        output_path: Path to save the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    badge_dict = {
        "schemaVersion": badge.schemaVersion,
        "label": badge.label,
        "message": badge.message,
        "color": badge.color,
    }
    if badge.namedLogo:
        badge_dict["namedLogo"] = badge.namedLogo
    if badge.logoColor:
        badge_dict["logoColor"] = badge.logoColor

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(badge_dict, f, indent=2)

    logger.info("Saved badge to %s", output_path)


def generate_all_badges(state_store: StateStore, badges_dir: Path) -> None:
    """Generate badges for all datasets in the state store.

    Args:
        state_store: State store containing dataset states.
        badges_dir: Directory to save badge JSON files.
    """
    state = state_store.load()

    if not state.datasets:
        logger.warning("No datasets found in state, skipping badge generation")
        return

    badges_dir.mkdir(parents=True, exist_ok=True)

    for name, dataset_state in state.datasets.items():
        # Generate uptime badge
        uptime_badge = generate_uptime_badge(dataset_state, name)
        uptime_path = badges_dir / f"{name}-uptime.json"
        save_badge_json(uptime_badge, uptime_path)

        # Generate status badge
        status_badge = generate_status_badge(dataset_state, name)
        status_path = badges_dir / f"{name}-status.json"
        save_badge_json(status_badge, status_path)

    # Generate summary badge for all datasets
    total_datasets = len(state.datasets)
    healthy_count = sum(1 for ds in state.datasets.values() if ds.is_healthy)

    if total_datasets > 0:
        overall_health = (healthy_count / total_datasets) * 100
        overall_color = get_uptime_color(overall_health)
    else:
        overall_health = 100.0
        overall_color = "lightgrey"

    summary_badge = BadgeConfig(
        schemaVersion=1,
        label="datasets health",
        message=f"{healthy_count}/{total_datasets} healthy",
        color=overall_color,
        namedLogo="databricks",
        logoColor="white",
    )
    save_badge_json(summary_badge, badges_dir / "summary.json")

    logger.info(
        "Generated %d badges for %d datasets",
        total_datasets * 2 + 1,
        total_datasets,
    )


def get_badge_url(repo_url: str, badge_name: str) -> str:
    """Generate a shields.io badge URL for a dataset.

    Args:
        repo_url: GitHub repository URL (e.g., 'owner/repo').
        badge_name: Name of the badge file (without .json extension).

    Returns:
        Shields.io badge URL.
    """
    # Use GitHub raw URL for the badge JSON
    raw_url = f"https://raw.githubusercontent.com/{repo_url}/main/badges/{badge_name}.json"
    return f"https://img.shields.io/endpoint?url={raw_url}"


def get_badge_markdown(repo_url: str, dataset_name: str) -> str:
    """Generate markdown for dataset badges.

    Args:
        repo_url: GitHub repository URL (e.g., 'owner/repo').
        dataset_name: Name of the dataset.

    Returns:
        Markdown string with badge images.
    """
    uptime_url = get_badge_url(repo_url, f"{dataset_name}-uptime")
    status_url = get_badge_url(repo_url, f"{dataset_name}-status")

    return f"![{dataset_name} uptime]({uptime_url}) ![{dataset_name} status]({status_url})"
