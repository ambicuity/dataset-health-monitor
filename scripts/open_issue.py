"""GitHub Issue management module for dataset health monitoring.

This module provides functionality to create, update, and close
GitHub issues based on dataset health check results.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"


@dataclass
class IssueResult:
    """Result of an issue operation."""

    success: bool
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    error_message: Optional[str] = None


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment.

    Returns:
        GitHub token or None if not set.
    """
    return os.environ.get("GITHUB_TOKEN")


def get_repo_info() -> tuple[Optional[str], Optional[str]]:
    """Get repository owner and name from environment.

    Returns:
        Tuple of (owner, repo) or (None, None) if not set.
    """
    repo_full = os.environ.get("GITHUB_REPOSITORY")
    if repo_full and "/" in repo_full:
        parts = repo_full.split("/", 1)
        return parts[0], parts[1]
    return None, None


def create_issue(
    title: str,
    body: str,
    labels: Optional[list[str]] = None,
    token: Optional[str] = None,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
) -> IssueResult:
    """Create a new GitHub issue.

    Args:
        title: Issue title.
        body: Issue body (Markdown supported).
        labels: List of labels to apply.
        token: GitHub token (uses env var if not provided).
        owner: Repository owner (uses env var if not provided).
        repo: Repository name (uses env var if not provided).

    Returns:
        IssueResult with operation details.
    """
    token = token or get_github_token()
    if not token:
        return IssueResult(
            success=False,
            error_message="GITHUB_TOKEN environment variable not set",
        )

    if owner is None or repo is None:
        owner, repo = get_repo_info()
    if not owner or not repo:
        return IssueResult(
            success=False,
            error_message="GITHUB_REPOSITORY environment variable not set",
        )

    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues"

    payload = {
        "title": title,
        "body": body,
    }
    if labels:
        payload["labels"] = labels

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        logger.info("Creating issue: %s", title)
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        issue_number = data["number"]
        issue_url = data["html_url"]

        logger.info("✓ Created issue #%d: %s", issue_number, issue_url)

        return IssueResult(
            success=True,
            issue_number=issue_number,
            issue_url=issue_url,
        )

    except RequestException as e:
        error_msg = f"Failed to create issue: {e}"
        logger.error("✗ %s", error_msg)
        return IssueResult(success=False, error_message=error_msg)


def close_issue(
    issue_number: int,
    comment: Optional[str] = None,
    token: Optional[str] = None,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
) -> IssueResult:
    """Close a GitHub issue with an optional comment.

    Args:
        issue_number: Issue number to close.
        comment: Optional comment to add before closing.
        token: GitHub token (uses env var if not provided).
        owner: Repository owner (uses env var if not provided).
        repo: Repository name (uses env var if not provided).

    Returns:
        IssueResult with operation details.
    """
    token = token or get_github_token()
    if not token:
        return IssueResult(
            success=False,
            error_message="GITHUB_TOKEN environment variable not set",
        )

    if owner is None or repo is None:
        owner, repo = get_repo_info()
    if not owner or not repo:
        return IssueResult(
            success=False,
            error_message="GITHUB_REPOSITORY environment variable not set",
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        # Add comment if provided
        if comment:
            comment_url = (
                f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
            )
            comment_response = requests.post(
                comment_url,
                json={"body": comment},
                headers=headers,
                timeout=30,
            )
            comment_response.raise_for_status()
            logger.info("✓ Added comment to issue #%d", issue_number)

        # Close the issue
        issue_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.patch(
            issue_url,
            json={"state": "closed"},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        logger.info("✓ Closed issue #%d", issue_number)

        return IssueResult(
            success=True,
            issue_number=issue_number,
            issue_url=response.json().get("html_url"),
        )

    except RequestException as e:
        error_msg = f"Failed to close issue: {e}"
        logger.error("✗ %s", error_msg)
        return IssueResult(success=False, error_message=error_msg)


def find_open_issue(
    dataset_name: str,
    token: Optional[str] = None,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
) -> Optional[int]:
    """Find an open issue for a specific dataset.

    Args:
        dataset_name: Name of the dataset.
        token: GitHub token (uses env var if not provided).
        owner: Repository owner (uses env var if not provided).
        repo: Repository name (uses env var if not provided).

    Returns:
        Issue number if found, None otherwise.
    """
    token = token or get_github_token()
    if not token:
        return None

    if owner is None or repo is None:
        owner, repo = get_repo_info()
    if not owner or not repo:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Search for open issues with the dataset name in the title
    search_url = f"{GITHUB_API_URL}/search/issues"
    query = f"repo:{owner}/{repo} is:issue is:open \"{dataset_name}\" in:title"

    try:
        response = requests.get(
            search_url,
            params={"q": query},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        if data["total_count"] > 0:
            return data["items"][0]["number"]

    except RequestException as e:
        logger.warning("Failed to search for existing issues: %s", e)

    return None


def format_issue_body(
    dataset_name: str,
    source_url: str,
    errors: list[str],
    last_known_good_state: Optional[dict] = None,
) -> str:
    """Format the issue body with diagnostic information.

    Args:
        dataset_name: Name of the failing dataset.
        source_url: URL of the dataset.
        errors: List of error messages.
        last_known_good_state: Previous healthy state information.

    Returns:
        Formatted Markdown issue body.
    """
    body_parts = [
        "## Dataset Health Check Failed",
        "",
        f"**Dataset:** `{dataset_name}`",
        f"**URL:** {source_url}",
        "",
        "### Detected Issues",
        "",
    ]

    for error in errors:
        body_parts.append(f"- ❌ {error}")

    body_parts.append("")

    if last_known_good_state:
        body_parts.extend([
            "### Last Known Good State",
            "",
        ])

        if last_known_good_state.get("last_success_timestamp"):
            body_parts.append(
                f"- **Last Successful Check:** {last_known_good_state['last_success_timestamp']}"
            )
        if last_known_good_state.get("checksum"):
            body_parts.append(
                f"- **Previous Checksum:** `{last_known_good_state['checksum']}`"
            )
        if last_known_good_state.get("schema"):
            schema_str = ", ".join(last_known_good_state["schema"])
            body_parts.append(f"- **Previous Schema:** `{schema_str}`")

        body_parts.append("")

    body_parts.extend([
        "### Next Steps",
        "",
        "1. Verify the dataset URL is accessible",
        "2. Check if the dataset has been intentionally updated",
        "3. Update `datasets/datasets.yaml` if the change is expected",
        "4. This issue will be auto-closed when the dataset is healthy again",
        "",
        "---",
        "*This issue was automatically created by Dataset Health Monitor.*",
    ])

    return "\n".join(body_parts)


def get_labels_for_errors(errors: list[str]) -> list[str]:
    """Determine appropriate labels based on error types.

    Args:
        errors: List of error messages.

    Returns:
        List of label names.
    """
    labels = ["dataset-health"]

    error_text = " ".join(errors).lower()

    if "broken" in error_text or "404" in error_text or "timeout" in error_text:
        labels.append("dataset-broken")
    if "checksum" in error_text:
        labels.append("checksum-change")
    if "missing" in error_text:
        labels.append("missing-files")
    if "schema" in error_text:
        labels.append("schema-change")

    return labels
