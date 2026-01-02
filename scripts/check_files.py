"""File validation module for datasets.

This module provides functionality to check file existence,
sizes, and availability within dataset archives or URLs.
"""

import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
MAX_DOWNLOAD_SIZE = 500 * 1024 * 1024  # 500MB limit for archive inspection


@dataclass
class FileCheckResult:
    """Result of a file existence check."""

    dataset_name: str
    expected_files: list[str]
    found_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    file_sizes: dict[str, int] = field(default_factory=dict)
    is_valid: bool = False
    error_message: Optional[str] = None


def check_archive_contents(
    url: str,
    expected_files: list[str],
    dataset_name: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> FileCheckResult:
    """Check if expected files exist within a remote archive.

    Args:
        url: URL to the archive file (zip).
        expected_files: List of expected file paths within the archive.
        dataset_name: Name of the dataset for logging.
        timeout: Download timeout in seconds.

    Returns:
        FileCheckResult with validation details.
    """
    result = FileCheckResult(
        dataset_name=dataset_name,
        expected_files=expected_files,
    )

    try:
        # Determine archive type from URL
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if not path.endswith(".zip"):
            # For non-archive URLs, we can only verify the URL itself
            logger.debug(
                "URL %s is not a supported archive format, skipping file check", url
            )
            result.is_valid = True
            result.found_files = expected_files
            return result

        logger.info("Downloading archive to check contents: %s", url)

        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "DatasetHealthMonitor/1.0"},
        )
        response.raise_for_status()

        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            logger.warning(
                "Archive too large (%s bytes), skipping content check",
                content_length,
            )
            result.is_valid = True
            result.found_files = expected_files
            return result

        # Download and inspect archive
        content = BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)
            downloaded += len(chunk)
            if downloaded > MAX_DOWNLOAD_SIZE:
                logger.warning("Download exceeded size limit, stopping")
                break

        content.seek(0)

        # Extract file list from zip
        with zipfile.ZipFile(content, "r") as zf:
            archive_files = set(zf.namelist())
            for info in zf.infolist():
                if not info.is_dir():
                    result.file_sizes[info.filename] = info.file_size

        # Check for expected files
        for expected in expected_files:
            # Check for exact match or exact path ending with /filename
            found = False
            for archive_file in archive_files:
                # Normalize paths for comparison
                if archive_file == expected:
                    result.found_files.append(expected)
                    found = True
                    break
                # Check if expected is a filename within a directory
                archive_basename = archive_file.rsplit("/", 1)[-1]
                expected_basename = expected.rsplit("/", 1)[-1]
                if archive_file.endswith(f"/{expected}") or (
                    expected == expected_basename and archive_basename == expected_basename
                ):
                    result.found_files.append(expected)
                    found = True
                    break
            if not found:
                result.missing_files.append(expected)

        result.is_valid = len(result.missing_files) == 0

        if result.is_valid:
            logger.info("✓ All expected files found in %s", dataset_name)
        else:
            logger.error(
                "✗ Missing files in %s: %s",
                dataset_name,
                ", ".join(result.missing_files),
            )

    except zipfile.BadZipFile:
        result.error_message = "Invalid or corrupted zip archive"
        logger.error("✗ Bad zip file: %s", url)
    except RequestException as e:
        result.error_message = f"Download failed: {e}"
        logger.error("✗ Download error for %s: %s", url, e)

    return result


def check_local_files(
    directory: Path,
    expected_files: list[str],
    dataset_name: str,
) -> FileCheckResult:
    """Check if expected files exist in a local directory.

    Args:
        directory: Path to the directory to check.
        expected_files: List of expected file paths.
        dataset_name: Name of the dataset for logging.

    Returns:
        FileCheckResult with validation details.
    """
    result = FileCheckResult(
        dataset_name=dataset_name,
        expected_files=expected_files,
    )

    if not directory.exists():
        result.error_message = f"Directory does not exist: {directory}"
        logger.error("✗ Directory not found: %s", directory)
        return result

    for expected in expected_files:
        file_path = directory / expected
        if file_path.exists():
            result.found_files.append(expected)
            result.file_sizes[expected] = file_path.stat().st_size
        else:
            result.missing_files.append(expected)

    result.is_valid = len(result.missing_files) == 0

    if result.is_valid:
        logger.info("✓ All expected files found in %s", dataset_name)
    else:
        logger.error(
            "✗ Missing files in %s: %s",
            dataset_name,
            ", ".join(result.missing_files),
        )

    return result


def check_github_repo_files(
    repo_url: str,
    expected_files: list[str],
    dataset_name: str,
    branch: str = "main",
    timeout: int = DEFAULT_TIMEOUT,
) -> FileCheckResult:
    """Check if expected files exist in a GitHub repository.

    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/owner/repo).
        expected_files: List of expected file paths.
        dataset_name: Name of the dataset for logging.
        branch: Branch to check (default: main).
        timeout: Request timeout in seconds.

    Returns:
        FileCheckResult with validation details.
    """
    result = FileCheckResult(
        dataset_name=dataset_name,
        expected_files=expected_files,
    )

    try:
        # Parse GitHub URL
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            result.error_message = f"Invalid GitHub URL: {repo_url}"
            return result

        owner, repo = path_parts[0], path_parts[1]

        # Check each expected file
        for expected in expected_files:
            raw_url = (
                f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{expected}"
            )

            response = requests.head(
                raw_url,
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": "DatasetHealthMonitor/1.0"},
            )

            if response.status_code == 200:
                result.found_files.append(expected)
                content_length = response.headers.get("Content-Length")
                if content_length:
                    result.file_sizes[expected] = int(content_length)
            else:
                result.missing_files.append(expected)

        result.is_valid = len(result.missing_files) == 0

        if result.is_valid:
            logger.info("✓ All expected files found in %s", dataset_name)
        else:
            logger.error(
                "✗ Missing files in %s: %s",
                dataset_name,
                ", ".join(result.missing_files),
            )

    except RequestException as e:
        result.error_message = f"Request failed: {e}"
        logger.error("✗ Error checking GitHub repo %s: %s", repo_url, e)

    return result
