"""Checksum validation module for datasets.

This module provides functionality to compute and verify
SHA256 checksums for remote files.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120
CHUNK_SIZE = 8192
MAX_DOWNLOAD_SIZE = 1024 * 1024 * 1024  # 1GB limit for checksum computation


@dataclass
class ChecksumResult:
    """Result of a checksum validation."""

    dataset_name: str
    url: str
    expected_checksum: Optional[str]
    computed_checksum: Optional[str] = None
    is_valid: bool = False
    checksum_changed: bool = False
    error_message: Optional[str] = None
    file_size: Optional[int] = None


def compute_checksum(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_size: int = MAX_DOWNLOAD_SIZE,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Compute SHA256 checksum of a remote file.

    Args:
        url: URL of the file to checksum.
        timeout: Download timeout in seconds.
        max_size: Maximum file size to download.

    Returns:
        Tuple of (checksum, file_size, error_message).
    """
    try:
        logger.debug("Computing checksum for: %s", url)

        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "DatasetHealthMonitor/1.0"},
        )
        response.raise_for_status()

        sha256_hash = hashlib.sha256()
        downloaded = 0

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            # Check size limit before processing to avoid partial results
            if downloaded + len(chunk) > max_size:
                return None, downloaded, f"File exceeds maximum size ({max_size} bytes)"
            sha256_hash.update(chunk)
            downloaded += len(chunk)

        checksum = f"sha256:{sha256_hash.hexdigest()}"
        return checksum, downloaded, None

    except RequestException as e:
        return None, None, f"Download failed: {e}"


def verify_checksum(
    url: str,
    expected_checksum: Optional[str],
    dataset_name: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> ChecksumResult:
    """Verify the checksum of a remote file.

    Args:
        url: URL of the file to verify.
        expected_checksum: Expected checksum in format 'sha256:hexdigest'.
        dataset_name: Name of the dataset for logging.
        timeout: Download timeout in seconds.

    Returns:
        ChecksumResult with validation details.
    """
    result = ChecksumResult(
        dataset_name=dataset_name,
        url=url,
        expected_checksum=expected_checksum,
    )

    computed, file_size, error = compute_checksum(url, timeout=timeout)

    if error:
        result.error_message = error
        logger.error("✗ Checksum computation failed for %s: %s", dataset_name, error)
        return result

    result.computed_checksum = computed
    result.file_size = file_size

    if expected_checksum is None:
        # No expected checksum, just record the computed one
        result.is_valid = True
        logger.info(
            "✓ Computed checksum for %s: %s (no expected value to compare)",
            dataset_name,
            computed,
        )
        return result

    # Normalize checksum format
    expected_normalized = expected_checksum
    if not expected_normalized.startswith("sha256:"):
        expected_normalized = f"sha256:{expected_normalized}"

    if computed == expected_normalized:
        result.is_valid = True
        logger.info("✓ Checksum verified for %s", dataset_name)
    else:
        result.is_valid = False
        result.checksum_changed = True
        logger.error(
            "✗ Checksum mismatch for %s:\n  Expected: %s\n  Computed: %s",
            dataset_name,
            expected_normalized,
            computed,
        )

    return result


def format_checksum(checksum: str) -> str:
    """Format a checksum for display (truncate if too long).

    Args:
        checksum: Full checksum string.

    Returns:
        Truncated checksum for display.
    """
    if checksum.startswith("sha256:"):
        hex_part = checksum[7:]
        if len(hex_part) > 16:
            return f"sha256:{hex_part[:8]}...{hex_part[-8:]}"
    return checksum
