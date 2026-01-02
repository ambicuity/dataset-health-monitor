"""Link validation module for dataset URLs.

This module provides functionality to validate HTTP/HTTPS URLs,
checking for broken links, timeouts, and server errors.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_RETRY_COUNT = 3


@dataclass
class LinkCheckResult:
    """Result of a link validation check."""

    url: str
    is_valid: bool
    status_code: Optional[int] = None
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


def check_link(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRY_COUNT,
    verify_ssl: bool = True,
) -> LinkCheckResult:
    """Validate a URL and return detailed check results.

    Args:
        url: The URL to validate.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on failure.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        LinkCheckResult with validation details.
    """
    last_error: Optional[str] = None

    for attempt in range(retries):
        try:
            logger.debug(
                "Checking URL: %s (attempt %d/%d)", url, attempt + 1, retries
            )

            response = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                verify=verify_ssl,
                headers={"User-Agent": "DatasetHealthMonitor/1.0"},
            )

            # Calculate response time
            response_time_ms = response.elapsed.total_seconds() * 1000

            # Some servers don't support HEAD, try GET if we get 405
            if response.status_code == 405:
                response = requests.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    verify=verify_ssl,
                    stream=True,
                    headers={"User-Agent": "DatasetHealthMonitor/1.0"},
                )
                response_time_ms = response.elapsed.total_seconds() * 1000

            is_valid = 200 <= response.status_code < 400

            content_length = response.headers.get("Content-Length")
            if content_length:
                content_length = int(content_length)

            return LinkCheckResult(
                url=url,
                is_valid=is_valid,
                status_code=response.status_code,
                content_length=content_length,
                content_type=response.headers.get("Content-Type"),
                error_message=None if is_valid else f"HTTP {response.status_code}",
                response_time_ms=response_time_ms,
            )

        except Timeout:
            last_error = f"Request timed out after {timeout}s"
            logger.warning("Timeout checking %s: %s", url, last_error)

        except RequestException as e:
            last_error = str(e)
            logger.warning(
                "Request error checking %s (attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                last_error,
            )

    return LinkCheckResult(
        url=url,
        is_valid=False,
        error_message=last_error,
    )


def check_links(
    urls: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRY_COUNT,
) -> list[LinkCheckResult]:
    """Validate multiple URLs.

    Args:
        urls: List of URLs to validate.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on failure.

    Returns:
        List of LinkCheckResult objects.
    """
    results = []
    for url in urls:
        result = check_link(url, timeout=timeout, retries=retries)
        results.append(result)
        if result.is_valid:
            logger.info("✓ Link OK: %s", url)
        else:
            logger.error("✗ Link FAILED: %s - %s", url, result.error_message)
    return results
