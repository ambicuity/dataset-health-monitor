"""Schema validation module for CSV and JSON datasets.

This module provides functionality to detect and compare
schema changes in CSV and JSON files.
"""

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from difflib import unified_diff
from io import StringIO
from typing import Any, Optional

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
MAX_SAMPLE_SIZE = 10 * 1024 * 1024  # 10MB for schema detection


@dataclass
class SchemaCheckResult:
    """Result of a schema validation check."""

    dataset_name: str
    url: str
    file_type: Optional[str] = None
    current_schema: Optional[list[str]] = None
    expected_schema: Optional[list[str]] = None
    schema_hash: Optional[str] = None
    is_valid: bool = False
    schema_changed: bool = False
    schema_diff: list[str] = field(default_factory=list)
    error_message: Optional[str] = None


def detect_file_type(url: str) -> Optional[str]:
    """Detect file type from URL extension.

    Args:
        url: URL to analyze.

    Returns:
        File type ('csv', 'json') or None if unsupported.
    """
    lower_url = url.lower()
    if lower_url.endswith(".csv"):
        return "csv"
    if lower_url.endswith(".json") or lower_url.endswith(".jsonl"):
        return "json"
    return None


def compute_schema_hash(schema: list[str]) -> str:
    """Compute a hash of the schema for comparison.

    Args:
        schema: List of column/field names.

    Returns:
        SHA256 hash of the schema.
    """
    schema_str = "|".join(sorted(schema))
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def extract_csv_schema(content: str) -> list[str]:
    """Extract column headers from CSV content.

    Args:
        content: CSV file content.

    Returns:
        List of column headers.
    """
    reader = csv.reader(StringIO(content))
    try:
        headers = next(reader)
        return [h.strip() for h in headers]
    except StopIteration:
        return []


def extract_json_schema(content: str) -> list[str]:
    """Extract field names from JSON content.

    Handles both JSON arrays and single objects.
    For JSONL, uses the first line.

    Args:
        content: JSON file content.

    Returns:
        List of field names.
    """
    try:
        # Try parsing as regular JSON first
        data = json.loads(content)

        if isinstance(data, list) and len(data) > 0:
            # Array of objects - use first object
            first_item = data[0]
            if isinstance(first_item, dict):
                return list(first_item.keys())
        elif isinstance(data, dict):
            return list(data.keys())

    except json.JSONDecodeError:
        # Try JSONL format (one JSON object per line)
        lines = content.strip().split("\n")
        if lines:
            try:
                first_obj = json.loads(lines[0])
                if isinstance(first_obj, dict):
                    return list(first_obj.keys())
            except json.JSONDecodeError:
                pass

    return []


def check_schema(
    url: str,
    dataset_name: str,
    expected_schema: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> SchemaCheckResult:
    """Check the schema of a remote CSV or JSON file.

    Args:
        url: URL of the file to check.
        dataset_name: Name of the dataset for logging.
        expected_schema: Expected column/field names.
        timeout: Download timeout in seconds.

    Returns:
        SchemaCheckResult with validation details.
    """
    result = SchemaCheckResult(
        dataset_name=dataset_name,
        url=url,
        expected_schema=expected_schema,
    )

    # Detect file type
    file_type = detect_file_type(url)
    if file_type is None:
        result.is_valid = True
        logger.debug("Skipping schema check for non-CSV/JSON file: %s", url)
        return result

    result.file_type = file_type

    try:
        # Download file content (with size limit)
        logger.debug("Fetching file for schema check: %s", url)

        response = requests.get(
            url,
            timeout=timeout,
            stream=True,
            headers={"User-Agent": "DatasetHealthMonitor/1.0"},
        )
        response.raise_for_status()

        # Read up to MAX_SAMPLE_SIZE as binary, then decode
        content_bytes = b""
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content_bytes += chunk
                downloaded += len(chunk)
                if downloaded >= MAX_SAMPLE_SIZE:
                    break
                # For schema detection, we only need the first few lines
                if b"\n" in content_bytes and downloaded > 1024:
                    break

        # Decode with error handling - try common encodings
        content = ""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                content = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Fallback: decode with replacement characters
            content = content_bytes.decode("utf-8", errors="replace")

        # Extract schema based on file type
        if file_type == "csv":
            result.current_schema = extract_csv_schema(content)
        elif file_type == "json":
            result.current_schema = extract_json_schema(content)

        if not result.current_schema:
            result.error_message = "Could not extract schema from file"
            logger.warning("✗ Schema extraction failed for %s", dataset_name)
            return result

        result.schema_hash = compute_schema_hash(result.current_schema)

        # Compare with expected schema
        if expected_schema is None:
            result.is_valid = True
            logger.info(
                "✓ Schema extracted for %s: %s (no expected value to compare)",
                dataset_name,
                result.current_schema,
            )
            return result

        # Check that all expected fields are present (subset check)
        # This allows for additional fields to be present
        expected_set = set(expected_schema)
        current_set = set(result.current_schema)
        missing_fields = expected_set - current_set

        if not missing_fields:
            result.is_valid = True
            logger.info("✓ Schema validated for %s", dataset_name)
            if current_set - expected_set:
                logger.debug(
                    "Note: %s has additional fields: %s",
                    dataset_name,
                    current_set - expected_set,
                )
        else:
            result.is_valid = False
            result.schema_changed = True

            # Generate diff
            diff = list(
                unified_diff(
                    sorted(expected_schema),
                    sorted(result.current_schema),
                    fromfile="expected",
                    tofile="current",
                    lineterm="",
                )
            )
            result.schema_diff = diff

            logger.error(
                "✗ Schema mismatch for %s:\n  Missing fields: %s\n  Current: %s",
                dataset_name,
                missing_fields,
                result.current_schema,
            )

    except RequestException as e:
        result.error_message = f"Download failed: {e}"
        logger.error("✗ Schema check failed for %s: %s", dataset_name, e)

    return result
