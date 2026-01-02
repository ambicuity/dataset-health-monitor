"""GitHub App authentication module.

Handles JWT generation and installation token management for GitHub App authentication.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jwt
import requests

logger = logging.getLogger(__name__)

# GitHub API endpoints
GITHUB_API_BASE = "https://api.github.com"


@dataclass
class AppCredentials:
    """GitHub App credentials."""

    app_id: str
    private_key: str
    webhook_secret: Optional[str] = None


@dataclass
class InstallationToken:
    """GitHub App installation token with expiration."""

    token: str
    expires_at: str
    installation_id: int
    permissions: dict


class GitHubAppAuth:
    """GitHub App authentication handler.

    Manages JWT generation and installation token lifecycle.
    """

    def __init__(self, credentials: AppCredentials):
        """Initialize GitHub App authentication.

        Args:
            credentials: GitHub App credentials.
        """
        self.credentials = credentials
        self._installation_tokens: dict[int, InstallationToken] = {}

    def generate_jwt(self, expiration_minutes: int = 10) -> str:
        """Generate a JWT for GitHub App authentication.

        Args:
            expiration_minutes: Token expiration time in minutes (max 10).

        Returns:
            JWT token string.
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds in the past to account for clock drift
            "exp": now + (min(expiration_minutes, 10) * 60),
            "iss": self.credentials.app_id,
        }

        token = jwt.encode(payload, self.credentials.private_key, algorithm="RS256")
        logger.debug("Generated JWT for app %s", self.credentials.app_id)
        return token

    def get_installation_token(self, installation_id: int) -> InstallationToken:
        """Get an installation access token.

        Caches tokens and refreshes when expired.

        Args:
            installation_id: GitHub App installation ID.

        Returns:
            Installation access token.

        Raises:
            GitHubAuthError: If token generation fails.
        """
        # Check cache
        cached = self._installation_tokens.get(installation_id)
        if cached and not self._is_token_expired(cached):
            return cached

        # Generate new token
        jwt_token = self.generate_jwt()
        url = f"{GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )

        if response.status_code != 201:
            raise GitHubAuthError(
                f"Failed to get installation token: {response.status_code} - {response.text}"
            )

        data = response.json()
        token = InstallationToken(
            token=data["token"],
            expires_at=data["expires_at"],
            installation_id=installation_id,
            permissions=data.get("permissions", {}),
        )

        # Cache the token
        self._installation_tokens[installation_id] = token
        logger.info("Generated installation token for installation %d", installation_id)

        return token

    def _is_token_expired(self, token: InstallationToken) -> bool:
        """Check if an installation token is expired or about to expire.

        Args:
            token: Installation token to check.

        Returns:
            True if token is expired or expires within 5 minutes.
        """
        expires_at = datetime.fromisoformat(token.expires_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        # Consider expired if less than 5 minutes remaining
        return (expires_at - now).total_seconds() < 300

    def get_app_info(self) -> dict:
        """Get information about the GitHub App.

        Returns:
            App information dict.
        """
        jwt_token = self.generate_jwt()
        response = requests.get(
            f"{GITHUB_API_BASE}/app",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise GitHubAuthError(f"Failed to get app info: {response.status_code}")

        return response.json()

    def list_installations(self) -> list[dict]:
        """List all installations of the GitHub App.

        Returns:
            List of installation objects.
        """
        jwt_token = self.generate_jwt()
        response = requests.get(
            f"{GITHUB_API_BASE}/app/installations",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise GitHubAuthError(f"Failed to list installations: {response.status_code}")

        return response.json()

    def clear_token_cache(self) -> None:
        """Clear all cached installation tokens."""
        self._installation_tokens.clear()
        logger.info("Cleared installation token cache")


class GitHubAuthError(Exception):
    """GitHub authentication error."""

    pass


def load_credentials_from_env() -> AppCredentials:
    """Load GitHub App credentials from environment variables.

    Expected environment variables:
    - GITHUB_APP_ID: GitHub App ID
    - GITHUB_PRIVATE_KEY: PEM-encoded private key (or path to file)
    - GITHUB_WEBHOOK_SECRET: Webhook secret (optional)

    Returns:
        AppCredentials object.

    Raises:
        ValueError: If required credentials are missing.
    """
    app_id = os.environ.get("GITHUB_APP_ID")
    if not app_id:
        raise ValueError("GITHUB_APP_ID environment variable is required")

    private_key = os.environ.get("GITHUB_PRIVATE_KEY", "")
    private_key_path = os.environ.get("GITHUB_PRIVATE_KEY_PATH")

    if private_key_path:
        # Load from file
        key_path = Path(private_key_path)
        if not key_path.exists():
            raise ValueError(f"Private key file not found: {private_key_path}")
        private_key = key_path.read_text()
    elif not private_key:
        raise ValueError(
            "GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH environment variable is required"
        )

    # Handle escaped newlines in environment variable
    if "\\n" in private_key:
        private_key = private_key.replace("\\n", "\n")

    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET")

    return AppCredentials(
        app_id=app_id,
        private_key=private_key,
        webhook_secret=webhook_secret,
    )
