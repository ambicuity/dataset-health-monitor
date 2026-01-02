"""Dataset Health Monitor GitHub App - FastAPI Application.

A GitHub App that monitors ML datasets for health and integrity issues.
Installable on any repository to automatically monitor datasets defined
in datasets.yaml.
"""

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import requests
import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.auth import (
    AppCredentials,
    GitHubAppAuth,
    GitHubAuthError,
    load_credentials_from_env,
)
from app.webhook import (
    WebhookHandler,
    WebhookPayload,
    WebhookVerificationError,
    create_default_handlers,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state
app_auth: Optional[GitHubAppAuth] = None
webhook_handler: Optional[WebhookHandler] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    app_name: Optional[str] = None


class WebhookResponse(BaseModel):
    """Webhook handling response."""

    status: str
    event: str
    action: Optional[str] = None
    results: Optional[list[Any]] = None


class InstallationInfo(BaseModel):
    """Installation information."""

    installation_id: int
    account: str
    repositories: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global app_auth, webhook_handler

    # Initialize on startup
    try:
        credentials = load_credentials_from_env()
        app_auth = GitHubAppAuth(credentials)
        webhook_handler = WebhookHandler(credentials.webhook_secret)
        create_default_handlers(webhook_handler)
        register_dataset_handlers(webhook_handler, app_auth)

        # Verify app credentials
        app_info = app_auth.get_app_info()
        logger.info("GitHub App initialized: %s (ID: %s)", app_info["name"], app_info["id"])

    except ValueError as e:
        logger.warning("GitHub App credentials not configured: %s", e)
        logger.info("Running in limited mode (no GitHub App features)")
    except GitHubAuthError as e:
        logger.error("Failed to initialize GitHub App: %s", e)
        raise

    yield

    # Cleanup on shutdown
    if app_auth:
        app_auth.clear_token_cache()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Dataset Health Monitor",
    description="GitHub App for monitoring ML dataset health and integrity",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    app_name = None
    if app_auth:
        try:
            app_info = app_auth.get_app_info()
            app_name = app_info.get("name")
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        app_name=app_name,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/webhook", response_model=WebhookResponse)
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str = Header(..., alias="X-GitHub-Event"),
    x_github_delivery: str = Header(..., alias="X-GitHub-Delivery"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
):
    """Handle incoming GitHub webhooks.

    Verifies signature, parses payload, and dispatches to handlers.
    """
    if not webhook_handler:
        raise HTTPException(
            status_code=503,
            detail="Webhook handler not initialized",
        )

    # Read raw body for signature verification
    body = await request.body()

    # Verify signature
    try:
        webhook_handler.verify_signature(body, x_hub_signature_256 or "")
    except WebhookVerificationError as e:
        logger.warning("Webhook verification failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e))

    # Parse payload
    try:
        payload_dict = await request.json()
    except Exception as e:
        logger.error("Failed to parse webhook payload: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    payload = webhook_handler.parse_payload(
        event_type=x_github_event,
        delivery_id=x_github_delivery,
        payload=payload_dict,
    )

    logger.info(
        "Received webhook: event=%s, action=%s, delivery=%s",
        payload.event,
        payload.action,
        payload.delivery_id,
    )

    # Handle event
    results = webhook_handler.handle_event(payload)

    return WebhookResponse(
        status="ok",
        event=payload.event,
        action=payload.action,
        results=results,
    )


@app.get("/installations")
async def list_installations():
    """List all installations of this GitHub App."""
    if not app_auth:
        raise HTTPException(
            status_code=503,
            detail="GitHub App not configured",
        )

    try:
        installations = app_auth.list_installations()
        result = []

        for inst in installations:
            account = inst.get("account", {})
            result.append(
                InstallationInfo(
                    installation_id=inst["id"],
                    account=account.get("login", "unknown"),
                    repositories=[],  # Would need separate API call per installation
                )
            )

        return {"installations": result, "total": len(result)}

    except GitHubAuthError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check/{installation_id}/{owner}/{repo}")
async def trigger_check(
    installation_id: int,
    owner: str,
    repo: str,
    background_tasks: BackgroundTasks,
):
    """Manually trigger a dataset health check for a repository.

    Args:
        installation_id: GitHub App installation ID.
        owner: Repository owner.
        repo: Repository name.
    """
    if not app_auth:
        raise HTTPException(
            status_code=503,
            detail="GitHub App not configured",
        )

    # Schedule check in background
    background_tasks.add_task(
        run_dataset_check,
        app_auth,
        installation_id,
        owner,
        repo,
    )

    return {
        "status": "scheduled",
        "repository": f"{owner}/{repo}",
        "installation_id": installation_id,
    }


def register_dataset_handlers(handler: WebhookHandler, auth: GitHubAppAuth) -> None:
    """Register dataset-specific webhook handlers.

    Args:
        handler: WebhookHandler instance.
        auth: GitHubAppAuth instance.
    """

    @handler.on("push")
    def handle_push(payload: WebhookPayload) -> dict:
        """Handle push events - check for datasets.yaml changes."""
        ref = payload.raw_payload.get("ref", "")
        commits = payload.raw_payload.get("commits", [])
        repo = payload.repository

        if not repo:
            return {"status": "skipped", "reason": "no repository"}

        # Check if datasets.yaml was modified
        datasets_modified = False
        for commit in commits:
            modified_files = (
                commit.get("modified", [])
                + commit.get("added", [])
            )
            if any("datasets.yaml" in f or "datasets.yml" in f for f in modified_files):
                datasets_modified = True
                break

        if datasets_modified:
            logger.info(
                "datasets.yaml modified in %s, scheduling health check",
                repo.get("full_name"),
            )
            # Could trigger a health check here
            return {
                "status": "ok",
                "action": "datasets_modified",
                "repository": repo.get("full_name"),
            }

        return {"status": "ok", "action": "no_dataset_changes"}


async def run_dataset_check(
    auth: GitHubAppAuth,
    installation_id: int,
    owner: str,
    repo: str,
) -> dict:
    """Run dataset health check for a repository.

    Args:
        auth: GitHub App authentication.
        installation_id: Installation ID.
        owner: Repository owner.
        repo: Repository name.

    Returns:
        Check results.
    """
    logger.info("Starting dataset check for %s/%s", owner, repo)

    try:
        # Get installation token
        token = auth.get_installation_token(installation_id)

        headers = {
            "Authorization": f"Bearer {token.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Fetch datasets.yaml from the repository
        config_url = f"https://api.github.com/repos/{owner}/{repo}/contents/datasets/datasets.yaml"
        response = requests.get(config_url, headers=headers, timeout=30)

        if response.status_code == 404:
            logger.warning("datasets/datasets.yaml not found in %s/%s", owner, repo)
            return {"status": "skipped", "reason": "no datasets.yaml"}

        if response.status_code != 200:
            logger.error(
                "Failed to fetch datasets.yaml: %d - %s",
                response.status_code,
                response.text,
            )
            return {"status": "error", "reason": "failed to fetch config"}

        # Decode content
        content_data = response.json()
        content = base64.b64decode(content_data["content"]).decode("utf-8")

        # Parse YAML
        config = yaml.safe_load(content)
        datasets = config.get("datasets", [])

        logger.info("Found %d datasets in %s/%s", len(datasets), owner, repo)

        # Run checks (simplified - in production would use full monitor.py logic)
        results = []
        for dataset in datasets:
            name = dataset.get("name", "unknown")
            source_url = dataset.get("source_url") or dataset.get("source", "")

            # Basic availability check
            check_result = {"name": name, "status": "healthy"}

            if source_url.startswith("http"):
                try:
                    resp = requests.head(source_url, timeout=10, allow_redirects=True)
                    if resp.status_code >= 400:
                        check_result["status"] = "broken"
                        check_result["error"] = f"HTTP {resp.status_code}"
                except requests.RequestException as e:
                    check_result["status"] = "broken"
                    check_result["error"] = str(e)

            results.append(check_result)

        # Report results
        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        broken_count = len(results) - healthy_count

        logger.info(
            "Dataset check complete for %s/%s: %d healthy, %d broken",
            owner,
            repo,
            healthy_count,
            broken_count,
        )

        # If there are broken datasets, create an issue
        if broken_count > 0:
            broken_datasets = [r for r in results if r["status"] != "healthy"]
            await create_issue_for_broken_datasets(
                auth,
                installation_id,
                owner,
                repo,
                broken_datasets,
            )

        return {
            "status": "completed",
            "total": len(results),
            "healthy": healthy_count,
            "broken": broken_count,
            "results": results,
        }

    except Exception as e:
        logger.error("Dataset check failed for %s/%s: %s", owner, repo, e)
        return {"status": "error", "reason": str(e)}


async def create_issue_for_broken_datasets(
    auth: GitHubAppAuth,
    installation_id: int,
    owner: str,
    repo: str,
    broken_datasets: list[dict],
) -> Optional[int]:
    """Create a GitHub issue for broken datasets.

    Args:
        auth: GitHub App authentication.
        installation_id: Installation ID.
        owner: Repository owner.
        repo: Repository name.
        broken_datasets: List of broken dataset check results.

    Returns:
        Issue number if created, None otherwise.
    """
    try:
        token = auth.get_installation_token(installation_id)

        headers = {
            "Authorization": f"Bearer {token.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Check for existing open issue
        search_url = "https://api.github.com/search/issues"
        search_query = f"repo:{owner}/{repo} is:issue is:open label:dataset-broken"
        search_response = requests.get(
            search_url,
            headers=headers,
            params={"q": search_query},
            timeout=30,
        )

        if search_response.status_code == 200:
            existing_issues = search_response.json().get("items", [])
            if existing_issues:
                logger.info("Existing dataset issue found, skipping creation")
                return existing_issues[0]["number"]

        # Create issue body
        body_lines = [
            "## 🔴 Dataset Health Check Failed",
            "",
            "The following datasets have issues:",
            "",
        ]

        for dataset in broken_datasets:
            name = dataset.get("name", "unknown")
            error = dataset.get("error", "Unknown error")
            body_lines.append(f"- **{name}**: {error}")

        body_lines.extend([
            "",
            "---",
            "*This issue was automatically created by [Dataset Health Monitor](https://github.com/ambicuity/dataset-health-monitor)*",
        ])

        # Create issue
        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        issue_data = {
            "title": "[Dataset Health] Health Check Failed",
            "body": "\n".join(body_lines),
            "labels": ["dataset-broken"],
        }

        response = requests.post(
            issue_url,
            headers=headers,
            json=issue_data,
            timeout=30,
        )

        if response.status_code == 201:
            issue_number = response.json()["number"]
            logger.info("Created issue #%d in %s/%s", issue_number, owner, repo)
            return issue_number
        else:
            logger.error(
                "Failed to create issue: %d - %s",
                response.status_code,
                response.text,
            )
            return None

    except Exception as e:
        logger.error("Failed to create issue: %s", e)
        return None


def main():
    """Run the application."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    logger.info("Starting Dataset Health Monitor on %s:%d", host, port)

    uvicorn.run(
        "app.app:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "").lower() == "true",
    )


if __name__ == "__main__":
    main()
