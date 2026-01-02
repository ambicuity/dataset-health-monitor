"""GitHub webhook handling module.

Handles incoming webhooks from GitHub and dispatches to appropriate handlers.
"""

import hashlib
import hmac
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Supported GitHub webhook events."""

    INSTALLATION = "installation"
    INSTALLATION_REPOSITORIES = "installation_repositories"
    REPOSITORY = "repository"
    WORKFLOW_RUN = "workflow_run"
    PUSH = "push"
    PING = "ping"


@dataclass
class WebhookPayload:
    """Parsed webhook payload."""

    event: str
    action: Optional[str]
    delivery_id: str
    installation_id: Optional[int]
    repository: Optional[dict]
    sender: Optional[dict]
    raw_payload: dict


class WebhookVerificationError(Exception):
    """Webhook signature verification failed."""

    pass


class WebhookHandler:
    """GitHub webhook handler.

    Verifies webhook signatures and dispatches events to handlers.
    """

    def __init__(self, webhook_secret: Optional[str] = None):
        """Initialize webhook handler.

        Args:
            webhook_secret: Secret for webhook signature verification.
        """
        self.webhook_secret = webhook_secret
        self._handlers: dict[str, list[Callable]] = {}

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature.

        Args:
            payload: Raw request body.
            signature: X-Hub-Signature-256 header value.

        Returns:
            True if signature is valid.

        Raises:
            WebhookVerificationError: If signature verification fails.
        """
        if not self.webhook_secret:
            logger.warning("Webhook secret not configured, skipping verification")
            return True

        if not signature:
            raise WebhookVerificationError("Missing X-Hub-Signature-256 header")

        if not signature.startswith("sha256="):
            raise WebhookVerificationError("Invalid signature format")

        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        actual_signature = signature[7:]  # Remove "sha256=" prefix

        if not hmac.compare_digest(expected_signature, actual_signature):
            raise WebhookVerificationError("Signature verification failed")

        return True

    def parse_payload(
        self,
        event_type: str,
        delivery_id: str,
        payload: dict,
    ) -> WebhookPayload:
        """Parse webhook payload into structured format.

        Args:
            event_type: X-GitHub-Event header value.
            delivery_id: X-GitHub-Delivery header value.
            payload: Parsed JSON payload.

        Returns:
            Parsed WebhookPayload.
        """
        installation = payload.get("installation", {})
        repository = payload.get("repository")
        sender = payload.get("sender")

        return WebhookPayload(
            event=event_type,
            action=payload.get("action"),
            delivery_id=delivery_id,
            installation_id=installation.get("id") if installation else None,
            repository=repository,
            sender=sender,
            raw_payload=payload,
        )

    def register_handler(self, event: str, handler: Callable[[WebhookPayload], Any]) -> None:
        """Register a handler for a webhook event.

        Args:
            event: Event type to handle (e.g., "installation", "push").
            handler: Callback function to handle the event.
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
        logger.debug("Registered handler for event: %s", event)

    def handle_event(self, payload: WebhookPayload) -> list[Any]:
        """Dispatch webhook event to registered handlers.

        Args:
            payload: Parsed webhook payload.

        Returns:
            List of handler results.
        """
        handlers = self._handlers.get(payload.event, [])
        if not handlers:
            logger.debug("No handlers registered for event: %s", payload.event)
            return []

        results = []
        for handler in handlers:
            try:
                result = handler(payload)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Handler error for event %s: %s",
                    payload.event,
                    e,
                    exc_info=True,
                )
                results.append({"error": str(e)})

        return results

    def on(self, event: str) -> Callable:
        """Decorator to register a webhook handler.

        Usage:
            @handler.on("installation")
            def handle_installation(payload: WebhookPayload):
                ...

        Args:
            event: Event type to handle.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable) -> Callable:
            self.register_handler(event, func)
            return func

        return decorator


def create_default_handlers(handler: WebhookHandler) -> None:
    """Register default webhook handlers.

    Args:
        handler: WebhookHandler to register handlers on.
    """

    @handler.on("ping")
    def handle_ping(payload: WebhookPayload) -> dict:
        """Handle ping event (GitHub App verification)."""
        logger.info("Received ping event from GitHub")
        return {"status": "ok", "zen": payload.raw_payload.get("zen")}

    @handler.on("installation")
    def handle_installation(payload: WebhookPayload) -> dict:
        """Handle app installation/uninstallation events."""
        action = payload.action
        installation_id = payload.installation_id
        sender = payload.sender.get("login") if payload.sender else "unknown"

        if action == "created":
            logger.info(
                "App installed by %s (installation_id: %d)",
                sender,
                installation_id,
            )
            return {
                "status": "installed",
                "installation_id": installation_id,
                "action": action,
            }
        elif action == "deleted":
            logger.info(
                "App uninstalled by %s (installation_id: %d)",
                sender,
                installation_id,
            )
            return {
                "status": "uninstalled",
                "installation_id": installation_id,
                "action": action,
            }
        else:
            logger.info(
                "Installation event: %s (installation_id: %d)",
                action,
                installation_id,
            )
            return {"status": "ok", "action": action}

    @handler.on("installation_repositories")
    def handle_installation_repositories(payload: WebhookPayload) -> dict:
        """Handle repository added/removed from installation."""
        action = payload.action
        installation_id = payload.installation_id

        repos_added = payload.raw_payload.get("repositories_added", [])
        repos_removed = payload.raw_payload.get("repositories_removed", [])

        if repos_added:
            repo_names = [r["full_name"] for r in repos_added]
            logger.info(
                "Repositories added to installation %d: %s",
                installation_id,
                repo_names,
            )

        if repos_removed:
            repo_names = [r["full_name"] for r in repos_removed]
            logger.info(
                "Repositories removed from installation %d: %s",
                installation_id,
                repo_names,
            )

        return {
            "status": "ok",
            "action": action,
            "repositories_added": len(repos_added),
            "repositories_removed": len(repos_removed),
        }

    @handler.on("workflow_run")
    def handle_workflow_run(payload: WebhookPayload) -> dict:
        """Handle workflow run events."""
        action = payload.action
        workflow = payload.raw_payload.get("workflow_run", {})
        repo = payload.repository

        workflow_name = workflow.get("name", "unknown")
        conclusion = workflow.get("conclusion")
        repo_name = repo.get("full_name") if repo else "unknown"

        if action == "completed":
            logger.info(
                "Workflow '%s' completed in %s with conclusion: %s",
                workflow_name,
                repo_name,
                conclusion,
            )

            # Could trigger dataset health check on workflow completion
            return {
                "status": "ok",
                "workflow": workflow_name,
                "conclusion": conclusion,
                "repository": repo_name,
            }

        return {"status": "ok", "action": action}
