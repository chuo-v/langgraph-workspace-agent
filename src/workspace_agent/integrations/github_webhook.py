import hashlib
import hmac
import logging
import os
import re
from typing import Any

from src.workspace_agent.core.config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "verify_github_signature",
    "parse_github_pr_action",
    "parse_agentic_ci_trigger",
]


# ==========================================
# Webhook Authentication
# ==========================================


def verify_github_signature(payload_body: bytes, x_hub_signature_256: str | None) -> bool:
    """Validates that the incoming webhook payload matches the configured GITHUB_WEBHOOK_SECRET.

    Args:
        payload_body (bytes): The raw request body bytes from the incoming GitHub webhook.
        x_hub_signature_256 (str | None): The value of the 'X-Hub-Signature-256' header.

    Returns:
        bool: True if the HMAC SHA-256 signature is valid; False if verification fails,
        if the signature header is missing, or if the secret is unconfigured.
    """
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        logger.error("GITHUB_WEBHOOK_SECRET is missing. Rejecting webhook to prevent spoofing.")
        return False

    if not x_hub_signature_256:
        return False

    expected_signature = (
        "sha256=" + hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected_signature, x_hub_signature_256)


# ==========================================
# Pull Request Lifecycle
# ==========================================


def parse_github_pr_action(payload: dict[str, Any]) -> str | None:
    """Analyzes the Pull Request event payload to handle agent branch lifecycle state transitions.

    Args:
        payload (dict[str, Any]): The raw JSON payload dictionary from the GitHub webhook event.

    Returns:
        str | None:
            - 'LGTM' if an agent-generated branch ('agent/*') was successfully merged.
            - 'abort' if an agent-generated branch was closed unmerged.
            - None if the event is not a PR closure or does not target an agent branch.
    """
    if payload.get("action") != "closed" or "pull_request" not in payload:
        return None

    pr = payload["pull_request"]
    branch_name = pr.get("head", {}).get("ref", "")

    # verify this is an agent-generated branch
    if not branch_name.startswith("agent/"):
        return None

    if pr.get("merged") is True:
        logger.info(f"GitHub Webhook: PR for {branch_name} merged successfully.")
        return "LGTM"
    else:
        logger.info(f"GitHub Webhook: PR for {branch_name} closed unmerged.")
        return "abort"


# ==========================================
# CI/CD & ChatOps Triggers
# ==========================================


def parse_agentic_ci_trigger(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Analyzes Webhook payloads to determine if an Agentic CI/CD run should be triggered.

    Evaluates both automatic PR events ('opened', 'synchronize') and manual ChatOps commands
    in PR comments (e.g., '@chatops retest'), enforcing strict user authorization guards.

    Args:
        payload (dict[str, Any]): The raw JSON payload dictionary from the GitHub webhook event.

    Returns:
        dict[str, Any] | None: A dictionary containing essential PR metadata (repo, PR number,
        commit SHA, target branch, or ChatOps flag) if a valid run is triggered; otherwise, None.
    """
    allowed_users = settings.agent.allowed_github_users
    chatops_name = settings.agent.chatops_name

    # 1. Base Security Guards
    if not allowed_users or not chatops_name:
        if not allowed_users:
            logger.error(
                "Security: 'allowed_github_users' is empty in config.yaml. Rejecting webhook."
            )
        if not chatops_name:
            logger.error("Security: 'chatops_name' is empty in config.yaml. Rejecting webhook.")
        return None

    base_name = chatops_name.lower()
    repo_info = payload.get("repository", {})
    repo_full_name = repo_info.get("full_name")
    repo_name = repo_info.get("name")

    # 2. Automatic Triggers (PR Opened or Synchronize)
    is_pr_action = "pull_request" in payload and payload.get("action") in ["opened", "synchronize"]

    if is_pr_action:
        pr = payload["pull_request"]
        pr_author = pr.get("user", {}).get("login")

        if pr_author in allowed_users:
            return {
                "repo_full_name": repo_full_name,
                "repo_name": repo_name,
                "pr_number": pr.get("number"),
                "commit_sha": pr.get("head", {}).get("sha"),
                "target_branch": pr.get("base", {}).get("ref"),
            }

        logger.warning(f"Security: Ignored CI trigger from unauthorized PR author: {pr_author}")

    # 3. ChatOps Triggers (Manual comments on PRs)
    elif (
        "issue" in payload
        and "comment" in payload
        and payload.get("action") == "created"
        and "pull_request" in payload["issue"]
    ):
        comment_author = payload["comment"].get("user", {}).get("login")

        # Reject unauthorized users immediately
        if comment_author not in allowed_users:
            logger.warning(
                f"Security: Ignored ChatOps command from unauthorized user: {comment_author}"
            )
            return None

        comment_body = payload["comment"].get("body", "").lower()
        github_username = os.getenv("GITHUB_USERNAME", "").lower()

        # Build trigger targets dynamically
        target_tags = [f"@{base_name}"]
        if github_username:
            target_tags.append(f"@{base_name}-{github_username}")

        tag_pattern = "|".join(target_tags)
        trigger_pattern = rf"(^|\s)({tag_pattern})\s+/?retest($|\s)"

        if re.search(trigger_pattern, comment_body):
            return {
                "repo_full_name": repo_full_name,
                "repo_name": repo_name,
                "pr_number": payload["issue"].get("number"),
                "is_chatops": True,
            }

    return None
