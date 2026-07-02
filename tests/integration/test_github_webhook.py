import hashlib
import hmac

from src.workspace_agent.integrations.github_webhook import (
    parse_agentic_ci_trigger,
    parse_github_pr_action,
    verify_github_signature,
)

# ==========================================
# Component: parse_github_pr_action
# ==========================================


def test_parse_github_pr_action_success_merged():
    """Green Path: Validates extraction of the 'LGTM' signal when an agent branch is merged."""
    mock_payload = {
        "action": "closed",
        "pull_request": {"merged": True, "head": {"ref": "agent/fix-typo-1234"}},
    }

    result = parse_github_pr_action(mock_payload)
    assert result == "LGTM"


def test_parse_github_pr_action_fallback_closed_unmerged():
    """
    Edge Path: Validates extraction of the 'abort' signal when an agent branch is closed manually.
    """
    mock_payload = {
        "action": "closed",
        "pull_request": {"merged": False, "head": {"ref": "agent/bad-code-4567"}},
    }

    result = parse_github_pr_action(mock_payload)
    assert result == "abort"


def test_parse_github_pr_action_fallback_ignored_branches():
    """
    Edge Path: Validates that PRs not created by the agent
    (e.g., humans merging 'feature' branches) are ignored.
    """
    mock_payload = {
        "action": "closed",
        "pull_request": {"merged": True, "head": {"ref": "feature/human-made-branch"}},
    }

    result = parse_github_pr_action(mock_payload)
    assert result is None


# ==========================================
# Component: verify_github_signature
# ==========================================


def test_verify_github_signature_success_valid(monkeypatch):
    """
    Green Path: Validates that the HMAC SHA-256 signature algorithm
    correctly matches a perfectly signed payload by generating the expected signature dynamically.
    """
    # 1. Setup Mock Environment
    secret = "my_super_secret_key"
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

    payload_body = b'{"action": "closed"}'

    # Dynamically compute the valid signature to avoid formatting-mismatch failures
    expected_hash = hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    valid_signature = f"sha256={expected_hash}"

    # 2. Execute & Assert
    result = verify_github_signature(payload_body, valid_signature)
    assert result is True


def test_verify_github_signature_error_invalid(monkeypatch):
    """
    Red Path: Validates that an attacker attempting to forge a webhook
    with an invalid signature is strictly rejected.
    """
    # 1. Setup Mock Environment
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "my_super_secret_key")

    payload_body = b'{"action": "malicious_payload"}'
    invalid_signature = "sha256=abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

    # 2. Execute & Assert
    result = verify_github_signature(payload_body, invalid_signature)
    assert result is False


def test_verify_github_signature_error_missing_secret(monkeypatch, caplog):
    """
    Red Path: If the user forgets to set the secret in .env,
    the gateway must fail closed and reject the payload to prevent spoofing attacks.
    """
    # 1. Setup Mock Environment (delete the key)
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)

    payload_body = b'{"action": "closed"}'
    random_signature = "sha256=doesntmatter"

    # 2. Execute & Assert
    result = verify_github_signature(payload_body, random_signature)

    # Must explicitly fail closed
    assert result is False

    # Verify the security error was logged properly
    assert "Rejecting webhook to prevent spoofing" in caplog.text


# ==========================================
# Component: parse_agentic_ci_trigger
# ==========================================


def test_parse_agentic_ci_trigger_success_automatic_opened(mocker):
    """Green Path: Validates extraction of CI metadata when a PR is newly opened."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 101,
            "head": {"sha": "abcdef123"},
            "base": {"ref": "main"},
            "user": {"login": "test_user"},
        },
        "repository": {"full_name": "owner/repo", "name": "repo"},
    }

    result = parse_agentic_ci_trigger(payload)

    assert result is not None
    assert result["pr_number"] == 101
    assert result["commit_sha"] == "abcdef123"


def test_parse_agentic_ci_trigger_success_automatic_synchronize(mocker):
    """Green Path: Validates extraction of CI metadata when a PR is updated with new commits."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    payload = {
        "action": "synchronize",
        "pull_request": {
            "number": 102,
            "head": {"sha": "0987654"},
            "base": {"ref": "main"},
            "user": {"login": "test_user"},
        },
        "repository": {"full_name": "owner/repo", "name": "repo"},
    }

    result = parse_agentic_ci_trigger(payload)

    assert result is not None
    assert result["commit_sha"] == "0987654"


def test_parse_agentic_ci_trigger_success_chatops_agent_test(mocker):
    """Green Path: Validates extraction of ChatOps triggers (@agent retest) on PR comments."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.chatops_name",
        "agent",
    )
    payload = {
        "action": "created",
        "issue": {"number": 202, "pull_request": {"url": "..."}},
        "comment": {"body": "@agent retest", "user": {"login": "test_user"}},
        "repository": {"full_name": "owner/repo", "name": "repo"},
    }

    result = parse_agentic_ci_trigger(payload)
    assert result is not None
    assert result.get("is_chatops") is True


def test_parse_agentic_ci_trigger_success_chatops_retest(mocker):
    """Green Path: Validates extraction of ChatOps triggers (@agent retest) on PR comments."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.chatops_name",
        "agent",
    )
    payload = {
        "action": "created",
        "issue": {
            "number": 201,
            "pull_request": {"url": "https://api.github.com/repos/owner/repo/pulls/201"},
        },
        "comment": {
            "body": "I fixed the bug. @agent retest please.",
            "user": {"login": "test_user"},
        },
        "repository": {"full_name": "owner/repo", "name": "repo"},
    }

    result = parse_agentic_ci_trigger(payload)

    assert result is not None
    assert result["pr_number"] == 201
    assert result.get("is_chatops") is True


def test_parse_agentic_ci_trigger_fallback_ignores_non_prs(mocker):
    """Edge Path: Ensures comments on standard issues (not PRs) are safely ignored."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    payload = {
        "action": "created",
        "issue": {"number": 301},
        "comment": {"body": "/retest", "user": {"login": "test_user"}},
    }

    result = parse_agentic_ci_trigger(payload)
    assert result is None


def test_parse_agentic_ci_trigger_fallback_ignores_unrelated_comments(mocker):
    """Edge Path: Ensures standard conversational comments on PRs don't trigger the CI."""
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )
    payload = {
        "action": "created",
        "issue": {"number": 302, "pull_request": {"url": "..."}},
        "comment": {"body": "Looks good to me!", "user": {"login": "test_user"}},
    }

    result = parse_agentic_ci_trigger(payload)
    assert result is None
