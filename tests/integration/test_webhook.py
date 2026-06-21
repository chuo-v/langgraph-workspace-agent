import hashlib
import hmac
import json
import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import src.workspace_agent.main
from src.workspace_agent.main import (
    IODependencies,
    _format_final_response,
    app,
    get_active_thread,
    process_agent_message,
    set_active_thread,
)

client = TestClient(app)

# ==========================================
# Helper: Test Fixtures
# ==========================================


@pytest.fixture(autouse=True)
def setup_app_state():
    """Ensure app.state has the required attributes for testing since lifespan is bypassed."""
    app.state.redis_client = None
    app.state.chroma_collection = None
    app.state.docker_client = None


@pytest.fixture
def setup_env(monkeypatch):
    """
    Injects deterministic environment variables for testing.
    This ensures the tests run perfectly in CI without needing the real .env file.
    """
    monkeypatch.setenv("TELEGRAM_SECRET_TOKEN", "test_secret_123")
    monkeypatch.setenv("AUTHORIZED_OWNER_CHAT_ID", "999888777")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_bot_token")


# ==========================================
# Component: active_thread (Session Management)
# ==========================================


def test_active_thread_success_redis_persistence(mocker):
    """Green Path: Verifies active threads are successfully stored and retrieved from Redis."""
    mock_redis = mocker.Mock()
    mock_redis.get.return_value = "thread_redis_123"

    # Pass the mock dynamically instead of patching a global
    set_active_thread("test_user", "thread_redis_123", redis_client=mock_redis)

    # Verify the setter sent the correct payload to Redis
    mock_redis.set.assert_called_once_with("active_thread_test_user", "thread_redis_123")

    fetched = get_active_thread("test_user", redis_client=mock_redis)

    # Verify the getter retrieves from Redis
    mock_redis.get.assert_called_once_with("active_thread_test_user")
    assert fetched == "thread_redis_123"


def test_active_thread_fallback_memory():
    """Edge Path: Verifies memory routing is used if Redis is unavailable."""
    # Passing None validates the fallback behavior correctly
    set_active_thread("test_user", "thread_mem_123", redis_client=None)
    fetched = get_active_thread("test_user", redis_client=None)

    assert fetched == "thread_mem_123"
    assert get_active_thread("unknown_user", redis_client=None) == "unknown_user"


# ==========================================
# Component: _format_final_response
# ==========================================


def test_format_final_response_success_list():
    """Green Path: Verifies list-based responses (Gemini style) are flattened before telemetry."""
    state = {
        "messages": [AIMessage(content=[{"type": "text", "text": "List output."}, " Extra str."])],
        "t1_base_calls": 0,
        "t2_standard_calls": 0,
        "t3_frontier_calls": 0,
    }
    result = _format_final_response(state)
    # The output should be a cleanly joined string
    assert "List output.\n Extra str." in result


def test_format_final_response_success_string():
    """Green Path: Verifies standard string responses are formatted with telemetry."""
    state = {
        "messages": [AIMessage(content="Task complete.")],
        "t1_base_calls": 2,
        "t2_standard_calls": 0,
        "t3_frontier_calls": 1,
    }
    # Temporarily force telemetry on for the test
    src.workspace_agent.main.settings.agent.show_telemetry = True

    result = _format_final_response(state)
    assert "Task complete." in result
    assert "T1: 2 | T2: 0 | T3: 1" in result


# ==========================================
# Workflow: Telegram Webhook Endpoint
# ==========================================


def test_webhook_success_reset_command(setup_env, mocker):
    """
    Green Path: Tests memory management. The /reset command should trigger
    a completely new thread ID in the runtime state.
    """
    mocker.patch("src.workspace_agent.main.is_agent_busy", return_value=False)
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "/reset"}}

    response = client.post("/webhook", headers=headers, json=payload)

    assert response.status_code == 200

    # LangGraph processor should not trigger for a system command
    mock_process.assert_not_called()

    # system should have replied via the API acknowledging the reset
    mock_send.assert_called_once()

    # verify acknowledgement message was sent
    args, _ = mock_send.call_args
    assert "Memory cleared" in args[1]


def test_webhook_success_valid_payload(setup_env, mocker):
    """
    Green Path: Perfect payload from the authorized owner.
    Verifies that the background task is correctly queued and executed.
    """
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "sync the repository"}}

    response = client.post("/webhook", headers=headers, json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # prove the handoff occurred exactly once
    mock_process.assert_called_once()

    # verify parameters passed to the background task match the payload
    args, _ = mock_process.call_args
    assert args[0] == "999888777"
    assert args[1] == "sync the repository"
    # args[2] is the thread_id
    assert isinstance(args[3], IODependencies)


def test_webhook_fallback_agent_busy(setup_env, mocker):
    """
    Edge Path: Tests the concurrency protection. If the agent is currently
    processing a graph execution, new webhooks should be intercepted.
    """
    # mock is_agent_busy to return True
    mocker.patch("src.workspace_agent.main.is_agent_busy", return_value=True)
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "do something else"}}

    response = client.post("/webhook", headers=headers, json=payload)

    assert response.status_code == 200

    # verify the background task was explicitly NOT called
    mock_process.assert_not_called()

    # verify the warning was sent to the user
    mock_send.assert_called_once()
    args, _ = mock_send.call_args
    assert "currently executing a task" in args[1]


def test_webhook_fallback_unauthorized_chat_id(setup_env, mocker):
    """
    Edge Path: Valid secret token, but the Telegram user ID is not yours.
    This tests the "200 OK Trap" designed to drop payloads without alerting the sender.
    """
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 111222333},  # Not the authorized 999888777
            "text": "run malicious code",
        }
    }

    response = client.post("/webhook", headers=headers, json=payload)

    # must return 200 OK to satisfy Telegram's retry logic
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # prove the core logic was entirely bypassed
    mock_process.assert_not_called()


def test_webhook_error_invalid_secret_token(setup_env):
    """
    Red Path: The request has a token, but it does not match the environment.
    """
    headers = {"X-Telegram-Bot-Api-Secret-Token": "malicious_hacker_token"}
    response = client.post("/webhook", headers=headers, json={"message": {"text": "hello"}})

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_webhook_error_missing_secret_token(setup_env):
    """
    Red Path: The request is missing the secret token header entirely.
    """
    response = client.post("/webhook", json={"message": {"text": "hello"}})

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


# ==========================================
# Workflow: LangGraph Integration (process_agent_message)
# ==========================================


def test_process_agent_message_success_human_approve_lgtm(setup_env, mocker):
    """
    Green Path: Explicit Human Approval.
    If the graph is suspended at a PR review, sending 'LGTM' must correctly
    update the state to `human_approved=True`.
    """
    mock_state = mocker.Mock()
    mock_state.next = ("human_pr_node",)
    mock_state.values = {"pending_pr_url": "http://github.com/pr/1"}

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state

    # execute the processor with the approval keyword
    process_agent_message("999888777", "LGTM", "thread_123")

    # verify the first atomic state update successfully captured the human approval
    state_update = mock_app.update_state.call_args_list[0].args[1]
    assert state_update.get("human_approved") is True


def test_process_agent_message_fallback_auto_recovery(setup_env, mocker):
    """
    Edge Path: Verifies that a stuck execution thread triggers auto-recovery,
    safely releasing the mutex lock and migrating the user to a clean thread.
    """
    # mock network call to prevent the successful execution from pinging Telegram API
    mocker.patch("src.workspace_agent.main.send_telegram_message")

    # mock the state to look stuck on an execution node (not human_node or clarify)
    mock_state = mocker.Mock()
    mock_state.next = ["execute_task"]

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state

    mock_set_thread = mocker.patch("src.workspace_agent.main.set_active_thread")

    # track thread_ids as they are passed to update_state to avoid Python pass-by-reference
    # mutation issues
    captured_updates = []

    def capture_update(cfg, state_update):
        captured_updates.append((cfg["configurable"]["thread_id"], state_update))
        return mock_state

    mock_app.update_state.side_effect = capture_update

    # execute
    process_agent_message("999888777", "New instruction", "stuck_thread_123")

    # 1. Verify it released the lock on the OLD broken thread FIRST
    assert captured_updates[0][0] == "stuck_thread_123"
    assert captured_updates[0][1] == {"is_busy": False}

    # Verify it generated a new thread and updated the routing registry
    mock_set_thread.assert_called_once()
    assert mock_set_thread.call_args[0][0] == "999888777"
    new_thread = mock_set_thread.call_args[0][1]
    assert new_thread != "stuck_thread_123"

    # 2. Verify the lock was successfully engaged on the NEW thread
    assert captured_updates[1][0] == new_thread
    assert captured_updates[1][1] == {"is_busy": True}

    # 3. Verify the lock was safely released at the end of the entire execution block
    assert captured_updates[2][0] == new_thread
    assert captured_updates[2][1] == {"is_busy": False}


def test_process_agent_message_fallback_human_abort(setup_env, mocker):
    """
    Edge Path: Explicit Human Abort.
    If the graph is suspended at a breakpoint and the user sends an abort keyword,
    the webhook must inject the is_aborted override flag into the state update.
    """
    # mock the LangGraph app to return a paused state
    mock_state = mocker.Mock()
    mock_state.next = ("human_pr_node",)
    mock_state.values = {"pending_pr_url": "http://github.com/pr/1"}

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state

    # execute the processor with the abort keyword
    process_agent_message("999888777", "abort", "thread_123")

    # verify the first atomic state update successfully injected the abort override
    state_update = mock_app.update_state.call_args_list[0].args[1]
    assert state_update.get("is_aborted") is True


def test_process_agent_message_error_catastrophic_crash(setup_env, mocker):
    """
    Red Path: If the LangGraph orchestrator violently crashes during execution,
    the gateway must catch the exception, notify the user, and release the mutex lock.
    """
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    # force LangGraph to crash unconditionally
    mock_app.invoke.side_effect = Exception("Catastrophic LangGraph Failure")

    # simulate a clean state (not paused)
    mock_state = mocker.Mock()
    mock_state.next = []
    mock_app.get_state.return_value = mock_state

    # execute
    process_agent_message("999888777", "do a task", "thread_123")

    # 1. Verify the user was notified of the critical failure
    mock_send.assert_called_once()
    assert "critical error occurred" in mock_send.call_args[0][1]

    # 2. Verify the `finally` block successfully released the lock
    # Process gets state -> updates lock (True) -> crashes -> updates lock (False)
    final_update = mock_app.update_state.call_args_list[-1]
    assert final_update.args[0]["configurable"]["thread_id"] == "thread_123"
    assert final_update.args[1]["is_busy"] is False


# ==========================================
# Workflow: GitHub Webhook (PR Events)
# ==========================================


def _create_github_payload(
    action: str, branch: str, merged: bool, secret: str
) -> tuple[dict, bytes, str]:
    """Helper to generate signed GitHub payloads for testing."""
    payload = {
        "action": action,
        "pull_request": {"merged": merged, "head": {"ref": branch}},
    }
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return payload, body, signature


def test_github_webhook_success_ci_trigger(setup_env, mocker):
    """
    Green Path: Tests GitHub webhook routing a PR Opened event
    directly into the stateless Agentic CI pipeline.
    """
    mock_process_ci = mocker.patch("src.workspace_agent.main.process_ci_trigger")

    # Mock the authorization configuration
    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    # Add the authorized user to the mock payload
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

    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": signature}

    response = client.post("/github/webhook", headers=headers, content=body)

    assert response.status_code == 200
    assert response.json() == {"status": "ci_triggered"}

    mock_process_ci.assert_called_once()


def test_github_webhook_success_merged(setup_env, mocker):
    """
    Green Path: Tests GitHub webhook catching a merged PR and
    injecting the 'LGTM' signal into the LangGraph orchestrator.
    """
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "agent/test-branch", True, secret)
    headers = {"X-Hub-Signature-256": signature}

    response = client.post("/github/webhook", headers=headers, content=body)

    assert response.status_code == 200
    assert response.json() == {"status": "acknowledged_merge"}

    # verify the orchestrator was commanded to proceed
    mock_process.assert_called_once()
    args, _ = mock_process.call_args
    assert args[0] == "999888777"
    assert args[1] == "LGTM"


def test_github_webhook_fallback_ignored_branch(setup_env, mocker):
    """
    Edge Path: Webhook should ignore PRs that were not created by the agent
    (i.e., branches that don't start with 'agent/').
    """
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "user-feature-branch", True, secret)
    headers = {"X-Hub-Signature-256": signature}

    response = client.post("/github/webhook", headers=headers, content=body)

    # API still acknowledges receipt, but no processing occurs
    assert response.status_code == 200
    mock_process.assert_not_called()


def test_github_webhook_fallback_unmerged_close(setup_env, mocker):
    """
    Edge Path: Tests GitHub webhook catching an unmerged PR closure and
    injecting the 'abort' signal into the LangGraph orchestrator.
    """
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "agent/test-branch", False, secret)
    headers = {"X-Hub-Signature-256": signature}

    response = client.post("/github/webhook", headers=headers, content=body)

    assert response.status_code == 200
    assert response.json() == {"status": "acknowledged_merge"}

    # verify the orchestrator was commanded to abort
    mock_process.assert_called_once()
    args, _ = mock_process.call_args
    assert args[1] == "abort"


def test_github_webhook_error_invalid_signature(setup_env, mocker):
    """
    Red Path: Requests with an invalid HMAC signature must be rejected.
    """
    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, _ = _create_github_payload("closed", "agent/test-branch", True, secret)
    headers = {"X-Hub-Signature-256": "sha256=invalid_signature"}

    response = client.post("/github/webhook", headers=headers, content=body)

    assert response.status_code == 401
    assert "Invalid GitHub signature" in response.json()["detail"]
