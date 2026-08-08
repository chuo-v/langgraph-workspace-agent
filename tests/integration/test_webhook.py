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


def _mock_telegram_file_download(mocker, is_ok=True, content=b"dummy content"):
    """Helper to mock the two-step httpx.AsyncClient process for Telegram file downloads."""
    # 1. Mock the getFile JSON response
    mock_info_resp = mocker.Mock()
    mock_info_resp.json.return_value = {"ok": is_ok, "result": {"file_path": "dummy/path.txt"}}

    # 2. Mock the actual binary file download response
    mock_download_resp = mocker.Mock()
    mock_download_resp.raise_for_status = mocker.Mock()
    mock_download_resp.content = content

    # Route the mock depending on which URL is being requested
    async def mock_get(url, *args, **kwargs):
        if "getFile" in str(url):
            return mock_info_resp
        return mock_download_resp

    mock_client = mocker.AsyncMock()
    mock_client.get.side_effect = mock_get

    # Wrap in an async context manager mock since we use `async with httpx.AsyncClient()...`
    mock_context_manager = mocker.MagicMock()
    mock_context_manager.__aenter__.return_value = mock_client

    mocker.patch("src.workspace_agent.main.httpx.AsyncClient", return_value=mock_context_manager)


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


# ==========================================
# Workflow: active_thread (Session Management)
# ==========================================


def test_active_thread_success_redis_persistence(mocker):
    """Green Path: Verifies active threads are successfully stored and retrieved from Redis."""
    # 1. Setup Mock Environment
    mock_redis = mocker.Mock()
    mock_redis.get.return_value = "thread_redis_123"

    # 2. Execute
    set_active_thread("test_user", "thread_redis_123", redis_client=mock_redis)

    fetched = get_active_thread("test_user", redis_client=mock_redis)

    # 3. Assertions
    mock_redis.set.assert_called_once_with("active_thread_test_user", "thread_redis_123")
    mock_redis.get.assert_called_once_with("active_thread_test_user")
    assert fetched == "thread_redis_123"


def test_active_thread_fallback_memory():
    """Edge Path: Verifies memory routing is used if Redis is unavailable."""
    # 1. Setup Mock Environment
    # Passing None validates the fallback behavior correctly

    # 2. Execute
    set_active_thread("test_user", "thread_mem_123", redis_client=None)
    fetched_mem = get_active_thread("test_user", redis_client=None)
    fetched_unknown = get_active_thread("unknown_user", redis_client=None)

    # 3. Assertions
    assert fetched_mem == "thread_mem_123"
    assert fetched_unknown == "unknown_user"


# ==========================================
# Workflow: _format_final_response (Response Formatting)
# ==========================================


def test_format_final_response_success_list():
    """Green Path: Verifies list-based responses (Gemini style) are flattened before telemetry."""
    # 1. Setup Mock Environment
    state = {
        "messages": [AIMessage(content=[{"type": "text", "text": "List output."}, " Extra str."])],
        "t1_base_calls": 0,
        "t2_standard_calls": 0,
        "t3_frontier_calls": 0,
    }

    # 2. Execute
    result = _format_final_response(state)

    # 3. Assertions
    assert "List output.\n Extra str." in result


def test_format_final_response_success_string():
    """Green Path: Verifies standard string responses are formatted with telemetry."""
    # 1. Setup Mock Environment
    state = {
        "messages": [AIMessage(content="Task complete.")],
        "t1_base_calls": 2,
        "t2_standard_calls": 0,
        "t3_frontier_calls": 1,
    }
    # Temporarily force telemetry on for the test
    src.workspace_agent.main.settings.agent.show_telemetry = True

    # 2. Execute
    result = _format_final_response(state)

    # 3. Assertions
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
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.main.is_agent_busy", return_value=False)
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "/reset"}}

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200
    mock_process.assert_not_called()
    mock_send.assert_called_once()

    args, _ = mock_send.call_args
    assert "Memory cleared" in args[1]


def test_webhook_success_valid_payload(setup_env, mocker):
    """
    Green Path: Perfect payload from the authorized owner.
    Verifies that the background task is correctly queued and executed.
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "sync the repository"}}

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_process.assert_called_once()

    args, _ = mock_process.call_args
    assert args[0] == "999888777"
    assert args[1] == "sync the repository"
    assert isinstance(args[3], IODependencies)


def test_webhook_success_manual_pause(setup_env, mocker):
    """
    Green Path: Tests that when the agent is busy, sending a pause command (/stop)
    intercepts the webhook, writes to the agent_store, and returns safely without
    queueing a new process_agent_message task.
    """
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.main.is_agent_busy", return_value=True)
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")
    mock_store = mocker.patch("src.workspace_agent.main.agent_store")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "/stop"}}

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200
    mock_process.assert_not_called()

    # Verify the abort signal was written to the store
    mock_store.put.assert_called_once()
    args, _ = mock_store.put.call_args
    assert args[0][0] == "abort_signals"
    assert args[1] == "abort"
    assert args[2]["stop_type"] == "pause"

    # Verify the user was notified of the pause
    mock_send.assert_called_once()
    args, _ = mock_send.call_args
    assert "Pausing current task" in args[1]


def test_webhook_fallback_agent_busy(setup_env, mocker):
    """
    Edge Path: Tests the concurrency protection. If the agent is currently
    processing a graph execution, new webhooks should be intercepted.
    """
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.main.is_agent_busy", return_value=True)
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {"message": {"chat": {"id": 999888777}, "text": "do something else"}}

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200
    mock_process.assert_not_called()

    mock_send.assert_called_once()
    args, _ = mock_send.call_args
    assert "currently executing a task" in args[1]


def test_webhook_fallback_unauthorized_chat_id(setup_env, mocker):
    """
    Edge Path: Valid secret token, but the Telegram user ID is not yours.
    This tests the "200 OK Trap" designed to drop payloads without alerting the sender.
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 111222333},  # Not the authorized 999888777
            "text": "run malicious code",
        }
    }

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_process.assert_not_called()


def test_webhook_error_invalid_secret_token(setup_env):
    """
    Red Path: The request has a token, but it does not match the environment.
    """
    # 1. Setup Mock Environment
    headers = {"X-Telegram-Bot-Api-Secret-Token": "malicious_hacker_token"}
    payload = {"message": {"text": "hello"}}

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_webhook_error_missing_secret_token(setup_env):
    """
    Red Path: The request is missing the secret token header entirely.
    """
    # 1. Setup Mock Environment
    payload = {"message": {"text": "hello"}}

    # 2. Execute
    response = client.post("/webhook", json=payload)

    # 3. Assertions
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


# ==========================================
# Workflow: Telegram Webhook Attachments
# ==========================================


def test_webhook_success_telegram_attachment_text(setup_env, mocker):
    """Green Path: Verifies standard text files are successfully downloaded and parsed."""
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    _mock_telegram_file_download(mocker, content=b"print('hello world')")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "text": "Review this script",
            "document": {"file_id": "file123", "file_name": "script.py"},
        }
    }

    # 2. Execute
    response = client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    assert response.status_code == 200

    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "Review this script" in final_instruction
    assert "--- Contents of script.py ---" in final_instruction
    assert "print('hello world')" in final_instruction


def test_webhook_success_telegram_attachment_pdf(setup_env, mocker):
    """Green Path: Verifies PDFs trigger the PyPDF reader integration."""
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    _mock_telegram_file_download(mocker, content=b"dummy binary pdf data")

    mock_pdf_reader = mocker.patch("src.workspace_agent.main.PdfReader")
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = "Extracted PDF text."
    mock_pdf_reader.return_value.pages = [mock_page]

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "text": "Summarize this paper",
            "document": {"file_id": "file123", "file_name": "research.pdf"},
        }
    }

    # 2. Execute
    client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "Summarize this paper" in final_instruction
    assert "Extracted PDF text." in final_instruction


def test_webhook_success_caption_fallback(setup_env, mocker):
    """Green Path: Verifies Telegram's quirk of sending 'caption' instead of 'text' for files."""
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    _mock_telegram_file_download(mocker, content=b"content here")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "caption": "Please review this",
            "document": {"file_id": "file123", "file_name": "script.py"},
        }
    }

    # 2. Execute
    client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "Please review this" in final_instruction
    assert "content here" in final_instruction


def test_webhook_fallback_unsupported_binary(setup_env, mocker):
    """
    Edge Path: Verifies unsupported true binaries (like .zip) are caught and gracefully rejected.
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    _mock_telegram_file_download(mocker, content=b"\xff\xfe\x00\x00")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "text": "Extract this",
            "document": {"file_id": "file123", "file_name": "archive.zip"},
        }
    }

    # 2. Execute
    client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "format is not currently supported for text extraction" in final_instruction


def test_webhook_fallback_file_too_large(setup_env, mocker):
    """Edge Path: Verifies the 20MB bot limit API failure does not crash the gateway."""
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")
    _mock_telegram_file_download(mocker, is_ok=False)

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "text": "Analyze this video",
            "document": {"file_id": "file123", "file_name": "movie.mp4"},
        }
    }

    # 2. Execute
    client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "exceeding Telegram's 20MB bot download limit" in final_instruction


def test_webhook_fallback_photo_attachment(setup_env, mocker):
    """Edge Path: Verifies compressed images sent via 'photo' are caught and rejected cleanly."""
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    headers = {"X-Telegram-Bot-Api-Secret-Token": "test_secret_123"}
    payload = {
        "message": {
            "chat": {"id": 999888777},
            "caption": "Look at this error",
            "photo": [{"file_id": "photo123"}],
        }
    }

    # 2. Execute
    client.post("/webhook", headers=headers, json=payload)

    # 3. Assertions
    args, _ = mock_process.call_args
    final_instruction = args[1]
    assert "Look at this error" in final_instruction
    assert "Images are not currently supported" in final_instruction


# ==========================================
# Workflow: LangGraph Integration (process_agent_message)
# ==========================================


def test_process_agent_message_success_human_approve_lgtm(setup_env, mocker):
    """
    Green Path: Explicit Human Approval.
    If the graph is suspended at a PR review, sending 'LGTM' must correctly
    update the state to `human_approved=True`.
    """
    # 1. Setup Mock Environment
    mock_state = mocker.Mock()
    mock_state.next = ("human_pr_node",)
    mock_state.values = {"pending_pr_url": "http://github.com/pr/1"}

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state

    # 2. Execute
    process_agent_message("999888777", "LGTM", "thread_123")

    # 3. Assertions
    state_update = mock_app.update_state.call_args_list[0].args[1]
    assert state_update.get("human_approved") is True


def test_process_agent_message_fallback_auto_recovery(setup_env, mocker):
    """
    Edge Path: Verifies that a stuck execution thread triggers auto-recovery,
    safely releasing the mutex lock and migrating the user to a clean thread.
    """
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.main.send_telegram_message")

    mock_state = mocker.Mock()
    mock_state.next = ["execute_task"]

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state
    mock_set_thread = mocker.patch("src.workspace_agent.main.set_active_thread")

    captured_updates = []

    def capture_update(cfg, state_update):
        captured_updates.append((cfg["configurable"]["thread_id"], state_update))
        return mock_state

    mock_app.update_state.side_effect = capture_update

    # 2. Execute
    process_agent_message("999888777", "New instruction", "stuck_thread_123")

    # 3. Assertions
    # Verify it released the lock on the OLD broken thread FIRST
    assert captured_updates[0][0] == "stuck_thread_123"
    assert captured_updates[0][1] == {"is_busy": False}

    # Verify it generated a new thread and updated the routing registry
    mock_set_thread.assert_called_once()
    assert mock_set_thread.call_args[0][0] == "999888777"
    new_thread = mock_set_thread.call_args[0][1]
    assert new_thread != "stuck_thread_123"

    # Verify the lock was successfully engaged on the NEW thread
    assert captured_updates[1][0] == new_thread
    assert captured_updates[1][1] == {"is_busy": True}

    # Verify the lock was safely released at the end of the entire execution block
    assert captured_updates[2][0] == new_thread
    assert captured_updates[2][1] == {"is_busy": False}


def test_process_agent_message_fallback_human_abort(setup_env, mocker):
    """
    Edge Path: Explicit Human Abort.
    If the graph is suspended at a breakpoint and the user sends an abort keyword,
    the webhook must inject the is_aborted override flag into the state update.
    """
    # 1. Setup Mock Environment
    mock_state = mocker.Mock()
    mock_state.next = ("human_pr_node",)
    mock_state.values = {"pending_pr_url": "http://github.com/pr/1"}

    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.get_state.return_value = mock_state

    # 2. Execute
    process_agent_message("999888777", "abort", "thread_123")

    # 3. Assertions
    state_update = mock_app.update_state.call_args_list[0].args[1]
    assert state_update.get("is_aborted") is True


def test_process_agent_message_error_catastrophic_crash(setup_env, mocker):
    """
    Red Path: If the LangGraph orchestrator violently crashes during execution,
    the gateway must catch the exception, notify the user, and release the mutex lock.
    """
    # 1. Setup Mock Environment
    mock_send = mocker.patch("src.workspace_agent.main.send_telegram_message")
    mock_app = mocker.patch("src.workspace_agent.main.agent_app")
    mock_app.invoke.side_effect = Exception("Catastrophic LangGraph Failure")

    mock_state = mocker.Mock()
    mock_state.next = []
    mock_app.get_state.return_value = mock_state

    # 2. Execute
    process_agent_message("999888777", "do a task", "thread_123")

    # 3. Assertions
    mock_send.assert_called_once()
    assert "critical error occurred" in mock_send.call_args[0][1]

    final_update = mock_app.update_state.call_args_list[-1]
    assert final_update.args[0]["configurable"]["thread_id"] == "thread_123"
    assert final_update.args[1]["is_busy"] is False


# ==========================================
# Workflow: GitHub Webhook (PR Events)
# ==========================================


def test_github_webhook_success_ci_trigger(setup_env, mocker):
    """
    Green Path: Tests GitHub webhook routing a PR Opened event
    directly into the stateless Agentic CI pipeline.
    """
    # 1. Setup Mock Environment
    mock_process_ci = mocker.patch("src.workspace_agent.main.process_ci_trigger")

    mocker.patch(
        "src.workspace_agent.integrations.github_webhook.settings.agent.allowed_github_users",
        ["test_user"],
    )

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

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

    # 2. Execute
    response = client.post("/github/webhook", headers=headers, content=body)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "ci_triggered"}
    mock_process_ci.assert_called_once()


def test_github_webhook_success_merged(setup_env, mocker):
    """
    Green Path: Tests GitHub webhook catching a merged PR and
    injecting the 'LGTM' signal into the LangGraph orchestrator.
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "agent/test-branch", True, secret)
    headers = {"X-Hub-Signature-256": signature}

    # 2. Execute
    response = client.post("/github/webhook", headers=headers, content=body)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "acknowledged_merge"}

    mock_process.assert_called_once()
    args, _ = mock_process.call_args
    assert args[0] == "999888777"
    assert args[1] == "LGTM"


def test_github_webhook_fallback_ignored_branch(setup_env, mocker):
    """
    Edge Path: Webhook should ignore PRs that were not created by the agent
    (i.e., branches that don't start with 'agent/').
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "user-feature-branch", True, secret)
    headers = {"X-Hub-Signature-256": signature}

    # 2. Execute
    response = client.post("/github/webhook", headers=headers, content=body)

    # 3. Assertions
    assert response.status_code == 200
    mock_process.assert_not_called()


def test_github_webhook_fallback_unmerged_close(setup_env, mocker):
    """
    Edge Path: Tests GitHub webhook catching an unmerged PR closure and
    injecting the 'abort' signal into the LangGraph orchestrator.
    """
    # 1. Setup Mock Environment
    mock_process = mocker.patch("src.workspace_agent.main.process_agent_message")

    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, signature = _create_github_payload("closed", "agent/test-branch", False, secret)
    headers = {"X-Hub-Signature-256": signature}

    # 2. Execute
    response = client.post("/github/webhook", headers=headers, content=body)

    # 3. Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "acknowledged_merge"}

    mock_process.assert_called_once()
    args, _ = mock_process.call_args
    assert args[1] == "abort"


def test_github_webhook_error_invalid_signature(setup_env, mocker):
    """
    Red Path: Requests with an invalid HMAC signature must be rejected.
    """
    # 1. Setup Mock Environment
    secret = "github_test_secret"
    mocker.patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": secret})

    _, body, _ = _create_github_payload("closed", "agent/test-branch", True, secret)
    headers = {"X-Hub-Signature-256": "sha256=invalid_signature"}

    # 2. Execute
    response = client.post("/github/webhook", headers=headers, content=body)

    # 3. Assertions
    assert response.status_code == 401
    assert "Invalid GitHub signature" in response.json()["detail"]
