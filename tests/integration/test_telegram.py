import logging
from unittest.mock import patch

import httpx

from src.workspace_agent.integrations.telegram import (
    MAX_MESSAGE_LENGTH,
    _chunk_message,
    _convert_markdown_to_telegram_html,
    send_telegram_message,
)

# ==========================================
# Component: _convert_markdown_to_telegram_html
# ==========================================


def test_convert_markdown_to_telegram_html_success_standard():
    """Green Path: Verifies custom Markdown to Telegram HTML conversion & escaping."""
    # 1. Setup Mock Environment
    markdown_text = (
        "This is **bold** and *italic*. Here is `inline code` and a [link](https://test.com)."
    )
    code_block = "```python\nprint('hello <world>')\n```"

    # 2. Execute
    html_text = _convert_markdown_to_telegram_html(markdown_text)
    html_code = _convert_markdown_to_telegram_html(code_block)

    # 3. Assertions
    assert "<b>bold</b>" in html_text
    assert "<i>italic</i>" in html_text
    assert "<code>inline code</code>" in html_text
    assert '<a href="https://test.com">link</a>' in html_text

    # Verify code block escaping (protecting raw HTML inside code blocks)
    assert "&lt;world&gt;" in html_code
    assert "<pre>print" in html_code


# ==========================================
# Component: _chunk_message
# ==========================================


def test_chunk_message_success_standard():
    """Green Path: Verifies message chunking respects limits and breaks gracefully at boundaries."""
    # 1. Setup Mock Environment
    # Create a string slightly over 4000 characters, with a clean paragraph break at 3000
    part1 = "A" * 3000 + "\n\n"
    part2 = "B" * 1500
    long_text = part1 + part2

    # 2. Execute
    chunks = _chunk_message(long_text, max_length=4000)

    # 3. Assertions
    assert len(chunks) == 2
    assert chunks[0] == "A" * 3000
    assert chunks[1] == "B" * 1500


def test_chunk_message_fallback_hard_split():
    """Edge Path: Verifies fallback to hard character split for massive unbroken strings."""
    # 1. Setup Mock Environment
    # create a massive string with ZERO spaces or newlines
    long_text = "A" * 5000

    # 2. Execute
    chunks = _chunk_message(long_text, max_length=MAX_MESSAGE_LENGTH)

    # 3. Assertions
    assert len(chunks) == 2
    # verify the hard split occurred exactly at the threshold
    assert len(chunks[0]) == MAX_MESSAGE_LENGTH
    assert len(chunks[1]) == 5000 - MAX_MESSAGE_LENGTH


# ==========================================
# Workflow: Telegram Message Dispatch
# ==========================================


@patch("src.workspace_agent.integrations.telegram.os.getenv")
@patch("src.workspace_agent.integrations.telegram.httpx.post")
def test_send_telegram_message_success_standard(mock_post, mock_getenv):
    """
    Green Path: Validates that the adapter correctly formats the
    HTTP payload with HTML parsing and makes the request.
    """
    # 1. Setup Mock Environment
    mock_getenv.return_value = "fake_bot_token"

    # Mock a successful 200 OK response from Telegram
    mock_response = httpx.Response(200, request=httpx.Request("POST", "https://fake.url"))
    mock_post.return_value = mock_response

    # 2. Execute
    send_telegram_message("12345", "Hello, **Agent**!")

    # 3. Assertions
    # Verify the environment variable was fetched
    mock_getenv.assert_called_with("TELEGRAM_BOT_TOKEN")

    # Verify httpx.post was called exactly once
    mock_post.assert_called_once()

    # Inspect the exact arguments passed to httpx.post
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "https://api.telegram.org/botfake_bot_token/sendMessage"
    assert call_kwargs["json"]["chat_id"] == "12345"

    # Verify the adapter successfully translated the Markdown to HTML
    assert call_kwargs["json"]["text"] == "Hello, <b>Agent</b>!"
    assert call_kwargs["json"]["parse_mode"] == "HTML"


@patch("src.workspace_agent.integrations.telegram.os.getenv")
@patch("src.workspace_agent.integrations.telegram.httpx.post")
def test_send_telegram_message_fallback_handling(mock_post, mock_getenv, caplog):
    """
    Edge Path: Validates that HTTP Status Errors from Telegram
    are correctly intercepted and trigger the plain-text fallback.
    """
    # 1. Setup Mock Environment
    mock_getenv.return_value = "fake_bot_token"

    # Mock a 400 Bad Request response from Telegram to simulate an HTML tag error
    mock_response = httpx.Response(400, request=httpx.Request("POST", "https://fake.url"))
    mock_post.return_value = mock_response

    # 2. Execute
    with caplog.at_level(logging.WARNING):
        send_telegram_message("12345", "Malformed *Markdown")

    # 3. Assertions
    # Ensure the adapter caught the 400 error and triggered the plain-text fallback
    assert "Attempting plain-text fallback for rejected chunk" in caplog.text


@patch("src.workspace_agent.integrations.telegram.os.getenv")
@patch("src.workspace_agent.integrations.telegram.httpx.post")
def test_send_telegram_message_error_network(mock_post, mock_getenv, caplog):
    """
    Red Path: Validates that catastrophic network errors (e.g., DNS failure, timeout)
    are caught and logged without crashing the worker thread.
    """
    # 1. Setup Mock Environment
    mock_getenv.return_value = "fake_bot_token"

    # Simulate a severe network drop instead of a clean HTTP error
    mock_post.side_effect = httpx.ConnectError("Network is unreachable")

    # 2. Execute
    with caplog.at_level(logging.ERROR):
        send_telegram_message("12345", "Test message")

    # 3. Assertions
    # Verify the broad Exception block caught the network drop
    assert "Network error sending to Telegram" in caplog.text
    assert "Network is unreachable" in caplog.text
