import html
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

# Telegram's hard physical limit is 4096 characters per message.
# We set the chunking threshold slightly lower to leave room for safety margins.
MAX_MESSAGE_LENGTH = 4000


def _convert_markdown_to_telegram_html(text: str) -> str:
    """
    Converts basic Markdown to Telegram-safe HTML.
    Telegram's HTML parser is immune to the stray underscore/asterisk crashes
    that plague its legacy Markdown parser.
    """
    # 1. Escape HTML entities to prevent unintended tag parsing (e.g. <, >, & in code)
    text = html.escape(text)

    # 2. Convert multi-line code blocks: ```language\ncode\n``` -> <pre>code</pre>
    # We use string multiplication ("`" * 3) to prevent UI markdown rendering issues
    code_block_pattern = "`" * 3 + r"(?:[a-zA-Z0-9\-]+)?\n(.*?)" + "`" * 3
    text = re.sub(
        code_block_pattern,
        r"<pre>\1</pre>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Convert inline code: `code` -> <code>code</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # 4. Convert bold: **text** -> <b>text</b>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    # 5. Convert italics: *text* -> <i>text</i>
    # Using negative lookbehinds to avoid matching the asterisks in bold tags
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)

    # 6. Convert links: [text](url) -> <a href="url">text</a>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    return text


def _chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """
    Splits a long message into chunks that fit within Telegram's character limits.
    Attempts to break cleanly at paragraph boundaries, then newlines, then spaces.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Attempt to find a graceful breaking point
        split_at = text.rfind("\n\n", 0, max_length)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            split_at = max_length  # Hard split if it's a massive unbroken block

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return chunks


def send_telegram_message(chat_id: str, text: str):
    """
    Encapsulates raw communication with the Telegram Bot API.
    Handles message chunking and automatic formatting fallbacks to ensure delivery.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is missing. Cannot send message.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Chunk the raw Markdown first to avoid splitting HTML tags down the middle
    chunks = _chunk_message(text)

    for chunk in chunks:
        # Convert each isolated chunk to HTML
        html_chunk = _convert_markdown_to_telegram_html(chunk)

        payload = {
            "chat_id": chat_id,
            "text": html_chunk,
            "parse_mode": "HTML",
        }

        try:
            # 10-second timeout prevents the background task thread from hanging indefinitely
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to push integration message to Telegram. Status: {e.response.status_code}"
            )
            logger.error(f"Telegram API Response: {e.response.text}")

            # FALLBACK: If HTML parsing fails (e.g. chunking broke a <pre> tag in half),
            # retry sending this specific chunk as pure plain-text to guarantee delivery.
            if e.response.status_code == httpx.codes.BAD_REQUEST:
                logger.warning("Attempting plain-text fallback for rejected chunk...")

                # Use the original, un-escaped Markdown chunk without any parse_mode
                fallback_payload = {"chat_id": chat_id, "text": chunk}
                try:
                    fallback_response = httpx.post(url, json=fallback_payload, timeout=10.0)
                    fallback_response.raise_for_status()
                    logger.info("Fallback delivery successful.")
                except Exception as fallback_e:
                    logger.error(f"Fallback plain-text delivery also failed: {fallback_e}")

        except Exception as e:
            logger.error(f"Network error sending to Telegram: {e}")
