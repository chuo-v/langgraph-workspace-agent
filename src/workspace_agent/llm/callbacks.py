import os

from langfuse.langchain import CallbackHandler

__all__ = ["get_langfuse_callback"]


# ==========================================
# Langfuse Callback Instrumentation
# ==========================================


def get_langfuse_callback(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list | None = None,
) -> CallbackHandler | None:
    """
    Initializes and configures the Langfuse callback handler for LLM tracing and observability.

    Preconditions:
        Requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to be set in the environment.
        Automatically retrieves the host, public key, and secret key from environment variables.

    Args:
        session_id: Optional unique identifier for tracking the execution or conversation session.
        user_id: Optional unique identifier for the user triggering the workflow.
        tags: Optional list of categorization tags to attach to the generated trace.

    Returns:
        Optional[CallbackHandler]: A configured Langfuse `CallbackHandler` instance with
        custom metadata attached if API credentials are present; otherwise returns `None`.

    State Transitions & Side Effects:
        Instantiates `CallbackHandler` without arguments to prevent signature mismatches in
        older SDK versions, then directly mutates the instance attributes (`session_id`,
        `user_id`, `tags`) if provided.
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None

    # Instantiate naked to avoid strict __init__ signature mismatch in older SDK versions
    handler = CallbackHandler()

    # Explicitly set the metadata on the instantiated handler
    if session_id:
        handler.session_id = session_id
    if user_id:
        handler.user_id = user_id
    if tags:
        handler.tags = tags

    return handler
