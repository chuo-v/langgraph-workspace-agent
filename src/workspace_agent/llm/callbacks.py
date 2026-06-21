import os

from langfuse.langchain import CallbackHandler


def get_langfuse_callback(session_id: str = None, user_id: str = None, tags: list = None):
    """
    Initializes the Langfuse handler.
    Returns None if critical variables (public or secret keys) are missing.
    Automatically retrieves the host, public key, and secret key from env variables.
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
