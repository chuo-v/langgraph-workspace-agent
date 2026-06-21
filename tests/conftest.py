import os
import warnings
from unittest.mock import MagicMock

import pytest

# Force the config loader to use the example template during test runs.
# This prevents the fail-fast security check from crashing pytest collection
# when a live config.yaml is not present on the developer's machine or in CI.
os.environ["WORKSPACE_AGENT_CONFIG_PATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.example.yaml")
)


# LangChain explicitly forces deprecation warnings to display at import time,
# overriding pyproject.toml settings; we force the import here, then squash it
try:
    import langchain_core  # noqa: F401
    import langgraph  # noqa: F401
except ImportError:
    pass

# push our ignore rule to the top of the stack after LangChain has loaded
warnings.filterwarnings(
    "ignore", message=".*The default value of `allowed_objects` will change in a future version.*"
)


@pytest.fixture
def mock_chroma_collection(mocker):
    """
    Creates a mock ChromaDB collection that returns static semantic matches.
    """
    # create a dummy object to represent the ChromaDB collection
    mock_collection = MagicMock()

    # simulate a typical ChromaDB query response
    mock_collection.query.return_value = {
        "ids": [["mock_id_1", "mock_id_2"]],
        "documents": [
            [
                "[USER]: How do I configure the Langfuse database?",
                "[ASSISTANT]: You should map the Postgres volume in docker-compose.yml.",
            ]
        ],
        "metadatas": [
            [
                {"thread_id": "test-thread", "role": "user"},
                {"thread_id": "test-thread", "role": "assistant"},
            ]
        ],
        "distances": [[0.15, 0.22]],
    }

    # patch the exact location where the singleton is used in utilities
    # whenever utils.py calls `vector_memory.get_collection()`, it will receive our mock instead
    mocker.patch(
        "src.workspace_agent.core.vector_memory.get_collection",
        return_value=mock_collection,
    )

    return mock_collection
