from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from src.workspace_agent.core.context import get_hybrid_context

# ==========================================
# Component: get_hybrid_context
# ==========================================


def test_get_hybrid_context_success_ai_last_message(mocker):
    """
    Green Path: Ensures the semantic search extracts the last *Human* message
    as the query, even if the most recent message was from the AI.
    """
    mock_search = mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()
    messages = [HumanMessage(content="Real human query"), AIMessage(content="AI response")]

    get_hybrid_context(messages, "t_1", "u_1", store, config={})
    mock_search.assert_called_once_with(
        "Real human query", collection=None, thread_id=None, limit=3
    )


def test_get_hybrid_context_success_combines_memories(mocker):
    """
    Green Path: Verifies that Entity (Store) and Episodic (Vector) memories
    are successfully merged into the token-zero SystemMessage.
    """
    # Mock the vector DB search to return a dummy episodic memory
    mock_search = mocker.patch("src.workspace_agent.core.context.semantic_search")
    mock_search.return_value = ["Always use type hints in Python."]

    # Setup the LangGraph Store with a dummy Entity memory
    store = InMemoryStore()
    store.put(("user_profile", "user_123"), "profile", {"preferences": ["Use black formatter"]})

    messages = [
        HumanMessage(content="What is my task?"),
        AIMessage(content="Working on routing."),
        HumanMessage(content="Write a function."),
    ]

    hybrid_msgs = get_hybrid_context(
        messages=messages,
        thread_id="thread_123",
        user_id="user_123",
        store=store,
        config={},
        is_frontier_tier=False,
    )

    # We expect 4 total messages: 1 SystemMessage (Memory) + the 3 existing messages
    assert len(hybrid_msgs) == 4
    assert isinstance(hybrid_msgs[0], SystemMessage)

    sys_prompt = hybrid_msgs[0].content

    # Verify Entity Memory was injected
    assert "Use black formatter" in sys_prompt
    # Verify Episodic Memory was injected
    assert "Always use type hints in Python." in sys_prompt

    # Verify it searched using the latest HumanMessage and the injected collection
    mock_search.assert_called_once_with(
        "Write a function.", collection=None, thread_id=None, limit=3
    )


def test_get_hybrid_context_success_frontier_scaling(mocker):
    """
    Green Path: Verifies that setting is_frontier_tier=True dynamically
    expands the short-term context window from 15 to 40 messages.
    """
    mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()

    # Create 50 dummy messages to exceed both default and frontier limits
    messages = [HumanMessage(content=f"Message {i}") for i in range(50)]

    # Test default (base/standard tier) -> should return 1 System + 15 Short Term = 16
    res_base = get_hybrid_context(messages, "t_1", "u_1", store, config={}, is_frontier_tier=False)
    assert len(res_base) == 16

    # Test frontier tier -> should return 1 System + 40 Short Term = 41
    res_frontier = get_hybrid_context(
        messages, "t_1", "u_1", store, config={}, is_frontier_tier=True
    )
    assert len(res_frontier) == 41


def test_get_hybrid_context_fallback_db_offline_or_empty(mocker):
    """
    Edge Path: If ChromaDB is empty or offline (returns []),
    the context assembly succeeds without appending the episodic section.
    """
    mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()

    messages = [HumanMessage(content="Hello")]
    result = get_hybrid_context(messages, "t_1", "u_1", store, config={})

    sys_prompt = result[0].content
    assert "EPISODIC MEMORY" not in sys_prompt


def test_get_hybrid_context_fallback_empty_messages(mocker):
    """Edge Path: If the message list is empty, handle gracefully."""
    mock_search = mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()

    result = get_hybrid_context(messages=[], thread_id="t_1", user_id="u_1", store=store, config={})

    # Should still generate exactly 1 memory SystemMessage
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    mock_search.assert_called_once_with("", collection=None, thread_id=None, limit=3)


def test_get_hybrid_context_fallback_gemini_tool_alignment(mocker):
    """
    Edge Path: If the sliding window slices the history exactly such that an
    AIMessage is dropped but its corresponding ToolMessage is kept, Gemini will crash.
    Verifies that the alignment logic safely strips orphaned ToolMessages.
    """
    mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()

    # Force a very small window to artificially slice right through an execution pair
    # (e.g. keeping only the last 3 messages drops the first AIMessage but keeps its ToolMessage)
    mocker.patch("src.workspace_agent.core.context.get_hybrid_context.__defaults__", (False,))

    # We call it directly but simulate the internal slicing by passing a sliced array
    # Let's say short_term_count is 2: it keeps [AIMessage(test2), ToolMessage(Result 2)]
    # Wait, if it keeps 3: [ToolMessage(Result 1), AIMessage(test2), ToolMessage(Result 2)]
    # In get_hybrid_context, short_term_count is hardcoded to 15. So we must pass 18 messages
    # to force it to slice off the human message.

    long_history = [HumanMessage(content="Way back instructions")]
    for i in range(8):  # 8 pairs = 16 messages
        long_history.append(
            AIMessage(
                content="",
                tool_calls=[{"name": f"test_{i}", "id": str(i), "args": {}}],
            )
        )
        long_history.append(
            ToolMessage(content=f"Result {i}", tool_call_id=str(i), name=f"test_{i}")
        )

    # Total length is 1 + 16 = 17. The slice [-15:] will keep from index 2 onwards.
    # Index 1 (AIMessage 0) is dropped. Index 2 (ToolMessage 0) is kept.
    # This is an orphaned ToolMessage.

    res = get_hybrid_context(long_history, "t_1", "u_1", store, config={}, is_frontier_tier=False)

    # 1. Verify the SystemMessage is first
    assert isinstance(res[0], SystemMessage)

    # 2. Verify the smart alignment found the actual HumanMessage and injected it
    assert isinstance(res[1], HumanMessage)
    assert res[1].content == "Way back instructions"

    # 3. Verify the orphaned ToolMessage 0 was successfully stripped, meaning
    # the next message should be AIMessage 1.
    assert isinstance(res[2], AIMessage)
    assert res[2].tool_calls[0]["id"] == "1"


def test_get_hybrid_context_fallback_missing_profile(mocker):
    """
    Edge Path: If the user profile doesn't exist in the LangGraph Store yet,
    it defaults to an empty dict gracefully.
    """
    mocker.patch("src.workspace_agent.core.context.semantic_search", return_value=[])
    store = InMemoryStore()  # Empty store

    messages = [HumanMessage(content="Hello")]
    result = get_hybrid_context(messages, "t_1", "u_1", store, config={})

    sys_prompt = result[0].content
    # An empty dict {} is dumped into the prompt string for the profile
    assert "{}" in sys_prompt
