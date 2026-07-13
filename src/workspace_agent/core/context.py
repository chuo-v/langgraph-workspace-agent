import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.core.vector_memory import semantic_search

__all__ = ["get_hybrid_context"]


# ==========================================
# Hybrid Context Assembly
# ==========================================


def get_hybrid_context(
    messages: list,
    thread_id: str,
    user_id: str,
    store: BaseStore,
    config: RunnableConfig,
    is_frontier_tier: bool = False,
) -> list:
    """Assembles the context window by injecting Long-Term Entity Memory (from Store),
    Episodic Semantic Memory (from VectorDB), and sliding Working Memory.

    State Transitions:
    - Evaluates raw message history to extract the latest human intent.
    - Constructs a comprehensive SystemMessage containing unified memory inputs.
    - Truncates and sanitizes the operational message history for API ingestion.

    Exceptions/Edge Cases:
    - Tolerates missing namespaces in the store, defaulting to empty profiles.
    - Handles malformed window boundaries by forcibly injecting missing HumanMessages
        if orphaned ToolMessages cause structural non-compliance.
    """
    user_profile = _fetch_user_profile(user_id, store)

    latest_human_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    collection = config.get("configurable", {}).get("chroma_collection")
    episodic_injection = _fetch_episodic_memory(latest_human_msg, collection)

    memory_prompt = PromptManager.get(
        "context",
        "system_memory_prompt",
        user_profile=json.dumps(user_profile, indent=2),
        episodic_injection=episodic_injection,
    ).strip()

    short_term_messages = _slide_working_memory(messages, is_frontier_tier)

    hybrid_messages = [SystemMessage(content=memory_prompt)] + short_term_messages
    return hybrid_messages


def _fetch_user_profile(user_id: str, store: BaseStore) -> dict:
    """Retrieves the Long-Term Entity Memory (User Profile) from the LangGraph Store.
    Defaults to an empty dictionary if the profile item does not exist to prevent
    unexpected runtime errors.
    """
    namespace = ("user_profile", user_id)
    profile_item = store.get(namespace, "profile")
    return profile_item.value if profile_item else {}


def _fetch_episodic_memory(latest_human_msg: str, collection: Any) -> str:
    """Executes a semantic search across past sessions to discover relevant technical context.
    Formats and returns the resulting episodic injection string, or an empty string if no
    relevant memories are matched.
    """
    episodic_memories = semantic_search(
        latest_human_msg, collection=collection, thread_id=None, limit=3
    )

    episodic_injection = ""
    if episodic_memories:
        episodic_list = "Relevant Technical Discoveries from Past Sessions:\n"
        for i, mem in enumerate(episodic_memories, 1):
            episodic_list += f"{i}. {mem}\n"

        episodic_injection = PromptManager.get(
            "context", "episodic_injection", episodic_context=episodic_list.strip()
        )

    return episodic_injection


def _slide_working_memory(messages: list, is_frontier_tier: bool) -> list:
    """Applies tier-based truncation to the message history while strictly enforcing
    downstream LLM compliance. Crucially strips orphaned ToolMessages at the window's
    boundary and guarantees the returned sequence begins with a HumanMessage.
    """
    short_term_count = 40 if is_frontier_tier else 15
    short_term_messages = messages[-short_term_count:] if messages else []

    # Check if the sliding window actually captured a HumanMessage
    has_human_message = any(isinstance(m, HumanMessage) for m in short_term_messages)

    if not has_human_message and messages:
        # Retrieve the actual last human message from the full history instead of a placeholder
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

        if last_human:
            # Gemini crashes if a ToolMessage doesn't have a preceding AIMessage.
            # Strip orphaned ToolMessages from the front before we inject the HumanMessage.
            while short_term_messages and isinstance(short_term_messages[0], ToolMessage):
                short_term_messages.pop(0)

            # Inject the actual instructions at the front of the operational memory
            short_term_messages.insert(0, last_human)

    # Final sweep to ensure strict Gemini API compliance (must start with HumanMessage)
    while short_term_messages and not isinstance(short_term_messages[0], HumanMessage):
        short_term_messages.pop(0)

    # Extreme fallback just in case the history is completely malformed
    if messages and not short_term_messages:
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            HumanMessage(content="[Prior context omitted]"),
        )
        short_term_messages = [last_human]

    return short_term_messages
