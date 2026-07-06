import json
import os
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.context import get_hybrid_context
from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.core.state import AgentState
from src.workspace_agent.core.vector_memory import save_memory
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
    get_execution_llm_sequence,
    get_tier_for_model,
)
from src.workspace_agent.tools.github import sync_repository
from src.workspace_agent.tools.registry import agent_tools, execute_tool_call

# ==========================================
# Runtime Fallback Helper
# ==========================================


def _build_fallback_chain(llms: list, tools: list = None, structured_schema=None):
    """
    Constructs a resilient LCEL chain that automatically falls back to higher-tier models
    if the primary model fails or times out.
    Applies tools or structured schemas to ALL models in the sequence.
    """
    if not llms:
        raise ValueError("Cannot build fallback chain: No LLMs provided.")

    chains = []
    for llm in llms:
        chain = llm
        if tools:
            chain = chain.bind_tools(tools)
        elif structured_schema:
            model_name = getattr(llm, "model_name", "")
            if "deepseek" in str(model_name).lower() or "gemini" in str(model_name).lower():
                chain = chain.with_structured_output(structured_schema, method="function_calling")
            else:
                chain = chain.with_structured_output(structured_schema)
        chains.append(chain)

    primary_chain = chains[0]
    if len(chains) > 1:
        return primary_chain.with_fallbacks(chains[1:])
    return primary_chain


def _check_circuit_breaker(messages: list) -> dict | None:
    """Analyzes message history to prevent runaway tool loops."""
    consecutive_agent_steps = 0
    for msg in reversed(messages):
        if msg.type == "human":
            break
        if msg.type in ["ai", "tool"]:
            consecutive_agent_steps += 1

    limit = settings.agent.max_consecutive_tool_steps

    if consecutive_agent_steps >= limit:
        loops = limit // 2
        abort_msg = (
            "⚠️ **Execution Aborted:** The agent entered a runaway loop by executing "
            f"tools {loops} times in a row without finalizing the task. Circuit breaker "
            "triggered."
        )
        return {
            "messages": [AIMessage(content=abort_msg)],
            "is_aborted": True,
        }
    return None


# ==========================================
# Task Execution Helpers
# ==========================================


def _perform_sync_check(
    state: AgentState, target_path: str, target_branch: str, target_ws: str
) -> dict | None:
    """Helper to verify and sync the target repository before execution."""
    is_tool_loop = state.get("messages") and state["messages"][-1].type == "tool"
    is_retry_loop = state.get("execution_retry_count", 0) > 0

    active_branch = state.get("active_agent_branch")
    sync_target = active_branch if active_branch else target_branch

    # Prevent sync if we are in the middle of a tool loop OR an error recovery loop
    if target_path and target_path != "None" and not is_tool_loop and not is_retry_loop:
        sync_result = json.loads(sync_repository(directory=target_path, target_branch=sync_target))
        if sync_result.get("status") == "error":
            error_msg = (
                f"⚠️ **Sync Failed**\n"
                f"Could not pull latest changes for `{target_ws}`. "
                f"Reason: {sync_result.get('reason')}.\n"
                f"**Git Error Details:** `{sync_result.get('details', 'No details provided')}`\n"
                f"*Please check your branch name, token, or resolve conflicts manually.*"
            )
            return {"messages": [AIMessage(content=error_msg)], "is_aborted": True}
    return None


def _build_cross_workspace_prompt(
    state: AgentState, target_ws: str, target_path: str
) -> SystemMessage:
    """Helper to construct the strict workspace boundary prompt."""
    workspace_map = {name: ws.path for name, ws in settings.workspaces.items()}

    prompt = PromptManager.get(
        "execution",
        "cross_workspace_prompt",
        workspace_map=json.dumps(workspace_map, indent=2),
        target_ws=target_ws,
        target_path=target_path,
    ).strip()

    match state.get("intent_category"):
        case "workspace_read_only":
            prompt += "\n\n" + PromptManager.get("execution", "read_only_suffix").strip()
        case _:
            prompt += "\n\n" + PromptManager.get("execution", "write_operation_suffix").strip()

    return SystemMessage(content=prompt)


def _sanitize_llm_response(response: AIMessage) -> None:
    """Helper to sanitize empty content to ensure multi-provider API compatibility."""
    if isinstance(response.content, list):
        # Deeply sanitize empty text blocks inside structural lists
        cleaned_content = []
        for part in response.content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text", "").strip()
                if text_val:
                    cleaned_content.append(text_val)
            elif isinstance(part, str):
                if part.strip():
                    cleaned_content.append(part.strip())

        response.content = "\n".join(cleaned_content)

    # Do not forcefully inject "Executing tools..." or "Task complete."
    # Modern LLM frameworks handle empty string contents gracefully,
    # and injecting generic text causes smaller models to overfit and
    # hallucinate the text instead of triggering the actual tool API.
    if response.content is None:
        response.content = ""


def _resolve_execution_tier(state: AgentState) -> int:
    """Helper to determine the execution tier based on state flags."""
    requested_model = state.get("requested_model")
    if requested_model:
        inferred_tier = get_tier_for_model(requested_model)
        if inferred_tier:
            return inferred_tier

    if state.get("force_frontier_tier"):
        return TIER_FRONTIER
    if state.get("force_standard_tier"):
        return TIER_STANDARD
    return TIER_BASE


def _extract_modified_tex_files(current_files: list[str], response: AIMessage) -> list[str]:
    """
    Helper to trap newly modified .tex files from tool calls and clean up deleted/renamed paths.
    """
    new_files = list(current_files)
    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            tool_name = tc.get("name")
            args = tc.get("args", {})

            # 1. Handle file modifications/creations
            if tool_name in ["write_file", "search_and_replace", "replace_text_block"]:
                fpath = str(args.get("file_path", ""))
                if fpath.lower().endswith(".tex") and fpath not in new_files:
                    new_files.append(fpath)

            # 2. Handle file deletions
            elif tool_name == "delete_file":
                fpath = str(args.get("file_path", ""))
                if fpath in new_files:
                    new_files.remove(fpath)

            # 3. Handle file renames
            elif tool_name == "rename_file":
                source_path = str(args.get("source_path", ""))
                dest_path = str(args.get("destination_path", ""))

                # Remove the old path if it was queued
                if source_path in new_files:
                    new_files.remove(source_path)

                # Add the new path if it is a .tex file
                if dest_path.lower().endswith(".tex") and dest_path not in new_files:
                    new_files.append(dest_path)

    return new_files


def _calculate_telemetry(state: AgentState, tier: int) -> tuple[int, int, int]:
    """Helper to increment tier telemetry."""
    return (
        state.get("t1_base_calls", 0) + (1 if tier == TIER_BASE else 0),
        state.get("t2_standard_calls", 0) + (1 if tier == TIER_STANDARD else 0),
        state.get("t3_frontier_calls", 0) + (1 if tier == TIER_FRONTIER else 0),
    )


def _filter_execution_context(messages: list) -> list:
    """
    Helper to strip orchestrator success markers and past hallucinated summaries,
    replacing them with safe boundary markers to prevent context pollution.
    """
    filtered_messages = []

    for i, m in enumerate(messages):
        if isinstance(m, AIMessage):
            # Extract string representation regardless of structural format
            content_str = ""
            if isinstance(m.content, str):
                content_str = m.content
            elif isinstance(m.content, list):
                parts = []
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                content_str = "".join(parts)

            # Replace Orchestrator markers with a safe boundary
            if content_str and any(
                marker in content_str
                for marker in [
                    "✅ **Execution Complete**",
                    "✅ **Revisions Applied**",
                    "🚫 **Workflow Aborted.**",
                    "🎉 **Pull Request Merged!**",
                ]
            ):
                filtered_messages.append(
                    AIMessage(
                        content=PromptManager.get("execution", "workflow_completed_note").strip(),
                        id=m.id,
                    )
                )
                continue

            # Prevent hallucination: Replace past AI summaries that lack tool calls
            # (We keep it only if it's the very last message in the queue so the
            # agent can maintain conversational flow if it just answered a question)
            if not getattr(m, "tool_calls", None) and i != len(messages) - 1:
                filtered_messages.append(
                    AIMessage(
                        content=PromptManager.get(
                            "execution", "conversational_acknowledged_note"
                        ).strip(),
                        id=m.id,
                    )
                )
                continue

        filtered_messages.append(m)

    return filtered_messages


def execute_task_node(  # noqa: PLR0911, PLR0915
    state: AgentState, config: RunnableConfig = None, store: BaseStore = None
) -> dict:
    """
    The Core Workhorse Node.
    Assembles hybrid context, injects the workspace map, and triggers the LLM.
    """
    config = config or {}
    store = store or InMemoryStore()

    thread_id = config.get("configurable", {}).get("thread_id", "default")
    user_id = config.get("configurable", {}).get("user_id", "default")

    namespace = ("abort_signals", thread_id)
    abort_item = store.get(namespace, "abort")

    if abort_item:
        store.delete(namespace, "abort")  # Acknowledge and clear the flag

        return {
            "messages": [
                AIMessage(content="🛑 **Execution Paused:** I have halted my current task.")
            ],
            "clarification_question": (
                "I stopped processing. Your branch and PR (if any) are perfectly safe.\n\n"
                "• **To continue:** Just tell me what you want me to do next.\n"
                "• **To abandon this task & delete the branch:** Type `/cleanup`"
            ),
            "latest_traceback_error": None,  # Ensure we don't trigger a retry loop
        }

    tier = _resolve_execution_tier(state)
    requested_model = state.get("requested_model")

    breaker_abort = _check_circuit_breaker(state.get("messages", []))
    if breaker_abort:
        return breaker_abort

    target_ws = state.get("inferred_workspace", "None")
    target_path = state.get("workspace_absolute_path", "None")
    target_branch = state.get("target_branch", "main")

    # Sync check
    sync_abort_state = _perform_sync_check(state, target_path, target_branch, target_ws)
    if sync_abort_state:
        return sync_abort_state

    # Instantiate resilient fallback chain
    try:
        llms = get_execution_llm_sequence(requested_tier=tier, requested_model_key=requested_model)
        execution_chain = _build_fallback_chain(llms, tools=agent_tools)
    except TerminalEscalationError as e:
        return {
            "messages": [AIMessage(content=f"⚠️ **Escalation Failed:** {str(e)}")],
            "is_aborted": True,
        }

    # Context assembly
    messages = get_hybrid_context(
        messages=state["messages"],
        thread_id=thread_id,
        user_id=user_id,
        store=store,
        config=config,
        is_frontier_tier=(tier == TIER_FRONTIER),
    )
    cross_workspace_prompt = _build_cross_workspace_prompt(state, target_ws, target_path)

    # Prevent context poisoning. Strip previous orchestrator success messages from the LLM's view.
    # Otherwise, the LLM mimics these messages and hallucinates task completion
    # without actually calling any tools.
    filtered_messages = _filter_execution_context(messages)

    if filtered_messages and isinstance(filtered_messages[0], SystemMessage):
        filtered_messages[0] = SystemMessage(
            content=filtered_messages[0].content + "\n\n" + cross_workspace_prompt.content
        )
    else:
        filtered_messages.insert(0, cross_workspace_prompt)

    # Grab the existing queue of latex files before attempting LLM invocation
    new_tex_files = list(state.get("modified_tex_files", []))

    # LLM execution & sanitization with API Resilience
    try:
        response = execution_chain.invoke(filtered_messages, config=config)
    except Exception as e:
        # Catch strict API rate limits, timeouts, or JSON truncation errors
        retry_count = state.get("execution_retry_count", 0)
        t1_calls, t2_calls, t3_calls = _calculate_telemetry(state, tier)

        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = f"⚠️ **Execution Aborted:** Fatal API or Parsing Error -> {str(e)}"
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "modified_tex_files": new_tex_files,
                "t1_base_calls": t1_calls,
                "t2_standard_calls": t2_calls,
                "t3_frontier_calls": t3_calls,
            }

        # Inject a system rejection so the graph waits and loops cleanly
        recovery_msg = PromptManager.get(
            "execution", "api_truncation_recovery", error_details=str(e)
        ).strip()

        # Sleep briefly to respect API rate limits/timeouts
        time.sleep(3)

        return {
            "messages": [HumanMessage(content=recovery_msg)],
            "latest_traceback_error": "api_invocation_error",
            "execution_retry_count": retry_count + 1,
            "modified_tex_files": new_tex_files,
            "t1_base_calls": t1_calls,
            "t2_standard_calls": t2_calls,
            "t3_frontier_calls": t3_calls,
        }

    _sanitize_llm_response(response)

    # Trap any newly modified .tex files from the successful response
    new_tex_files = _extract_modified_tex_files(new_tex_files, response)

    # Increment telemetry
    t1_calls, t2_calls, t3_calls = _calculate_telemetry(state, tier)

    # Return the AIMessage natively. The graph will decide if tools should execute.
    return {
        "messages": [response],
        "modified_tex_files": new_tex_files,
        "t1_base_calls": t1_calls,
        "t2_standard_calls": t2_calls,
        "t3_frontier_calls": t3_calls,
        "latest_traceback_error": None,
    }


def _is_tool_error(tool_name: str, res_str: str) -> bool:
    """Helper to evaluate if a tool execution result constitutes a hard error."""
    # 1. Hard Invocation Errors
    if res_str.startswith("Tool execution failed:") or res_str.startswith("Error: Tool"):
        return True

    # 2. Execution & Sandbox Tools
    if tool_name in ["run_python_script", "run_pytest", "compile_latex_document"]:
        sandbox_signatures = [
            "Traceback (most recent call last):",
            "=== ERROR",
            "System Error:",
            "Execution Failed",
            "FAILED",
        ]
        if res_str.startswith("Error:") or any(sig in res_str for sig in sandbox_signatures):
            return True

    # 3. Read, Write, and OS Tools
    elif (
        res_str.startswith("Error:")
        or res_str.startswith("Failed:")
        or res_str.startswith("Security Exception:")
    ):
        return True

    return False


def _post_process_tool_messages(
    new_msgs: list, tool_calls: list, initial_retries: int
) -> tuple[str | None, list | None, str | None, int]:
    """Helper to extract business logic and error states from executed tools."""
    clarify_q = None
    disambig_opts = None
    latest_err = None
    retries = initial_retries

    for msg in new_msgs:
        tc = next((t for t in tool_calls if t["id"] == msg.tool_call_id), None)
        if tc:
            tool_name = tc["name"]
            if tool_name == "ask_user_for_clarification":
                clarify_q = tc["args"].get("question")
                disambig_opts = tc["args"].get("options")
                continue

            res_str = str(msg.content)
            if _is_tool_error(tool_name, res_str):
                retries += 1
                latest_err = res_str
        else:
            # It was an invalid tool call! Treat it as a hard error so retries increment
            retries += 1
            latest_err = str(msg.content)

    return clarify_q, disambig_opts, latest_err, retries


def workspace_tools_node(
    state: AgentState, config: RunnableConfig = None, store: BaseStore = None
) -> dict:
    """
    Custom node to natively execute tools sequentially.
    File system agents MUST execute tools sequentially to prevent race conditions
    when multiple tools attempt to edit the same file simultaneously.
    """
    store = store or InMemoryStore()
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    namespace = ("abort_signals", thread_id)

    new_msgs = []
    last_ai_msg = state["messages"][-1]
    tool_calls = getattr(last_ai_msg, "tool_calls", [])
    invalid_tool_calls = getattr(last_ai_msg, "invalid_tool_calls", [])

    is_paused = False

    # 1. Force sequential execution for valid tools
    for tc in tool_calls:
        tool_name = tc["name"]

        abort_item = store.get(namespace, "abort")
        if abort_item:
            is_paused = True
            store.delete(namespace, "abort")

        if is_paused:
            new_msgs.append(
                ToolMessage(
                    content="Error: Execution manually paused by user.",
                    name=tool_name,
                    tool_call_id=tc["id"],
                )
            )
            continue

        # Trap clarifications before executing physical tools
        if tool_name == "ask_user_for_clarification":
            res = "Clarification requested. Pausing execution."
        else:
            try:
                # Pass config into the registry execution
                res = execute_tool_call(tc, config)
            except Exception as e:
                res = f"Error: Tool execution failed: {str(e)}"

        new_msgs.append(ToolMessage(content=str(res), name=tool_name, tool_call_id=tc["id"]))

    # 2. Handle invalid tool calls directly so the LLM gets syntax feedback
    for itc in invalid_tool_calls:
        err = itc.get("error", "Invalid JSON arguments")
        new_msgs.append(
            ToolMessage(
                content=f"Error: Tool invocation failed due to malformed JSON. {err}",
                name=itc.get("name", "unknown_tool"),
                tool_call_id=itc.get("id", "unknown_id"),
            )
        )

    # 3. Extract context and post-process tool messages
    retries = state.get("execution_retry_count", 0)
    clarify_q, disambig_opts, latest_err, retries = _post_process_tool_messages(
        new_msgs, tool_calls, retries
    )

    # If the agent explicitly marked the task as complete, clear transient errors
    if any(tc.get("name") == "mark_task_already_completed" for tc in tool_calls):
        return {
            "messages": new_msgs,
            "execution_retry_count": state.get("execution_retry_count", 0),
            "latest_traceback_error": None,
        }

    if is_paused:
        pause_msg = "🛑 **Execution Paused:** I stopped the current tool loop."
        clarify_msg = (
            "I halted the tool execution early.\n\n"
            "• **To continue:** Just tell me what you want me to do next.\n"
            "• **To abandon this task & delete the branch:** Type `/cleanup`"
        )
        return {
            "messages": new_msgs + [AIMessage(content=pause_msg)],
            "clarification_question": clarify_msg,
            "latest_traceback_error": None,
        }

    # 5. Check for max retries abort condition
    if latest_err and retries >= settings.agent.max_sandbox_retries:
        abort_msg = (
            f"⚠️ **Execution Failed**\n"
            f"I was unable to fix the code after {retries} attempts.\n\n"
            f"**Latest Error:**\n`{latest_err}`\n\n"
            f"*Workflow safely aborted.*"
        )
        return {
            "messages": new_msgs + [AIMessage(content=abort_msg)],
            "is_aborted": True,
            "latest_traceback_error": None,
            "execution_retry_count": retries,
        }

    return {
        "messages": new_msgs,
        "clarification_question": clarify_q,
        "disambiguation_options": disambig_opts,
        "execution_retry_count": retries,
        "latest_traceback_error": latest_err,
    }


def force_tool_retry_node(state: AgentState) -> dict:
    """
    Appends a system message forcing the LLM to emit a tool call
    when it mistakenly replies with plain text during an operation intent.
    """
    prompt = PromptManager.get("execution", "force_tool_retry").strip()
    return {
        "messages": [SystemMessage(content=prompt)],
        "execution_retry_count": state.get("execution_retry_count", 0) + 1,
    }


# ==========================================
# Entity Memory & Profile Management
# ==========================================


class MemoryExtraction(BaseModel):
    """Schema for extracting both Entity and Episodic memories."""

    core_interests: list[str] = Field(
        description="Broad technical or personal topics the user cares about."
    )
    active_projects: dict[str, str] = Field(
        description="Map of project names to brief descriptions."
    )
    preferences: list[str] = Field(
        description=(
            "Explicit constraints, styling guidelines, or operational rules dictated directly by "
            "the user regarding how they want the agent to behave."
        )
    )
    operational_insights: list[str] = Field(
        description=(
            "Implicit systemic rules, codebase constraints, environmental quirks, or non-technical "
            "file constraints discovered autonomously by the agent during tool execution and "
            "sandbox runtime."
        )
    )


def update_memory_node(
    state: AgentState, config: RunnableConfig = None, store: BaseStore = None
) -> dict:
    """
    Asynchronous Memory extraction.
    Writes structured facts to the LangGraph Store and embeds operational insights into ChromaDB.
    """
    config = config or {}
    store = store or InMemoryStore()

    user_id = config.get("configurable", {}).get("user_id", "default")
    thread_id = config.get("configurable", {}).get("thread_id", "default")

    try:
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE)
        if not llms:
            return {}
        extractor = _build_fallback_chain(llms, structured_schema=MemoryExtraction)
    except TerminalEscalationError:
        return {}

    # Read the current entity profile from the Store
    namespace = ("user_profile", user_id)
    profile_item = store.get(namespace, "profile")
    current_profile = profile_item.value if profile_item else {}

    recent_messages = state["messages"][-10:]

    prompt = PromptManager.get(
        "memory",
        "extraction",
        current_profile=json.dumps(current_profile),
        recent_messages=recent_messages,
    ).strip()

    # Strip intermediate tool calls and system rejections to prevent context window poisoning
    messages_to_remove = []
    for m in state.get("messages", []):
        if isinstance(m, ToolMessage) or (
            isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        ):
            messages_to_remove.append(RemoveMessage(id=m.id))
        elif (
            isinstance(m, HumanMessage)
            and isinstance(m.content, str)
            and (m.content.startswith("SYSTEM REJECTION") or m.content.startswith("SYSTEM ERROR"))
        ):
            messages_to_remove.append(RemoveMessage(id=m.id))

    try:
        extraction = extractor.invoke(prompt, config=config)

        # Extract the ChromaDB collection from config
        collection = config.get("configurable", {}).get("chroma_collection")

        # 1. Save Entity Memory to LangGraph Store
        updated_profile = {
            "core_interests": extraction.core_interests,
            "active_projects": extraction.active_projects,
            "preferences": extraction.preferences,
        }
        store.put(namespace, "profile", updated_profile)

        # Mirror to disk for persistence across container restarts
        default_profile_path = os.path.join(os.getcwd(), "agent_state", "user_profile.json")
        profile_path = os.getenv("AGENT_PROFILE_PATH", default_profile_path)

        os.makedirs(os.path.dirname(profile_path), exist_ok=True)

        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(updated_profile, f, indent=2)
        except Exception as e:
            print(f"Failed to backup profile to disk: {e}")

        # 2. Save Episodic Memory to ChromaDB Vector Store
        for insight in extraction.operational_insights:
            if collection:  # Only save if collection successfully injected
                save_memory(
                    text=insight,
                    thread_id=thread_id,
                    collection=collection,
                    metadata={"type": "operational_insight"},
                )

        return {
            "t1_base_calls": state.get("t1_base_calls", 0) + 1,
            "messages": messages_to_remove,
            "is_busy": False,
        }
    except Exception as e:
        print(f"Memory extraction failed: {e}")
        return {
            "t1_base_calls": state.get("t1_base_calls", 0) + 1,
            "messages": messages_to_remove,
            "is_busy": False,
        }
