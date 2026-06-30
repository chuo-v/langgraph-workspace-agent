import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid

import psutil
import redis
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
from src.workspace_agent.core.state import AgentState, PRState
from src.workspace_agent.core.vector_memory import save_memory
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
    get_execution_llm_sequence,
    get_intent_router,
    get_tier_for_model,
)
from src.workspace_agent.tools.github import (
    cleanup_local_branch,
    comment_on_pull_request,
    create_branch_and_commit,
    get_git_diff,
    get_git_diff_blueprint,
    open_pull_request,
    set_commit_status,
    sync_repository,
    update_pull_request,
)
from src.workspace_agent.tools.registry import agent_tools, execute_tool_call

# ==========================================
# Module Configuration Constants
# ==========================================


MAX_CONTEXT_LENGTH = 500
MAX_DIFF_LENGTH = 40000
MAX_GITHUB_COMMENT_LENGTH = 60000
MAX_FAST_PATH_LEN = 15


# ==========================================
# Redis Initialization
# ==========================================
try:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.Redis.from_url(redis_url)
except Exception as e:
    print(f"Warning: Failed to initialize Redis in nodes.py: {e}")
    redis_client = None


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


# ==========================================
# Core Orchestration Nodes
# ==========================================


def _extract_tier_command(instruction: str) -> tuple[str, bool, bool, str | None]:
    """
    Detects tier and model override commands in a user instruction.
    Supports /use:standard, /use:frontier, and /use:<model_alias>.
    Returns (cleaned_instruction, has_frontier, has_standard, requested_model).
    """
    if not instruction:
        return "", False, False, None

    has_frontier = False
    has_standard = False
    requested_model = None

    # 1. Look for specific tier overrides (using whitespace boundaries to prevent path collisions)
    if re.search(r"(?i)(?:^|\s)/use:frontier(?=\s|$)", instruction):
        has_frontier = True
    elif re.search(r"(?i)(?:^|\s)/use:standard(?=\s|$)", instruction):
        has_standard = True

    # 2. Look for specific model aliases
    # Matches /use: followed by word characters, but ignores standard/frontier
    model_match = re.search(
        r"(?i)(?:^|\s)/use:(?!standard\b|frontier\b)([a-zA-Z0-9_]+)(?=\s|$)", instruction
    )
    if model_match:
        requested_model = model_match.group(1)

    # If no commands were found, return early
    if not (has_frontier or has_standard or requested_model):
        return instruction, False, False, None

    # 3. Clean the instruction safely by removing the command token and any extra spaces it leaves
    clean_instruction = re.sub(
        r"(?i)(?:^|\s)/use:(frontier|standard|[a-zA-Z0-9_]+)(?=\s|$)", "", instruction
    ).strip()

    clean_instruction = re.sub(r"\s+", " ", clean_instruction)

    return clean_instruction, has_frontier, has_standard, requested_model


def _invoke_escalating_router(payload: dict, initial_tier: int, config: RunnableConfig = None):
    """Helper to manage the timeout and tier-escalation logic for the unified router chain."""
    config = config or {}
    decision = None
    error_log = []
    current_tier = initial_tier

    while current_tier <= TIER_FRONTIER:
        router_chain = get_intent_router(requested_tier=current_tier)

        # if the requested tier is not configured (e.g., no API keys), skip to the next
        if not router_chain:
            error_log.append(f"Tier {current_tier} skipped: Not configured or missing API keys.")
            current_tier += 1
            continue

        def _invoke_router(chain=router_chain):
            # Create a localized configuration that strips out the async Langfuse callbacks.
            # This completely decouples the background thread from global telemetry locks,
            # preventing cold-boot cascading deadlocks.
            safe_config = config.copy()
            safe_config["callbacks"] = []

            # Pass the generic dictionary payload natively into the router chain
            return chain.invoke(payload, config=safe_config)

        # isolate the blocking HTTP call in a background thread manually.
        # a 'with' block is not used so we can execute a non-blocking
        # shutdown if the API takes too long, saving the server from freezing.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_invoke_router)
            # Pull the safe, extended timeout from config
            decision = future.result(timeout=settings.agent.router_timeout_seconds)

            # Clean up cleanly on success
            executor.shutdown(wait=False, cancel_futures=True)
            break  # Successful classification, exit the escalation loop

        except concurrent.futures.TimeoutError:
            timeout_sec = settings.agent.router_timeout_seconds
            error_log.append(f"Tier {current_tier} error: Thread timed out after {timeout_sec}s.")
            executor.shutdown(wait=False, cancel_futures=True)
            current_tier += 1

        except Exception as e:
            error_log.append(f"Tier {current_tier} error: {str(e)}")
            executor.shutdown(wait=False, cancel_futures=True)
            current_tier += 1

    latest_error = "\n".join(error_log) if error_log else None

    return decision, latest_error


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


def _get_latest_human_instruction(messages: list, fallback: str) -> str:
    """Extracts the absolute latest user feedback, explicitly ignoring system traps."""
    latest_human_msg = next(
        (
            m.content
            for m in reversed(messages)
            if m.type == "human"
            and not str(m.content).startswith("SYSTEM REJECTION")
            and not str(m.content).startswith("SYSTEM ERROR")
        ),
        "",
    )
    return latest_human_msg if latest_human_msg else fallback


def _check_conversational_fast_path(clean_instruction: str) -> bool:
    """
    Fast-path for simple conversational declines/acknowledgments to prevent
    router confusion (e.g., when the user says "No" to an AI-generated
    question that wasn't a formal tool interrupt).
    """
    quick_responses = {
        "no",
        "nope",
        "nah",
        "stop",
        "done",
        "thanks",
        "thank you",
        "no thanks",
        "goodbye",
        "bye",
        "ok",
        "okay",
    }
    check_str = re.sub(r"[^a-zA-Z\s]", "", clean_instruction.strip()).strip().lower()
    return len(check_str) < MAX_FAST_PATH_LEN and check_str in quick_responses


def _build_recent_context(messages: list) -> str:
    """Builds a truncated sliding window of the conversation history."""
    if not messages or len(messages) <= 1:
        return ""

    recent_msgs = messages[-4:]
    history_lines = []

    for m in recent_msgs:
        content_str = str(m.content)
        if len(content_str) > MAX_CONTEXT_LENGTH:
            content_str = content_str[:MAX_CONTEXT_LENGTH] + "... [TRUNCATED FOR ROUTING]"
        speaker = "User" if m.type == "human" else "Agent"
        history_lines.append(f"{speaker}: {content_str}")

    context_str = "\n".join(history_lines)
    return f"\nRecent Conversation Context:\n{context_str}\n"


def parse_intent_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Tier 1 Routing Node with Escalation.
    Analyzes the user's initial instruction and forces a structured JSON decision.
    Uses limited recent message history to handle conversational follow-ups naturally.
    """
    config = config or {}

    instruction = _get_latest_human_instruction(
        state.get("messages", []), state.get("original_instruction", "")
    )

    clean_instruction, has_frontier, has_standard, requested_model = _extract_tier_command(
        instruction
    )

    # 1. Fast-Path Bypass
    if _check_conversational_fast_path(clean_instruction):
        state_update = {
            "intent_category": "conversational",
            "inferred_workspace": None,
            "workspace_absolute_path": None,
            "target_branch": None,
            "router_confidence": 1.0,
            "force_frontier_tier": False,
            "force_standard_tier": False,
            "requested_model": requested_model,
            "clarification_question": None,
            "t1_base_calls": state.get("t1_base_calls", 0),
        }
        if has_frontier or has_standard or requested_model:
            state_update["original_instruction"] = clean_instruction
            if state.get("messages") and state["messages"][-1].type == "human":
                updated_message = HumanMessage(
                    content=clean_instruction, id=state["messages"][-1].id
                )
                state_update["messages"] = [updated_message]
        return state_update

    # 2. Build Payload
    payload = {
        "instruction": clean_instruction,
        "recent_context": _build_recent_context(state.get("messages", [])),
    }

    initial_tier = TIER_FRONTIER if has_frontier else (TIER_STANDARD if has_standard else TIER_BASE)

    # 3. Escalate & Route
    decision, latest_error = _invoke_escalating_router(payload, initial_tier, config)

    if not decision:
        abort_msg = (
            "⚠️ **Router Escalation Failed:** All router tiers timed out or failed "
            "while trying to classify this highly complex prompt. The connection has "
            "been safely severed to prevent the server from freezing.\n\n"
            f"**Latest Error:** `{latest_error}`\n\n"
            "*Tip: Try simplifying the initial prompt or verify your API keys for higher tiers.*"
        )
        return {
            "messages": state.get("messages", []) + [AIMessage(content=abort_msg)],
            "is_aborted": True,
        }

    # 4. Resolve Context & Workspace State
    absolute_path = None
    target_branch = None

    if decision.inferred_workspace and decision.inferred_workspace in settings.workspaces:
        workspace_config = settings.workspaces[decision.inferred_workspace]
        absolute_path = workspace_config.path
        target_branch = workspace_config.target_branch or settings.agent.target_branch

    clarification_question = None
    if decision.is_context_missing and decision.clarification_question_to_ask:
        clarification_question = decision.clarification_question_to_ask
        absolute_path = None

    force_frontier = has_frontier or (decision.task_complexity == "high")
    force_standard = (has_standard and not has_frontier) or (
        not force_frontier and (decision.task_complexity == "medium")
    )

    state_update = {
        "intent_category": decision.intent_category,
        "inferred_workspace": decision.inferred_workspace,
        "workspace_absolute_path": absolute_path,
        "target_branch": target_branch,
        "router_confidence": decision.router_confidence,
        "force_frontier_tier": force_frontier,
        "force_standard_tier": force_standard,
        "requested_model": requested_model,
        "clarification_question": clarification_question,
        "t1_base_calls": state.get("t1_base_calls", 0) + 1,
    }

    if has_frontier or has_standard or requested_model:
        state_update["original_instruction"] = clean_instruction
        if state.get("messages") and state["messages"][-1].type == "human":
            updated_message = HumanMessage(content=clean_instruction, id=state["messages"][-1].id)
            state_update["messages"] = [updated_message]

    return state_update


def conversational_reply_node(state: AgentState, config: RunnableConfig = None) -> dict:
    config = config or {}

    try:
        # Fetch sequence and construct fallback chain
        llms = get_execution_llm_sequence(
            requested_tier=TIER_BASE, requested_model_key=state.get("requested_model")
        )
        execution_chain = _build_fallback_chain(llms)
    except TerminalEscalationError as e:
        return {
            "messages": [AIMessage(content=f"⚠️ **Escalation Failed:** {str(e)}")],
            "is_aborted": True,
        }

    # Inject a system prompt to guide the conversational response and prevent hallucination
    system_prompt = SystemMessage(
        content=(
            "You are a helpful AI workspace assistant. Respond to the user's conversational "
            "message concisely. If the user is declining further assistance, saying goodbye, "
            "or acknowledging completion, respond politely and terminate the interaction. "
            "Do NOT hallucinate or simulate tool outputs, logs, or test results."
        )
    )

    # Prepend the system prompt dynamically without permanently saving it to state
    messages_to_send = [system_prompt] + state.get("messages", [])

    response = execution_chain.invoke(messages_to_send, config=config)

    return {
        "messages": [response],
        "t1_base_calls": state.get("t1_base_calls", 0) + 1,
        "is_busy": False,  # Ensure the workflow is safely unlocked for the next task
    }


def clarification_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop Node.
    Handles dynamic LLM questions, file disambiguation, and workspace routing errors.
    """
    question = state.get("clarification_question")
    options = state.get("disambiguation_options")

    # LLM asked a specific question
    if question:
        msg = f"🤔 **Question from Agent:**\n\n{question}"
        if options:
            formatted_options = "\n".join([f"- {opt}" for opt in options])
            msg += f"\n\n*Options:*\n{formatted_options}"

    # file disambiguation
    elif options:
        formatted_options = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        msg = (
            f"⚠️ **Multiple matching files found.**\n\n"
            f"Please specify which file you meant:\n{formatted_options}\n\n"
            f"*Reply with the exact path or number.*"
        )

    # workspace routing error
    else:
        available = "\n".join([f"- {key}" for key in settings.workspaces.keys()])
        msg = (
            f"⚠️ **Ambiguous Workspace Request.**\n\n"
            f"I wasn't sure which repository you meant. Here are your configured workspaces:\n"
            f"{available}\n\n"
            f"*Please reply with the target workspace.*"
        )

    # do not clear state here, or the route_after_human router will lose its breadcrumbs
    return {"messages": [AIMessage(content=msg)]}


def cleanup_workflow_node(state: AgentState) -> dict:
    """
    Final exit ramp for aborted tasks in the parent graph.
    Resets the busy lock, clears dangling conversational state,
    and handles Git repository hygiene.
    """
    last_message = state["messages"][-1] if state.get("messages") else None

    # Strip intermediate tool calls and system rejections
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

    if isinstance(last_message, HumanMessage):
        final_msg = "🚫 **Workflow Aborted.** All temporary state has been cleared."
        state_update = {"messages": messages_to_remove + [AIMessage(content=final_msg)]}
    else:
        state_update = {"messages": messages_to_remove}

    target_path = state.get("workspace_absolute_path")
    target_branch = state.get("target_branch", "main")
    agent_branch = state.get("active_agent_branch")

    if target_path and target_branch and agent_branch:
        try:
            cleanup_local_branch(target_path, target_branch, agent_branch)
        except Exception as e:
            # Safely trap git failures so the workflow can still unlock
            warning = f"\n\n*Git Cleanup Warning: {e}*"
            if "final_msg" in locals():
                final_msg += warning
                state_update["messages"][-1] = AIMessage(content=final_msg)
            else:
                state_update["messages"].append(AIMessage(content=warning))

    # Make sure to reset the requested_model when a graph cycle aborts/cleans up
    state_update.update(
        {
            "is_aborted": False,
            "latest_traceback_error": None,
            "execution_retry_count": 0,
            "clarification_question": None,
            "disambiguation_options": None,
            "modified_tex_files": [],
            "is_busy": False,
            "active_agent_branch": None,
            "pending_pr_url": None,
            "human_approved": False,
            "requested_model": None,
        }
    )
    return state_update


# ==========================================
# Task Execution Helpers
# ==========================================


def _perform_sync_check(
    state: AgentState, target_path: str, target_branch: str, target_ws: str
) -> dict | None:
    """Helper to verify and sync the target repository before execution."""
    is_tool_loop = state.get("messages") and state["messages"][-1].type == "tool"
    active_branch = state.get("active_agent_branch")
    sync_target = active_branch if active_branch else target_branch

    # only sync if there's a target path and we are not in the middle of a tool loop
    if target_path and target_path != "None" and not is_tool_loop:
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
        # deeply sanitize empty text blocks inside structural lists
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
                        content=(
                            "[System Note: The requested action has been completed "
                            "and the workflow cycle has concluded.]"
                        ),
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
                        content="[System Note: Conversational reply acknowledged.]",
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
    tier = _resolve_execution_tier(state)
    requested_model = state.get("requested_model")

    breaker_abort = _check_circuit_breaker(state.get("messages", []))
    if breaker_abort:
        return breaker_abort

    target_ws = state.get("inferred_workspace", "None")
    target_path = state.get("workspace_absolute_path", "None")
    target_branch = state.get("target_branch", "main")

    # sync check
    sync_abort_state = _perform_sync_check(state, target_path, target_branch, target_ws)
    if sync_abort_state:
        return sync_abort_state

    # instantiate resilient fallback chain
    try:
        llms = get_execution_llm_sequence(requested_tier=tier, requested_model_key=requested_model)
        execution_chain = _build_fallback_chain(llms, tools=agent_tools)
    except TerminalEscalationError as e:
        return {
            "messages": [AIMessage(content=f"⚠️ **Escalation Failed:** {str(e)}")],
            "is_aborted": True,
        }

    # context assembly
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

    # grab the existing queue of latex files before attempting LLM invocation
    new_tex_files = list(state.get("modified_tex_files", []))

    # LLM execution & sanitization with API Resilience
    try:
        response = execution_chain.invoke(filtered_messages, config=config)
    except Exception as e:
        # catch strict API rate limits, timeouts, or JSON truncation errors
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

        # inject a system rejection so the graph waits and loops cleanly
        recovery_msg = PromptManager.get(
            "execution", "api_truncation_recovery", error_details=str(e)
        ).strip()

        # sleep briefly to respect API rate limits/timeouts
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

    # trap any newly modified .tex files from the successful response
    new_tex_files = _extract_modified_tex_files(new_tex_files, response)

    # increment telemetry
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


def workspace_tools_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Custom node to natively execute tools sequentially.
    File system agents MUST execute tools sequentially to prevent race conditions
    when multiple tools attempt to edit the same file simultaneously.
    """
    new_msgs = []
    last_ai_msg = state["messages"][-1]
    tool_calls = getattr(last_ai_msg, "tool_calls", [])
    invalid_tool_calls = getattr(last_ai_msg, "invalid_tool_calls", [])

    # 1. Force sequential execution for valid tools
    for tc in tool_calls:
        tool_name = tc["name"]

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

    # 3. Extract context for state updates
    retries = state.get("execution_retry_count", 0)
    clarify_q = None
    disambig_opts = None
    latest_err = None

    # 4. Post-process the ToolMessages to enforce business logic
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

    # If the agent explicitly marked the task as complete, clear transient errors
    if any(tc.get("name") == "mark_task_already_completed" for tc in tool_calls):
        return {
            "messages": new_msgs,
            "execution_retry_count": state.get("execution_retry_count", 0),
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


def compile_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Automated graph node that forcefully compiles any modified LaTeX files.
    Provides an autonomous feedback loop to the LLM if compilation fails.
    """
    files_to_compile = state.get("modified_tex_files", [])
    retry_count = state.get("execution_retry_count", 0)

    errors = []
    for file_path in files_to_compile:
        tc = {"name": "compile_latex_document", "args": {"tex_file_path": file_path}}
        res = execute_tool_call(tc, config)

        if "Compilation Finished (Exit Code: 0)" not in res:
            errors.append(f"File {file_path}:\n{res}")

    if errors:
        retry_count += 1
        error_str = "\n\n".join(errors)

        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = (
                f"⚠️ **Compilation Failed**\n"
                f"I was unable to fix the LaTeX syntax after {retry_count} attempts.\n\n"
                f"**Latest Error:**\n`{error_str}`\n\n"
                f"*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }
        else:
            msg = PromptManager.get("compilation", "latex_rejection", error_str=error_str).strip()

            return {
                "messages": [HumanMessage(content=msg)],
                "latest_traceback_error": "latex_compilation_error",
                "execution_retry_count": retry_count,
            }

    return {
        "modified_tex_files": [],  # successfully compiled, clear the queue
        "latest_traceback_error": None,
    }


def run_pre_commit_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Executes pre-commit tools (e.g., formatters, linters, type-checkers).
    If a tool fails and cannot auto-fix, it routes the error back to the LLM.
    """
    target_path = state.get("workspace_absolute_path")
    retry_count = state.get("execution_retry_count", 0)

    workspace_config = next(
        (ws for ws in settings.workspaces.values() if ws.path == target_path), None
    )

    # Bypass if not configured for this specific workspace
    if not workspace_config or not workspace_config.pre_commit_suites:
        return {"latest_traceback_error": None}

    expanded_target_path = os.path.expanduser(target_path)
    errors = []

    for suite in workspace_config.pre_commit_suites:
        try:
            process = subprocess.run(
                suite.command,
                cwd=expanded_target_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=suite.timeout_seconds,
                check=False,
            )

            # Non-zero exit code indicates an unfixable error (e.g., strict typing issue)
            if process.returncode != 0:
                errors.append(f"Suite '{suite.name}' failed:\n{process.stdout}\n{process.stderr}")

        except Exception as e:
            errors.append(f"Suite '{suite.name}' OS Error: {str(e)}")

    if errors:
        retry_count += 1
        error_str = "\n\n".join(errors)

        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = (
                f"⚠️ **Pre-Commit Checks Failed**\n"
                f"I was unable to satisfy the repository's pre-commit rules after "
                f"{retry_count} attempts.\n\n"
                f"**Latest Error:**\n`{error_str}`\n\n"
                f"*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }

        # Inject the pre-commit feedback directly into the agent's context
        msg = (
            f"SYSTEM ERROR: The pre-commit checks failed. Please review the output below "
            f"and use your tools to fix the remaining issues before I can open the Pull "
            f"Request:\n\n```text\n{error_str}\n```"
        )
        return {
            "messages": [HumanMessage(content=msg)],
            "latest_traceback_error": "pre_commit_error",
            "execution_retry_count": retry_count,
        }

    # Successfully passed all checks
    return {"latest_traceback_error": None}


def _get_evaluation_diffs(target_path: str, target_branch: str) -> tuple[str | None, str | None]:
    """Helper to stage intent-to-add files and fetch git diffs."""
    try:
        try:
            # Stage intent-to-add for all untracked files so they appear in git diff
            subprocess.run(
                ["git", "add", "-N", "."], cwd=os.path.expanduser(target_path), check=False
            )
        except Exception:
            pass

        # The cumulative diff for the whole PR (against the target branch)
        raw_diff = get_git_diff(directory=target_path, target_branch=target_branch)
        # The incremental diff for the new uncommitted changes (against HEAD)
        incremental_diff = get_git_diff(directory=target_path)
        return raw_diff, incremental_diff
    except Exception:
        return None, None


def _handle_evaluation_pass(
    state: PRState,
    is_empty_diff: bool,
    is_empty_incremental: bool,
    explicit_escape: bool,
    retry_count: int,
) -> dict:
    """Helper to process a successful critic evaluation, trapping LLM hallucinations."""
    if is_empty_diff and is_empty_incremental:
        # Trap LLM Tool Hallucinations: If it was supposed to operate on the workspace
        # but didn't explicitly use the escape hatch, it likely hallucinated task completion.
        if not explicit_escape and state.get("intent_category") == "workspace_operation":
            if retry_count < settings.agent.max_sandbox_retries:
                msg = (
                    "SYSTEM ERROR: No files were modified. You must use the provided tools "
                    "(like write_file or search_and_replace) to fulfill the user's request "
                    "before stating you are finished."
                )
                return {
                    "messages": [HumanMessage(content=msg)],
                    "latest_traceback_error": "hallucinated_success",
                    "execution_retry_count": retry_count + 1,
                    "t1_base_calls": state.get("t1_base_calls", 0) + 1,
                }

            abort_msg = (
                "⚠️ **Execution Failed:** I was unable to invoke the necessary tools "
                f"to satisfy the requirements after {retry_count} attempts.\n\n"
                "*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }

        # Valid read-only pass
        return {"latest_traceback_error": None, "intent_category": "workspace_read_only"}

    # Standard pass with valid changes
    return {"latest_traceback_error": None}


def _handle_evaluation_fail(eval_result: str, state: PRState, retry_count: int) -> dict:
    """Helper to process a failed critic evaluation, triggering a retry loop or aborting."""
    if retry_count >= settings.agent.max_sandbox_retries:
        abort_msg = (
            "⚠️ **Execution Failed:** I was unable to satisfy the requirements after "
            f"{retry_count} attempts.\n\n"
            f"*Critic Feedback:* {eval_result}\n\n*Workflow safely aborted.*"
        )
        return {
            "messages": [AIMessage(content=abort_msg)],
            "is_aborted": True,
            "latest_traceback_error": None,
            "execution_retry_count": retry_count,
        }

    feedback = eval_result.replace("FAIL:", "").strip()
    msg = PromptManager.get("evaluation", "semantic_rejection", feedback=feedback).strip()

    return {
        "messages": [HumanMessage(content=msg)],
        "latest_traceback_error": "semantic_review_rejection",
        "execution_retry_count": retry_count + 1,
        "t1_base_calls": state.get("t1_base_calls", 0) + 1,
    }


def evaluate_diff_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Reflection Node: Uses git diff and an LLM critic to verify the agent actually
    completed the requested task before opening a PR.
    """
    config = config or {}
    target_path = state.get("workspace_absolute_path")
    target_branch = state.get("target_branch", "main")
    retry_count = state.get("execution_retry_count", 0)
    messages = state.get("messages", [])

    # Scan the most recent messages to see if the agent explicitly used the escape hatch
    explicit_escape = False
    for msg in reversed(messages):
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            if any(tc["name"] == "mark_task_already_completed" for tc in msg.tool_calls):
                explicit_escape = True
                # The agent correctly identified the task was already done.
                # Convert the intent to read_only to gracefully bypass PR creation!
                return {"latest_traceback_error": None, "intent_category": "workspace_read_only"}
            break  # Only evaluate the single most recent AI action

    raw_diff, incremental_diff = _get_evaluation_diffs(target_path, target_branch)
    if raw_diff is None:
        # If git fails locally, pass it through; review_pr_node has fallback handlers
        return {"latest_traceback_error": None}

    # Determine if diffs are effectively empty
    is_empty_diff = (
        not raw_diff or "No uncommitted changes" in raw_diff or "No changes compared to" in raw_diff
    )
    is_empty_incremental = (
        not incremental_diff
        or "No uncommitted changes" in incremental_diff
        or "No changes compared to" in incremental_diff
    )

    # Extract the true user request (ignoring our injected SYSTEM REJECTIONS)
    latest_human_msg = next(
        (
            m.content
            for m in reversed(state.get("messages", []))
            if m.type == "human"
            and not str(m.content).startswith("SYSTEM REJECTION")
            and not str(m.content).startswith("SYSTEM ERROR")
        ),
        "",
    )
    original_instruction = state.get("original_instruction", "")
    instruction_context = latest_human_msg if latest_human_msg else original_instruction

    # Extract the agent's final summary from the message history
    agent_summary = next(
        (str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "No summary provided.",
    )

    critic_prompt = PromptManager.get(
        "evaluation",
        "critic_prompt",
        instruction_context=instruction_context,
        agent_summary=agent_summary,
        raw_diff=raw_diff if not is_empty_diff else "[NO CUMULATIVE CHANGES TO REPOSITORY]",
        incremental_diff=incremental_diff
        if not is_empty_incremental
        else "[NO NEW UNCOMMITTED CHANGES]",
    ).strip()

    try:
        # Use the base tier LLM for fast, reliable evaluation, wrapping it in fallbacks
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.0)
        eval_chain = _build_fallback_chain(llms)
        eval_result = eval_chain.invoke(critic_prompt, config=config).content.strip()
    except Exception:
        # Degrade gracefully if API fails and pass it through
        return {"latest_traceback_error": None}

    if eval_result.startswith("PASS"):
        return _handle_evaluation_pass(
            state, is_empty_diff, is_empty_incremental, explicit_escape, retry_count
        )

    return _handle_evaluation_fail(eval_result, state, retry_count)


def force_tool_retry_node(state: AgentState) -> dict:
    """
    Appends a system message forcing the LLM to emit a tool call
    when it mistakenly replies with plain text during an operation intent.
    """
    return {
        "messages": [
            SystemMessage(
                content=(
                    "CRITICAL: You are performing a workspace operation but failed "
                    "to invoke any tools. Do not narrate your intentions. You MUST "
                    "execute the appropriate tool function now."
                )
            )
        ],
        "execution_retry_count": state.get("execution_retry_count", 0) + 1,
    }


def _execute_ci_suites(
    target_path: str,
    expanded_target_path: str,
    active_commit_sha: str,
    ci_suites: list,
    repo_full_name: str | None,
    run_id: str,
) -> list[dict]:
    """Helper to isolate CI subprocess executions and status reporting."""
    # 1. Post pending statuses for all suites upfront
    for suite in ci_suites:
        context_str = f"Agentic CI / {suite.name}"
        set_commit_status(
            directory=target_path,
            commit_sha=active_commit_sha,
            state="pending",
            context_str=context_str,
            description="Evaluation is running...",
            repo_full_name=repo_full_name,
        )

    results = []

    # Inject the uv .venv into the execution environment
    env = os.environ.copy()
    venv_path = os.path.join(expanded_target_path, ".venv")
    if os.path.exists(venv_path):
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = f"{os.path.join(venv_path, 'bin')}:{env.get('PATH', '')}"

    # 2. Execute each suite strictly sequentially
    for suite in ci_suites:
        # Check if a newer commit preempted us before we even start this suite
        if redis_client and redis_client.get(f"ci_superseded:{run_id}"):
            raise InterruptedError("Superseded by a newer commit.")

        context_str = f"Agentic CI / {suite.name}"
        try:
            process = subprocess.Popen(
                suite.command,
                cwd=expanded_target_path,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            # Store the PID so a newer commit can preemptively kill this process
            if redis_client:
                redis_client.set(f"ci_pid:{target_path}", process.pid)

            try:
                stdout, stderr = process.communicate(timeout=suite.timeout_seconds)
                passed = process.returncode == 0
                logs = f"{stdout}\n{stderr}".strip()
            except subprocess.TimeoutExpired:
                # Force kill the process tree instantly using psutil
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

                # Grab whatever output was generated before the timeout kill
                stdout, stderr = process.communicate()
                logs = (
                    f"{stdout}\n{stderr}\n\n"
                    f"[ERROR: TimeoutExpired] Command forcefully terminated after "
                    f"{suite.timeout_seconds} seconds."
                ).strip()
                passed = False
            finally:
                # If we were killed by a newer commit, psutil causes communicate()
                # to return instantly. We trap that here and abort cleanly before
                # posting false test results to GitHub.
                if redis_client and redis_client.get(f"ci_superseded:{run_id}"):
                    raise InterruptedError("Superseded by a newer commit.")

            set_commit_status(
                directory=target_path,
                commit_sha=active_commit_sha,
                state="success" if passed else "failure",
                context_str=context_str,
                description="Evaluation passed"
                if passed
                else "Execution timed out"
                if not passed and "[ERROR: TimeoutExpired]" in logs
                else "Evaluation failed",
                repo_full_name=repo_full_name,
            )

            results.append({"name": suite.name, "passed": passed, "logs": logs})

        except InterruptedError:
            # Bubble up the intentional abort signal
            raise
        except Exception as e:
            set_commit_status(
                directory=target_path,
                commit_sha=active_commit_sha,
                state="error",
                context_str=context_str,
                description="OS error during execution",
                repo_full_name=repo_full_name,
            )
            results.append({"name": suite.name, "passed": False, "logs": f"OS Error: {str(e)}"})

    return results


def _handle_concurrency_rejection(
    target_path: str,
    pr_number: int,
    commit_sha: str | None,
    repo_full_name: str | None,
    is_already_running: bool,
    active_jobs: int,
) -> None:
    """Helper to post rejection messages when the CI concurrency limit is reached."""
    max_jobs = settings.agent.max_concurrent_ci_jobs
    reason = (
        "A CI/CD job is already running for this repository"
        if is_already_running
        else f"Global concurrency limit ({max_jobs}) reached"
    )
    print(f"Agentic CI/CD skipped: {reason}. ({active_jobs}/{max_jobs})")

    comment_body = (
        f"⚠️ **Agentic CI/CD Skipped**\n\n"
        f"{reason} to preserve system resources. "
        f"If necessary, please comment `/retest` to try again once the current tests finish."
    )

    comment_on_pull_request(
        directory=target_path,
        pr_number=pr_number,
        body=comment_body,
        repo_full_name=repo_full_name,
    )

    if commit_sha:
        set_commit_status(
            directory=target_path,
            commit_sha=commit_sha,
            state="error",
            context_str="Agentic CI/CD / Concurrency",
            description="Concurrency limit reached. If necessary, comment /retest later.",
            repo_full_name=repo_full_name,
        )


def _post_ci_results_comment(
    target_path: str, pr_number: int, repo_full_name: str | None, results: list[dict]
) -> None:
    """Helper to format and post the consolidated CI results to GitHub."""
    if not results:
        return

    summary_table = "| Suite | Status |\n|---|---|\n"
    for r in results:
        status_icon = "✅ Pass" if r["passed"] else "❌ Fail"
        summary_table += f"| {r['name']} | {status_icon} |\n"

    details_sections = ""
    for r in results:
        safe_logs = (
            r["logs"][:MAX_GITHUB_COMMENT_LENGTH] + "\n...[TRUNCATED]"
            if len(r["logs"]) > MAX_GITHUB_COMMENT_LENGTH
            else r["logs"]
        )

        details_sections += (
            f"\n<details><summary>Logs: {r['name']}</summary>\n\n"
            f"```text\n{safe_logs}\n```\n</details>\n"
        )

    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    comment_body = f"## {prefix_str}Agentic CI/CD Results\n\n{summary_table}\n{details_sections}"

    comment_on_pull_request(
        directory=target_path,
        pr_number=pr_number,
        body=comment_body,
        repo_full_name=repo_full_name,
    )


def _preempt_stale_ci_run(workspace_lock_id: str, pid_key: str) -> None:
    """Helper to assassinate an older, superseded Agentic CI test run."""
    if not redis_client:
        return

    old_run_id = redis_client.get(workspace_lock_id)
    if not old_run_id:
        return

    old_run_id = old_run_id.decode("utf-8") if isinstance(old_run_id, bytes) else old_run_id
    print(f"Agentic CI: Preempting older execution {old_run_id} for newest commit.")

    # Mark old run as superseded
    redis_client.setex(f"ci_superseded:{old_run_id}", 3600, "1")

    # Send SIGKILL to the old test suite process tree
    old_pid = redis_client.get(pid_key)
    if old_pid:
        try:
            parent = psutil.Process(int(old_pid))
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except (psutil.NoSuchProcess, TypeError, ValueError):
            pass

    # Brief pause to let the old LangGraph thread trap
    # the InterruptedError and exit cleanly
    time.sleep(1.0)


def _acquire_preemptive_lock(
    target_path: str,
    run_id: str,
    pr_number: int,
    commit_sha: str | None,
    repo_full_name: str | None,
) -> bool:
    """Attempts to acquire the CI lock, preempting older runs if necessary."""
    if not redis_client:
        return True

    workspace_lock_id = f"ci_job:{target_path}"
    pid_key = f"ci_pid:{target_path}"

    try:
        is_already_running = redis_client.sismember("active_ci_jobs", workspace_lock_id)
        active_jobs = redis_client.scard("active_ci_jobs")
        limit_reached = active_jobs >= settings.agent.max_concurrent_ci_jobs

        # Only reject if we hit the limit AND it's not our own repo we can preempt
        if limit_reached and not is_already_running:
            _handle_concurrency_rejection(
                target_path,
                pr_number,
                commit_sha,
                repo_full_name,
                False,
                active_jobs,
            )
            return False

        if is_already_running:
            _preempt_stale_ci_run(workspace_lock_id, pid_key)

        # Acquire Lock for this new run
        redis_client.sadd("active_ci_jobs", workspace_lock_id)
        redis_client.set(workspace_lock_id, run_id)
        return True
    except Exception as e:
        print(f"Warning: Redis concurrency check failed: {e}")
        # Fail-open if Redis crashes temporarily so workflow isn't fully blocked
        return True


def _release_preemptive_lock_and_cleanup(
    target_path: str,
    run_id: str,
    tmp_dir: str | None = None,
) -> None:
    """Releases the redis lock if we still own it, and safely GC's the tmp dir."""
    workspace_lock_id = f"ci_job:{target_path}"
    pid_key = f"ci_pid:{target_path}"

    if redis_client:
        try:
            current_owner = redis_client.get(workspace_lock_id)
            if current_owner:
                current_owner = (
                    current_owner.decode("utf-8")
                    if isinstance(current_owner, bytes)
                    else current_owner
                )

                if current_owner == run_id:
                    # We still own the lock, clean it up
                    redis_client.srem("active_ci_jobs", workspace_lock_id)
                    redis_client.delete(workspace_lock_id)
                    redis_client.delete(pid_key)
                # If a newer commit preempted us, we just leave the lock alone.
        except Exception as e:
            print(f"Warning: Failed to verify Redis lock ownership: {e}")

    # 5. Cleanup: Delete the ephemeral directory.
    # We do this regardless of lock ownership because every run gets a unique tmp_dir!
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
            print(f"Agentic CI: Ephemeral workspace {tmp_dir} cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Failed to GC temporary CI directory {tmp_dir}: {e}")


def agentic_ci_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Agentic CI/CD Execution Node.
    Iterates through configured CI suites, executes them via OS-level subprocess isolation,
    and natively reports status checks and log aggregations back to the GitHub Pull Request.
    """
    target_path = state.get("workspace_absolute_path")
    repo_full_name = state.get("repo_full_name")
    commit_sha = state.get("commit_sha")
    pr_number = state.get("pr_number")

    if not target_path or not pr_number:
        return {}

    workspace_config = next(
        (ws for ws in settings.workspaces.values() if ws.path == target_path), None
    )

    if not workspace_config or not workspace_config.ci_suites:
        return {"ci_results": []}

    # ==========================================
    # 1. Redis Concurrency Lock (Preemptive)
    # ==========================================
    run_id = str(uuid.uuid4())

    if not _acquire_preemptive_lock(target_path, run_id, pr_number, commit_sha, repo_full_name):
        return {"ci_results": []}

    # Lock acquired (or bypassed via fail-open), proceed with execution
    expanded_target_path = os.path.expanduser(target_path)
    tmp_dir = None

    try:
        # 0. Generate unique ephemeral directory
        tmp_dir = tempfile.mkdtemp(prefix=f"ci_job_{run_id}_")
        print(f"Agentic CI: Creating ephemeral workspace at {tmp_dir}")

        # 1. Isolate: Clone the host repository to the /tmp directory locally (extremely fast)
        subprocess.run(
            ["git", "clone", expanded_target_path, tmp_dir], check=True, capture_output=True
        )

        # 2. Checkout exact PR commit from origin
        # We inject the credential helper just in case the remote requires auth
        github_token = os.getenv("GITHUB_TOKEN", "")
        cred_helper = (
            f'!f() {{ echo "username=x-access-token"; echo "password={github_token}"; }}; f'
        )

        # Explicitly build the GitHub URL so we don't fetch from the local 'origin'
        github_url = f"https://github.com/{repo_full_name}.git"

        subprocess.run(
            [
                "git",
                "-c",
                f"credential.helper={cred_helper}",
                "fetch",
                github_url,
                f"pull/{pr_number}/head",
            ],
            cwd=tmp_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "FETCH_HEAD"], cwd=tmp_dir, check=True, capture_output=True
        )

        active_commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_dir)
            .decode("utf-8")
            .strip()
        )

        # 3. Dependency Caching: Use uv to build an ephemeral .venv lightning fast
        print(f"Agentic CI: Building isolated .venv using uv in {tmp_dir}...")
        subprocess.run(["uv", "venv"], cwd=tmp_dir, check=True, capture_output=True)

        req_file = None
        if os.path.exists(os.path.join(tmp_dir, "requirements-dev.txt")):
            req_file = "requirements-dev.txt"
        elif os.path.exists(os.path.join(tmp_dir, "requirements.txt")):
            req_file = "requirements.txt"

        if req_file:
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = os.path.join(tmp_dir, ".venv")
            env["PATH"] = f"{os.path.join(tmp_dir, '.venv', 'bin')}:{env.get('PATH', '')}"
            subprocess.run(
                ["uv", "pip", "install", "-r", req_file],
                cwd=tmp_dir,
                env=env,
                check=True,
                capture_output=True,
            )
            print(f"Agentic CI: Dependencies installed successfully from {req_file}.")

        # 4. Execute all suites in the isolated environment
        results = _execute_ci_suites(
            target_path=target_path,
            expanded_target_path=tmp_dir,
            active_commit_sha=active_commit_sha,
            ci_suites=workspace_config.ci_suites,
            repo_full_name=repo_full_name,
            run_id=run_id,
        )

        # 5. Post Consolidated PR Comment
        _post_ci_results_comment(target_path, pr_number, repo_full_name, results)

        return {"ci_results": results}

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        print(f"Agentic CI setup error: {err_msg}", flush=True)
        if commit_sha:
            set_commit_status(
                directory=target_path,
                commit_sha=commit_sha,
                state="error",
                context_str="Agentic CI / Setup",
                description="Failed to prepare isolated CI environment.",
                repo_full_name=repo_full_name,
            )
        return {"ci_results": []}

    except InterruptedError as e:
        # We were gracefully aborted by a newer commit, exit quietly.
        print(f"Agentic CI gracefully aborted: {e}", flush=True)
        return {"ci_results": []}

    finally:
        # 6. Release Concurrency Lock & GC the isolated directory
        _release_preemptive_lock_and_cleanup(target_path, run_id, tmp_dir=tmp_dir)


# ==========================================
# PR Review Helpers
# ==========================================


def _generate_commit_message(
    instruction_context: str, incremental_diff: str, blueprint: str, config: RunnableConfig = None
) -> str:
    """Helper to generate a semantic commit message using the LLM based on the actual diff."""
    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    if not instruction_context and not incremental_diff:
        return f"{prefix_str}Apply human feedback revisions"

    config = config or {}
    try:
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.2)
        eval_chain = _build_fallback_chain(llms)

        # Fast track for huge diffs: just use the blueprint and the first chunk
        chunks = _chunk_git_diff(incremental_diff, MAX_DIFF_LENGTH)
        diff_snippet = chunks[0] if chunks else ""

        diff_str = f"\nACTUAL CODE DIFF SNIPPET:\n{diff_snippet}\n" if diff_snippet else ""

        prompt = PromptManager.get(
            "pr_generation",
            "commit_message",
            instruction_context=instruction_context,
            blueprint=blueprint,
            diff_snippet=diff_str,
        ).strip()

        res = eval_chain.invoke(prompt, config=config).content.strip().strip("\"'")
        if res:
            return f"{prefix_str}{res}"
    except Exception as e:
        print(f"Error generating commit message: {e}")

    return f"{prefix_str}Apply human feedback revisions"


def _generate_pr_metadata(
    instruction_context: str, raw_diff: str, blueprint: str, config: RunnableConfig = None
) -> tuple[str, str, str]:
    """Helper to generate semantic branch names and PR descriptions using Map-Reduce."""
    config = config or {}
    try:
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.2)
        eval_chain = _build_fallback_chain(llms)
        chunks = _chunk_git_diff(raw_diff, MAX_DIFF_LENGTH)

        # Map phase: Anchor each chunk with the Blueprint and parse file-by-file
        if len(chunks) > 1:
            chunk_summaries = []
            for i, chunk_text in enumerate(chunks):
                map_prompt = PromptManager.get(
                    "pr_generation",
                    "map_phase",
                    instruction_context=instruction_context,
                    blueprint=blueprint,
                    part_num=i + 1,
                    total_parts=len(chunks),
                    chunk_text=chunk_text,
                ).strip()

                res = eval_chain.invoke(map_prompt, config=config).content.strip()
                chunk_summaries.append(res)

            synthesized_diff = "\n\n".join(
                [f"--- Part {i + 1} Summary ---\n{s}" for i, s in enumerate(chunk_summaries)]
            )
        else:
            synthesized_diff = chunks[0] if chunks else ""

        syn_diff_str = (
            f"\nACTUAL SYNTHESIZED DIFF SUMMARY:\n{synthesized_diff}\n" if synthesized_diff else ""
        )

        summary_prompt = PromptManager.get(
            "pr_generation",
            "summary_phase",
            instruction_context=instruction_context,
            blueprint=blueprint,
            synthesized_diff=syn_diff_str,
        ).strip()

        generated_summary = (
            eval_chain.invoke(summary_prompt, config=config).content.strip().strip("\"'")
        )

        body_prompt = PromptManager.get(
            "pr_generation",
            "body_phase",
            instruction_context=instruction_context,
            blueprint=blueprint,
            synthesized_diff=syn_diff_str,
        ).strip()

        generated_body = eval_chain.invoke(body_prompt, config=config).content.strip()
    except Exception as e:
        print(f"Error in PR generation map-reduce: {e}")
        generated_summary = "Automated agent modifications"
        generated_body = "Automated PR generated by the LangGraph Agent."

    clean_slug = re.sub(r"[^a-zA-Z0-9\s]", "", generated_summary).strip()
    clean_slug = re.sub(r"\s+", "-", clean_slug).lower()
    short_hash = str(uuid.uuid4())[:4]

    branch_name = f"agent/{clean_slug}-{short_hash}"

    return branch_name, generated_body, generated_summary


def _chunk_git_diff(raw_diff: str, max_chunk_length: int = MAX_DIFF_LENGTH) -> list[str]:
    """Splits a raw git diff into manageable chunks safely along file boundaries."""
    if not raw_diff or "No uncommitted changes" in raw_diff or "No changes compared to" in raw_diff:
        return []

    parts = raw_diff.split("diff --git ")
    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue
        file_diff = "diff --git " + part

        # If a single file diff is too massive, truncate just that specific file
        if len(file_diff) > max_chunk_length:
            half = max_chunk_length // 2
            file_diff = (
                file_diff[:half] + "\n\n...[single file diff truncated]...\n\n" + file_diff[-half:]
            )

        # If adding this file pushes the current chunk over the limit, flush the chunk
        if current_chunk and len(current_chunk) + len(file_diff) > max_chunk_length:
            chunks.append(current_chunk.strip())
            current_chunk = file_diff
        else:
            current_chunk += "\n" + file_diff if current_chunk else file_diff

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _attempt_pr_update(
    directory: str,
    pending_pr_url: str,
    pr_title: str,
    generated_body: str,
    generated_summary: str,
) -> str:
    """Attempts to patch an existing GitHub PR. Returns a status message string."""
    if not directory or not pending_pr_url:
        return ""

    try:
        pr_number = int(pending_pr_url.rstrip("/").split("/")[-1])
        update_res = json.loads(
            update_pull_request(
                directory=directory,
                pr_number=pr_number,
                title=pr_title,
                body=generated_body,
            )
        )
        if update_res.get("status") == "success":
            return (
                "\n*I also updated the PR title to reflect the "
                f"latest direction: '{generated_summary}'*"
            )
    except Exception:
        # gracefully degrade if URL parsing or API call fails
        pass

    return ""


def _handle_commit_failure(commit_response: dict, retry_count: int, branch_name: str) -> dict:
    """Helper to handle self-healing loops or aborts on commit failures."""
    if commit_response.get("reason") == "no_changes_to_commit":
        if retry_count >= settings.agent.max_sandbox_retries:
            return {
                "messages": [
                    AIMessage(
                        content="⚠️ **Execution Failed:** I was unable to successfully "
                        "modify any files after multiple attempts. *Workflow safely aborted.*"
                    )
                ],
                "is_aborted": True,
            }
        return {
            "messages": [
                HumanMessage(
                    content=PromptManager.get("pr_generation", "no_changes_rejection").strip()
                )
            ],
            "latest_traceback_error": "no_changes_to_commit",
            "execution_retry_count": retry_count + 1,
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    f"⚠️ **Commit & Push Failed:** Could not push updates to `{branch_name}`.\n"
                    f"**Reason:** {commit_response.get('reason')}\n"
                    f"**Details:** `{commit_response.get('details', 'No details provided')}`"
                )
            )
        ],
        "is_aborted": True,
    }


def review_pr_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Branch-and-Link execution.
    Branches the code, commits, pushes, and suspends for human review.
    """
    config = config or {}
    target_path = state.get("workspace_absolute_path")
    active_branch = state.get("active_agent_branch")
    pending_pr_url = state.get("pending_pr_url")

    # determine if we are updating an existing PR or making a new one
    is_refining = bool(active_branch and pending_pr_url)
    repo_name = target_path.split("/")[-1]
    target_branch = state.get("target_branch", "main")

    # Extract the absolute latest user feedback, explicitly ignoring system traps
    latest_human_msg = next(
        (
            m.content
            for m in reversed(state.get("messages", []))
            if m.type == "human"
            and not str(m.content).startswith("SYSTEM REJECTION")
            and not str(m.content).startswith("SYSTEM ERROR")
        ),
        "",
    )
    original_instruction = state.get("original_instruction", "")

    # Build a holistic instruction context so the LLM doesn't lose the forest for the trees
    if is_refining and latest_human_msg and latest_human_msg != original_instruction:
        instruction_context = (
            f"Original PR Goal: {original_instruction}\n"
            f"Latest Revision Requested: {latest_human_msg}"
        )
    else:
        instruction_context = latest_human_msg if latest_human_msg else original_instruction

    # 1. Grab the raw diffs and blueprints
    try:
        # The cumulative diff for the whole PR (against the target branch)
        raw_diff = get_git_diff(directory=target_path, target_branch=target_branch)
        blueprint = get_git_diff_blueprint(directory=target_path, target_branch=target_branch)

        # The incremental diff for the new commit (against HEAD)
        incremental_diff = get_git_diff(directory=target_path)
        incremental_blueprint = get_git_diff_blueprint(directory=target_path)
    except Exception:
        raw_diff = ""
        blueprint = "Unknown Blueprint"
        incremental_diff = ""
        incremental_blueprint = "Unknown Blueprint"

    # 2. Generate the dynamic commit message based strictly on the uncommitted incremental changes
    commit_message = _generate_commit_message(
        latest_human_msg, incremental_diff, incremental_blueprint, config
    )

    # 3. Generate branch and PR metadata using the Map-Reduce pipeline
    if is_refining:
        branch_name = active_branch
        _, generated_body, generated_summary = _generate_pr_metadata(
            instruction_context, raw_diff, blueprint, config
        )
    else:
        branch_name, generated_body, generated_summary = _generate_pr_metadata(
            instruction_context, raw_diff, blueprint, config
        )

    # 4. Branch and Commit
    commit_response = json.loads(
        create_branch_and_commit(
            directory=target_path, new_branch=branch_name, commit_message=commit_message
        )
    )

    if commit_response.get("status") == "error":
        return _handle_commit_failure(
            commit_response, state.get("execution_retry_count", 0), branch_name
        )

    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    # 5. Handle existing PR push success and dynamically update Title/Body
    if is_refining:
        pr_title = f"{prefix_str}{generated_summary}"
        update_status_msg = _attempt_pr_update(
            target_path, pending_pr_url, pr_title, generated_body, generated_summary
        )

        msg = (
            "✅ **Revisions Applied**\n"
            f"📦 `[{repo_name}]`\n\n"
            "I have pushed the requested updates to the existing Pull Request:\n"
            f"🔗 {pending_pr_url}\n{update_status_msg}\n\n"
            "*Merge the PR on GitHub to finalize, or provide feedback here for further edits.*"
        )
        return {"messages": [AIMessage(content=msg)]}

    # 6. Open a new PR
    pr_title = f"{prefix_str}{generated_summary}"
    pr_response = open_pull_request(
        directory=target_path,
        title=pr_title,
        head_branch=branch_name,
        base_branch=state.get("target_branch", "main"),
        body=generated_body,
    )

    try:
        pr_data = json.loads(pr_response)
        if pr_data.get("status") == "error":
            return {
                "active_agent_branch": branch_name,
                "messages": [
                    AIMessage(
                        content=(
                            "⚠️ **PR Creation Failed:** The code was pushed to GitHub, but "
                            "the Pull Request API rejected the request.\n"
                            f"**Reason:** {pr_data.get('reason')}\n"
                            f"**Details:** `{pr_data.get('details', '')}`\n\n"
                            "*Aborting and cleaning up temporary branches.*"
                        )
                    )
                ],
                "is_aborted": True,
            }

        pr_url = pr_data.get("pr_url", "[Error Fetching URL]")
    except json.JSONDecodeError:
        pr_url = "[Error Fetching URL]"

    msg = (
        "✅ **Execution Complete**\n"
        f"📦 `[{repo_name}]`\n\n"
        "I have pushed the changes to a new branch and opened a Pull Request for your review:\n"
        f"🔗 {pr_url}\n\n"
        "*Merge the PR on GitHub to finalize, or provide feedback here for further edits.*"
    )

    return {
        "active_agent_branch": branch_name,
        "pending_pr_url": pr_url,
        "messages": [AIMessage(content=msg)],
    }


def pr_merged_node(state: AgentState) -> dict:
    """
    Finalization step after human review.
    Cleans up the active branch locally and resets the global Git state.
    """
    target_path = state.get("workspace_absolute_path")
    target_branch = state.get("target_branch", "main")
    agent_branch = state.get("active_agent_branch")

    repo_name = target_path.split("/")[-1] if target_path else "Unknown Repo"

    if target_path and target_branch and agent_branch:
        cleanup_local_branch(target_path, target_branch, agent_branch)

    msg = (
        "🎉 **Pull Request Merged!**\n"
        f"📦 `[{repo_name}]`\n\n"
        f"🧹 *Cleanup Complete:* I have switched your local repository back to `{target_branch}` "
        f"and deleted the temporary `{agent_branch}` branch."
    )

    return {
        "messages": [AIMessage(content=msg)],
        "human_approved": False,
        "pending_pr_url": None,
        "active_agent_branch": None,
        "modified_tex_files": [],
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


# ==========================================
# Interaction & Utility Nodes
# ==========================================


def human_node(state: AgentState) -> dict:
    """
    Dummy breakpoint node for human-in-the-loop.
    The graph pauses before this node to wait for user input via Telegram.
    """
    return {}
