import concurrent.futures
import json
import os
import re
import subprocess
import time
import uuid

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
    get_execution_llm,
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
MAX_CONSECUTIVE_TOOL_STEPS = 30
MAX_GITHUB_COMMENT_LENGTH = 60000


# ==========================================
# Core Orchestration Nodes
# ==========================================


def _extract_tier_command(instruction: str) -> tuple[str, bool, bool, str | None]:
    """
    Detects tier and model override commands in a user instruction.
    Returns (cleaned_instruction, has_frontier, has_standard, requested_model).
    """
    if not instruction:
        return "", False, False, None

    instruction_lower = instruction.lower()
    has_frontier = "/frontier" in instruction_lower
    has_standard = "/standard" in instruction_lower

    requested_model = None
    model_match = re.search(r"(?i)/model:([a-zA-Z0-9_]+)", instruction)
    if model_match:
        requested_model = model_match.group(1)

    if not (has_frontier or has_standard or requested_model):
        return instruction, False, False, None

    # Case-insensitively remove the commands
    clean_instruction = re.sub(
        r"(?i)/frontier|/standard|/model:[a-zA-Z0-9_]+", "", instruction
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

    if consecutive_agent_steps >= MAX_CONSECUTIVE_TOOL_STEPS:
        loops = MAX_CONSECUTIVE_TOOL_STEPS // 2
        abort_msg = (
            "⚠️ **Execution Aborted:** The agent entered a runaway loop by executing "
            f"tools {loops} times in a row without finalizing the task. Circuit breaker triggered."
        )
        return {
            "messages": [AIMessage(content=abort_msg)],
            "is_aborted": True,
        }
    return None


def parse_intent_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """
    Tier 1 Routing Node with Escalation.
    Analyzes the user's initial instruction and forces a structured JSON decision.
    Uses limited recent message history to handle conversational follow-ups naturally.
    """
    config = config or {}
    instruction = state.get("original_instruction", "")

    # extract the command and clean the instruction immediately
    clean_instruction, has_frontier_override, has_standard_override, requested_model = (
        _extract_tier_command(instruction)
    )

    # Build conversational context sliding window for the payload
    context_block = ""
    if state.get("messages") and len(state["messages"]) > 1:
        # grab the last 4 messages (2 conversational turns)
        recent_msgs = state["messages"][-4:]
        history_lines = []

        for m in recent_msgs:
            # Aggressively truncate history to prevent context poisoning from long responses
            content_str = str(m.content)
            if len(content_str) > MAX_CONTEXT_LENGTH:
                content_str = content_str[:MAX_CONTEXT_LENGTH] + "... [TRUNCATED FOR ROUTING]"
            speaker = "User" if m.type == "human" else "Agent"
            history_lines.append(f"{speaker}: {content_str}")

        context_str = "\n".join(history_lines)
        context_block = f"\nRecent Conversation Context:\n{context_str}\n"

    # Package the variables into a generic dictionary payload for the unified chain
    payload = {
        "instruction": clean_instruction,
        "recent_context": context_block,
    }

    # dynamically set the initial routing tier based on human overrides
    if has_frontier_override:
        initial_tier = TIER_FRONTIER
    elif has_standard_override:
        initial_tier = TIER_STANDARD
    else:
        initial_tier = TIER_BASE

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

    # resolve the inferred workspace friendly name to its absolute path from config.yaml
    absolute_path = None
    target_branch = None

    if decision.inferred_workspace and decision.inferred_workspace in settings.workspaces:
        workspace_config = settings.workspaces[decision.inferred_workspace]
        absolute_path = workspace_config.path
        # use the workspace override if it exists, otherwise use the global default
        target_branch = workspace_config.target_branch or settings.agent.target_branch

    # trap the new CoT context missing flag
    clarification_question = None
    if decision.is_context_missing and decision.clarification_question_to_ask:
        clarification_question = decision.clarification_question_to_ask
        # force path to None so the graph strictly routes to the clarify node
        absolute_path = None

    # determine the execution tier based on user override OR autonomous triage
    force_frontier = has_frontier_override or (decision.task_complexity == "high")
    force_standard = (has_standard_override and not has_frontier_override) or (
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
        "requested_model": requested_model,  # <--- INJECT MODEL
        "clarification_question": clarification_question,
        "t1_base_calls": state.get("t1_base_calls", 0) + 1,
    }

    # overwrite the original instruction in the state if we stripped the command
    if has_frontier_override or has_standard_override or requested_model:
        state_update["original_instruction"] = clean_instruction

        if state.get("messages"):
            last_message = state["messages"][-1]
            if last_message.type == "human":
                updated_message = HumanMessage(content=clean_instruction, id=last_message.id)
                state_update["messages"] = [updated_message]

    return state_update


def conversational_reply_node(state: AgentState, config: RunnableConfig = None) -> dict:
    config = config or {}

    # Map conversational queries to the requested model if provided
    llm = get_execution_llm(
        requested_tier=TIER_BASE, requested_model_key=state.get("requested_model")
    )
    response = llm.invoke(state["messages"], config=config)
    return {"messages": [response], "t1_base_calls": state.get("t1_base_calls", 0) + 1}


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
    """Helper to trap newly modified .tex files from tool calls."""
    new_files = list(current_files)
    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            if tc.get("name") in ["write_file", "search_and_replace", "replace_text_block"]:
                fpath = str(tc.get("args", {}).get("file_path", ""))
                if fpath.lower().endswith(".tex") and fpath not in new_files:
                    new_files.append(fpath)
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

    # instantiate LLM
    try:
        llm = get_execution_llm(requested_tier=tier, requested_model_key=requested_model)
    except TerminalEscalationError as e:
        return {
            "messages": [AIMessage(content=f"⚠️ **Escalation Failed:** {str(e)}")],
            "is_aborted": True,
        }

    llm_with_tools = llm.bind_tools(agent_tools)

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
        response = llm_with_tools.invoke(filtered_messages, config=config)
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
            "FAILED (",
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


def evaluate_diff_node(state: PRState, config: RunnableConfig = None) -> dict:  # noqa: PLR0911
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

    try:
        # The cumulative diff for the whole PR (against the target branch)
        raw_diff = get_git_diff(directory=target_path, target_branch=target_branch)
        # The incremental diff for the new uncommitted changes (against HEAD)
        incremental_diff = get_git_diff(directory=target_path)
    except Exception:
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

    critic_prompt = PromptManager.get(
        "evaluation",
        "critic_prompt",
        instruction_context=instruction_context,
        raw_diff=raw_diff if not is_empty_diff else "[NO CUMULATIVE CHANGES TO REPOSITORY]",
        incremental_diff=incremental_diff
        if not is_empty_incremental
        else "[NO NEW UNCOMMITTED CHANGES]",
    ).strip()

    try:
        # We use the base tier LLM for fast, reliable evaluation
        llm = get_execution_llm(requested_tier=TIER_BASE, temperature=0.0)
        eval_result = llm.invoke(critic_prompt, config=config).content.strip()
    except Exception:
        # Degrade gracefully if API fails and pass it through
        return {"latest_traceback_error": None}

    if eval_result.startswith("PASS"):
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
                else:
                    abort_msg = (
                        "⚠️ **Execution Failed:** I was unable to invoke the necessary tools "
                        f"to satisfy the requirements after {retry_count} attempts.\n\n"
                        "*Workflow safely aborted.*"
                    )
                    return {"messages": [AIMessage(content=abort_msg)], "is_aborted": True}

            # Valid read-only pass
            return {"latest_traceback_error": None, "intent_category": "workspace_read_only"}

        # Standard pass with valid changes
        return {"latest_traceback_error": None}

    else:
        # Semantic FAIL -> Bounce it back to the agent
        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = (
                "⚠️ **Execution Failed:** I was unable to satisfy the requirements after "
                f"{retry_count} attempts.\n\n"
                f"*Critic Feedback:* {eval_result}\n\n*Workflow safely aborted.*"
            )
            return {"messages": [AIMessage(content=abort_msg)], "is_aborted": True}

        feedback = eval_result.replace("FAIL:", "").strip()
        msg = PromptManager.get("evaluation", "semantic_rejection", feedback=feedback).strip()

        return {
            "messages": [HumanMessage(content=msg)],
            "latest_traceback_error": "semantic_review_rejection",
            "execution_retry_count": retry_count + 1,
            "t1_base_calls": state.get("t1_base_calls", 0) + 1,
        }


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


def agentic_ci_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Agentic CI/CD Execution Node.
    Iterates through configured CI suites, executes them via OS-level subprocess isolation,
    and natively reports status checks and log aggregations back to the GitHub Pull Request.
    """
    target_path = state.get("workspace_absolute_path")
    commit_sha = state.get("commit_sha")
    pr_number = state.get("pr_number")

    if not target_path or not commit_sha or not pr_number:
        # Skip if missing necessary GitHub webhook context
        return {}

    workspace_config = next(
        (ws for ws in settings.workspaces.values() if ws.path == target_path), None
    )

    if not workspace_config or not workspace_config.ci_suites:
        return {"ci_results": []}

    expanded_target_path = os.path.expanduser(target_path)

    # 1. Post pending statuses for all suites upfront
    for suite in workspace_config.ci_suites:
        context_str = f"Agentic CI / {suite.name}"
        set_commit_status(
            directory=target_path,
            commit_sha=commit_sha,
            state="pending",
            context_str=context_str,
            description="Evaluation is running...",
        )

    results = []

    # 2. Execute each suite strictly sequentially
    for suite in workspace_config.ci_suites:
        context_str = f"Agentic CI / {suite.name}"
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

            passed = process.returncode == 0
            logs = f"{process.stdout}\n{process.stderr}".strip()

            set_commit_status(
                directory=target_path,
                commit_sha=commit_sha,
                state="success" if passed else "failure",
                context_str=context_str,
                description="Evaluation passed" if passed else "Evaluation failed",
            )

            results.append({"name": suite.name, "passed": passed, "logs": logs})

        except subprocess.TimeoutExpired as e:
            stdout_str = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
            )
            stderr_str = (
                e.stderr
                if isinstance(e.stderr, str)
                else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
            )

            logs = (
                f"{stdout_str}\n{stderr_str}\n\n"
                f"[ERROR: TimeoutExpired] Command timed out after {suite.timeout_seconds} seconds."
            ).strip()

            set_commit_status(
                directory=target_path,
                commit_sha=commit_sha,
                state="failure",
                context_str=context_str,
                description="Execution timed out",
            )

            results.append({"name": suite.name, "passed": False, "logs": logs})
        except Exception as e:
            set_commit_status(
                directory=target_path,
                commit_sha=commit_sha,
                state="error",
                context_str=context_str,
                description="OS error during execution",
            )
            results.append({"name": suite.name, "passed": False, "logs": f"OS Error: {str(e)}"})

    # 3. Post Consolidated PR Comment
    if results:
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

        comment_body = (
            f"## {prefix_str}Agentic CI/CD Results\n\n{summary_table}\n{details_sections}"
        )

        comment_on_pull_request(directory=target_path, pr_number=pr_number, body=comment_body)

    return {"ci_results": results}


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
        llm = get_execution_llm(requested_tier=TIER_BASE, temperature=0.2)

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

        res = llm.invoke(prompt, config=config).content.strip().strip("\"'")
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
    llm = get_execution_llm(requested_tier=TIER_BASE, temperature=0.2)

    chunks = _chunk_git_diff(raw_diff, MAX_DIFF_LENGTH)

    try:
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

                res = llm.invoke(map_prompt, config=config).content.strip()
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

        generated_summary = llm.invoke(summary_prompt, config=config).content.strip().strip("\"'")

        body_prompt = PromptManager.get(
            "pr_generation",
            "body_phase",
            instruction_context=instruction_context,
            blueprint=blueprint,
            synthesized_diff=syn_diff_str,
        ).strip()

        generated_body = llm.invoke(body_prompt, config=config).content.strip()
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
        llm = get_execution_llm(requested_tier=TIER_BASE)
        if not llm:
            return {}
    except TerminalEscalationError:
        return {}

    extractor = llm.with_structured_output(MemoryExtraction)

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

        # mirror to disk for persistence across container restarts
        profile_path = os.path.join(os.getcwd(), "user_profile.json")
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
