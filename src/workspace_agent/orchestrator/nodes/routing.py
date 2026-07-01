import concurrent.futures
import re

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.core.state import AgentState
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
    get_execution_llm_sequence,
    get_intent_router,
)
from src.workspace_agent.tools.github import cleanup_local_branch

from .execution import _build_fallback_chain

# ==========================================
# Module Configuration Constants
# ==========================================

MAX_CONTEXT_LENGTH = 500
MAX_FAST_PATH_LEN = 15


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

    clean_instruction = re.sub(r"[ \t]+", " ", clean_instruction)

    return clean_instruction, has_frontier, has_standard, requested_model


def _invoke_escalating_router(payload: dict, initial_tier: int, config: RunnableConfig = None):
    """Helper to manage the timeout and tier-escalation logic for the unified router chain."""
    config = config or {}
    decision = None
    error_log = []
    current_tier = initial_tier

    while current_tier <= TIER_FRONTIER:
        router_chain = get_intent_router(requested_tier=current_tier)

        # If the requested tier is not configured (e.g., no API keys), skip to the next
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

        # Isolate the blocking HTTP call in a background thread manually.
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
    prompt_content = PromptManager.get("router", "conversational_reply").strip()
    system_prompt = SystemMessage(content=prompt_content)

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

    # File disambiguation
    elif options:
        formatted_options = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
        msg = (
            f"⚠️ **Multiple matching files found.**\n\n"
            f"Please specify which file you meant:\n{formatted_options}\n\n"
            f"*Reply with the exact path or number.*"
        )

    # Workspace routing error
    else:
        available = "\n".join([f"- {key}" for key in settings.workspaces.keys()])
        msg = (
            f"⚠️ **Ambiguous Workspace Request.**\n\n"
            f"I wasn't sure which repository you meant. Here are your configured workspaces:\n"
            f"{available}\n\n"
            f"*Please reply with the target workspace.*"
        )

    # Do not clear state here, or the route_after_human router will lose its breadcrumbs
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


def human_node(state: AgentState) -> dict:
    """
    Dummy breakpoint node for human-in-the-loop.
    The graph pauses before this node to wait for user input via Telegram.
    """
    return {}
