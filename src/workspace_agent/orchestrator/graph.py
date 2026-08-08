import json
import os

import redis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.state import AgentState, PRState
from src.workspace_agent.orchestrator.nodes import execution, github_lifecycle, routing

__all__ = ["pr_app", "agent_app"]

MIN_ROUTER_CONFIDENCE = 0.85

# ==========================================
# 1. Pull Request Sub-Graph
# ==========================================
pr_workflow = StateGraph(PRState)

pr_workflow.add_node("run_pre_commit", github_lifecycle.run_pre_commit_node)
pr_workflow.add_node("evaluate_diff", github_lifecycle.evaluate_diff_node)
pr_workflow.add_node("compile_latex", github_lifecycle.compile_node)
pr_workflow.add_node("review_pr", github_lifecycle.review_pr_node)
pr_workflow.add_node("agentic_ci", github_lifecycle.agentic_ci_node)


def _route_pr_entry(state: PRState) -> str:
    """Routes initial PR tasks based on the presence of CI triggers, modified tex files,
    or normal execution.
    """
    # If commit_sha is present, this is a CI trigger
    if state.get("commit_sha") or state.get("pr_number"):
        return "agentic_ci"
    if state.get("modified_tex_files"):
        return "compile_latex"
    # Execute pre-commit hooks before evaluating the diff
    return "run_pre_commit"


def _route_after_pre_commit(state: PRState) -> str:
    """Evaluates pre-commit hook results to route to diff evaluation or abort back to
    parent on unrecoverable errors.
    """
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error") == "pre_commit_error":
        # Pass control back up to parent graph to retry tool execution
        return END
    return "evaluate_diff"


def _route_after_compilation(state: PRState) -> str:
    """Evaluates LaTeX compilation results, routing to PR review on success or
    exiting on failure.
    """
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error") == "latex_compilation_error":
        # Pass control back up to parent graph to retry tool execution
        return END
    return "review_pr"


def _route_after_evaluation(state: PRState) -> str:
    """Determines if the semantic diff evaluation succeeded, routing to PR review
    or exiting on read-only/error states.
    """
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error"):
        # Semantic failure: Exit back to parent graph to resume tool execution
        return END
    if state.get("intent_category") == "workspace_read_only":
        # Safe exit (bypass PR entirely)
        return END
    return "review_pr"


pr_workflow.set_conditional_entry_point(_route_pr_entry)
pr_workflow.add_conditional_edges("run_pre_commit", _route_after_pre_commit)
pr_workflow.add_conditional_edges("compile_latex", _route_after_compilation)
pr_workflow.add_conditional_edges("evaluate_diff", _route_after_evaluation)

# Review PR and Agentic CI always exit.
pr_workflow.add_edge("review_pr", END)
pr_workflow.add_edge("agentic_ci", END)

# Compile the subgraph. It is now stateless from an interrupt perspective.
pr_app = pr_workflow.compile()


# ==========================================
# 2. Main Parent DAG Construction
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("parse_intent", routing.parse_intent_node)
workflow.add_node("conversational_reply", routing.conversational_reply_node)
workflow.add_node("clarify", routing.clarification_node)
workflow.add_node("execute_task", execution.execute_task_node)
workflow.add_node("workspace_tools", execution.workspace_tools_node)
workflow.add_node("update_memory", execution.update_memory_node)
workflow.add_node("cleanup_workflow", routing.cleanup_workflow_node)
workflow.add_node("human_clarify_node", routing.human_node)
workflow.add_node("human_pr_node", routing.human_node)
workflow.add_node("pr_merged", github_lifecycle.pr_merged_node)
workflow.add_node("force_tool_retry", execution.force_tool_retry_node)


def _pull_request_subgraph_node(state: AgentState, config: RunnableConfig) -> dict:
    """Explicitly maps the subgraph's terminal state flags back to the parent AgentState."""
    result = pr_app.invoke(state, config)
    return {
        # Relies on LangGraph's add_messages reducer to deduplicate by ID
        "messages": result.get("messages", []),
        "is_aborted": result.get("is_aborted", False),
        "latest_traceback_error": result.get("latest_traceback_error"),
        "execution_retry_count": result.get(
            "execution_retry_count", state.get("execution_retry_count", 0)
        ),
        "active_agent_branch": result.get("active_agent_branch"),
        "pending_pr_url": result.get("pending_pr_url"),
        "human_approved": result.get("human_approved", False),
        "intent_category": result.get("intent_category"),
        "modified_tex_files": result.get("modified_tex_files", []),
    }


workflow.add_node("pull_request_subgraph", _pull_request_subgraph_node)


def _route_after_intent(state: AgentState) -> str:
    """Routes based on the Tier 1 model's intent classification, checking confidence
    and workspace path resolution.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"
    if state.get("intent_category") == "conversational":
        return "conversational_reply"

    # catch any CoT clarification question before checking confidence
    if state.get("clarification_question"):
        return "clarify"

    # check for missing workspace resolution or low confidence
    if (
        not state.get("workspace_absolute_path")
        or state.get("router_confidence", 0.0) < MIN_ROUTER_CONFIDENCE
    ):
        return "clarify"

    return "execute_task"


def _route_after_llm(state: AgentState) -> str:
    """Routes to the ToolNode if the LLM generated tool calls, forces retries on
    un-called tools, or evaluates the diff.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"

    messages = state.get("messages", [])
    if not messages:
        return "pull_request_subgraph"
    last_message = messages[-1]

    # 1. If the LLM returned valid OR invalid tool calls, route to tools node to handle them
    if getattr(last_message, "tool_calls", None) or getattr(
        last_message, "invalid_tool_calls", None
    ):
        return "workspace_tools"

    # 2. Force a retry if the task is an operation but the LLM forgot to use tools entirely
    if state.get("intent_category") == "workspace_operation":
        # Check if the agent has already used tools in the current task
        # (Previous task tools are stripped out by the memory node)
        has_used_tools = any(m.type == "tool" for m in messages)

        if not has_used_tools:
            retry_count = state.get("execution_retry_count", 0)
            if retry_count < settings.agent.max_sandbox_retries:
                return "force_tool_retry"

    # 3. If it didn't use any tools and retries are exhausted, or if it already used tools and
    #    finished, pass forward
    return _route_after_execution(state)


def _route_after_execution(state: AgentState) -> str:  # noqa: PLR0911
    """Evaluates the execution node's output to determine if the workflow should proceed
    to the evaluation phase or terminate early.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"

    if state.get("clarification_question") or state.get("disambiguation_options"):
        return "clarify"

    messages = state.get("messages", [])

    is_recoverable_error = (
        state.get("latest_traceback_error")
        and state.get("execution_retry_count", 0) < settings.agent.max_sandbox_retries
    )

    is_tool_loop = bool(messages) and messages[-1].type == "tool"

    if is_tool_loop:
        # Robustly find the last AIMessage regardless of how many tools were executed
        last_ai_msg = next((m for m in reversed(messages) if m.type == "ai"), None)
        if last_ai_msg and hasattr(last_ai_msg, "tool_calls"):
            if any(tc["name"] == "mark_task_already_completed" for tc in last_ai_msg.tool_calls):
                is_tool_loop = False

    if is_recoverable_error or is_tool_loop:
        return "execute_task"

    if state.get("intent_category") == "workspace_read_only" and not state.get(
        "modified_tex_files"
    ):
        return "update_memory"

    return "pull_request_subgraph"


def _route_after_subgraph(state: AgentState) -> str:
    """Evaluates the state payload returned from the isolated PR subgraph to route to
    human review or execution retries.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"

    # Handle Agentic CI completion gracefully (it doesn't need human review or memory update)
    if state.get("repo_full_name") and (state.get("commit_sha") or state.get("pr_number")):
        return END

    # The subgraph hit a failure, or the human rejected the PR for changes with feedback
    messages = state.get("messages", [])
    if state.get("latest_traceback_error") or (messages and messages[-1].type == "human"):
        return "execute_task"

    # The critic passed an empty diff -> bypass PR and update memory
    if state.get("intent_category") == "workspace_read_only":
        return "update_memory"

    # If the subgraph successfully generated a PR, we need human approval
    if state.get("pending_pr_url") and not state.get("human_approved"):
        return "human_pr_node"

    return "update_memory"


def _route_after_human_clarify(state: AgentState) -> str:
    """Evaluates human clarification input to either execute tasks, parse new intent,
    or handle external merges.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"

    # Route to cleanup if a merge (LGTM) occurs while paused
    if state.get("human_approved"):
        return "pr_merged"

    # Ensure the physical path is resolved before executing tools.
    # If a workflow aborted and the path is missing, force it through the parser
    # so the human's response can be mapped to an absolute path first.
    if not state.get("workspace_absolute_path"):
        return "parse_intent"
    if state.get("clarification_question") or state.get("disambiguation_options"):
        return "execute_task"

    # If no options or questions, we were disambiguating the workspace intent; route back to parsing
    return "parse_intent"


def _route_after_human_pr(state: AgentState) -> str:
    """Routes execution after the user has reviewed the pending Pull Request, returning
    to execution on rejection.
    """
    if state.get("is_aborted"):
        return "cleanup_workflow"
    if state.get("human_approved"):
        return "pr_merged"

    # Human rejected PR with written feedback -> route back to execution to apply it
    return "execute_task"


workflow.set_entry_point("parse_intent")

# entry routing
workflow.add_conditional_edges("parse_intent", _route_after_intent)
workflow.add_edge("conversational_reply", END)

# cyclic loops and execution
workflow.add_conditional_edges("execute_task", _route_after_llm)
workflow.add_conditional_edges("workspace_tools", _route_after_execution)
workflow.add_edge("force_tool_retry", "execute_task")

workflow.add_edge("clarify", "human_clarify_node")
workflow.add_conditional_edges("human_clarify_node", _route_after_human_clarify)

# Route into and out of the isolated subgraph
workflow.add_conditional_edges("pull_request_subgraph", _route_after_subgraph)
workflow.add_conditional_edges("human_pr_node", _route_after_human_pr)

workflow.add_edge("pr_merged", "update_memory")
workflow.add_edge("update_memory", END)
workflow.add_edge("cleanup_workflow", END)


# ==========================================
# 3. State Management & Compilation
# ==========================================
agent_store = InMemoryStore()

# hydrate the store from disk to survive FastAPI/Docker restarts
default_profile_path = os.path.join(os.getcwd(), "agent_state", "user_profile.json")
PROFILE_PATH = os.getenv("AGENT_PROFILE_PATH", default_profile_path)

if os.path.exists(PROFILE_PATH):
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            profile_data = json.load(f)
            # fetch the configured admin chat ID to namespace the loaded memory correctly
            chat_id = os.getenv("AUTHORIZED_OWNER_CHAT_ID", "default")
            agent_store.put(("user_profile", str(chat_id)), "profile", profile_data)
    except Exception as e:
        print(f"Failed to hydrate LangGraph Store from disk: {e}")


def _get_checkpointer():
    """Attempts to connect to the Redis container for persistent memory across webhooks,
    falling back to ephemeral MemorySaver.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        # test the connection to fail fast if offline
        r = redis.Redis.from_url(redis_url)
        r.ping()

        # initialize the saver and create the required database indices
        saver = RedisSaver(redis_url)
        saver.setup()
        return saver
    except Exception as e:
        print(f"Warning: Redis checkpointer offline, falling back to MemorySaver. ({e})")
        return MemorySaver()


agent_app = workflow.compile(
    checkpointer=_get_checkpointer(),
    store=agent_store,
    interrupt_before=["human_clarify_node", "human_pr_node"],
)
