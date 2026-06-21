import json
import os

import redis
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.state import AgentState, PRState
from src.workspace_agent.orchestrator import nodes

# ==========================================
# Constants
# ==========================================
MIN_ROUTER_CONFIDENCE = 0.85

# ==========================================
# 1. Pull Request Sub-Graph (Bounded Context)
# ==========================================
pr_workflow = StateGraph(PRState)

pr_workflow.add_node("run_pre_commit", nodes.run_pre_commit_node)
pr_workflow.add_node("evaluate_diff", nodes.evaluate_diff_node)
pr_workflow.add_node("compile_latex", nodes.compile_node)
pr_workflow.add_node("review_pr", nodes.review_pr_node)
pr_workflow.add_node("agentic_ci", nodes.agentic_ci_node)


def route_pr_entry(state: PRState) -> str:
    # If commit_sha is present, this is a CI trigger
    if state.get("commit_sha") or state.get("pr_number"):
        return "agentic_ci"
    if state.get("modified_tex_files"):
        return "compile_latex"
    # Execute pre-commit hooks before evaluating the diff
    return "run_pre_commit"


def route_after_pre_commit(state: PRState) -> str:
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error") == "pre_commit_error":
        # Pass control back up to parent graph to retry tool execution
        return END
    return "evaluate_diff"


def route_after_compilation(state: PRState) -> str:
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error") == "latex_compilation_error":
        # Pass control back up to parent graph to retry tool execution
        return END
    return "review_pr"


def route_after_evaluation(state: PRState) -> str:
    if state.get("is_aborted"):
        return END
    if state.get("latest_traceback_error"):
        # Semantic failure: Exit back to parent graph to resume tool execution
        return END
    if state.get("intent_category") == "workspace_read_only":
        # Safe exit (bypass PR entirely)
        return END
    return "review_pr"


pr_workflow.set_conditional_entry_point(route_pr_entry)
pr_workflow.add_conditional_edges("run_pre_commit", route_after_pre_commit)
pr_workflow.add_conditional_edges("compile_latex", route_after_compilation)
pr_workflow.add_conditional_edges("evaluate_diff", route_after_evaluation)

# Review PR and Agentic CI always exit.
pr_workflow.add_edge("review_pr", END)
pr_workflow.add_edge("agentic_ci", END)

# Compile the subgraph. It is now stateless from an interrupt perspective.
pr_app = pr_workflow.compile()


# ==========================================
# 2. Main Parent DAG Construction
# ==========================================
workflow = StateGraph(AgentState)

# add all nodes to the graph
workflow.add_node("parse_intent", nodes.parse_intent_node)
workflow.add_node("conversational_reply", nodes.conversational_reply_node)
workflow.add_node("clarify", nodes.clarification_node)
workflow.add_node("execute_task", nodes.execute_task_node)
workflow.add_node("workspace_tools", nodes.workspace_tools_node)
workflow.add_node("update_memory", nodes.update_memory_node)
workflow.add_node("cleanup_workflow", nodes.cleanup_workflow_node)
workflow.add_node("human_clarify_node", nodes.human_node)
workflow.add_node("human_pr_node", nodes.human_node)
workflow.add_node("pr_merged", nodes.pr_merged_node)
workflow.add_node("force_tool_retry", nodes.force_tool_retry_node)

# Inject the Sub-Graph as a standard functional node
workflow.add_node("pull_request_subgraph", pr_app)


# ==========================================
# Parent Conditional Edge Logic
# ==========================================
def route_after_intent(state: AgentState) -> str:
    """Routes based on the Tier 1 model's intent classification."""
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


def route_after_llm(state: AgentState) -> str:
    """Routes to the ToolNode if the LLM generated tool calls, otherwise evaluates the diff."""
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
    return route_after_execution(state)


def route_after_execution(state: AgentState) -> str:  # noqa: PLR0911
    """
    Evaluates the execution node's output to determine if the workflow
    should proceed to the evaluation phase or terminate early.
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


def route_after_subgraph(state: AgentState) -> str:
    """Evaluates the state payload returned from the isolated PR subgraph."""
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


def route_after_human_clarify(state: AgentState) -> str:
    if state.get("is_aborted"):
        return "cleanup_workflow"

    # Ensure the physical path is resolved before executing tools.
    # If a workflow aborted and the path is missing, force it through the parser
    # so the human's response can be mapped to an absolute path first.
    if not state.get("workspace_absolute_path"):
        return "parse_intent"
    if state.get("clarification_question") or state.get("disambiguation_options"):
        return "execute_task"

    # if no options or questions, we were disambiguating the workspace intent; route back to parsing
    return "parse_intent"


def route_after_human_pr(state: AgentState) -> str:
    """Routes execution after the user has reviewed the pending Pull Request."""
    if state.get("is_aborted"):
        return "cleanup_workflow"
    if state.get("human_approved"):
        return "pr_merged"

    # Human rejected PR with written feedback -> route back to execution to apply it
    return "execute_task"


# ==========================================
# Parent Edge Mapping
# ==========================================

workflow.set_entry_point("parse_intent")

# entry routing
workflow.add_conditional_edges("parse_intent", route_after_intent)
workflow.add_edge("conversational_reply", END)

# cyclic loops and execution
workflow.add_conditional_edges("execute_task", route_after_llm)
workflow.add_conditional_edges("workspace_tools", route_after_execution)

workflow.add_edge("force_tool_retry", "execute_task")

workflow.add_edge("clarify", "human_clarify_node")
workflow.add_conditional_edges("human_clarify_node", route_after_human_clarify)

# Route into and out of the isolated subgraph
workflow.add_conditional_edges("pull_request_subgraph", route_after_subgraph)
workflow.add_conditional_edges("human_pr_node", route_after_human_pr)

workflow.add_edge("pr_merged", "update_memory")

workflow.add_edge("update_memory", END)
workflow.add_edge("cleanup_workflow", END)


# ==========================================
# Compilation & Checkpointer Attachment
# ==========================================

agent_store = InMemoryStore()

# hydrate the store from disk to survive FastAPI/Docker restarts
PROFILE_PATH = os.path.join(os.getcwd(), "user_profile.json")
if os.path.exists(PROFILE_PATH):
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            profile_data = json.load(f)
            # fetch the configured admin chat ID to namespace the loaded memory correctly
            chat_id = os.getenv("AUTHORIZED_OWNER_CHAT_ID", "default")
            agent_store.put(("user_profile", str(chat_id)), "profile", profile_data)
    except Exception as e:
        print(f"Failed to hydrate LangGraph Store from disk: {e}")


def get_checkpointer():
    """
    Attempts to connect to the Redis container for persistent memory across webhooks.
    Falls back to ephemeral MemorySaver if Redis is offline.
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
    checkpointer=get_checkpointer(),
    store=agent_store,
    interrupt_before=["human_clarify_node", "human_pr_node"],
)
