from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 1. Core Conversation Memory
    # The 'add_messages' reducer intelligently appends new messages,
    # OR overwrites existing messages if their IDs match.
    messages: Annotated[list[BaseMessage], add_messages]
    original_instruction: str

    # 2. Context-Aware Routing
    intent_category: Literal["conversational", "workspace_operation", "workspace_read_only"]
    inferred_workspace: str | None
    workspace_absolute_path: str | None
    target_branch: str | None
    router_confidence: float | None
    force_frontier_tier: bool
    force_standard_tier: bool
    requested_model: str | None

    # 3. Execution & Mid-Task Disambiguation
    clarification_question: str | None
    disambiguation_options: list[str] | None
    execution_retry_count: int
    latest_traceback_error: str | None
    modified_tex_files: list[str]

    # 4. Observability & Lifecycle
    t1_base_calls: int
    t2_standard_calls: int
    t3_frontier_calls: int
    is_aborted: bool
    is_busy: bool

    # 5. Git Lifecycle (Restored to global state to persist across execution loops)
    active_agent_branch: str | None
    pending_pr_url: str | None
    human_approved: bool


class PRState(TypedDict):
    """Bounded context for the PR Sub-Graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    workspace_absolute_path: str | None
    target_branch: str | None
    intent_category: Literal["conversational", "workspace_operation", "workspace_read_only"]
    original_instruction: str
    execution_retry_count: int
    latest_traceback_error: str | None
    is_aborted: bool
    modified_tex_files: list[str]

    # Telemetry metrics used by evaluate_diff_node
    t1_base_calls: int
    t2_standard_calls: int
    t3_frontier_calls: int

    # Git Lifecycle tracking used by review_pr_node
    active_agent_branch: str | None
    pending_pr_url: str | None

    # GitHub Webhook Context for Agentic CI/CD
    repo_full_name: str | None
    pr_number: int | None
    commit_sha: str | None
    ci_results: list[dict]
