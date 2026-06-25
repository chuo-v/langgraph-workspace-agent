from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from src.workspace_agent.orchestrator.graph import (
    get_checkpointer,
    route_after_compilation,
    route_after_evaluation,
    route_after_execution,
    route_after_human_clarify,
    route_after_human_pr,
    route_after_intent,
    route_after_llm,
    route_after_pre_commit,
    route_after_subgraph,
    route_pr_entry,
)

# ==========================================
# Component: route_after_intent
# ==========================================


def test_route_after_intent_success_conversational():
    """Green Path: Chat intent bypasses tools and goes to the conversational node."""
    state = {"intent_category": "conversational", "is_aborted": False}
    assert route_after_intent(state) == "conversational_reply"


def test_route_after_intent_success_high_confidence():
    """Green Path: High confidence and valid workspace triggers immediate execution."""
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "example-project",
        "workspace_absolute_path": "/fake/path",
        "router_confidence": 0.95,
        "is_aborted": False,
    }
    assert route_after_intent(state) == "execute_task"


def test_route_after_intent_fallback_cot_clarification():
    """Edge Path: The Structured Chain-of-Thought immediately trapped missing context."""
    state = {
        "intent_category": "workspace_operation",
        "clarification_question": "Which repository did you mean?",
        "is_aborted": False,
    }
    assert route_after_intent(state) == "clarify"


def test_route_after_intent_fallback_low_confidence():
    """Edge Path: Low confidence score triggers the human disambiguation node."""
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "example-project",
        "workspace_absolute_path": "/fake/path",
        "router_confidence": 0.40,  # below the 0.85 threshold
        "is_aborted": False,
    }
    assert route_after_intent(state) == "clarify"


def test_route_after_intent_fallback_missing_workspace():
    """
    Edge Path: High confidence, but the inferred workspace didn't match a valid absolute path.
    """
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "unknown-project",
        "workspace_absolute_path": None,
        "router_confidence": 0.99,
        "is_aborted": False,
    }
    assert route_after_intent(state) == "clarify"


# ==========================================
# Component: route_after_execution
# ==========================================


def test_route_after_execution_success_escape_hatch():
    """Green Path: If the agent marks task as complete, force route into subgraph."""
    messages = [
        AIMessage(
            content="", tool_calls=[{"name": "mark_task_already_completed", "args": {}, "id": "1"}]
        ),
        ToolMessage(content="Task complete", tool_call_id="1", name="mark_task_already_completed"),
    ]
    state = {"messages": messages, "intent_category": "workspace_operation"}
    assert route_after_execution(state) == "pull_request_subgraph"


def test_route_after_execution_success_read_only():
    """Green Path: Execution succeeds for a pure read task. Safely bypass PR generation."""
    state = {
        "intent_category": "workspace_read_only",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "modified_tex_files": [],
    }
    assert route_after_execution(state) == "update_memory"


def test_route_after_execution_success_tool_loop():
    """Green Path: The last message was a tool result, loop back to the LLM to process it."""
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "disambiguation_options": None,
        "clarification_question": None,
        "messages": [
            ToolMessage(content="File contents here", tool_call_id="call_123", name="read_file")
        ],
    }
    assert route_after_execution(state) == "execute_task"


def test_route_after_execution_success_workspace_operation():
    """Green Path: Execution succeeds. Route to the pull_request_subgraph to handle verification."""
    state = {
        "intent_category": "workspace_operation",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [
            ToolMessage(content="File written.", tool_call_id="call_123", name="write_file"),
            AIMessage(content="Task is done."),
        ],
    }
    # Parent graph delegates all evaluation/PR logic to the subgraph
    assert route_after_execution(state) == "pull_request_subgraph"


def test_route_after_execution_success_workspace_operation_no_modifications():
    """Green Path: Execution concludes without using tools. Delegate evaluation to subgraph."""
    state = {
        "intent_category": "workspace_operation",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [
            ToolMessage(
                content="File read successfully.", tool_call_id="call_1", name="read_files"
            ),
            AIMessage(content="I reviewed the file. No changes were necessary."),
        ],
    }
    assert route_after_execution(state) == "pull_request_subgraph"


def test_route_after_execution_fallback_clarification_question():
    """Edge Path: Agent asked a clarification question."""
    state = {
        "disambiguation_options": None,
        "clarification_question": "Should I use a list comprehension?",
        "is_aborted": False,
    }
    assert route_after_execution(state) == "clarify"


def test_route_after_execution_fallback_disambiguate_files():
    """Edge Path: File disambiguation required from human."""
    state = {
        "disambiguation_options": ["/src/main.py", "/tests/main.py"],
        "is_aborted": False,
    }
    assert route_after_execution(state) == "clarify"


def test_route_after_execution_fallback_empty_messages():
    """Edge Path: Safe fallback to subgraph if message trace is empty."""
    state = {
        "intent_category": "workspace_operation",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [],
    }
    assert route_after_execution(state) == "pull_request_subgraph"


def test_route_after_execution_fallback_retry_loop():
    """Edge Path: Execution crashed and retries remain. Loop back to execution."""
    state = {
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": "SyntaxError: invalid syntax",
        "execution_retry_count": 1,
    }
    assert route_after_execution(state) == "execute_task"


def test_route_after_execution_error_max_retries(mocker):
    """Red Path: Execution crashed and max retries reached. Graph aborts to cleanup."""
    mocker.patch(
        "src.workspace_agent.orchestrator.graph.settings",
        mocker.Mock(agent=mocker.Mock(max_sandbox_retries=3)),
    )
    state = {
        "disambiguation_options": None,
        "is_aborted": True,
        "latest_traceback_error": None,
        "execution_retry_count": 3,
    }
    assert route_after_execution(state) == "cleanup_workflow"


# ==========================================
# Component: route_pr_entry
# ==========================================


def test_route_pr_entry_success_agentic_ci():
    """Green Path: If PR context is present, route to agentic CI node."""
    state = {"commit_sha": "abc1234", "pr_number": 42}
    assert route_pr_entry(state) == "agentic_ci"


def test_route_pr_entry_success_compile_latex():
    """Green Path: If LaTeX files modified, intercept and compile them first."""
    state = {"modified_tex_files": ["main.tex"]}
    assert route_pr_entry(state) == "compile_latex"


def test_route_pr_entry_success_run_pre_commit():
    """Green Path: If no LaTeX files modified, run pre-commit checks first."""
    state = {"modified_tex_files": []}
    assert route_pr_entry(state) == "run_pre_commit"


# ==========================================
# Component: route_after_evaluation
# ==========================================


def test_route_after_evaluation_success_pass():
    """Green Path: Critic passed the diff, proceed to PR review."""
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "intent_category": "workspace_operation",
    }
    assert route_after_evaluation(state) == "review_pr"


def test_route_after_evaluation_success_read_only_pass():
    """Green Path: Critic confirmed safe bypass, exit the subgraph."""
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "intent_category": "workspace_read_only",
    }
    assert route_after_evaluation(state) == END


def test_route_after_evaluation_fallback_rejection():
    """Edge Path: Critic rejected diff, exit subgraph to resume execution node."""
    state = {"is_aborted": False, "latest_traceback_error": "semantic_review_rejection"}
    assert route_after_evaluation(state) == END


# ==========================================
# Component: route_after_pre_commit
# ==========================================


def test_route_after_pre_commit_success_standard():
    """Green Path: Pre-commit succeeded, proceed to diff evaluation."""
    state = {"is_aborted": False, "latest_traceback_error": None}
    assert route_after_pre_commit(state) == "evaluate_diff"


def test_route_after_pre_commit_fallback_error():
    """Edge Path: Pre-commit failed, exit subgraph to resume execution node."""
    state = {"is_aborted": False, "latest_traceback_error": "pre_commit_error"}
    assert route_after_pre_commit(state) == END


def test_route_after_pre_commit_error_max_retries():
    """Red Path: Pre-commit max retries reached. Graph aborts and exits subgraph."""
    state = {"is_aborted": True}
    assert route_after_pre_commit(state) == END


# ==========================================
# Component: route_after_compilation
# ==========================================


def test_route_after_compilation_success_standard():
    """Green Path: Compilation succeeded, proceed to PR review."""
    state = {"is_aborted": False, "latest_traceback_error": None}
    assert route_after_compilation(state) == "review_pr"


def test_route_after_compilation_fallback_syntax_error():
    """Edge Path: Compilation failed, exit subgraph to resume execution node."""
    state = {"is_aborted": False, "latest_traceback_error": "latex_compilation_error"}
    assert route_after_compilation(state) == END


def test_route_after_compilation_error_max_retries():
    """Red Path: Compilation max retries reached. Graph aborts and exits subgraph."""
    state = {"is_aborted": True}
    assert route_after_compilation(state) == END


# ==========================================
# Component: route_after_subgraph
# ==========================================


def test_route_after_subgraph_success_agentic_ci():
    """Green Path: Agentic CI completes cleanly and exits the parent workflow."""
    state = {"repo_full_name": "owner/repo", "commit_sha": "abc1234", "is_aborted": False}
    assert route_after_subgraph(state) == END


def test_route_after_subgraph_success_standard():
    """Green Path: Subgraph completes and generates PR, requiring human review."""
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "pending_pr_url": "https://github.com/pr",
        "human_approved": False,
    }
    assert route_after_subgraph(state) == "human_pr_node"


def test_route_after_subgraph_fallback_error():
    """Edge Path: Subgraph propagated a semantic or compilation error."""
    state = {"is_aborted": False, "latest_traceback_error": "semantic_review_rejection"}
    assert route_after_subgraph(state) == "execute_task"


def test_route_after_subgraph_fallback_hallucination_retry():
    """Edge Path: If the subgraph hits a hallucination trap, route back to execute_task."""
    state = {"latest_traceback_error": "hallucinated_success", "is_aborted": False}
    assert route_after_subgraph(state) == "execute_task"


def test_route_after_subgraph_fallback_human_feedback():
    """Edge Path: Subgraph ended because a human provided written feedback to refine PR."""
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [HumanMessage(content="Please change the color to blue.")],
    }
    assert route_after_subgraph(state) == "execute_task"


# ==========================================
# Component: route_after_human_pr
# ==========================================


def test_route_after_human_pr_success_approved():
    """Green Path: PR approved by user, route to parent graph merge node."""
    state = {
        "human_approved": True,
        "is_aborted": False,
    }
    assert route_after_human_pr(state) == "pr_merged"


def test_route_after_human_pr_fallback_feedback():
    """Edge Path: User provided feedback instead of approval, route to execution."""
    state = {
        "human_approved": False,
        "is_aborted": False,
    }
    assert route_after_human_pr(state) == "execute_task"


# ==========================================
# Component: route_after_human_clarify
# ==========================================


def test_route_after_human_clarify_fallback_mid_task():
    """Edge Path: User resolved file conflict mid-task."""
    state = {
        "workspace_absolute_path": "/fake/path",
        "disambiguation_options": ["fileA.py"],
        "is_aborted": False,
    }
    assert route_after_human_clarify(state) == "execute_task"


def test_route_after_human_clarify_fallback_missing_path():
    """Edge Path: Path is still missing, force re-evaluation of intent."""
    state = {
        "workspace_absolute_path": None,
        "clarification_question": "Should I capitalize it?",
        "is_aborted": False,
    }
    assert route_after_human_clarify(state) == "parse_intent"


def test_route_after_human_clarify_fallback_workspace():
    """Edge Path: User clarified initial workspace mapping."""
    state = {
        "workspace_absolute_path": "/fake/path",
        "disambiguation_options": None,
        "clarification_question": None,
        "is_aborted": False,
    }
    assert route_after_human_clarify(state) == "parse_intent"


# ==========================================
# Component: route_after_llm
# ==========================================


def test_route_after_llm_success_tool_calls():
    """Green Path: LLM generates tool calls, direct graph to execute them."""
    state = {
        "is_aborted": False,
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "read_files", "args": {}, "id": "1"}])
        ],
    }
    assert route_after_llm(state) == "workspace_tools"


def test_route_after_llm_success_no_tools_max_retries(mocker):
    """Green Path: Retries exhausted, proceed to evaluation/execution logic."""
    mocker.patch(
        "src.workspace_agent.orchestrator.graph.settings",
        mocker.Mock(agent=mocker.Mock(max_sandbox_retries=3)),
    )
    state = {
        "is_aborted": False,
        "intent_category": "workspace_operation",
        "execution_retry_count": 3,
        "messages": [AIMessage(content="I have reviewed the task and it is complete.")],
    }
    # Retries exhausted -> delegates to route_after_execution -> pull_request_subgraph
    assert route_after_llm(state) == "pull_request_subgraph"


def test_route_after_llm_success_read_only_no_tools():
    """Green Path: Read-only tasks without tools safely delegate to evaluation logic."""
    state = {
        "is_aborted": False,
        "intent_category": "workspace_read_only",
        "messages": [AIMessage(content="I have reviewed the task and it is complete.")],
        "modified_tex_files": [],
    }
    # Read-only bypasses tool retries -> delegates to route_after_execution -> update_memory
    assert route_after_llm(state) == "update_memory"


def test_route_after_llm_fallback_missing_tools():
    """Edge Path: LLM responds without tool calls during an operation. Force retry."""
    state = {
        "is_aborted": False,
        "intent_category": "workspace_operation",
        "execution_retry_count": 0,
        "messages": [AIMessage(content="I have reviewed the task and it is complete.")],
    }
    # With the new behavior, this should intercept and force a tool retry
    assert route_after_llm(state) == "force_tool_retry"


def test_route_after_llm_fallback_empty_messages():
    """Edge Path: Safe fallback to subgraph if message trace is unexpectedly empty."""
    state = {
        "is_aborted": False,
        "messages": [],
    }
    assert route_after_llm(state) == "pull_request_subgraph"


# ==========================================
# Workflow: Global Abort Override
# ==========================================


def test_global_routing_error_abort_override():
    """Red Path: The is_aborted flag forces an immediate exit to respective cleanup nodes."""
    parent_state = {"is_aborted": True}
    assert route_after_intent(parent_state) == "cleanup_workflow"
    assert route_after_llm(parent_state) == "cleanup_workflow"
    assert route_after_execution(parent_state) == "cleanup_workflow"
    assert route_after_subgraph(parent_state) == "cleanup_workflow"
    assert route_after_human_clarify(parent_state) == "cleanup_workflow"
    assert route_after_human_pr(parent_state) == "cleanup_workflow"

    subgraph_state = {"is_aborted": True}
    assert route_after_evaluation(subgraph_state) == END
    assert route_after_compilation(subgraph_state) == END
    assert route_after_pre_commit(subgraph_state) == END


# ==========================================
# Component: get_checkpointer
# ==========================================


def test_get_checkpointer_success_redis(mocker):
    """Green Path: Successfully connects to Redis and initializes the RedisSaver checkpointer."""
    mock_redis_instance = mocker.Mock()

    mocker.patch(
        "src.workspace_agent.orchestrator.graph.redis.Redis.from_url",
        return_value=mock_redis_instance,
    )
    mock_saver = mocker.patch("src.workspace_agent.orchestrator.graph.RedisSaver")

    checkpointer = get_checkpointer()

    # Verify Redis was actively pinged to test connection validity
    mock_redis_instance.ping.assert_called_once()

    # Verify the LangGraph saver was properly initialized
    mock_saver.return_value.setup.assert_called_once()
    assert checkpointer == mock_saver.return_value


def test_get_checkpointer_fallback_memory_saver(mocker, capsys):
    """
    Edge Path: Redis connection fails (or is offline), falls back gracefully to MemorySaver.
    """
    # Force the ping to throw an exception mimicking an offline container
    mock_redis_instance = mocker.Mock()
    mock_redis_instance.ping.side_effect = Exception("Connection refused")
    mocker.patch(
        "src.workspace_agent.orchestrator.graph.redis.Redis.from_url",
        return_value=mock_redis_instance,
    )

    mock_memory_saver = mocker.patch("src.workspace_agent.orchestrator.graph.MemorySaver")

    checkpointer = get_checkpointer()

    captured = capsys.readouterr()
    assert "Warning: Redis checkpointer offline" in captured.out

    # Ensure it cleanly fell back to ephemeral memory
    assert checkpointer == mock_memory_saver.return_value
