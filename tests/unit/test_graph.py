from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from src.workspace_agent.orchestrator.graph import (
    _get_checkpointer,
    _route_after_compilation,
    _route_after_evaluation,
    _route_after_execution,
    _route_after_human_clarify,
    _route_after_human_pr,
    _route_after_intent,
    _route_after_llm,
    _route_after_pre_commit,
    _route_after_subgraph,
    _route_pr_entry,
)

# ==========================================
# Workflow: Graph State Initialization
# ==========================================


def test_get_checkpointer_success_redis(mocker):
    """Green Path: Successfully connects to Redis and initializes the RedisSaver checkpointer."""
    # 1. Setup Mock Environment
    mock_redis_instance = mocker.Mock()
    mocker.patch(
        "src.workspace_agent.orchestrator.graph.redis.Redis.from_url",
        return_value=mock_redis_instance,
    )
    mock_saver = mocker.patch("src.workspace_agent.orchestrator.graph.RedisSaver")

    # 2. Execute
    checkpointer = _get_checkpointer()

    # 3. Assertions
    mock_redis_instance.ping.assert_called_once()
    mock_saver.return_value.setup.assert_called_once()
    assert checkpointer == mock_saver.return_value


def test_get_checkpointer_fallback_memory_saver(mocker, capsys):
    """Edge Path: Redis connection fails (or is offline), falls back gracefully to MemorySaver."""
    # 1. Setup Mock Environment
    mock_redis_instance = mocker.Mock()
    mock_redis_instance.ping.side_effect = Exception("Connection refused")
    mocker.patch(
        "src.workspace_agent.orchestrator.graph.redis.Redis.from_url",
        return_value=mock_redis_instance,
    )
    mock_memory_saver = mocker.patch("src.workspace_agent.orchestrator.graph.MemorySaver")

    # 2. Execute
    checkpointer = _get_checkpointer()
    captured = capsys.readouterr()

    # 3. Assertions
    assert "Warning: Redis checkpointer offline" in captured.out
    assert checkpointer == mock_memory_saver.return_value


# ==========================================
# Workflow: Intent Parsing & Routing
# ==========================================


def test_route_after_intent_success_conversational():
    """Green Path: Chat intent bypasses tools and goes to the conversational node."""
    # 1. Setup Mock Environment
    state = {"intent_category": "conversational", "is_aborted": False}

    # 2. Execute
    result = _route_after_intent(state)

    # 3. Assertions
    assert result == "conversational_reply"


def test_route_after_intent_success_high_confidence():
    """Green Path: High confidence and valid workspace triggers immediate execution."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "example-project",
        "workspace_absolute_path": "/fake/path",
        "router_confidence": 0.95,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_intent(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_intent_fallback_cot_clarification():
    """Edge Path: The Structured Chain-of-Thought immediately trapped missing context."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_operation",
        "clarification_question": "Which repository did you mean?",
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_intent(state)

    # 3. Assertions
    assert result == "clarify"


def test_route_after_intent_fallback_low_confidence():
    """Edge Path: Low confidence score triggers the human disambiguation node."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "example-project",
        "workspace_absolute_path": "/fake/path",
        "router_confidence": 0.40,  # below the 0.85 threshold
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_intent(state)

    # 3. Assertions
    assert result == "clarify"


def test_route_after_intent_fallback_missing_workspace():
    """Edge Path: High confidence, but the inferred workspace didn't match a valid absolute path."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_operation",
        "inferred_workspace": "unknown-project",
        "workspace_absolute_path": None,
        "router_confidence": 0.99,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_intent(state)

    # 3. Assertions
    assert result == "clarify"


# ==========================================
# Workflow: Agent Execution Loop
# ==========================================


def test_route_after_llm_success_tool_calls():
    """Green Path: LLM generates tool calls, direct graph to execute them."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "read_files", "args": {}, "id": "1"}])
        ],
    }

    # 2. Execute
    result = _route_after_llm(state)

    # 3. Assertions
    assert result == "workspace_tools"


def test_route_after_llm_success_no_tools_max_retries(mocker):
    """Green Path: Retries exhausted, proceed to evaluation/execution logic."""
    # 1. Setup Mock Environment
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

    # 2. Execute
    result = _route_after_llm(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_llm_success_read_only_no_tools():
    """Green Path: Read-only tasks without tools safely delegate to evaluation logic."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "intent_category": "workspace_read_only",
        "messages": [AIMessage(content="I have reviewed the task and it is complete.")],
        "modified_tex_files": [],
    }

    # 2. Execute
    result = _route_after_llm(state)

    # 3. Assertions
    assert result == "update_memory"


def test_route_after_execution_success_escape_hatch():
    """Green Path: If the agent marks task as complete, force route into subgraph."""
    # 1. Setup Mock Environment
    messages = [
        AIMessage(
            content="", tool_calls=[{"name": "mark_task_already_completed", "args": {}, "id": "1"}]
        ),
        ToolMessage(content="Task complete", tool_call_id="1", name="mark_task_already_completed"),
    ]
    state = {"messages": messages, "intent_category": "workspace_operation"}

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_execution_success_read_only():
    """Green Path: Execution succeeds for a pure read task. Safely bypass PR generation."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_read_only",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "modified_tex_files": [],
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "update_memory"


def test_route_after_execution_success_tool_loop():
    """Green Path: The last message was a tool result, loop back to the LLM to process it."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "disambiguation_options": None,
        "clarification_question": None,
        "messages": [
            ToolMessage(content="File contents here", tool_call_id="call_123", name="read_file")
        ],
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_execution_success_workspace_operation():
    """Green Path: Execution succeeds. Route to the pull_request_subgraph to handle verification."""
    # 1. Setup Mock Environment
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

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_execution_success_workspace_operation_no_modifications():
    """Green Path: Execution concludes without using tools. Delegate evaluation to subgraph."""
    # 1. Setup Mock Environment
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

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_llm_fallback_missing_tools():
    """Edge Path: LLM responds without tool calls during an operation. Force retry."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "intent_category": "workspace_operation",
        "execution_retry_count": 0,
        "messages": [AIMessage(content="I have reviewed the task and it is complete.")],
    }

    # 2. Execute
    result = _route_after_llm(state)

    # 3. Assertions
    assert result == "force_tool_retry"


def test_route_after_llm_fallback_empty_messages():
    """Edge Path: Safe fallback to subgraph if message trace is unexpectedly empty."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "messages": [],
    }

    # 2. Execute
    result = _route_after_llm(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_execution_fallback_clarification_question():
    """Edge Path: Agent asked a clarification question."""
    # 1. Setup Mock Environment
    state = {
        "disambiguation_options": None,
        "clarification_question": "Should I use a list comprehension?",
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "clarify"


def test_route_after_execution_fallback_disambiguate_files():
    """Edge Path: File disambiguation required from human."""
    # 1. Setup Mock Environment
    state = {
        "disambiguation_options": ["/src/main.py", "/tests/main.py"],
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "clarify"


def test_route_after_execution_fallback_empty_messages():
    """Edge Path: Safe fallback to subgraph if message trace is empty."""
    # 1. Setup Mock Environment
    state = {
        "intent_category": "workspace_operation",
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [],
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "pull_request_subgraph"


def test_route_after_execution_fallback_retry_loop():
    """Edge Path: Execution crashed and retries remain. Loop back to execution."""
    # 1. Setup Mock Environment
    state = {
        "disambiguation_options": None,
        "is_aborted": False,
        "latest_traceback_error": "SyntaxError: invalid syntax",
        "execution_retry_count": 1,
    }

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_execution_error_max_retries(mocker):
    """Red Path: Execution crashed and max retries reached. Graph aborts to cleanup."""
    # 1. Setup Mock Environment
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

    # 2. Execute
    result = _route_after_execution(state)

    # 3. Assertions
    assert result == "cleanup_workflow"


# ==========================================
# Workflow: Human-in-the-Loop Clarification
# ==========================================


def test_route_after_human_clarify_success_approved():
    """Green Path: User merges a PR ('LGTM') while the agent is paused at clarification."""
    # 1. Setup Mock Environment
    state = {
        "human_approved": True,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_clarify(state)

    # 3. Assertions
    assert result == "pr_merged"


def test_route_after_human_clarify_fallback_mid_task():
    """Edge Path: User resolved file conflict mid-task."""
    # 1. Setup Mock Environment
    state = {
        "workspace_absolute_path": "/fake/path",
        "disambiguation_options": ["fileA.py"],
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_clarify(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_human_clarify_fallback_missing_path():
    """Edge Path: Path is still missing, force re-evaluation of intent."""
    # 1. Setup Mock Environment
    state = {
        "workspace_absolute_path": None,
        "clarification_question": "Should I capitalize it?",
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_clarify(state)

    # 3. Assertions
    assert result == "parse_intent"


def test_route_after_human_clarify_fallback_workspace():
    """Edge Path: User clarified initial workspace mapping."""
    # 1. Setup Mock Environment
    state = {
        "workspace_absolute_path": "/fake/path",
        "disambiguation_options": None,
        "clarification_question": None,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_clarify(state)

    # 3. Assertions
    assert result == "parse_intent"


# ==========================================
# Workflow: Pull Request Evaluation Subgraph
# ==========================================


def test_route_pr_entry_success_agentic_ci():
    """Green Path: If PR context is present, route to agentic CI node."""
    # 1. Setup Mock Environment
    state = {"commit_sha": "abc1234", "pr_number": 42}

    # 2. Execute
    result = _route_pr_entry(state)

    # 3. Assertions
    assert result == "agentic_ci"


def test_route_pr_entry_success_compile_latex():
    """Green Path: If LaTeX files modified, intercept and compile them first."""
    # 1. Setup Mock Environment
    state = {"modified_tex_files": ["main.tex"]}

    # 2. Execute
    result = _route_pr_entry(state)

    # 3. Assertions
    assert result == "compile_latex"


def test_route_pr_entry_success_run_pre_commit():
    """Green Path: If no LaTeX files modified, run pre-commit checks first."""
    # 1. Setup Mock Environment
    state = {"modified_tex_files": []}

    # 2. Execute
    result = _route_pr_entry(state)

    # 3. Assertions
    assert result == "run_pre_commit"


def test_route_after_pre_commit_success_standard():
    """Green Path: Pre-commit succeeded, proceed to diff evaluation."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": None}

    # 2. Execute
    result = _route_after_pre_commit(state)

    # 3. Assertions
    assert result == "evaluate_diff"


def test_route_after_compilation_success_standard():
    """Green Path: Compilation succeeded, proceed to PR review."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": None}

    # 2. Execute
    result = _route_after_compilation(state)

    # 3. Assertions
    assert result == "review_pr"


def test_route_after_evaluation_success_pass():
    """Green Path: Critic passed the diff, proceed to PR review."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "intent_category": "workspace_operation",
    }

    # 2. Execute
    result = _route_after_evaluation(state)

    # 3. Assertions
    assert result == "review_pr"


def test_route_after_evaluation_success_read_only_pass():
    """Green Path: Critic confirmed safe bypass, exit the subgraph."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "intent_category": "workspace_read_only",
    }

    # 2. Execute
    result = _route_after_evaluation(state)

    # 3. Assertions
    assert result == END


def test_route_after_pre_commit_fallback_error():
    """Edge Path: Pre-commit failed, exit subgraph to resume execution node."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": "pre_commit_error"}

    # 2. Execute
    result = _route_after_pre_commit(state)

    # 3. Assertions
    assert result == END


def test_route_after_compilation_fallback_syntax_error():
    """Edge Path: Compilation failed, exit subgraph to resume execution node."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": "latex_compilation_error"}

    # 2. Execute
    result = _route_after_compilation(state)

    # 3. Assertions
    assert result == END


def test_route_after_evaluation_fallback_rejection():
    """Edge Path: Critic rejected diff, exit subgraph to resume execution node."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": "semantic_review_rejection"}

    # 2. Execute
    result = _route_after_evaluation(state)

    # 3. Assertions
    assert result == END


def test_route_after_pre_commit_error_max_retries():
    """Red Path: Pre-commit max retries reached. Graph aborts and exits subgraph."""
    # 1. Setup Mock Environment
    state = {"is_aborted": True}

    # 2. Execute
    result = _route_after_pre_commit(state)

    # 3. Assertions
    assert result == END


def test_route_after_compilation_error_max_retries():
    """Red Path: Compilation max retries reached. Graph aborts and exits subgraph."""
    # 1. Setup Mock Environment
    state = {"is_aborted": True}

    # 2. Execute
    result = _route_after_compilation(state)

    # 3. Assertions
    assert result == END


# ==========================================
# Workflow: Post-Subgraph & Human PR Review
# ==========================================


def test_route_after_subgraph_success_agentic_ci():
    """Green Path: Agentic CI completes cleanly and exits the parent workflow."""
    # 1. Setup Mock Environment
    state = {"repo_full_name": "owner/repo", "commit_sha": "abc1234", "is_aborted": False}

    # 2. Execute
    result = _route_after_subgraph(state)

    # 3. Assertions
    assert result == END


def test_route_after_subgraph_success_standard():
    """Green Path: Subgraph completes and generates PR, requiring human review."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "pending_pr_url": "https://github.com/pr",
        "human_approved": False,
    }

    # 2. Execute
    result = _route_after_subgraph(state)

    # 3. Assertions
    assert result == "human_pr_node"


def test_route_after_human_pr_success_approved():
    """Green Path: PR approved by user, route to parent graph merge node."""
    # 1. Setup Mock Environment
    state = {
        "human_approved": True,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_pr(state)

    # 3. Assertions
    assert result == "pr_merged"


def test_route_after_subgraph_fallback_error():
    """Edge Path: Subgraph propagated a semantic or compilation error."""
    # 1. Setup Mock Environment
    state = {"is_aborted": False, "latest_traceback_error": "semantic_review_rejection"}

    # 2. Execute
    result = _route_after_subgraph(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_subgraph_fallback_hallucination_retry():
    """Edge Path: If the subgraph hits a hallucination trap, route back to execute_task."""
    # 1. Setup Mock Environment
    state = {"latest_traceback_error": "hallucinated_success", "is_aborted": False}

    # 2. Execute
    result = _route_after_subgraph(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_subgraph_fallback_human_feedback():
    """Edge Path: Subgraph ended because a human provided written feedback to refine PR."""
    # 1. Setup Mock Environment
    state = {
        "is_aborted": False,
        "latest_traceback_error": None,
        "messages": [HumanMessage(content="Please change the color to blue.")],
    }

    # 2. Execute
    result = _route_after_subgraph(state)

    # 3. Assertions
    assert result == "execute_task"


def test_route_after_human_pr_fallback_feedback():
    """Edge Path: User provided feedback instead of approval, route to execution."""
    # 1. Setup Mock Environment
    state = {
        "human_approved": False,
        "is_aborted": False,
    }

    # 2. Execute
    result = _route_after_human_pr(state)

    # 3. Assertions
    assert result == "execute_task"


# ==========================================
# Workflow: Global Abort Override
# ==========================================


def test_global_routing_error_abort_override():
    """Red Path: The is_aborted flag forces an immediate exit to respective cleanup nodes."""
    # 1. Setup Mock Environment
    state = {"is_aborted": True}

    # 2. Execute
    result_intent = _route_after_intent(state)
    result_llm = _route_after_llm(state)
    result_exec = _route_after_execution(state)
    result_subgraph = _route_after_subgraph(state)
    result_clarify = _route_after_human_clarify(state)
    result_pr = _route_after_human_pr(state)

    result_eval = _route_after_evaluation(state)
    result_comp = _route_after_compilation(state)
    result_pre = _route_after_pre_commit(state)

    # 3. Assertions
    assert result_intent == "cleanup_workflow"
    assert result_llm == "cleanup_workflow"
    assert result_exec == "cleanup_workflow"
    assert result_subgraph == "cleanup_workflow"
    assert result_clarify == "cleanup_workflow"
    assert result_pr == "cleanup_workflow"

    assert result_eval == END
    assert result_comp == END
    assert result_pre == END
