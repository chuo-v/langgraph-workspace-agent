import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.workspace_agent.orchestrator.nodes.routing import (
    _extract_tier_command,
    _invoke_escalating_router,
    clarification_node,
    cleanup_workflow_node,
    conversational_reply_node,
    human_node,
    parse_intent_node,
)
from src.workspace_agent.orchestrator.router import TIER_BASE

# ==========================================
# Component: _extract_tier_command
# ==========================================


def test_extract_tier_command_success_case_insensitive():
    """Green Path: Handles weird casing."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/USE:STANDARD execute the script"
    )
    assert has_frontier is False
    assert has_standard is True
    assert req_model is None
    assert clean == "execute the script"


def test_extract_tier_command_success_frontier_prefix():
    """Green Path: Frontier command at the beginning."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/use:frontier do a security audit"
    )
    assert has_frontier is True
    assert has_standard is False
    assert req_model is None
    assert clean == "do a security audit"


def test_extract_tier_command_success_middle():
    """Green Path: Command buried in the middle with awkward spacing."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "in langgraph_workspace_agent_test_arena  /use:frontier   change the title"
    )
    assert has_frontier is True
    assert has_standard is False
    assert req_model is None
    assert clean == "in langgraph_workspace_agent_test_arena change the title"


def test_extract_tier_command_success_model_override():
    """Green Path: Successfully parses dynamic model override keys."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/use:qwen_local summarize this"
    )
    assert has_frontier is False
    assert has_standard is False
    assert req_model == "qwen_local"
    assert clean == "summarize this"


def test_extract_tier_command_success_standard_suffix():
    """Green Path: Standard command at the end."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "change the log level /use:standard"
    )
    assert has_frontier is False
    assert has_standard is True
    assert req_model is None
    assert clean == "change the log level"


def test_extract_tier_command_fallback_empty():
    """Edge Path: Empty string."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command("")
    assert has_frontier is False
    assert has_standard is False
    assert req_model is None
    assert clean == ""


def test_extract_tier_command_error_missing():
    """Red Path: No command present."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "execute the script normally"
    )
    assert has_frontier is False
    assert has_standard is False
    assert req_model is None
    assert clean == "execute the script normally"


# ==========================================
# Component: _invoke_escalating_router
# ==========================================


def test_invoke_escalating_router_fallback_timeout(mocker):
    """Edge Path: Router isolates blocking requests and escalates cleanly on timeout."""
    # Force the router timeout to be effectively zero so it instantly fails
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.settings.agent.router_timeout_seconds", 0.01
    )

    # Create a mock router LLM that deliberately sleeps longer than the timeout
    mock_llm = mocker.Mock()

    def slow_invoke(*args, **kwargs):
        time.sleep(0.1)
        return "Too Slow!"

    mock_llm.invoke.side_effect = slow_invoke

    # Force the factory to return our slow LLM for all tiers so the loop finishes
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_intent_router", return_value=mock_llm
    )

    decision, latest_error = _invoke_escalating_router(
        {"instruction": "Test Prompt", "recent_context": ""}, TIER_BASE
    )

    # Verify the thread executor safely detached and returned the timeout error
    assert decision is None
    assert "timed out after" in latest_error


# ==========================================
# Workflow: Parse Intent Node
# ==========================================


def test_parse_intent_node_success_forces_model(mocker):
    """Green Path: Verifies parsing a model command translates into state and scrubs instruction."""
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "workspace_operation"
    mock_decision.inferred_workspace = None
    mock_decision.task_complexity = "low"
    mock_decision.is_context_missing = False
    mock_decision.router_confidence = 0.95

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    state = {"original_instruction": "generate report /use:gemini_pro"}
    result = parse_intent_node(state)

    assert result["requested_model"] == "gemini_pro"
    assert result["original_instruction"] == "generate report"


def test_parse_intent_node_success_forces_tier(mocker):
    """Green Path: Verifies that parsing a manual command translates into state flags."""
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "workspace_operation"
    mock_decision.inferred_workspace = None
    mock_decision.task_complexity = "low"  # low complexity to prevent auto-escalation
    mock_decision.is_context_missing = False
    mock_decision.router_confidence = 0.95

    # Mock the escalating router to return the decision and no error
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    # 1. Test the command
    state_standard = {"original_instruction": "/use:standard optimize the script"}
    result_standard = parse_intent_node(state_standard)

    assert result_standard["force_standard_tier"] is True
    assert result_standard["force_frontier_tier"] is False
    assert result_standard["original_instruction"] == "optimize the script"

    # 2. Test the command
    state_frontier = {"original_instruction": "/use:frontier rewrite the core engine"}
    result_frontier = parse_intent_node(state_frontier)

    assert result_frontier["force_standard_tier"] is False
    assert result_frontier["force_frontier_tier"] is True
    assert result_frontier["original_instruction"] == "rewrite the core engine"


def test_parse_intent_node_success_sliding_window(mocker):
    """
    Green Path: parse_intent_node successfully injects recent conversation
    history into the Tier 1 routing prompt for follow-up context.
    """
    # mock the structured output router decision
    mock_router = mocker.Mock()
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "workspace_operation"
    mock_decision.inferred_workspace = "test_arena"
    mock_decision.is_context_missing = False
    mock_decision.clarification_question_to_ask = None
    mock_decision.task_complexity = "medium"
    mock_decision.router_confidence = 0.99
    mock_router.invoke.return_value = mock_decision

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_intent_router", return_value=mock_router
    )

    # create a state simulating a mid-conversation clarification
    state = {
        "original_instruction": "Run the script",
        "messages": [
            HumanMessage(content="Where is the file?"),
            AIMessage(content="I cannot find it."),
            HumanMessage(content="It is in scripts/data_processor.py"),
        ],
    }

    parse_intent_node(state)

    # Extract the payload dictionary sent to the chain
    prompt_sent = mock_router.invoke.call_args[0][0]
    recent_context = prompt_sent["recent_context"]

    # verify the sliding window context was successfully appended
    assert "Recent Conversation Context:" in recent_context
    assert "It is in scripts/data_processor.py" in recent_context


def test_parse_intent_node_success_sliding_window_truncation(mocker):
    """
    Green Path: Verifies that extremely long conversational messages are aggressively truncated.
    """
    mock_router = mocker.Mock()
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "conversational"
    mock_router.invoke.return_value = mock_decision
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_intent_router", return_value=mock_router
    )

    # Generate a string way over the 500 char limit
    long_content = "A" * 1000

    state = {
        "original_instruction": "Run the script",
        "messages": [AIMessage(content="Hello"), HumanMessage(content=long_content)],
    }

    parse_intent_node(state)

    # Extract the payload dictionary sent to the chain
    prompt_sent = mock_router.invoke.call_args[0][0]
    recent_context = prompt_sent["recent_context"]

    # Verify the truncation text was appended
    assert "[TRUNCATED FOR ROUTING]" in recent_context
    # The prompt should contain exactly 500 'A's, not 1000
    assert "A" * 501 not in recent_context
    assert "A" * 500 in recent_context


def test_parse_intent_node_success_fast_path_conversational(mocker):
    """
    Green Path: Verifies that simple decline keywords bypass the router and return conversational
    intent.
    """
    # mock the escalating router to crash if called, proving the fast path safely bypassed it
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing._invoke_escalating_router",
        side_effect=Exception("Router should not be called"),
    )

    state = {
        "original_instruction": "Run pytest",
        "messages": [HumanMessage(content="No.")],
    }

    result = parse_intent_node(state)

    assert result["intent_category"] == "conversational"
    assert result["router_confidence"] == 1.0
    assert result["t1_base_calls"] == 0


def test_parse_intent_node_success_latest_human_message_override(mocker):
    """
    Green Path: Verifies that the latest human conversational message correctly
    overrides the original_instruction to prevent task loops.
    """
    mock_router = mocker.Mock()
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "conversational"
    mock_decision.inferred_workspace = None
    mock_decision.is_context_missing = False
    mock_decision.clarification_question_to_ask = None
    mock_decision.task_complexity = "low"
    mock_decision.router_confidence = 0.99
    mock_router.invoke.return_value = mock_decision

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_intent_router", return_value=mock_router
    )

    state = {
        "original_instruction": "Run the pytest suite",
        "messages": [
            HumanMessage(content="Run the pytest suite"),
            AIMessage(content="All tests passed."),
            # Changed from "No" to a command that forces it to hit the router
            HumanMessage(content="Actually, compile the report instead."),
        ],
    }

    parse_intent_node(state)

    # Extract the payload dictionary sent to the chain
    prompt_sent = mock_router.invoke.call_args[0][0]

    # Verify the routing instruction is the latest conversational message, not the original task
    assert prompt_sent["instruction"] == "Actually, compile the report instead."


def test_parse_intent_node_fallback_context_missing_trap(mocker):
    """
    Edge Path: Verifies that if the LLM flags missing context, the node
    traps the clarification question and safely nullifies the target path.
    """
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "workspace_operation"
    mock_decision.inferred_workspace = "test_arena"
    mock_decision.task_complexity = "medium"

    # Trigger the missing context trap
    mock_decision.is_context_missing = True
    mock_decision.clarification_question_to_ask = "Which specific file do you want to edit?"
    mock_decision.router_confidence = 0.99

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    state = {"original_instruction": "fix the bug in the test arena"}
    result = parse_intent_node(state)

    # Verify the trap successfully injected the question into the state
    assert result["clarification_question"] == "Which specific file do you want to edit?"

    # Verify the path was forcefully nullified to prevent blind execution
    assert result["workspace_absolute_path"] is None


def test_parse_intent_node_fallback_ignores_system_traps(mocker):
    """
    Edge Path: Verifies that system-injected rejection or error messages
    are safely bypassed when dynamically extracting the latest human instruction.
    """
    mock_router = mocker.Mock()
    mock_decision = mocker.Mock()
    mock_decision.intent_category = "workspace_operation"
    mock_decision.inferred_workspace = None
    mock_decision.is_context_missing = False
    mock_decision.clarification_question_to_ask = None
    mock_decision.task_complexity = "low"
    mock_decision.router_confidence = 0.99
    mock_router.invoke.return_value = mock_decision

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_intent_router", return_value=mock_router
    )

    state = {
        "original_instruction": "Fix the compiler bug",
        "messages": [
            HumanMessage(content="Fix the compiler bug"),
            AIMessage(content="I tried."),
            HumanMessage(content="SYSTEM REJECTION: SyntaxError"),  # Should be ignored
            AIMessage(content="I tried again."),
            HumanMessage(content="SYSTEM ERROR: API Timeout"),  # Should be ignored
        ],
    }

    parse_intent_node(state)

    # Extract the payload dictionary sent to the chain
    prompt_sent = mock_router.invoke.call_args[0][0]

    # Verify it bypassed the system traps and locked onto the actual human request
    assert prompt_sent["instruction"] == "Fix the compiler bug"


def test_parse_intent_node_error_escalation_failure(mocker):
    """Red Path: parse_intent_node aborts safely if all router tiers fail completely."""
    # mock the escalating router to return None (meaning it exhausted all tiers)
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing._invoke_escalating_router",
        return_value=(None, "API Timeout across all tiers"),
    )

    state = {"original_instruction": "do something"}
    result = parse_intent_node(state)

    # verify the workflow safely aborts and formats the error
    assert result.get("is_aborted") is True
    assert "Router Escalation Failed" in result["messages"][0].content
    assert "API Timeout across all tiers" in result["messages"][0].content


# ==========================================
# Workflow: Human-in-the-Loop & Clarification
# ==========================================


def test_human_node_success_strict_noop():
    """
    Green Path: The human_node must remain an absolute no-op breakpoint.
    It must never mutate, inject, or alter the graph state, as its only purpose
    is to trigger a LangGraph interrupt.
    """
    complex_state = {
        "messages": ["msg1", "msg2"],
        "workspace_absolute_path": "/tmp/safe",
        "execution_retry_count": 2,
        "latest_traceback_error": "SyntaxError",
        "force_frontier_tier": True,
    }

    # Execute the node
    result = human_node(complex_state)

    # It must return an empty state delta, proving it introduces absolutely zero
    # side-effects to the graph.
    assert result == {}, (
        "Architectural Violation: human_node returned a state delta. "
        "This node must remain a pure empty breakpoint for LangGraph interrupts."
    )


def test_clarification_node_success_with_question():
    """Green Path: Prioritizes rendering the dynamic LLM clarification question."""
    state = {
        "clarification_question": "Do you want light or dark mode?",
        "disambiguation_options": ["light", "dark"],
    }
    result = clarification_node(state)

    assert "Question from Agent" in result["messages"][0].content
    assert "dark mode" in result["messages"][0].content


def test_clarification_node_fallback_legacy_files():
    """Edge Path: Renders legacy file selection if only options are provided."""
    state = {
        "clarification_question": None,
        "disambiguation_options": ["/src/main.py", "/tests/main.py"],
    }
    result = clarification_node(state)

    assert "Multiple matching files found" in result["messages"][0].content
    assert "/src/main.py" in result["messages"][0].content


def test_clarification_node_error_workspace_ambiguous():
    """Red Path: Triggers global workspace error if no question or options exist."""
    state = {"clarification_question": None, "disambiguation_options": None}
    result = clarification_node(state)

    assert "Ambiguous Workspace Request" in result["messages"][0].content


# ==========================================
# Workflow: Terminal & Cleanup Nodes
# ==========================================


def test_conversational_reply_node_success_standard(mocker):
    """Green Path: Conversational node successfully invokes the LLM without tools."""
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = AIMessage(content="Hello there!")
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.get_execution_llm_sequence",
        return_value=[mock_llm],
    )

    state = {"messages": [HumanMessage(content="Hi")], "t1_base_calls": 0}
    result = conversational_reply_node(state)

    assert result["messages"][0].content == "Hello there!"
    assert result["t1_base_calls"] == 1
    assert result["is_busy"] is False

    # Verify system prompt was injected to prevent hallucination
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert isinstance(prompt_sent[0], SystemMessage)
    assert "helpful AI workspace assistant" in prompt_sent[0].content


def test_cleanup_workflow_node_success_hygiene():
    """Green Path: cleanup_workflow_node wipes all dangling operational state variables."""
    dirty_state = {
        "is_aborted": True,
        "active_agent_branch": "agent/update-123",
        "pending_pr_url": "http://github.com/pr",
        "latest_traceback_error": "SyntaxError",
        "execution_retry_count": 2,
        "disambiguation_options": ["fileA.py"],
        "clarification_question": "What is the meaning of life?",
        "requested_model": "gemini_pro",
    }

    result = cleanup_workflow_node(dirty_state)

    assert result["is_aborted"] is False
    assert result["active_agent_branch"] is None
    assert result["pending_pr_url"] is None
    assert result["latest_traceback_error"] is None
    assert result["execution_retry_count"] == 0
    assert result["disambiguation_options"] is None
    assert result["clarification_question"] is None
    assert result["requested_model"] is None
    assert result["is_busy"] is False


def test_cleanup_workflow_node_fallback_git_cleanup_failure(mocker):
    """Edge Path: Ensure Git exceptions during cleanup are trapped and logged."""
    # force the local branch cleanup helper to throw an exception
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.routing.cleanup_local_branch",
        side_effect=Exception("Git index locked"),
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "target_branch": "main",
        "active_agent_branch": "agent/test-branch",
        "messages": [HumanMessage(content="Aborting execution")],
    }

    result = cleanup_workflow_node(state)

    # verify the state was still successfully cleaned up despite the Git failure
    assert result["is_aborted"] is False
    assert result["active_agent_branch"] is None

    # verify the warning was successfully appended to the final message
    final_message = result["messages"][-1].content
    assert "Git Cleanup Warning" in final_message
    assert "Git index locked" in final_message
