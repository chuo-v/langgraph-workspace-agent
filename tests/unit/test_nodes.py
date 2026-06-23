import subprocess
import time

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from src.workspace_agent.orchestrator.nodes import (
    MAX_CONSECUTIVE_TOOL_STEPS,
    _build_cross_workspace_prompt,
    _chunk_git_diff,
    _extract_modified_tex_files,
    _extract_tier_command,
    _filter_execution_context,
    _generate_commit_message,
    _generate_pr_metadata,
    _invoke_escalating_router,
    _is_tool_error,
    _resolve_execution_tier,
    _sanitize_llm_response,
    agentic_ci_node,
    clarification_node,
    cleanup_workflow_node,
    compile_node,
    conversational_reply_node,
    evaluate_diff_node,
    execute_task_node,
    human_node,
    parse_intent_node,
    pr_merged_node,
    review_pr_node,
    update_memory_node,
    workspace_tools_node,
)
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
)

# ==========================================
# Component: _extract_tier_command
# ==========================================


def test_extract_tier_command_success_case_insensitive():
    """Green Path: Handles weird casing."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/STANDARD execute the script"
    )
    assert has_frontier is False
    assert has_standard is True
    assert req_model is None
    assert clean == "execute the script"


def test_extract_tier_command_success_frontier_prefix():
    """Green Path: Frontier command at the beginning."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/frontier do a security audit"
    )
    assert has_frontier is True
    assert has_standard is False
    assert req_model is None
    assert clean == "do a security audit"


def test_extract_tier_command_success_middle():
    """Green Path: Command buried in the middle with awkward spacing."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "in langgraph_workspace_agent_test_arena  /frontier   change the title"
    )
    assert has_frontier is True
    assert has_standard is False
    assert req_model is None
    assert clean == "in langgraph_workspace_agent_test_arena change the title"


def test_extract_tier_command_success_model_override():
    """Green Path: Successfully parses dynamic model override keys."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "/model:qwen_local summarize this"
    )
    assert has_frontier is False
    assert has_standard is False
    assert req_model == "qwen_local"
    assert clean == "summarize this"


def test_extract_tier_command_success_standard_suffix():
    """Green Path: Standard command at the end."""
    clean, has_frontier, has_standard, req_model = _extract_tier_command(
        "change the log level /standard"
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
        "src.workspace_agent.orchestrator.nodes.settings.agent.router_timeout_seconds", 0.01
    )

    # Create a mock router LLM that deliberately sleeps longer than the timeout
    mock_llm = mocker.Mock()

    def slow_invoke(*args, **kwargs):
        time.sleep(0.1)
        return "Too Slow!"

    mock_llm.invoke.side_effect = slow_invoke

    # Force the factory to return our slow LLM for all tiers so the loop finishes
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_intent_router", return_value=mock_llm)

    decision, latest_error = _invoke_escalating_router(
        {"instruction": "Test Prompt", "recent_context": ""}, TIER_BASE
    )

    # Verify the thread executor safely detached and returned the timeout error
    assert decision is None
    assert "timed out after" in latest_error


# ==========================================
# Component: _resolve_execution_tier
# ==========================================


def test_resolve_execution_tier_success_priorities(mocker):
    """Green Path: Verifies that tier overrides are correctly prioritized."""

    # 1. Test Explicit Model Override (Highest Priority)
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_tier_for_model", return_value=TIER_STANDARD
    )
    assert (
        _resolve_execution_tier(
            {
                "requested_model": "deepseek_fast",
                "force_frontier_tier": True,  # Explicit model should beat the frontier flag
            }
        )
        == TIER_STANDARD
    )

    # Reset mock for default flag checks
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_tier_for_model", return_value=None)

    # 2. Test Frontier Override
    assert (
        _resolve_execution_tier(
            {"requested_model": None, "force_frontier_tier": True, "force_standard_tier": False}
        )
        == TIER_FRONTIER
    )

    # 3. Test Standard Override
    assert (
        _resolve_execution_tier(
            {"requested_model": None, "force_frontier_tier": False, "force_standard_tier": True}
        )
        == TIER_STANDARD
    )

    # 4. Test Conflict (Frontier must win)
    assert (
        _resolve_execution_tier(
            {"requested_model": None, "force_frontier_tier": True, "force_standard_tier": True}
        )
        == TIER_FRONTIER
    )


# ==========================================
# Component: _build_cross_workspace_prompt
# ==========================================


def test_build_cross_workspace_prompt_success_operation():
    """Green Path: Verifies that standard operations receive the strict write constraint."""
    state = {"intent_category": "workspace_operation"}
    prompt_msg = _build_cross_workspace_prompt(state, "test_arena", "/tmp/path")

    # Should not have the read-only flag
    assert "TASK TYPE: READ ONLY" not in prompt_msg.content

    # Should contain the new write operation guardrails
    assert "TASK TYPE: WRITE OPERATION" in prompt_msg.content
    assert "physically accomplish the task" in prompt_msg.content


def test_build_cross_workspace_prompt_success_read_only():
    """Green Path: Verifies that the read-only constraint is forcefully appended."""
    state = {"intent_category": "workspace_read_only"}
    prompt_msg = _build_cross_workspace_prompt(state, "test_arena", "/tmp/path")

    assert "TASK TYPE: READ ONLY" in prompt_msg.content
    assert "Do NOT attempt to write files" in prompt_msg.content


# ==========================================
# Component: _sanitize_llm_response
# ==========================================


def test_sanitize_llm_response_success_empty_string():
    """Green Path: Preserves empty string content without injecting hallucination risks."""
    msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "call_123"}])
    _sanitize_llm_response(msg)
    assert msg.content == ""


def test_sanitize_llm_response_success_list_format_empty():
    """Green Path: Cleans out empty text blocks hidden inside structural lists."""
    msg = AIMessage(
        content=[{"type": "text", "text": "   "}, {"type": "text", "text": ""}],
        tool_calls=[{"name": "test", "args": {}, "id": "call_123"}],
    )
    _sanitize_llm_response(msg)
    assert msg.content == ""


def test_sanitize_llm_response_success_list_format_valid():
    """Green Path: Preserves valid text blocks from lists and flattens them to a string."""
    msg = AIMessage(
        content=[{"type": "text", "text": "Valid Text"}, {"type": "text", "text": "   "}],
        tool_calls=[{"name": "test", "args": {}, "id": "call_123"}],
    )
    _sanitize_llm_response(msg)
    assert isinstance(msg.content, str)
    assert msg.content == "Valid Text"


# ==========================================
# Component: _extract_modified_tex_files
# ==========================================


def test_extract_modified_tex_files_success_extraction():
    """Green Path: Validates that LaTeX file modifications are successfully trapped."""
    # Set up a fake LLM response containing tool calls to modify a .tex file and a .py file
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/src/report.tex"}, "id": "1"},
            {"name": "search_and_replace", "args": {"file_path": "/src/main.py"}, "id": "2"},
            {"name": "replace_text_block", "args": {"file_path": "/src/thesis.TEX"}, "id": "3"},
        ],
    )

    # Extract
    new_files = _extract_modified_tex_files(["/existing.tex"], mock_response)

    # The .tex files should be added, the .py file ignored, and existing preserved
    assert "/existing.tex" in new_files
    assert "/src/report.tex" in new_files
    assert "/src/thesis.TEX" in new_files
    assert "/src/main.py" not in new_files


def test_extract_modified_tex_files_success_deletion():
    """Green Path: Validates that deleted LaTeX files are removed from the compilation queue."""
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "delete_file", "args": {"file_path": "/src/report.tex"}, "id": "1"},
        ],
    )

    new_files = _extract_modified_tex_files(["/src/report.tex", "/src/keep.tex"], mock_response)

    assert "/src/report.tex" not in new_files
    assert "/src/keep.tex" in new_files


def test_extract_modified_tex_files_success_rename():
    """Green Path: Validates that renamed LaTeX files update the queue correctly."""
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "rename_file",
                "args": {"source_path": "/src/old.tex", "destination_path": "/src/new.tex"},
                "id": "1",
            },
            {
                "name": "rename_file",
                "args": {
                    "source_path": "/src/ignore.txt",
                    "destination_path": "/src/ignore_new.txt",
                },
                "id": "2",
            },
        ],
    )

    new_files = _extract_modified_tex_files(["/src/old.tex", "/src/other.tex"], mock_response)

    # old.tex should be removed
    assert "/src/old.tex" not in new_files
    # new.tex should be added
    assert "/src/new.tex" in new_files
    # other.tex should remain untouched
    assert "/src/other.tex" in new_files
    # .txt files should be ignored completely
    assert "/src/ignore_new.txt" not in new_files


# ==========================================
# Component: _filter_execution_context
# ==========================================


def test_filter_execution_context_success_prevents_hallucination():
    """Green Path: Replaces tool-less AI messages unless it is the very last message."""
    messages = [
        AIMessage(content="I claimed I did work here but used no tools."),  # should be replaced
        HumanMessage(content="Do another task"),  # should be kept
        AIMessage(content="I am the last message without tools."),  # should be kept natively
    ]
    filtered = _filter_execution_context(messages)

    assert len(filtered) == 3
    assert "System Note" in filtered[0].content
    assert "Do another task" in filtered[1].content
    assert "I am the last message" in filtered[2].content


def test_filter_execution_context_success_replaces_markers():
    """Green Path: Ensures orchestrator success markers are replaced with boundaries."""
    messages = [
        HumanMessage(content="Do a task"),
        AIMessage(content="✅ **Execution Complete**"),
        AIMessage(content="✅ **Revisions Applied**"),
        AIMessage(
            content="Normal AI Message", tool_calls=[{"name": "test", "args": {}, "id": "call_123"}]
        ),
    ]
    filtered = _filter_execution_context(messages)

    assert len(filtered) == 4
    assert "System Note" in filtered[1].content
    assert "System Note" in filtered[2].content
    assert "Normal AI Message" in filtered[3].content


def test_filter_execution_context_fallback_handles_list_content():
    """
    Edge Path: Verifies that context filtering clears out orchestration success markers
    even when they are wrapped in structured lists (Gemini provider payload style).
    """
    messages = [
        HumanMessage(content="Refactor values"),
        AIMessage(
            content=[
                {"type": "text", "text": "✅ **Execution Complete**"},
                {"type": "text", "text": "Unrelated structural text metadata block"},
            ]
        ),
        AIMessage(
            content="Normal response text",
            tool_calls=[{"name": "read_files", "id": "1", "args": {}}],
        ),
    ]

    filtered = _filter_execution_context(messages)

    assert len(filtered) == 3
    assert "System Note" in filtered[1].content
    assert "Normal response text" in filtered[2].content


# ==========================================
# Component: _is_tool_error
# ==========================================


def test_is_tool_error_success_identifies_errors():
    """Green Path: Explicitly tests the internal string matching bounds of _is_tool_error."""

    # 1. Hard Invocation Failures
    assert _is_tool_error("write_file", "Tool execution failed: No access") is True

    # 2. Sandbox Execution tools
    assert _is_tool_error("run_python_script", "Traceback (most recent call last):") is True
    assert _is_tool_error("compile_latex_document", "=== ERROR: Runaway argument") is True
    assert _is_tool_error("run_pytest", "FAILED (failures=1)") is True

    # 3. Standard Read/Write OS tools
    assert _is_tool_error("read_files", "Error: File not found") is True

    # Non-errors should pass cleanly
    assert _is_tool_error("run_python_script", "Execution Finished (Exit Code: 0)") is False
    assert _is_tool_error("read_files", "File content here.") is False


# ==========================================
# Component: _generate_commit_message
# ==========================================


def test_generate_commit_message_success_semantic(mocker):
    """Green Path: Successfully generates a semantic commit message from user feedback."""
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "Update toggle settings"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    msg = _generate_commit_message(
        "Please update the toggle settings to false", "+new code", "M\tfile.py"
    )
    assert msg == "🤖 Update toggle settings"


def test_generate_commit_message_fallback_api_error(mocker):
    """Edge Path: Returns the default legacy message if the LLM fails."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_execution_llm",
        side_effect=Exception("API Outage"),
    )

    msg = _generate_commit_message(
        "Please update the toggle settings to false", "+new code", "M\tfile.py"
    )
    assert msg == "🤖 Apply human feedback revisions"


def test_generate_commit_message_fallback_empty(mocker):
    """Edge Path: Returns default legacy message if no context is provided."""
    msg = _generate_commit_message("", "", "")
    assert msg == "🤖 Apply human feedback revisions"


# ==========================================
# Component: _generate_pr_metadata
# ==========================================


def test_generate_pr_metadata_success_map_reduce(mocker):
    """Green Path: Ensures large diffs trigger the map-reduce summarization pipeline."""
    # Force the chunk size to be tiny so our small mock string splits into multiple chunks
    mocker.patch("src.workspace_agent.orchestrator.nodes.MAX_DIFF_LENGTH", 40)

    mock_llm = mocker.Mock()
    # Each map and reduce step will return this generic response
    mock_llm.invoke.return_value.content = "mocked map-reduce response"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    raw_diff = (
        "diff --git a/file1.py b/file1.py\n"
        "+print('hello')\n"
        "diff --git a/file2.py b/file2.py\n"
        "+print('world')\n"
    )
    blueprint = "M\tfile1.py\nM\tfile2.py"

    branch, body, summary = _generate_pr_metadata("Update files", raw_diff, blueprint)

    # It should process 2 chunks (2 map calls) + 1 summary reduce + 1 body reduce = 4 API calls
    assert mock_llm.invoke.call_count == 4
    assert summary == "mocked map-reduce response"
    assert body == "mocked map-reduce response"
    # The branch name is derived from the LLM's summary output
    assert branch.startswith("agent/mocked-mapreduce-response-")


def test_generate_pr_metadata_success_single_chunk(mocker):
    """Green Path: Ensures small diffs bypass the map phase and go straight to reduce."""
    # Ensure the max length is huge so the diff stays as a single chunk
    mocker.patch("src.workspace_agent.orchestrator.nodes.MAX_DIFF_LENGTH", 40000)
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked generic response"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    raw_diff = "diff --git a/file1.py b/file1.py\n+print('hello')"
    blueprint = "M\tfile1.py"

    branch, body, summary = _generate_pr_metadata("Update files", raw_diff, blueprint)

    # It should process 0 map calls + 1 summary reduce + 1 body reduce = 2 total API calls
    assert mock_llm.invoke.call_count == 2
    assert summary == "mocked generic response"
    assert branch.startswith("agent/mocked-generic-response-")


def test_generate_pr_metadata_fallback_on_error(mocker):
    """Edge Path: Ensures PR metadata safely falls back to defaults if the LLM API crashes."""
    # Force the LLM's invoke method to raise an exception (simulating an API timeout)
    mock_llm = mocker.Mock()
    mock_llm.invoke.side_effect = Exception("API Outage")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    branch, body, summary = _generate_pr_metadata("Update files", "raw diff", "M\tfile.py")

    # The agent should catch the error, log it, and return safe generic defaults
    assert summary == "Automated agent modifications"
    assert body == "Automated PR generated by the LangGraph Agent."
    assert branch.startswith("agent/automated-agent-modifications-")


# ==========================================
# Component: _chunk_git_diff
# ==========================================


def test_chunk_git_diff_success_normal():
    """Green Path: Successfully splits multiple file diffs into chunks without severing logic."""
    raw_diff = (
        "diff --git a/file1.py b/file1.py\n"
        "+print('hello')\n"
        "diff --git a/file2.py b/file2.py\n"
        "+print('world')\n"
    )
    # set max length small enough to force a split across files
    chunks = _chunk_git_diff(raw_diff, max_chunk_length=50)

    assert len(chunks) == 2
    assert "file1.py" in chunks[0]
    assert "file2.py" in chunks[1]


def test_chunk_git_diff_fallback_empty():
    """Edge Path: Returns an empty list if the diff is empty or signifies no changes."""
    assert _chunk_git_diff("") == []
    assert _chunk_git_diff("No uncommitted changes") == []
    assert _chunk_git_diff("No changes compared to target") == []


def test_chunk_git_diff_fallback_mega_file():
    """Edge Path: Safely truncates the middle of a massive single-file diff."""
    raw_diff = "diff --git a/mega.py b/mega.py\n" + ("X" * 5000)

    # set max length specifically to test truncation
    chunks = _chunk_git_diff(raw_diff, max_chunk_length=100)

    assert len(chunks) == 1
    assert "[single file diff truncated]" in chunks[0]
    assert len(chunks[0]) <= 150  # half + half + placeholder string


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
        "src.workspace_agent.orchestrator.nodes._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    state = {"original_instruction": "generate report /model:gemini_pro"}
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
        "src.workspace_agent.orchestrator.nodes._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    # 1. Test the /standard command
    state_standard = {"original_instruction": "/standard optimize the script"}
    result_standard = parse_intent_node(state_standard)

    assert result_standard["force_standard_tier"] is True
    assert result_standard["force_frontier_tier"] is False
    assert result_standard["original_instruction"] == "optimize the script"

    # 2. Test the /frontier command
    state_frontier = {"original_instruction": "/frontier rewrite the core engine"}
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
        "src.workspace_agent.orchestrator.nodes.get_intent_router", return_value=mock_router
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
        "src.workspace_agent.orchestrator.nodes.get_intent_router", return_value=mock_router
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
        "src.workspace_agent.orchestrator.nodes._invoke_escalating_router",
        return_value=(mock_decision, None),
    )

    state = {"original_instruction": "fix the bug in the test arena"}
    result = parse_intent_node(state)

    # Verify the trap successfully injected the question into the state
    assert result["clarification_question"] == "Which specific file do you want to edit?"

    # Verify the path was forcefully nullified to prevent blind execution
    assert result["workspace_absolute_path"] is None


def test_parse_intent_node_error_escalation_failure(mocker):
    """Red Path: parse_intent_node aborts safely if all router tiers fail completely."""
    # mock the escalating router to return None (meaning it exhausted all tiers)
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes._invoke_escalating_router",
        return_value=(None, "API Timeout across all tiers"),
    )

    state = {"original_instruction": "do something"}
    result = parse_intent_node(state)

    # verify the workflow safely aborts and formats the error
    assert result.get("is_aborted") is True
    assert "Router Escalation Failed" in result["messages"][0].content
    assert "API Timeout across all tiers" in result["messages"][0].content


# ==========================================
# Workflow: Workspace Tools Node
# ==========================================


def test_workspace_tools_node_success_config_injection(mocker):
    """
    Green Path: Ensures the RunnableConfig is successfully propagated down
    from the graph layer to the execution registry (Dependency Injection validation).
    """
    tool_calls = [{"name": "run_python_script", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    # Mock the injected config containing our external clients
    config = {"configurable": {"docker_client": "mock_client"}}

    mock_executor = mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call", return_value="Success"
    )

    workspace_tools_node(state, config)

    mock_executor.assert_called_once()

    # Extract arguments and verify the config was explicitly passed to the registry
    args, kwargs = mock_executor.call_args
    assert args[0] == state["messages"][0].tool_calls[0]
    assert args[1] == config


def test_workspace_tools_node_fallback_clarification_trap():
    """Edge Path: Ensures the tool loop intercepts human-in-the-loop requests without breaking."""
    tool_calls = [
        {
            "name": "ask_user_for_clarification",
            "args": {"question": "Are you sure?"},
            "id": "call_123",
        }
    ]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    # Clarification traps bypass physical execution, so no mocker.patch is needed
    result = workspace_tools_node(state)

    assert result["clarification_question"] == "Are you sure?"
    assert result["disambiguation_options"] is None
    assert "Clarification requested" in result["messages"][0].content
    assert result["execution_retry_count"] == 0
    assert result["latest_traceback_error"] is None


def test_workspace_tools_node_fallback_escape_hatch(mocker):
    """
    Edge Path: Ensures that if the agent explicitly uses the escape hatch tool,
    transient errors generated by other tools in the same parallel batch are safely ignored.
    """
    tool_calls = [
        {"name": "run_python_script", "args": {}, "id": "call_fail"},
        {"name": "mark_task_already_completed", "args": {}, "id": "call_success"},
    ]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 1}

    # Simulate the python script failing, but the escape hatch succeeding
    def mock_tool_execution(tc, config):
        if tc["name"] == "run_python_script":
            return "Execution Failed: SyntaxError"
        return "Task explicitly marked complete."

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call", side_effect=mock_tool_execution
    )

    result = workspace_tools_node(state)

    # The SyntaxError should be entirely scrubbed, and retries preserved but NOT incremented
    assert result["latest_traceback_error"] is None
    assert result["execution_retry_count"] == 1


def test_workspace_tools_node_fallback_ignores_read_traceback(mocker):
    """Edge Path: Ensures read tools ignore Tracebacks in file contents."""
    tool_calls = [{"name": "read_files", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Traceback (most recent call last): False positive from reading a log file",
    )

    result = workspace_tools_node(state)

    assert result["execution_retry_count"] == 0
    assert result["latest_traceback_error"] is None


def test_workspace_tools_node_fallback_pytest_capture(mocker):
    """Edge Path: Ensures pytest failure signatures correctly increment the retry counter."""
    tool_calls = [{"name": "run_pytest", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="FAILED (failures=1)\nAssertionError: 2 != 3",
    )

    result = workspace_tools_node(state)

    assert result["execution_retry_count"] == 1
    assert "FAILED (" in result["latest_traceback_error"]


def test_workspace_tools_node_fallback_traceback_capture(mocker):
    """Edge Path: Ensures sandbox script crashes increment the retry counter."""
    tool_calls = [{"name": "run_python_script", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Traceback (most recent call last): File missing",
    )

    result = workspace_tools_node(state)

    assert result["execution_retry_count"] == 1
    assert "Traceback" in result["latest_traceback_error"]


def test_workspace_tools_node_error_sandbox_crash_max_retries(mocker):
    """Red Path: workspace_tools_node aborts if sandbox crashes hit the retry limit."""
    tool_calls = [{"name": "run_python_script", "id": "1", "args": {}}]
    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 3,
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
    }

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Traceback (most recent call last): error",
    )

    result = workspace_tools_node(state)

    assert result.get("is_aborted") is True
    assert result.get("latest_traceback_error") is None
    assert "Execution Failed" in result["messages"][-1].content


# ==========================================
# Workflow: Execute Task Node
# ==========================================


def test_execute_task_node_success_standard(mocker):
    """Green Path: Successfully invokes the execution LLM API without aborting."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.sync_repository",
        return_value='{"status": "success"}',
    )
    # Mock the LLM to return a standard AIMessage with a tool call
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(
        content="I will do the task now.",
        tool_calls=[{"name": "read_files", "args": {}, "id": "call_123"}],
    )
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
        "messages": [HumanMessage(content="Do the task")],
    }
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # Verify the workflow did not trigger circuit breakers, abort, or increment retries
    assert result.get("is_aborted", False) is False
    assert result.get("latest_traceback_error") is None
    assert result.get("execution_retry_count", 0) == 0

    # Verify the AI message was properly appended to state
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "I will do the task now."


def test_execute_task_node_fallback_api_invocation_crash(mocker):
    """Edge Path: execute_task_node catches LLM API crash and returns a SYSTEM ERROR."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.sync_repository",
        return_value='{"status": "success"}',
    )

    # mock LLM to throw an exception directly on invoke to simulate a Rate Limit or Timeout
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.side_effect = Exception("Rate Limit Exceeded")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # Mock time.sleep to not actually delay the test suite
    mocker.patch("src.workspace_agent.orchestrator.nodes.time.sleep")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
        "messages": [],
    }
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # Ensure the workflow did not abort, but instead injected the self-healing prompt
    assert result.get("is_aborted", False) is False
    assert "SYSTEM ERROR: The LLM API failed" in result["messages"][0].content
    assert result["latest_traceback_error"] == "api_invocation_error"
    assert result["execution_retry_count"] == 1


def test_execute_task_node_error_api_invocation_crash_max_retries(mocker):
    """Red Path: execute_task_node aborts if API crashes hit the retry limit."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.sync_repository",
        return_value='{"status": "success"}',
    )

    # Mock LLM to throw an exception directly
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.side_effect = Exception("Rate Limit Exceeded")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # Force state to be at the maximum retry limit (default is 3)
    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 3,
        "messages": [],
    }
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    assert result.get("is_aborted") is True
    assert "Fatal API or Parsing Error" in result["messages"][0].content


def test_execute_task_node_error_circuit_breaker():
    """
    Red Path: Verifies that if the agent executes MAX_CONSECUTIVE_TOOL_STEPS
    without human interaction, the circuit breaker safely aborts the workflow.
    """
    messages = [HumanMessage(content="Start loop")]

    # Generate enough consecutive non-human messages to trigger the breaker
    for i in range(MAX_CONSECUTIVE_TOOL_STEPS + 1):
        messages.append(AIMessage(content=f"Thinking {i}"))

    state = {"workspace_absolute_path": "/tmp", "messages": messages}

    result = execute_task_node(state)

    assert result.get("is_aborted") is True
    assert "Circuit breaker triggered" in result["messages"][0].content


def test_execute_task_node_error_escalation_crash(mocker):
    """Red Path: execute_task_node catches TerminalEscalationError from the router."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.sync_repository",
        return_value='{"status": "success"}',
    )

    # force the router to fail escalating to Tier 3
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_execution_llm",
        side_effect=TerminalEscalationError("API Offline"),
    )

    state = {
        "force_frontier_tier": True,
        "workspace_absolute_path": "/tmp/test",
        "messages": [],
    }
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    assert result.get("is_aborted") is True
    assert "Escalation Failed" in result["messages"][0].content


def test_execute_task_node_error_sync_abort(mocker):
    """Red Path: Ensure execute_task_node aborts immediately if repository sync fails."""

    # Mock the sync repository to return a realistic error status
    mock_error_json = (
        '{"status": "error", "reason": "git_command_failed", '
        '"details": "merge conflict in README.md"}'
    )

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.sync_repository",
        return_value=mock_error_json,
    )

    state = {
        "inferred_workspace": "test_arena",
        "workspace_absolute_path": "/tmp/test",
        "target_branch": "main",
        "messages": [],
    }

    result = execute_task_node(state)

    # Verify the node successfully trapped the sync failure and aborted
    assert result.get("is_aborted") is True
    assert "Sync Failed" in result["messages"][0].content
    assert "merge conflict in README.md" in result["messages"][0].content


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
# Workflow: Evaluation & Reflection Node
# ==========================================


def test_evaluate_diff_node_success_ignores_system_messages(mocker):
    """
    Green Path: Ensure evaluate_diff_node extracts the real user request, skipping system errors.
    """
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="+new code")
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "PASS"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "original_instruction": "Original instruction fallback",
        "messages": [
            HumanMessage(content="Real user request to fix the bug"),
            AIMessage(content="Bad code"),
            HumanMessage(content="SYSTEM REJECTION: The code reviewer rejected your changes."),
            AIMessage(content="More bad code"),
            HumanMessage(content="SYSTEM ERROR: API timeout."),
        ],
    }

    evaluate_diff_node(state)

    # Extract the actual prompt sent to the critic LLM
    prompt_sent = mock_llm.invoke.call_args[0][0]

    # The critic should be evaluating the "Real user request", NOT the "SYSTEM REJECTION"
    assert "Real user request to fix the bug" in prompt_sent
    assert "SYSTEM REJECTION" not in prompt_sent
    assert "SYSTEM ERROR" not in prompt_sent


def test_evaluate_diff_node_success_pass(mocker):
    """Green Path: Critic approves the changes."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="+new code")
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "PASS"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"workspace_absolute_path": "/tmp/test", "messages": []}
    result = evaluate_diff_node(state)

    assert result["latest_traceback_error"] is None


def test_evaluate_diff_node_fallback_critic_api_crash_handling(mocker):
    """
    Edge Path: Asserts that evaluate_diff_node gracefully fails open
    without crashing the thread if the validation LLM provider goes offline.
    """
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="+some_code")

    # Force the base tier client factory to return an execution engine that crashes
    mock_llm = mocker.Mock()
    mock_llm.invoke.side_effect = Exception("Ollama connection refused / out of memory")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"workspace_absolute_path": "/tmp/test", "messages": []}
    result = evaluate_diff_node(state)

    # Confirm it returns a clean state to let the workflow proceed as a fallback
    assert result == {"latest_traceback_error": None}


def test_evaluate_diff_node_fallback_empty_pass(mocker):
    """Edge Path: Critic approves an empty diff (e.g., file was already correct)."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="")
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "PASS"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"workspace_absolute_path": "/tmp/test", "messages": []}
    result = evaluate_diff_node(state)

    # Must flip the intent to read_only so it safely offramps
    assert result["latest_traceback_error"] is None
    assert result["intent_category"] == "workspace_read_only"


def test_evaluate_diff_node_fallback_escape_hatch(mocker):
    """Edge Path: Bypasses the trap if the agent legitimately used the escape hatch."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="")
    mock_llm = mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm")

    # Mock the AIMessage with the explicit tool call (Added the required 'id' field)
    mock_msg = AIMessage(
        content="",
        tool_calls=[{"name": "mark_task_already_completed", "args": {}, "id": "call_123"}],
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "intent_category": "workspace_operation",
        "messages": [mock_msg],
    }

    result = evaluate_diff_node(state)

    # It should immediately downgrade the intent to read_only and clear errors
    assert result["intent_category"] == "workspace_read_only"
    assert result["latest_traceback_error"] is None
    # The LLM critic should be completely bypassed to save API calls
    mock_llm.assert_not_called()


def test_evaluate_diff_node_fallback_fail_retry(mocker):
    """Edge Path: Critic rejects the changes, injecting feedback to loop back."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="+wrong code")
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "FAIL: You used the wrong variable name."
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
        "t1_base_calls": 0,
        "messages": [],
    }
    result = evaluate_diff_node(state)

    assert result["latest_traceback_error"] == "semantic_review_rejection"
    assert result["execution_retry_count"] == 1
    assert result["t1_base_calls"] == 1
    assert "SYSTEM REJECTION" in result["messages"][0].content
    assert "wrong variable name" in result["messages"][0].content


def test_evaluate_diff_node_fallback_git_error(mocker):
    """Edge Path: If git fails locally, pass it through for review_pr_node to handle."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_git_diff", side_effect=Exception("Git locked")
    )

    state = {"workspace_absolute_path": "/tmp/test"}
    result = evaluate_diff_node(state)

    assert result["latest_traceback_error"] is None


def test_evaluate_diff_node_fallback_hallucination_trap_triggered(mocker):
    """Edge Path: Traps an LLM hallucination when it outputs PASS but the diff is empty."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="")

    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = mocker.Mock(content="PASS")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # Agent thinks it did a workspace operation, but no tool calls were used
    state = {
        "workspace_absolute_path": "/tmp/test",
        "intent_category": "workspace_operation",
        "messages": [AIMessage(content="I have updated the file.")],
        "execution_retry_count": 0,
        "t1_base_calls": 0,
    }

    result = evaluate_diff_node(state)

    # Verify the trap caught the hallucination and incremented the retry counter
    assert result["latest_traceback_error"] == "hallucinated_success"
    assert result["execution_retry_count"] == 1
    assert "SYSTEM ERROR: No files were modified" in result["messages"][0].content


def test_evaluate_diff_node_fallback_revert_trap(mocker):
    """
    Edge Path: Verifies that if the cumulative diff is empty (e.g., a revert),
    but the incremental diff shows the agent's actual uncommitted work, the prompt
    correctly injects both states so the Critic doesn't falsely fail the agent.
    """

    def mock_git_diff(directory, target_branch=None):
        if target_branch:
            # Cumulative PR diff is empty because the revert exactly matches the base branch
            return ""
        # Incremental diff shows the physical uncommitted revert operation
        return "+features_enabled: true"

    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", side_effect=mock_git_diff)

    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "PASS"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"workspace_absolute_path": "/tmp/test", "messages": []}
    evaluate_diff_node(state)

    # Extract the actual prompt sent to the critic LLM
    prompt_sent = mock_llm.invoke.call_args[0][0]

    # Verify both diff contexts are accurately represented to the LLM
    assert "[NO CUMULATIVE CHANGES TO REPOSITORY]" in prompt_sent
    assert "+features_enabled: true" in prompt_sent


def test_evaluate_diff_node_error_fail_max_retries(mocker):
    """Red Path: Critic rejects the changes, but max retries have been hit."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="+wrong code")
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "FAIL: Still wrong."
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"workspace_absolute_path": "/tmp/test", "execution_retry_count": 3, "messages": []}
    result = evaluate_diff_node(state)

    assert result.get("is_aborted") is True
    assert "Execution Failed" in result["messages"][0].content
    assert "Still wrong" in result["messages"][0].content


def test_evaluate_diff_node_error_hallucination_trap_aborted(mocker):
    """Red Path: Aborts the workflow if the agent hallucinates too many times."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="")

    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = mocker.Mock(content="PASS")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "intent_category": "workspace_operation",
        "execution_retry_count": 3,  # Max retries hit
        "messages": [AIMessage(content="I have updated the file.")],
    }

    result = evaluate_diff_node(state)

    assert result.get("is_aborted") is True
    assert "Execution Failed" in result["messages"][0].content


# ==========================================
# Workflow: Agentic CI/CD Execution
# ==========================================


def test_agentic_ci_node_success_all_pass(mocker):
    """Green Path: Executes multiple suites successfully and reports back to GitHub."""
    # mock workspace config
    mock_suite_1 = mocker.Mock(name="Unit Tests", command="pytest", timeout_seconds=60)
    mock_suite_1.name = "Unit Tests"

    mock_suite_2 = mocker.Mock(name="E2E Tests", command="make e2e", timeout_seconds=120)
    mock_suite_2.name = "E2E Tests"

    mock_workspace = mocker.Mock()
    mock_workspace.path = "/tmp/test"
    mock_workspace.ci_suites = [mock_suite_1, mock_suite_2]

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.settings.workspaces", {"test_ws": mock_workspace}
    )

    # mock subprocess.run for both suites
    mock_process = mocker.Mock()
    mock_process.returncode = 0
    mock_process.stdout = "Test Passed"
    mock_process.stderr = ""
    mock_run = mocker.patch(
        "src.workspace_agent.orchestrator.nodes.subprocess.run", return_value=mock_process
    )

    # mock github API calls
    mock_set_status = mocker.patch("src.workspace_agent.orchestrator.nodes.set_commit_status")
    mock_comment = mocker.patch("src.workspace_agent.orchestrator.nodes.comment_on_pull_request")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "repo_full_name": "owner/repo",
        "commit_sha": "sha123",
        "pr_number": 42,
    }

    result = agentic_ci_node(state)

    # verify OS processes ran
    assert mock_run.call_count == 2

    # verify github pending & success statuses sent
    assert mock_set_status.call_count == 4  # 2 pending, 2 success

    # verify consolidated markdown comment sent
    mock_comment.assert_called_once()
    comment_body = mock_comment.call_args[1]["body"]
    assert "✅ Pass" in comment_body
    assert "Unit Tests" in comment_body
    assert "E2E Tests" in comment_body

    assert len(result["ci_results"]) == 2
    assert result["ci_results"][0]["passed"] is True


def test_agentic_ci_node_fallback_missing_context():
    """Edge Path: Ensure node bypasses execution if missing webhook context."""
    state = {
        "workspace_absolute_path": "/tmp/test",
        "commit_sha": None,  # Missing context explicitly
        "pr_number": 42,
    }

    result = agentic_ci_node(state)
    assert result == {}


def test_agentic_ci_node_error_timeout(mocker):
    """Red Path: Safely traps OS timeouts and reports them as failures back to the PR."""
    mock_suite = mocker.Mock(name="Slow Test", command="sleep 10", timeout_seconds=1)
    mock_suite.name = "Slow Test"

    mock_workspace = mocker.Mock()
    mock_workspace.path = "/tmp/test"
    mock_workspace.ci_suites = [mock_suite]

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.settings.workspaces", {"test_ws": mock_workspace}
    )

    # Raise TimeoutExpired to simulate a hanging execution
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="sleep 10", timeout=1, output="hanging..."),
    )

    mock_set_status = mocker.patch("src.workspace_agent.orchestrator.nodes.set_commit_status")
    mock_comment = mocker.patch("src.workspace_agent.orchestrator.nodes.comment_on_pull_request")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "repo_full_name": "owner/repo",
        "commit_sha": "sha123",
        "pr_number": 42,
    }

    result = agentic_ci_node(state)

    # Should report pending, then update to failure
    assert mock_set_status.call_count == 2
    final_status_call = mock_set_status.call_args_list[1]
    assert final_status_call[1]["state"] == "failure"
    assert final_status_call[1]["description"] == "Execution timed out"

    assert result["ci_results"][0]["passed"] is False
    assert "[ERROR: TimeoutExpired]" in result["ci_results"][0]["logs"]

    mock_comment.assert_called_once()
    assert "❌ Fail" in mock_comment.call_args[1]["body"]


# ==========================================
# Workflow: Automated Compilation Node
# ==========================================


def test_compile_node_success_standard(mocker):
    """Green Path: compile_node successfully builds the document and clears the queue."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Compilation Finished (Exit Code: 0)\n\nLogs...",
    )

    state = {
        "modified_tex_files": ["/tmp/doc.tex"],
        "execution_retry_count": 0,
    }
    result = compile_node(state)

    assert result["modified_tex_files"] == []
    assert result["latest_traceback_error"] is None


def test_compile_node_fallback_syntax_error(mocker):
    """
    Edge Path: compile_node catches a syntax error and returns a
    system rejection to force the LLM to self-heal.
    """
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Compilation Finished (Exit Code: 1)\n\nRunaway argument?",
    )

    state = {
        "modified_tex_files": ["/tmp/broken.tex"],
        "execution_retry_count": 0,
    }
    result = compile_node(state)

    assert "SYSTEM REJECTION" in result["messages"][0].content
    assert "Runaway argument?" in result["messages"][0].content
    assert result["latest_traceback_error"] == "latex_compilation_error"
    assert result["execution_retry_count"] == 1


def test_compile_node_error_max_retries(mocker):
    """
    Red Path: compile_node aborts the workflow if the LLM cannot fix
    the LaTeX syntax after the maximum allowed retries.
    """
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execute_tool_call",
        return_value="Compilation Finished (Exit Code: 1)\n\nFatal error",
    )

    # simulate hitting the max retry limit
    state = {
        "modified_tex_files": ["/tmp/broken.tex"],
        "execution_retry_count": 3,
    }
    result = compile_node(state)

    assert result.get("is_aborted") is True
    assert "Compilation Failed" in result["messages"][0].content
    assert result["execution_retry_count"] == 4


# ==========================================
# Workflow: PR & Git Review Node
# ==========================================


def test_review_pr_node_success_new_pr(mocker):
    """
    Green Path: If the agent has no pending PR, it successfully opens a new one after
    committing the changes to a newly generated branch.
    """
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "success"}',
    )
    # mock environment and diff snippets
    mocker.patch("src.workspace_agent.orchestrator.nodes.os.getenv", return_value="test_owner")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="diff snippet")
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_git_diff_blueprint",
        return_value="A\tnew_file.py",
    )
    # mock LLM for PR metadata generation
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)
    # mock the PR creation tool explicitly
    mock_open_pr = mocker.patch(
        "src.workspace_agent.orchestrator.nodes.open_pull_request",
        return_value='{"status": "success", "pr_url": "https://github.com/owner/repo/pull/456"}',
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "original_instruction": "add new feature",
        "pending_pr_url": None,  # Explicitly None to trigger new PR logic
    }
    result = review_pr_node(state)

    # Verify workflow succeeded
    assert result.get("is_aborted", False) is False
    assert "Execution Complete" in result["messages"][0].content
    assert result.get("pending_pr_url") == "https://github.com/owner/repo/pull/456"

    # Verify the PR tool was called with the correct keyword parameters
    mock_open_pr.assert_called_once()
    _, kwargs = mock_open_pr.call_args
    assert kwargs["directory"] == "/tmp/test"
    # The branch name is dynamically generated using the LLM summary + UUID
    assert kwargs["head_branch"].startswith("agent/mocked-summary-")
    assert "mocked summary" in kwargs["title"]


def test_review_pr_node_success_refining_existing_pr(mocker):
    """
    Green Path: If the agent is refining an existing PR, it should bypass PR generation,
    push directly to the active branch, and patch the existing PR's title.
    """
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "success", "branch": "agent/test-branch"}',
    )

    # mock the new environment, diff snippet, and PR update dependencies
    mocker.patch("src.workspace_agent.orchestrator.nodes.os.getenv", return_value="test_owner")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_git_diff", return_value="diff snippet")
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_git_diff_blueprint", return_value="M\tREADME.md"
    )
    mock_update = mocker.patch(
        "src.workspace_agent.orchestrator.nodes.update_pull_request",
        return_value='{"status": "success"}',
    )

    # mock LLM to avoid calling Ollama during unit tests
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # spy on the PR creation tool to ensure it is NEVER called
    mock_open_pr = mocker.patch("src.workspace_agent.orchestrator.nodes.open_pull_request")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "active_agent_branch": "agent/test-branch",
        "pending_pr_url": "https://github.com/owner/repo/pull/123",
        "original_instruction": "fix the typo",
        "messages": [HumanMessage(content="actually, do something else")],
    }

    result = review_pr_node(state)

    # verify workflow succeeded
    assert result.get("is_aborted", False) is False
    assert "Revisions Applied" in result["messages"][0].content

    # verify the PR title update status message was injected
    assert "updated the PR title" in result["messages"][0].content

    # verify it bypassed opening a new PR
    mock_open_pr.assert_not_called()

    # verify it successfully extracted the PR number and sent the patch request
    mock_update.assert_called_once()
    args, kwargs = mock_update.call_args
    assert kwargs["pr_number"] == 123
    assert kwargs["directory"] == "/tmp/test"


def test_review_pr_node_fallback_diff_extraction_failure(mocker):
    """Edge Path: review_pr_node proceeds with defaults if git diff extraction fails locally."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.os.getenv", return_value="test_owner")

    # Force git diff functions to fail (e.g., corrupted local index)
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_git_diff", side_effect=Exception("Git locked")
    )

    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "success", "branch": "agent/test-branch"}',
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.open_pull_request",
        return_value='{"status": "success", "pr_url": "https://github.com/mock/pr/1"}',
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "original_instruction": "add new feature",
    }

    result = review_pr_node(state)

    # Verify the workflow did not abort and successfully completed the PR generation
    assert result.get("is_aborted", False) is False
    assert result.get("pending_pr_url") == "https://github.com/mock/pr/1"


def test_review_pr_node_fallback_self_healing_trap(mocker):
    """
    Edge Path: review_pr_node intercepts the `no_changes_to_commit` error,
    increments the retry counter, and loops back with a SYSTEM REJECTION.
    """
    # mock Git execution to return the specific no_changes error
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "error", "reason": "no_changes_to_commit"}',
    )

    # mock LLM to avoid calling Ollama during tests
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
    }
    result = review_pr_node(state)

    # workflow should not abort; it should loop back to the execution node
    assert result.get("is_aborted", False) is False
    assert result.get("latest_traceback_error") == "no_changes_to_commit"
    assert result.get("execution_retry_count") == 1

    # verify the LLM is explicitly scolded
    assert "SYSTEM REJECTION" in result["messages"][0].content
    assert "NO CHANGES" in result["messages"][0].content


def test_review_pr_node_error_generic_commit_failure(mocker):
    """Red Path: A generic Git commit failure aborts the workflow safely."""
    mocker.patch("src.workspace_agent.orchestrator.nodes.os.getenv", return_value="test_user")

    # mock LLM generation metadata
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # simulate a severe git failure that is NOT 'no_changes_to_commit'
    mock_error_json = (
        '{"status": "error", "reason": "git_command_failed", "details": "Merge conflict"}'
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value=mock_error_json,
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "original_instruction": "add new feature",
    }

    result = review_pr_node(state)

    # verify it aborts instead of looping
    assert result.get("is_aborted") is True
    assert "Commit & Push Failed" in result["messages"][0].content
    assert "Merge conflict" in result["messages"][0].content


def test_review_pr_node_error_github_api_rejection(mocker):
    """
    Red Path: If pushing the branch succeeds but the GitHub API rejects the PR creation,
    the workflow must safely abort.
    """
    mocker.patch("src.workspace_agent.orchestrator.nodes.os.getenv", return_value="test_user")

    # mock LLM generation metadata
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "success", "branch": "agent/new-branch"}',
    )

    # mock GitHub API returning an error, split across lines to satisfy Ruff E501
    mock_error_json = (
        '{"status": "error", "reason": "validation_failed", "details": "Branch protected"}'
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.open_pull_request",
        return_value=mock_error_json,
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "original_instruction": "add new feature",
    }

    result = review_pr_node(state)

    assert result.get("is_aborted") is True
    assert "PR Creation Failed" in result["messages"][0].content
    assert "Branch protected" in result["messages"][0].content


def test_review_pr_node_error_self_healing_trap_max_retries(mocker):
    """Red Path: review_pr_node aborts if no_changes_to_commit hits the retry limit."""
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.create_branch_and_commit",
        return_value='{"status": "error", "reason": "no_changes_to_commit"}',
    )

    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value.content = "mocked summary"
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 3,
    }
    result = review_pr_node(state)

    assert result.get("is_aborted") is True
    assert "Execution Failed" in result["messages"][0].content


# ==========================================
# Workflow: Terminal & Cleanup Nodes
# ==========================================


def test_conversational_reply_node_success_standard(mocker):
    """Green Path: Conversational node successfully invokes the LLM without tools."""
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = AIMessage(content="Hello there!")
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    state = {"messages": [HumanMessage(content="Hi")], "t1_base_calls": 0}
    result = conversational_reply_node(state)

    assert result["messages"][0].content == "Hello there!"
    assert result["t1_base_calls"] == 1


def test_pr_merged_node_success_standard(mocker):
    """Green Path: pr_merged_node successfully cleans up branches and formats response."""
    mock_cleanup = mocker.patch("src.workspace_agent.orchestrator.nodes.cleanup_local_branch")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "target_branch": "main",
        "active_agent_branch": "agent/test-branch",
    }
    result = pr_merged_node(state)

    assert "Pull Request Merged!" in result["messages"][0].content
    assert result["active_agent_branch"] is None
    assert result["human_approved"] is False
    assert result["pending_pr_url"] is None

    # verify the local cleanup helper was called correctly
    mock_cleanup.assert_called_once_with("/tmp/test", "main", "agent/test-branch")


def test_update_memory_node_success_message_stripping(mocker):
    """Green Path: Ensures update_memory_node strips system rejections and tool calls."""
    mock_llm = mocker.Mock()
    mock_extractor = mocker.Mock()

    # Mock a generic extraction
    mock_extraction = mocker.Mock(
        core_interests=[], active_projects={}, preferences=[], operational_insights=[]
    )
    mock_extractor.invoke.return_value = mock_extraction
    mock_llm.with_structured_output.return_value = mock_extractor

    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)
    mocker.patch("src.workspace_agent.orchestrator.nodes.open", mocker.mock_open())
    mocker.patch("src.workspace_agent.orchestrator.nodes.save_memory")

    store = InMemoryStore()

    # Validation update: Inject the mock collection
    config = {"configurable": {"user_id": "test_user", "chroma_collection": mocker.Mock()}}

    state = {
        "messages": [
            HumanMessage(content="Do task"),
            AIMessage(content="", tool_calls=[{"name": "test", "id": "1", "args": {}}]),
            ToolMessage(content="done", tool_call_id="1", name="test"),
            HumanMessage(content="SYSTEM REJECTION: Please fix the code"),
            HumanMessage(content="SYSTEM ERROR: API Timeout"),
            HumanMessage(content="Normal followup"),
        ],
        "t1_base_calls": 0,
    }

    result = update_memory_node(state, config, store)
    messages_update = result["messages"]

    # Isolate the RemoveMessage objects
    removals = [m for m in messages_update if isinstance(m, RemoveMessage)]

    # Should flag: 1 AIMessage (tool call), 1 ToolMessage, 2 System Rejection/Error HumanMessages
    assert len(removals) == 4


def test_update_memory_node_success_standard(mocker):
    """
    Green Path: Memory extraction successfully writes a structured profile to LangGraph Store and
    VectorDB.
    """
    mock_llm = mocker.Mock()
    mock_extractor = mocker.Mock()

    # mock the Pydantic structured output attributes explicitly
    mock_extraction = mocker.Mock()
    mock_extraction.core_interests = ["AI Engineering"]
    mock_extraction.active_projects = {}
    mock_extraction.preferences = []
    mock_extraction.operational_insights = ["Discovered a fix for YAML quotes."]

    mock_extractor.invoke.return_value = mock_extraction
    mock_llm.with_structured_output.return_value = mock_extractor

    mocker.patch("src.workspace_agent.orchestrator.nodes.open", mocker.mock_open())
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)
    mock_save_memory = mocker.patch("src.workspace_agent.orchestrator.nodes.save_memory")

    store = InMemoryStore()

    # Validation update: Inject the mock collection so `save_memory` actually executes
    mock_chroma = mocker.Mock()
    config = {
        "configurable": {
            "user_id": "test_user",
            "thread_id": "test_thread",
            "chroma_collection": mock_chroma,
        }
    }

    state = {"messages": [HumanMessage(content="I like AI")], "t1_base_calls": 0}
    result = update_memory_node(state, config, store)

    # verify it returns the incremented call count and DOES NOT return user_profile to state
    assert result["t1_base_calls"] == 1
    assert "user_profile" not in result

    # 1. Verify the Entity memory was stored properly in LangGraph Store API
    profile = store.get(("user_profile", "test_user"), "profile")
    assert profile.value["core_interests"] == ["AI Engineering"]

    # 2. Verify the Episodic memory was extracted and sent to ChromaDB
    mock_save_memory.assert_called_once_with(
        text="Discovered a fix for YAML quotes.",
        thread_id="test_thread",
        collection=mock_chroma,
        metadata={"type": "operational_insight"},
    )


def test_update_memory_node_fallback_disk_backup_fails(mocker, capsys):
    """
    Edge Path: If writing the local backup file fails, it should still update the
    LangGraph store.
    """
    mock_llm = mocker.Mock()
    mock_extractor = mocker.Mock()

    # Mock a successful extraction
    mock_extraction = mocker.Mock()
    mock_extraction.core_interests = ["Docker"]
    mock_extraction.active_projects = {}
    mock_extraction.preferences = []
    mock_extraction.operational_insights = []

    mock_extractor.invoke.return_value = mock_extraction
    mock_llm.with_structured_output.return_value = mock_extractor
    mocker.patch("src.workspace_agent.orchestrator.nodes.get_execution_llm", return_value=mock_llm)

    # Force the local file write to throw a Permission Error
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.open", side_effect=PermissionError("Access denied")
    )

    store = InMemoryStore()

    # Validation update: Inject the mock collection
    config = {"configurable": {"user_id": "test_user", "chroma_collection": mocker.Mock()}}

    state = {"messages": [], "t1_base_calls": 0}
    result = update_memory_node(state, config, store)

    # Verify the node still succeeded and successfully released the busy lock
    assert result["is_busy"] is False

    # Verify the permission error was trapped and printed to stdout
    captured = capsys.readouterr()
    assert "Failed to backup profile to disk" in captured.out

    # Verify the LangGraph store was still updated despite the disk failure
    profile = store.get(("user_profile", "test_user"), "profile")
    assert profile.value["core_interests"] == ["Docker"]


def test_update_memory_node_fallback_llm_unavailable(mocker):
    """Edge Path: If the base tier LLM is offline, safely return without crashing."""
    # Simulate the router failing to find an active Base Tier LLM
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.get_execution_llm",
        side_effect=TerminalEscalationError("No models available"),
    )

    state = {"messages": [], "t1_base_calls": 0}
    result = update_memory_node(state)

    # Should gracefully return an empty dict, skipping memory extraction entirely
    assert result == {}


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
        "src.workspace_agent.orchestrator.nodes.cleanup_local_branch",
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
