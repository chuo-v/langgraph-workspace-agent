from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.store.memory import InMemoryStore

from src.workspace_agent.core.config import settings
from src.workspace_agent.orchestrator.nodes.execution import (
    _build_cross_workspace_prompt,
    _extract_modified_tex_files,
    _filter_execution_context,
    _is_tool_error,
    _resolve_execution_tier,
    _sanitize_llm_response,
    execute_task_node,
    update_memory_node,
    workspace_tools_node,
)
from src.workspace_agent.orchestrator.router import (
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
)

# ==========================================
# Component: _resolve_execution_tier
# ==========================================


def test_resolve_execution_tier_success_priorities(mocker):
    """Green Path: Verifies that tier overrides are correctly prioritized."""
    # 1. Setup Mock Environment
    mock_tier = mocker.patch("src.workspace_agent.orchestrator.nodes.execution.get_tier_for_model")

    # 2. Execute
    # Test Explicit Model Override (Highest Priority)
    mock_tier.return_value = TIER_STANDARD
    result_explicit = _resolve_execution_tier(
        {
            "requested_model": "deepseek_fast",
            "force_frontier_tier": True,  # Explicit model should beat the frontier flag
        }
    )

    # Reset mock for default flag checks
    mock_tier.return_value = None

    # Test Frontier Override
    result_frontier = _resolve_execution_tier(
        {"requested_model": None, "force_frontier_tier": True, "force_standard_tier": False}
    )

    # Test Standard Override
    result_standard = _resolve_execution_tier(
        {"requested_model": None, "force_frontier_tier": False, "force_standard_tier": True}
    )

    # Test Conflict (Frontier must win)
    result_conflict = _resolve_execution_tier(
        {"requested_model": None, "force_frontier_tier": True, "force_standard_tier": True}
    )

    # 3. Assertions
    assert result_explicit == TIER_STANDARD
    assert result_frontier == TIER_FRONTIER
    assert result_standard == TIER_STANDARD
    assert result_conflict == TIER_FRONTIER


# ==========================================
# Component: _build_cross_workspace_prompt
# ==========================================


def test_build_cross_workspace_prompt_success_operation():
    """Green Path: Verifies that standard operations receive the strict write constraint."""
    # 1. Setup Mock Environment
    state = {"intent_category": "workspace_operation"}

    # 2. Execute
    prompt_msg = _build_cross_workspace_prompt(state, "test_arena", "/tmp/path")

    # 3. Assertions
    # Should not have the read-only flag
    assert "TASK TYPE: READ ONLY" not in prompt_msg.content

    # Should contain the new write operation guardrails
    assert "TASK TYPE: WRITE OPERATION" in prompt_msg.content
    assert "physically accomplish the task" in prompt_msg.content


def test_build_cross_workspace_prompt_success_read_only():
    """Green Path: Verifies that the read-only constraint is forcefully appended."""
    # 1. Setup Mock Environment
    state = {"intent_category": "workspace_read_only"}

    # 2. Execute
    prompt_msg = _build_cross_workspace_prompt(state, "test_arena", "/tmp/path")

    # 3. Assertions
    assert "TASK TYPE: READ ONLY" in prompt_msg.content
    assert "Do NOT attempt to write files" in prompt_msg.content


# ==========================================
# Component: _sanitize_llm_response
# ==========================================


def test_sanitize_llm_response_success_empty_string():
    """Green Path: Preserves empty string content without injecting hallucination risks."""
    # 1. Setup Mock Environment
    msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "call_123"}])

    # 2. Execute
    _sanitize_llm_response(msg)

    # 3. Assertions
    assert msg.content == ""


def test_sanitize_llm_response_success_list_format_empty():
    """Green Path: Cleans out empty text blocks hidden inside structural lists."""
    # 1. Setup Mock Environment
    msg = AIMessage(
        content=[{"type": "text", "text": " "}, {"type": "text", "text": ""}],
        tool_calls=[{"name": "test", "args": {}, "id": "call_123"}],
    )

    # 2. Execute
    _sanitize_llm_response(msg)

    # 3. Assertions
    assert msg.content == ""


def test_sanitize_llm_response_success_list_format_valid():
    """Green Path: Preserves valid text blocks from lists and flattens them to a string."""
    # 1. Setup Mock Environment
    msg = AIMessage(
        content=[{"type": "text", "text": "Valid Text"}, {"type": "text", "text": " "}],
        tool_calls=[{"name": "test", "args": {}, "id": "call_123"}],
    )

    # 2. Execute
    _sanitize_llm_response(msg)

    # 3. Assertions
    assert isinstance(msg.content, str)
    assert msg.content == "Valid Text"


# ==========================================
# Component: _extract_modified_tex_files
# ==========================================


def test_extract_modified_tex_files_success_extraction():
    """Green Path: Validates that LaTeX file modifications are successfully trapped."""
    # 1. Setup Mock Environment
    # Set up a fake LLM response containing tool calls to modify a .tex file and a .py file
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/src/report.tex"}, "id": "1"},
            {"name": "search_and_replace", "args": {"file_path": "/src/main.py"}, "id": "2"},
            {"name": "replace_text_block", "args": {"file_path": "/src/thesis.TEX"}, "id": "3"},
        ],
    )

    # 2. Execute
    new_files = _extract_modified_tex_files(["/existing.tex"], mock_response)

    # 3. Assertions
    # The .tex files should be added, the .py file ignored, and existing preserved
    assert "/existing.tex" in new_files
    assert "/src/report.tex" in new_files
    assert "/src/thesis.TEX" in new_files
    assert "/src/main.py" not in new_files


def test_extract_modified_tex_files_success_deletion():
    """Green Path: Validates that deleted LaTeX files are removed from the compilation queue."""
    # 1. Setup Mock Environment
    mock_response = AIMessage(
        content="",
        tool_calls=[
            {"name": "delete_file", "args": {"file_path": "/src/report.tex"}, "id": "1"},
        ],
    )

    # 2. Execute
    new_files = _extract_modified_tex_files(["/src/report.tex", "/src/keep.tex"], mock_response)

    # 3. Assertions
    assert "/src/report.tex" not in new_files
    assert "/src/keep.tex" in new_files


def test_extract_modified_tex_files_success_rename():
    """Green Path: Validates that renamed LaTeX files update the queue correctly."""
    # 1. Setup Mock Environment
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

    # 2. Execute
    new_files = _extract_modified_tex_files(["/src/old.tex", "/src/other.tex"], mock_response)

    # 3. Assertions
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
    # 1. Setup Mock Environment
    messages = [
        AIMessage(content="I claimed I did work here but used no tools."),  # should be replaced
        HumanMessage(content="Do another task"),  # should be kept
        AIMessage(content="I am the last message without tools."),  # should be kept natively
    ]

    # 2. Execute
    filtered = _filter_execution_context(messages)

    # 3. Assertions
    assert len(filtered) == 3
    assert "System Note" in filtered[0].content
    assert "Do another task" in filtered[1].content
    assert "I am the last message" in filtered[2].content


def test_filter_execution_context_success_replaces_markers():
    """Green Path: Ensures orchestrator success markers are replaced with boundaries."""
    # 1. Setup Mock Environment
    messages = [
        HumanMessage(content="Do a task"),
        AIMessage(content="✅ **Execution Complete**"),
        AIMessage(content="✅ **Revisions Applied**"),
        AIMessage(
            content="Normal AI Message", tool_calls=[{"name": "test", "args": {}, "id": "call_123"}]
        ),
    ]

    # 2. Execute
    filtered = _filter_execution_context(messages)

    # 3. Assertions
    assert len(filtered) == 4
    assert "System Note" in filtered[1].content
    assert "System Note" in filtered[2].content
    assert "Normal AI Message" in filtered[3].content


def test_filter_execution_context_fallback_handles_list_content():
    """
    Edge Path: Verifies that context filtering clears out orchestration success markers
    even when they are wrapped in structured lists (Gemini provider payload style).
    """
    # 1. Setup Mock Environment
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

    # 2. Execute
    filtered = _filter_execution_context(messages)

    # 3. Assertions
    assert len(filtered) == 3
    assert "System Note" in filtered[1].content
    assert "Normal response text" in filtered[2].content


# ==========================================
# Component: _is_tool_error
# ==========================================


def test_is_tool_error_success_identifies_errors():
    """Green Path: Explicitly tests the internal string matching bounds of _is_tool_error."""
    # 1. Setup Mock Environment
    # (No external mocking required)

    # 2. Execute
    # Hard Invocation Failures
    res_write = _is_tool_error("write_file", "Tool execution failed: No access")
    # Sandbox Execution tools
    res_py_err = _is_tool_error("run_python_script", "Traceback (most recent call last):")
    res_tex_err = _is_tool_error("compile_latex_document", "=== ERROR: Runaway argument")
    res_pytest_err = _is_tool_error("run_pytest", "FAILED (failures=1)")
    # Standard Read/Write OS tools
    res_read_err = _is_tool_error("read_files", "Error: File not found")
    # Non-errors should pass cleanly
    res_py_ok = _is_tool_error("run_python_script", "Execution Finished (Exit Code: 0)")
    res_read_ok = _is_tool_error("read_files", "File content here.")

    # 3. Assertions
    assert res_write is True
    assert res_py_err is True
    assert res_tex_err is True
    assert res_pytest_err is True
    assert res_read_err is True
    assert res_py_ok is False
    assert res_read_ok is False


# ==========================================
# Workflow: Workspace Tools Node
# ==========================================


def test_workspace_tools_node_success_config_injection(mocker):
    """
    Green Path: Ensures the RunnableConfig is successfully propagated down
    from the graph layer to the execution registry (Dependency Injection validation).
    """
    # 1. Setup Mock Environment
    tool_calls = [{"name": "run_python_script", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    # Mock the injected config containing our external clients
    config = {"configurable": {"docker_client": "mock_client"}}

    mock_executor = mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call", return_value="Success"
    )

    # 2. Execute
    workspace_tools_node(state, config)

    # 3. Assertions
    mock_executor.assert_called_once()

    # Extract arguments and verify the config was explicitly passed to the registry
    args, kwargs = mock_executor.call_args
    assert args[0] == state["messages"][0].tool_calls[0]
    assert args[1] == config


def test_workspace_tools_node_success_manual_pause(mocker):
    """
    Green Path: Ensures workspace_tools_node halts mid-execution if an abort signal is present.
    """
    # 1. Setup Mock Environment
    tool_calls = [
        {"name": "read_files", "args": {}, "id": "1"},
        {"name": "write_file", "args": {}, "id": "2"},
    ]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
    config = {"configurable": {"thread_id": "thread_pause_123"}}
    store = InMemoryStore()

    # Inject the pause signal
    store.put(("abort_signals", "thread_pause_123"), "abort", {"stop_type": "pause"})

    # Mock physical execution so it doesn't crash without real files
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call", return_value="Success"
    )

    # 2. Execute
    result = workspace_tools_node(state, config, store)

    # 3. Assertions
    new_messages = result["messages"]

    # It should iterate through the loop, inject the pause error for BOTH tools, and append the
    # AI pause message
    assert len(new_messages) == 3
    assert "Error: Execution manually paused" in new_messages[0].content
    assert "Error: Execution manually paused" in new_messages[1].content
    assert "🛑 **Execution Paused:**" in new_messages[2].content

    assert "To abandon this task & delete the branch" in result["clarification_question"]

    # Verify the flag was properly cleared
    assert store.get(("abort_signals", "thread_pause_123"), "abort") is None


def test_workspace_tools_node_fallback_clarification_trap():
    """Edge Path: Ensures the tool loop intercepts human-in-the-loop requests without breaking."""
    # 1. Setup Mock Environment
    tool_calls = [
        {
            "name": "ask_user_for_clarification",
            "args": {"question": "Are you sure?"},
            "id": "call_123",
        }
    ]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    # 2. Execute
    # Clarification traps bypass physical execution, so no mocker.patch is needed
    result = workspace_tools_node(state)

    # 3. Assertions
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
    # 1. Setup Mock Environment
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
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call",
        side_effect=mock_tool_execution,
    )

    # 2. Execute
    result = workspace_tools_node(state)

    # 3. Assertions
    # The SyntaxError should be entirely scrubbed, and retries preserved but NOT incremented
    assert result["latest_traceback_error"] is None
    assert result["execution_retry_count"] == 1


def test_workspace_tools_node_fallback_ignores_read_traceback(mocker):
    """Edge Path: Ensures read tools ignore Tracebacks in file contents."""
    # 1. Setup Mock Environment
    tool_calls = [{"name": "read_files", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call",
        return_value="Traceback (most recent call last): False positive from reading a log file",
    )

    # 2. Execute
    result = workspace_tools_node(state)

    # 3. Assertions
    assert result["execution_retry_count"] == 0
    assert result["latest_traceback_error"] is None


def test_workspace_tools_node_fallback_pytest_capture(mocker):
    """Edge Path: Ensures pytest failure signatures correctly increment the retry counter."""
    # 1. Setup Mock Environment
    tool_calls = [{"name": "run_pytest", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call",
        return_value="FAILED (failures=1)\nAssertionError: 2 != 3",
    )

    # 2. Execute
    result = workspace_tools_node(state)

    # 3. Assertions
    assert result["execution_retry_count"] == 1
    assert "FAILED (" in result["latest_traceback_error"]


def test_workspace_tools_node_fallback_traceback_capture(mocker):
    """Edge Path: Ensures sandbox script crashes increment the retry counter."""
    # 1. Setup Mock Environment
    tool_calls = [{"name": "run_python_script", "args": {}, "id": "call_123"}]
    state = {"messages": [AIMessage(content="", tool_calls=tool_calls)], "execution_retry_count": 0}

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call",
        return_value="Traceback (most recent call last): File missing",
    )

    # 2. Execute
    result = workspace_tools_node(state)

    # 3. Assertions
    assert result["execution_retry_count"] == 1
    assert "Traceback" in result["latest_traceback_error"]


def test_workspace_tools_node_error_sandbox_crash_max_retries(mocker):
    """Red Path: workspace_tools_node aborts if sandbox crashes hit the retry limit."""
    # 1. Setup Mock Environment
    tool_calls = [{"name": "run_python_script", "id": "1", "args": {}}]
    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 3,
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
    }

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.execute_tool_call",
        return_value="Traceback (most recent call last): error",
    )

    # 2. Execute
    result = workspace_tools_node(state)

    # 3. Assertions
    assert result.get("is_aborted") is True
    assert result.get("latest_traceback_error") is None
    assert "Execution Failed" in result["messages"][-1].content


# ==========================================
# Workflow: Execute Task Node
# ==========================================


def test_execute_task_node_success_standard(mocker):
    """Green Path: Successfully invokes the execution LLM API without aborting."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.sync_repository",
        return_value='{"status": "success"}',
    )
    # Mock the LLM to return a standard AIMessage with a tool call
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(
        content="I will do the task now.",
        tool_calls=[{"name": "read_files", "args": {}, "id": "call_123"}],
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
        "messages": [HumanMessage(content="Do the task")],
    }

    # 2. Execute
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # 3. Assertions
    # Verify the workflow did not trigger circuit breakers, abort, or increment retries
    assert result.get("is_aborted", False) is False
    assert result.get("latest_traceback_error") is None
    assert result.get("execution_retry_count", 0) == 0

    # Verify the AI message was properly appended to state
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "I will do the task now."


def test_execute_task_node_success_manual_pause():
    """Green Path: Ensures execute_task_node halts immediately if an abort signal is present."""
    # 1. Setup Mock Environment
    state = {"messages": [HumanMessage(content="do a task")]}
    config = {"configurable": {"thread_id": "thread_pause_123"}}
    store = InMemoryStore()

    # Inject the pause signal into the database before execution
    store.put(("abort_signals", "thread_pause_123"), "abort", {"stop_type": "pause"})

    # 2. Execute
    result = execute_task_node(state, config, store)

    # 3. Assertions
    assert "🛑 **Execution Paused:**" in result["messages"][0].content
    assert "To abandon this task & delete the branch" in result["clarification_question"]
    assert result.get("latest_traceback_error") is None

    # Verify the flag was properly cleared from memory
    assert store.get(("abort_signals", "thread_pause_123"), "abort") is None


def test_execute_task_node_fallback_api_invocation_crash(mocker):
    """Edge Path: execute_task_node catches LLM API crash and returns a SYSTEM ERROR."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.sync_repository",
        return_value='{"status": "success"}',
    )

    # mock LLM to throw an exception directly on invoke to simulate a Rate Limit or Timeout
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.side_effect = Exception("Rate Limit Exceeded")
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )

    # Mock time.sleep to not actually delay the test suite
    mocker.patch("src.workspace_agent.orchestrator.nodes.execution.time.sleep")

    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 0,
        "messages": [],
    }

    # 2. Execute
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # 3. Assertions
    # Ensure the workflow did not abort, but instead injected the self-healing prompt
    assert result.get("is_aborted", False) is False
    assert "SYSTEM ERROR: The LLM API failed" in result["messages"][0].content
    assert result["latest_traceback_error"] == "api_invocation_error"
    assert result["execution_retry_count"] == 1


def test_execute_task_node_error_api_invocation_crash_max_retries(mocker):
    """Red Path: execute_task_node aborts if API crashes hit the retry limit."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.sync_repository",
        return_value='{"status": "success"}',
    )

    # Mock LLM to throw an exception directly
    mock_llm = mocker.Mock()
    mock_llm.bind_tools.return_value.invoke.side_effect = Exception("Rate Limit Exceeded")
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )

    # Force state to be at the maximum retry limit (default is 3)
    state = {
        "workspace_absolute_path": "/tmp/test",
        "execution_retry_count": 3,
        "messages": [],
    }

    # 2. Execute
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # 3. Assertions
    assert result.get("is_aborted") is True
    assert "Fatal API or Parsing Error" in result["messages"][0].content


def test_execute_task_node_error_circuit_breaker():
    """
    Red Path: Verifies that if the agent executes max_consecutive_tool_steps
    without human interaction, the circuit breaker safely aborts the workflow.
    """
    # 1. Setup Mock Environment
    messages = [HumanMessage(content="Start loop")]

    limit = settings.agent.max_consecutive_tool_steps

    # Generate enough consecutive non-human messages to trigger the breaker
    for i in range(limit + 1):
        messages.append(AIMessage(content=f"Thinking {i}"))

    state = {"workspace_absolute_path": "/tmp", "messages": messages}

    # 2. Execute
    result = execute_task_node(state)

    # 3. Assertions
    assert result.get("is_aborted") is True
    assert "Circuit breaker triggered" in result["messages"][0].content


def test_execute_task_node_error_escalation_crash(mocker):
    """Red Path: execute_task_node catches TerminalEscalationError from the router."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.sync_repository",
        return_value='{"status": "success"}',
    )

    # force the router to fail escalating to Tier 3
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        side_effect=TerminalEscalationError("API Offline"),
    )

    state = {
        "force_frontier_tier": True,
        "workspace_absolute_path": "/tmp/test",
        "messages": [],
    }

    # 2. Execute
    result = execute_task_node(state, {"configurable": {"thread_id": "123"}})

    # 3. Assertions
    assert result.get("is_aborted") is True
    assert "Escalation Failed" in result["messages"][0].content


def test_execute_task_node_error_sync_abort(mocker):
    """Red Path: Ensure execute_task_node aborts immediately if repository sync fails."""
    # 1. Setup Mock Environment
    # Mock the sync repository to return a realistic error status
    mock_error_json = (
        '{"status": "error", "reason": "git_command_failed", '
        '"details": "merge conflict in README.md"}'
    )

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.sync_repository",
        return_value=mock_error_json,
    )

    state = {
        "inferred_workspace": "test_arena",
        "workspace_absolute_path": "/tmp/test",
        "target_branch": "main",
        "messages": [],
    }

    # 2. Execute
    result = execute_task_node(state)

    # 3. Assertions
    # Verify the node successfully trapped the sync failure and aborted
    assert result.get("is_aborted") is True
    assert "Sync Failed" in result["messages"][0].content
    assert "merge conflict in README.md" in result["messages"][0].content


# ==========================================
# Workflow: Memory Extraction Node
# ==========================================


def test_update_memory_node_success_message_stripping(mocker):
    """Green Path: Ensures update_memory_node strips system rejections and tool calls."""
    # 1. Setup Mock Environment
    mock_llm = mocker.Mock()
    mock_extractor = mocker.Mock()

    # Mock a generic extraction
    mock_extraction = mocker.Mock(
        core_interests=[], active_projects={}, preferences=[], operational_insights=[]
    )
    mock_extractor.invoke.return_value = mock_extraction
    mock_llm.with_structured_output.return_value = mock_extractor

    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )
    mocker.patch("src.workspace_agent.orchestrator.nodes.execution.open", mocker.mock_open())
    mocker.patch("src.workspace_agent.orchestrator.nodes.execution.save_memory")

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

    # 2. Execute
    result = update_memory_node(state, config, store)

    # 3. Assertions
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
    # 1. Setup Mock Environment
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

    mocker.patch("src.workspace_agent.orchestrator.nodes.execution.open", mocker.mock_open())
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )
    mock_save_memory = mocker.patch("src.workspace_agent.orchestrator.nodes.execution.save_memory")

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

    # 2. Execute
    result = update_memory_node(state, config, store)

    # 3. Assertions
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
    # 1. Setup Mock Environment
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
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        return_value=[mock_llm],
    )

    # Force the local file write to throw a Permission Error
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.open",
        side_effect=PermissionError("Access denied"),
    )

    store = InMemoryStore()

    # Validation update: Inject the mock collection
    config = {"configurable": {"user_id": "test_user", "chroma_collection": mocker.Mock()}}

    state = {"messages": [], "t1_base_calls": 0}

    # 2. Execute
    result = update_memory_node(state, config, store)

    # 3. Assertions
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
    # 1. Setup Mock Environment
    # Simulate the router failing to find an active Base Tier LLM
    mocker.patch(
        "src.workspace_agent.orchestrator.nodes.execution.get_execution_llm_sequence",
        side_effect=TerminalEscalationError("No models available"),
    )

    state = {"messages": [], "t1_base_calls": 0}

    # 2. Execute
    result = update_memory_node(state)

    # 3. Assertions
    # Should gracefully return an empty dict, skipping memory extraction entirely
    assert result == {}
