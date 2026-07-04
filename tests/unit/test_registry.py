from src.workspace_agent.tools.registry import (
    agent_tools,
    ask_user_for_clarification,
    execute_tool_call,
    mark_task_already_completed,
)

# ==========================================
# Workflow: Tool Registry Initialization
# ==========================================


def test_agent_tools_success_registration():
    """
    Green Path: Validates that all critical tools, including new GitHub integrations, are
    registered.
    """
    # 1. Setup Mock Environment
    expected_github_tools = ["comment_on_pull_request", "set_commit_status"]
    expected_standard_tools = ["read_files", "run_python_script"]

    # 2. Execute
    tool_names = [tool.name for tool in agent_tools]

    # 3. Assertions
    for tool in expected_github_tools:
        assert tool in tool_names

    for tool in expected_standard_tools:
        assert tool in tool_names


# ==========================================
# Workflow: Tool Execution Engine
# ==========================================


def test_execute_tool_call_success_standard(mocker):
    """Green Path: Validates that the client correctly maps string names to function invocations."""
    # 1. Setup Mock Environment
    # create a pure mock object to completely bypass Pydantic's strict model restrictions
    mock_tool = mocker.MagicMock()
    mock_tool.name = "read_file"
    mock_tool.invoke.return_value = "mocked file contents"

    # replace the actual tools list in the client module with our synthetic list
    mocker.patch("src.workspace_agent.tools.registry.agent_tools", [mock_tool])

    mock_tool_call = {
        "name": "read_file",
        "args": {"absolute_path": "/safe/path.py"},
        "id": "call_456",
    }

    # 2. Execute
    result = execute_tool_call(mock_tool_call)

    # 3. Assertions
    assert "mocked file contents" in result
    # Validation update: It must pass an empty config dictionary natively
    mock_tool.invoke.assert_called_once_with({"absolute_path": "/safe/path.py"}, config={})


def test_execute_tool_call_error_invalid_tool():
    """Red Path: Ensures the orchestrator doesn't crash if the LLM hallucinates a tool name."""
    # 1. Setup Mock Environment
    mock_tool_call = {
        "name": "hallucinated_tool_that_does_not_exist",
        "args": {"path": "/tmp"},
        "id": "call_123",
    }

    # 2. Execute
    result = execute_tool_call(mock_tool_call)

    # 3. Assertions
    assert "Error: Tool 'hallucinated_tool_that_does_not_exist' not found." in result


def test_execute_tool_call_error_exception(mocker):
    """
    Red Path: Ensures that if a tool crashes during execution (e.g., unhandled Python exception),
    the error is safely caught and returned as a string for the LLM to read and fix.
    """
    # 1. Setup Mock Environment
    mock_tool = mocker.MagicMock()
    mock_tool.name = "run_python_script"
    # force the tool to throw a runtime exception when invoked
    mock_tool.invoke.side_effect = Exception("Simulated sandbox timeout")

    mocker.patch("src.workspace_agent.tools.registry.agent_tools", [mock_tool])

    mock_tool_call = {
        "name": "run_python_script",
        "args": {"script_path": "/safe/infinite_loop.py"},
        "id": "call_789",
    }

    # 2. Execute
    result = execute_tool_call(mock_tool_call)

    # 3. Assertions
    assert "Tool execution failed" in result
    assert "Simulated sandbox timeout" in result


# ==========================================
# Workflow: Control Flow Escape Hatches
# ==========================================


def test_ask_user_for_clarification_success_legacy_options():
    """
    Green Path: Ensures the clarification tool returns the correct breakpoint string using
    legacy options.
    """
    # 1. Setup Mock Environment
    options = ["fileA.py", "fileB.py"]

    # 2. Execute
    result = ask_user_for_clarification(options=options)

    # 3. Assertions
    assert "Clarification requested" in result


def test_ask_user_for_clarification_success_specific_question():
    """
    "Green Path: Ensures the clarification tool returns the correct breakpoint string using
    a specific question.
    """
    # 1. Setup Mock Environment
    question = "Which version should I use?"

    # 2. Execute
    result = ask_user_for_clarification(question=question)

    # 3. Assertions
    assert "Clarification requested" in result


def test_mark_task_already_completed_success_standard():
    """Green Path: Verifies the escape hatch tool returns the correct payload."""
    # 1. Setup Mock Environment
    reason = "The toggle was already set to false."

    # 2. Execute
    result = mark_task_already_completed(reason=reason)

    # 3. Assertions
    assert "Task marked as successfully completed" in result
    assert "The toggle was already set to false." in result
