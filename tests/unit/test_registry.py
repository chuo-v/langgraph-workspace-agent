from src.workspace_agent.tools.registry import (
    agent_tools,
    ask_user_for_clarification,
    execute_tool_call,
    mark_task_already_completed,
)

# ==========================================
# Component: agent_tools
# ==========================================


def test_agent_tools_success_registration():
    """
    Green Path: Validates that all critical tools, including new GitHub integrations, are
    registered.
    """
    tool_names = [tool.name for tool in agent_tools]

    # Verify the newly added integrations
    assert "comment_on_pull_request" in tool_names
    assert "set_commit_status" in tool_names

    # Verify a sample of original tools to ensure standard loading
    assert "read_files" in tool_names
    assert "run_python_script" in tool_names


# ==========================================
# Component: ask_user_for_clarification
# ==========================================


def test_ask_user_for_clarification_success_standard():
    """Green Path: Ensures the clarification tool returns the correct breakpoint string."""
    # test with legacy options
    result = ask_user_for_clarification(options=["fileA.py", "fileB.py"])
    assert "Clarification requested" in result

    # test with a specific question
    result_with_question = ask_user_for_clarification(question="Which version should I use?")
    assert "Clarification requested" in result_with_question


# ==========================================
# Component: mark_task_already_completed
# ==========================================


def test_mark_task_already_completed_success_standard():
    """Green Path: Verifies the escape hatch tool returns the correct payload."""
    res = mark_task_already_completed(reason="The toggle was already set to false.")

    assert "Task marked as successfully completed" in res
    assert "The toggle was already set to false." in res


# ==========================================
# Component: execute_tool_call
# ==========================================


def test_execute_tool_call_success_standard(mocker):
    """Green Path: Validates that the client correctly maps string names to function invocations."""
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

    result = execute_tool_call(mock_tool_call)

    assert "mocked file contents" in result
    # Validation update: It must pass an empty config dictionary natively
    mock_tool.invoke.assert_called_once_with({"absolute_path": "/safe/path.py"}, config={})


def test_execute_tool_call_error_invalid_tool():
    """Red Path: Ensures the orchestrator doesn't crash if the LLM hallucinates a tool name."""
    mock_tool_call = {
        "name": "hallucinated_tool_that_does_not_exist",
        "args": {"path": "/tmp"},
        "id": "call_123",
    }

    result = execute_tool_call(mock_tool_call)
    assert "Error: Tool 'hallucinated_tool_that_does_not_exist' not found." in result


def test_execute_tool_call_error_exception(mocker):
    """
    Red Path: Ensures that if a tool crashes during execution (e.g., unhandled Python exception),
    the error is safely caught and returned as a string for the LLM to read and fix.
    """
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

    result = execute_tool_call(mock_tool_call)

    assert "Tool execution failed" in result
    assert "Simulated sandbox timeout" in result
