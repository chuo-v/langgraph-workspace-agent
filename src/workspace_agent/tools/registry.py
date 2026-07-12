from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from src.workspace_agent.tools.filesystem import (
    delete_file,
    get_code_skeleton,
    get_workspace_tree,
    grep_workspace,
    read_file_section,
    read_files,
    rename_file,
    replace_text_block,
    search_and_replace,
    search_workspace,
    write_file,
)
from src.workspace_agent.tools.github import (
    apply_git_patch,
    comment_on_pull_request,
    get_git_diff,
    set_commit_status,
    sync_repository,
)
from src.workspace_agent.tools.sandbox import (
    compile_latex_document,
    run_pytest,
    run_python_script,
)

__all__ = [
    "agent_tools",
    "ask_user_for_clarification",
    "execute_tool_call",
    "mark_task_already_completed",
]


# ==========================================
# State & Lifecycle Tools
# ==========================================


def ask_user_for_clarification(
    options: list[str] | None = None, question: str | None = None
) -> str:
    """
    State-modification tool.
    Called by the LLM if it finds multiple conflicting files or needs
    to ask the user a specific question before proceeding.
    """
    # this tool's presence simply signals the graph to pause
    # the actual state update is handled inside the execute_task_node loop
    return "Clarification requested. Pausing execution."


def mark_task_already_completed(reason: str) -> str:
    """
    Call this tool immediately if you evaluate the workspace and determine that the
    user's requested changes are ALREADY present, or no action is actually required.
    Provide a brief reason explaining why no changes were needed.
    """
    return f"Task marked as successfully completed without changes. Reason: {reason}"


# ==========================================
# Tool Registry
# ==========================================

# Aggregate all capabilities into a LangChain-compatible list, grouped by domain
agent_tools: list[StructuredTool] = [
    # === Filesystem Tools ===
    StructuredTool.from_function(read_files),
    StructuredTool.from_function(write_file),
    StructuredTool.from_function(rename_file),
    StructuredTool.from_function(delete_file),
    StructuredTool.from_function(search_and_replace),
    StructuredTool.from_function(replace_text_block),
    StructuredTool.from_function(search_workspace),
    StructuredTool.from_function(grep_workspace),
    StructuredTool.from_function(get_workspace_tree),
    StructuredTool.from_function(get_code_skeleton),
    StructuredTool.from_function(read_file_section),
    # === Sandbox & Execution Tools ===
    StructuredTool.from_function(run_python_script),
    StructuredTool.from_function(compile_latex_document),
    StructuredTool.from_function(run_pytest),
    # === Git & Repository Tools ===
    StructuredTool.from_function(get_git_diff),
    StructuredTool.from_function(sync_repository),
    StructuredTool.from_function(apply_git_patch),
    StructuredTool.from_function(comment_on_pull_request),
    StructuredTool.from_function(set_commit_status),
    # === State & Lifecycle Tools ===
    StructuredTool.from_function(ask_user_for_clarification),
    StructuredTool.from_function(mark_task_already_completed),
]


# ==========================================
# Tool Execution API
# ==========================================


def execute_tool_call(tool_call: dict[str, Any], config: RunnableConfig | None = None) -> str:
    """
    Manually invoke a tool from the registry and return its string result.

    Passes the `RunnableConfig` down into tools that require dependency injection
    or execution context.

    Args:
        tool_call: A dictionary containing 'name' (str) and 'args' (dict) of the tool to invoke.
        config: Optional LangChain RunnableConfig for dependency injection.

    Returns:
        str: The string output of the invoked tool, or an error message if invocation fails.

    Exceptions:
        No runtime exceptions are raised directly to the caller; execution exceptions
        raised during tool invocation are caught and returned as formatted error strings.
    """
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    config = config or {}

    if not tool_name:
        return "Error: Tool name missing from tool call."

    tool_map = {tool.name: tool for tool in agent_tools}
    if tool_name in tool_map:
        try:
            return str(tool_map[tool_name].invoke(tool_args, config=config))
        except Exception as e:
            return f"Tool execution failed: {e}"
    return f"Error: Tool '{tool_name}' not found."
