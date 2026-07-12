"""Model Context Protocol (MCP) server for the workspace agent.

Exposes the workspace agent's native tools (filesystem, sandbox execution,
and Git operations) as a standard MCP server for external clients such as
Claude Desktop, Cursor, or Zed.
"""

from mcp.server.fastmcp import FastMCP

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
from src.workspace_agent.tools.github import get_git_diff, sync_repository
from src.workspace_agent.tools.sandbox import (
    compile_latex_document,
    run_pytest,
    run_python_script,
)

__all__ = ["mcp"]

# ==========================================
# Server Setup & Tool Registration
# ==========================================

# Initialize the standalone MCP server with explicit type annotation
mcp: FastMCP = FastMCP("WorkspaceAgentTools")

# === Filesystem Tools ===
mcp.tool()(read_files)
mcp.tool()(write_file)
mcp.tool()(rename_file)
mcp.tool()(delete_file)
mcp.tool()(search_and_replace)
mcp.tool()(replace_text_block)
mcp.tool()(search_workspace)
mcp.tool()(grep_workspace)
mcp.tool()(get_workspace_tree)
mcp.tool()(get_code_skeleton)
mcp.tool()(read_file_section)

# === Sandbox Execution Tools ===
mcp.tool()(run_python_script)
mcp.tool()(compile_latex_document)
mcp.tool()(run_pytest)

# === Git Operations ===
mcp.tool()(get_git_diff)
mcp.tool()(sync_repository)

# ==========================================
# Server Execution
# ==========================================

if __name__ == "__main__":
    # Start the server using standard input/output transport
    mcp.run(transport="stdio")
