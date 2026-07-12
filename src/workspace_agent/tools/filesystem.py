import ast
import os
from pathlib import Path

__all__ = [
    "get_allowed_paths",
    "secure_resolve_path",
    "write_file",
    "search_and_replace",
    "replace_text_block",
    "search_workspace",
    "read_files",
    "get_workspace_tree",
    "get_code_skeleton",
    "read_file_section",
    "grep_workspace",
    "rename_file",
    "delete_file",
]

# ==========================================
# Filesystem Tools
# ==========================================

# === Security & Path Resolution ===


def get_allowed_paths() -> list[str]:
    """
    Retrieves the whitelisted directories from the environment.
    Defaults to the standard git workspace if not explicitly set.
    """
    default_workspace = str(Path.home() / "git")
    paths_str = os.getenv("ALLOWED_PATHS", default_workspace)
    return [p.strip() for p in paths_str.split(",") if p.strip()]


def secure_resolve_path(requested_path: str | Path, allowed_paths: list[str | Path]) -> Path:
    """
    Resolves a requested file path and strictly verifies it resides
    within one of the whitelisted workspace directories.

    Raises:
        PermissionError: If the path attempts to traverse outside allowed zones.
    """
    # Resolve the requested path to its absolute, canonical form
    # This automatically flattens all `../` attempts and resolves symlinks
    target_path = Path(requested_path).resolve()

    # Protects against Host-to-Container path evasion and variant secret extensions
    blocked_dirs = {".git", ".secrets", "langgraph-workspace-agent"}
    blocked_prefixes = (".ssh", ".aws", ".kube")
    blocked_extensions = (".pem", ".key", ".cert", ".pkcs12")

    # Explicit whitelist for safe template files
    allowed_env_templates = {".env.example", ".env.template", ".env.sample", ".env.dist"}

    for part in target_path.parts:
        # Block .env files unless they are explicitly whitelisted templates
        is_blocked_env = part.startswith(".env") and part not in allowed_env_templates
        if (
            part in blocked_dirs
            or part.startswith(blocked_prefixes)
            or part.endswith(blocked_extensions)
            or is_blocked_env
        ):
            raise PermissionError(
                f"Security Exception: Access to hidden, sensitive, or system "
                f"infrastructure path is strictly blocked -> {target_path}"
            )

    # Existing Workspace Boundary Check
    for allowed in allowed_paths:
        safe_base = Path(allowed).resolve()

        # Verify the target sits inside the safe base
        if target_path.is_relative_to(safe_base):
            return target_path

    # Trigger the circuit breaker if the loop finishes without returning
    raise PermissionError(
        f"Security Exception: The agent attempted to access an unauthorized path -> {target_path}"
    )


# === Core Filesystem Functions (Native Tools) ===


def write_file(file_path: str, content: str) -> str:
    """Writes content to a file, creating parent directories if necessary, post-security check."""
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())

        # ensure the parent directories exist within the sandbox
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

        return f"Success: Wrote {len(content)} characters to {safe_path}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error writing file: {str(e)}"


def search_and_replace(file_path: str, old_text: str, new_text: str) -> str:
    """
    Precisely replaces an exact string in a file with a new string.
    The old_text must match exactly and be unique in the file to prevent accidental overwrites.
    """
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())
        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: File not found at {safe_path}"

        content = safe_path.read_text(encoding="utf-8")

        occurrences = content.count(old_text)
        if occurrences == 0:
            return (
                f"Error: The exact string {repr(old_text)} was not found in the file. "
                "Ensure you have the exact string (check quotes, whitespace, and indentation). "
                "You MUST use the `read_files` tool to look at the file again before retrying."
            )
        if occurrences > 1:
            return (
                f"Error: The exact string {repr(old_text)} was found {occurrences} times. "
                "Please provide a larger, more unique snippet of text to ensure the correct "
                "instance is replaced."
            )

        new_content = content.replace(old_text, new_text, 1)
        safe_path.write_text(new_content, encoding="utf-8")

        return f"Success: Replaced 1 instance of text in {safe_path.name}."

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error modifying file: {str(e)}"


def replace_text_block(file_path: str, start_marker: str, end_marker: str, new_text: str) -> str:
    """
    Replaces a block of text bounded by start_marker and end_marker (INCLUSIVE of both markers)
    with new_text.
    Useful for replacing entire sections, functions, or environments.
    """
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())
        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: File not found at {safe_path}"

        content = safe_path.read_text(encoding="utf-8")

        start_idx = content.find(start_marker)
        if start_idx == -1:
            return (
                f"Error: The start_marker {repr(start_marker)} was not found in the file. "
                "Ensure you have the exact string (check quotes, whitespace, and indentation). "
                "You MUST use the `read_files` tool to look at the file again before retrying."
            )

        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            return (
                f"Error: The end_marker {repr(end_marker)} was not found after the start_marker. "
                "Ensure you have the exact string (check quotes, whitespace, and indentation). "
                "You MUST use the `read_files` tool to look at the file again before retrying."
            )

        # calculate the end index including the end_marker itself
        full_end_idx = end_idx + len(end_marker)

        new_content = content[:start_idx] + new_text + content[full_end_idx:]
        safe_path.write_text(new_content, encoding="utf-8")

        return (
            f"Success: Replaced block from '{start_marker}' to '{end_marker}' in {safe_path.name}."
        )

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error modifying file: {str(e)}"


def search_workspace(pattern: str, directory: str = "") -> str:
    """
    Searches for files matching a glob pattern (e.g., '*.py') within the allowed workspaces.
    If directory is provided, it searches within that specific sub-directory.
    """
    try:
        allowed = get_allowed_paths()

        # determine the base directory to search from
        if directory:
            search_base = secure_resolve_path(directory, allowed)
        else:
            # default to the first allowed workspace if none is specified
            search_base = Path(allowed[0]).resolve()

        if not search_base.exists() or not search_base.is_dir():
            return f"Error: Directory not found or is not a valid directory -> {search_base}"

        # perform a recursive glob search
        results = list(search_base.rglob(pattern))

        if not results:
            return f"No files found matching '{pattern}' in {search_base}"

        # format results, truncating to prevent massive LLM context bloat
        max_results = 50
        formatted_results = [str(p) for p in results[:max_results]]

        response = f"Found {len(results)} matches"
        if len(results) > max_results:
            response += f" (showing first {max_results}):\n"
        else:
            response += ":\n"

        response += "\n".join(formatted_results)
        return response

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error searching workspace: {str(e)}"


# === Batch Operations ===


def read_files(file_paths: list[str]) -> str:
    """
    Reads the contents of one or multiple files in a single batch operation.
    Supports reading files across different allowed workspaces.
    Returns a formatted string delineating the contents of each file.
    To read a single file, just pass a list with one path.
    """
    try:
        allowed = get_allowed_paths()
        results = []

        for path_str in file_paths:
            file_header = f"=== FILE: {path_str} ===\n"
            try:
                safe_path = secure_resolve_path(path_str, allowed)

                if not safe_path.exists():
                    results.append(f"{file_header}Error: File not found.")
                elif not safe_path.is_file():
                    results.append(f"{file_header}Error: Path is not a file.")
                else:
                    content = safe_path.read_text(encoding="utf-8")
                    results.append(f"{file_header}{content}")

            except PermissionError as e:
                results.append(f"{file_header}Security Exception: {str(e)}")
            except Exception as e:
                results.append(f"{file_header}Error reading file: {str(e)}")

        # cleanly separate the file outputs for the LLM context window
        return "\n\n".join(results)

    except Exception as e:
        return f"Error executing batch read: {str(e)}"


# === Context-Optimized Reading Tools ===


def get_workspace_tree(directory: str = "", max_depth: int = 3) -> str:
    """
    Generates a visual tree structure of the workspace or a specific directory.
    Useful for understanding the repository layout without reading full files.
    """
    try:
        allowed = get_allowed_paths()

        if directory:
            search_base = secure_resolve_path(directory, allowed)
        else:
            search_base = Path(allowed[0]).resolve()

        if not search_base.exists() or not search_base.is_dir():
            return f"Error: Directory not found or is not a valid directory -> {search_base}"

        tree_lines = []

        def walk_dir(current_path: Path, current_depth: int, prefix: str = ""):
            if current_depth > max_depth:
                return

            try:
                # sort directories first, then files
                paths = sorted(
                    current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                )
            except PermissionError:
                return

            for i, path in enumerate(paths):
                # Skip hidden directories like .git, .venv, etc. to reduce noise
                if path.name.startswith(".") and path.is_dir():
                    continue

                is_last = i == len(paths) - 1
                connector = "└── " if is_last else "├── "

                tree_lines.append(f"{prefix}{connector}{path.name}")

                if path.is_dir():
                    extension = "    " if is_last else "│   "
                    walk_dir(path, current_depth + 1, prefix + extension)

        tree_lines.append(search_base.name + "/")
        walk_dir(search_base, 1)

        return "\n".join(tree_lines)

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error generating workspace tree: {str(e)}"


def get_code_skeleton(file_path: str) -> str:
    """
    Parses a Python file and extracts only the imports, class names, and function signatures.
    Extremely useful for understanding large files without consuming massive context limits.
    """
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())
        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: File not found at {safe_path}"

        if safe_path.suffix != ".py":
            return "Error: get_code_skeleton currently only supports Python (.py) files."

        content = safe_path.read_text(encoding="utf-8")
        return _parse_python_skeleton(content)

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error extracting skeleton: {str(e)}"


def _parse_python_skeleton(content: str) -> str:
    """
    Private helper function to parse Python code and extract its skeleton.
    Separated to comply with Ruff cyclomatic complexity and branching limits.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return f"Error: Could not parse Python file (SyntaxError): {e}"

    skeleton_lines = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                skeleton_lines.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            skeleton_lines.append(f"from {node.module} import {names}")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(b.id for b in node.bases if isinstance(b, ast.Name))
            base_str = f"({bases})" if bases else ""
            skeleton_lines.append(f"\nclass {node.name}{base_str}:")

            # add method stubs
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    skeleton_lines.append(f"    def {class_node.name}(...): pass")
        elif isinstance(node, ast.FunctionDef):
            skeleton_lines.append(f"\ndef {node.name}(...): pass")

    if not skeleton_lines:
        return "File contains no imports, classes, or functions."

    return "\n".join(skeleton_lines)


def read_file_section(file_path: str, start_marker: str, end_marker: str) -> str:
    """
    Reads only a specific section of a file bounded by start_marker and end_marker (inclusive).
    Useful for analyzing targeted sections of large documents like LaTeX files or logs.
    """
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())
        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: File not found at {safe_path}"

        content = safe_path.read_text(encoding="utf-8")

        start_idx = content.find(start_marker)
        if start_idx == -1:
            return f"Error: The start_marker {repr(start_marker)} was not found in the file."

        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            return f"Error: The end_marker {repr(end_marker)} was not found after the start_marker."

        # calculate the end index including the end_marker itself
        full_end_idx = end_idx + len(end_marker)

        return content[start_idx:full_end_idx]

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file section: {str(e)}"


def grep_workspace(search_string: str, directory: str = "") -> str:
    """
    Searches for an exact string across all files in a directory or the active workspace.
    Returns the file paths and line numbers where the string is found.
    Ignores hidden folders (like .git) and binary caches (like __pycache__).
    """
    try:
        allowed = get_allowed_paths()
        if directory:
            search_base = secure_resolve_path(directory, allowed)
        else:
            search_base = Path(allowed[0]).resolve()

        if not search_base.exists() or not search_base.is_dir():
            return f"Error: Directory not found -> {search_base}"

        results = []

        for root, dirs, files in os.walk(search_base):
            # skip hidden directories and __pycache__ to reduce noise
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.startswith("."):
                    continue

                file_path = Path(root) / file
                results.extend(_search_file_for_string(file_path, search_base, search_string))

        if not results:
            return f"No matches found for '{search_string}'"

        return "\n".join(results)

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error executing grep: {str(e)}"


def _search_file_for_string(file_path: Path, search_base: Path, search_string: str) -> list[str]:
    """
    Private helper function to scan a single file for a search string.
    Separated to comply with Ruff cyclomatic complexity limits.
    """
    results = []
    try:
        content = file_path.read_text(encoding="utf-8")
        if search_string in content:
            # only split into lines if the file contains the string (saves memory/time)
            lines = content.splitlines()
            rel_path = file_path.relative_to(search_base)
            for i, line in enumerate(lines, 1):
                if search_string in line:
                    results.append(f"{rel_path}:{i}: {line.strip()}")
    except UnicodeDecodeError:
        # silently ignore binary files (PDFs, images, etc.)
        pass

    return results


# === File Structure & Refactoring Operations ===


def rename_file(old_path: str, new_path: str) -> str:
    """
    Renames or moves a file from old_path to new_path.
    Fails safely if the destination already exists to prevent accidental overwrites.
    """
    try:
        allowed = get_allowed_paths()
        safe_old_path = secure_resolve_path(old_path, allowed)
        safe_new_path = secure_resolve_path(new_path, allowed)

        if not safe_old_path.exists() or not safe_old_path.is_file():
            return f"Error: Source file not found at {safe_old_path}"

        if safe_new_path.exists():
            return (
                f"Error: Destination already exists at {safe_new_path}. Please delete it first or "
                "choose another name."
            )

        # ensure the target directory exists if the file is being moved to a new folder
        safe_new_path.parent.mkdir(parents=True, exist_ok=True)

        safe_old_path.rename(safe_new_path)
        return f"Success: Moved/Renamed file to {safe_new_path.name}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error renaming file: {str(e)}"


def delete_file(file_path: str) -> str:
    """
    Permanently deletes a file from the repository.
    Relies on Git for version control safety.
    """
    try:
        safe_path = secure_resolve_path(file_path, get_allowed_paths())

        if not safe_path.exists():
            return f"Error: File not found at {safe_path}"

        if not safe_path.is_file():
            return "Error: Path is a directory, not a file. Only files can be deleted."

        safe_path.unlink()
        return f"Success: Deleted file {safe_path.name}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error deleting file: {str(e)}"
