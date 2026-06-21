import pytest

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
    secure_resolve_path,
    write_file,
)


@pytest.fixture
def setup_workspaces(tmp_path, monkeypatch):
    """
    Creates a temporary safe workspace and a temporary forbidden zone
    to simulate the filesystem securely during tests.
    """
    # define the safe zone (simulating your allowed workspace)
    safe_dir = tmp_path / "git"
    safe_dir.mkdir()

    # define the forbidden zone (simulating restricted system directories)
    forbidden_dir = tmp_path / "ssh"
    forbidden_dir.mkdir()

    # explicitly point the environment variable to the safe zone
    monkeypatch.setenv("ALLOWED_PATHS", str(safe_dir))

    return {"safe": safe_dir, "forbidden": forbidden_dir}


# ==========================================
# Component: secure_resolve_path
# ==========================================


def test_secure_resolve_path_success_valid(setup_workspaces):
    """Green Path: Valid path inside the allowed workspace."""
    safe_dir = setup_workspaces["safe"]
    target = safe_dir / "project" / "main.py"

    resolved = secure_resolve_path(target, [safe_dir])
    assert resolved == target.resolve()


def test_secure_resolve_path_error_blocks_infrastructure(tmp_path):
    """Red Path: Ensure path resolution blocks sensitive infrastructure paths."""
    allowed = [str(tmp_path)]

    blocked_cases = [
        tmp_path / "project" / ".git" / "config",
        tmp_path / "project" / ".secrets" / "keys.json",
        tmp_path / "project" / ".env.local",
        tmp_path / "project" / "private.pem",
    ]

    for evil_path in blocked_cases:
        with pytest.raises(PermissionError, match="Security Exception.*strictly blocked"):
            secure_resolve_path(evil_path, allowed)


def test_secure_resolve_path_error_blocks_self_modification(tmp_path):
    """Red Path: Ensure the agent cannot traverse into its own orchestrator codebase."""
    allowed = [str(tmp_path)]

    # Using the specific directory name blocked in the filesystem.py array
    evil_path = tmp_path / "langgraph-workspace-agent" / "src" / "main.py"

    with pytest.raises(PermissionError, match="Security Exception.*strictly blocked"):
        secure_resolve_path(evil_path, allowed)


def test_secure_resolve_path_error_traversal_blocked(setup_workspaces):
    """Red Path: Malicious payload attempting directory traversal."""
    safe_dir = setup_workspaces["safe"]

    # simulate a path attempting to go up and into the forbidden zone
    malicious_path = safe_dir / ".." / "ssh" / "id_rsa"

    with pytest.raises(PermissionError, match="Security Exception"):
        secure_resolve_path(malicious_path, [safe_dir])


# ==========================================
# Component: write_file
# ==========================================


def test_write_file_success_standard(setup_workspaces):
    """Green Path: Successfully write a file and create parent directories."""
    safe_dir = setup_workspaces["safe"]
    # simulating a typical tracked output file
    new_file = safe_dir / "logs" / "01-08-training-logs.txt"

    result = write_file(str(new_file), "Epoch 1: Loss 0.04")

    assert "Success" in result
    assert new_file.exists()
    assert new_file.read_text(encoding="utf-8") == "Epoch 1: Loss 0.04"


def test_write_file_error_unauthorized(setup_workspaces):
    """Red Path: Graceful string error returned for unauthorized write."""
    forbidden_dir = setup_workspaces["forbidden"]
    malicious_file = forbidden_dir / "authorized_keys"

    result = write_file(str(malicious_file), "ssh-rsa hacker_key")

    assert "Security Exception" in result
    assert not malicious_file.exists()


# ==========================================
# Component: search_workspace
# ==========================================


def test_search_workspace_success_explicit_directory(setup_workspaces):
    """Green Path: Successfully search within a specific nested subdirectory."""
    safe_dir = setup_workspaces["safe"]

    nested_dir = safe_dir / "backend" / "api"
    nested_dir.mkdir(parents=True)
    (nested_dir / "routes.py").write_text("pass", encoding="utf-8")

    # create a file outside the nested dir to ensure it is NOT found
    (safe_dir / "routes.py").write_text("pass", encoding="utf-8")

    result = search_workspace("routes.py", directory=str(nested_dir))

    assert "Found 1 matches" in result
    assert "backend/api/routes.py" in result.replace("\\", "/")


def test_search_workspace_success_root(setup_workspaces):
    """Green Path: Successfully search and aggregate files from the root workspace."""
    safe_dir = setup_workspaces["safe"]

    # create a nested structure to test recursive rglob
    (safe_dir / "src").mkdir()
    (safe_dir / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (safe_dir / "test.py").write_text("def test(): pass", encoding="utf-8")

    result = search_workspace("*.py", directory="")

    assert "Found 2 matches" in result
    assert "main.py" in result
    assert "test.py" in result


def test_search_workspace_fallback_no_matches(setup_workspaces):
    """Edge Path: Graceful response when no files match the glob pattern."""
    safe_dir = setup_workspaces["safe"]

    result = search_workspace("*.rb", directory=str(safe_dir))
    assert "No files found matching" in result


def test_search_workspace_fallback_truncation(setup_workspaces):
    """Edge Path: Verifies that result lists > 50 are truncated to prevent context bloat."""
    safe_dir = setup_workspaces["safe"]
    logs_dir = safe_dir / "logs"
    logs_dir.mkdir()

    # create 55 dummy files
    for i in range(55):
        (logs_dir / f"log_{i}.txt").write_text("test", encoding="utf-8")

    result = search_workspace("*.txt", directory=str(logs_dir))

    assert "Found 55 matches" in result
    assert "(showing first 50)" in result

    # check truncation string split (header + 50 files = 51 lines)
    assert len(result.split("\n")) == 51


def test_search_workspace_error_invalid_directory(setup_workspaces):
    """Red Path: Graceful error if the explicit search directory doesn't exist."""
    safe_dir = setup_workspaces["safe"]
    missing_dir = safe_dir / "does_not_exist"

    result = search_workspace("*.py", directory=str(missing_dir))
    assert "Error: Directory not found" in result


def test_search_workspace_error_unauthorized_directory(setup_workspaces):
    """Red Path: Block search queries pointed at restricted domains."""
    forbidden_dir = setup_workspaces["forbidden"]

    result = search_workspace("*.txt", directory=str(forbidden_dir))
    assert "Security Exception" in result


# ==========================================
# Component: search_and_replace
# ==========================================


def test_search_and_replace_success_standard(setup_workspaces):
    """Green Path: Successfully replaces a unique string in the file."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.tex"
    test_file.write_text("Fact 1\nFact 2: Grass is green\nFact 3", encoding="utf-8")

    result = search_and_replace(str(test_file), "Grass is green", "Roses are red")

    assert "Success" in result
    assert test_file.read_text(encoding="utf-8") == "Fact 1\nFact 2: Roses are red\nFact 3"


def test_search_and_replace_error_ambiguous(setup_workspaces):
    """Red Path: Safely rejects the edit if the target string appears multiple times."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.tex"
    test_file.write_text("Fact 1: blue\nFact 2: blue", encoding="utf-8")

    result = search_and_replace(str(test_file), "blue", "red")

    assert "Error" in result
    assert "found 2 times" in result
    # file should remain completely untouched
    assert test_file.read_text(encoding="utf-8") == "Fact 1: blue\nFact 2: blue"


def test_search_and_replace_error_not_found(setup_workspaces):
    """Red Path: Safely rejects if the string doesn't exist."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.tex"
    test_file.write_text("Fact 1", encoding="utf-8")

    result = search_and_replace(str(test_file), "Fact 2", "Fact 3")
    assert "not found in the file" in result


# ==========================================
# Component: replace_text_block
# ==========================================


def test_replace_text_block_success_standard(setup_workspaces):
    """Green Path: Successfully replaces a chunk of text bounded by markers."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.tex"
    initial_content = "Header\n\\section{Old}\nThis is old\n\\end{section}\nFooter"
    test_file.write_text(initial_content, encoding="utf-8")

    result = replace_text_block(
        str(test_file),
        "\\section{Old}",
        "\\end{section}",
        "\\section{New}\nThis is new\n\\end{section}",
    )

    assert "Success" in result
    expected_content = "Header\n\\section{New}\nThis is new\n\\end{section}\nFooter"
    assert test_file.read_text(encoding="utf-8") == expected_content


def test_replace_text_block_error_missing_marker(setup_workspaces):
    """Red Path: Aborts if either marker is missing."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.tex"
    test_file.write_text("Header\n\\section{Old}\nContent", encoding="utf-8")

    result = replace_text_block(str(test_file), "\\section{Old}", "\\end{section}", "New")

    # Assert the new dynamic error string, escaping the backslash as expected by repr()
    assert "The end_marker '\\\\end{section}' was not found" in result


# ==========================================
# Component: read_files
# ==========================================


def test_read_files_success_cross_workspace(setup_workspaces):
    """Green Path: Successfully read multiple files located in different repositories."""
    safe_dir = setup_workspaces["safe"]

    # repo A
    repo_a = safe_dir / "gateway"
    repo_a.mkdir()
    file_a = repo_a / "main.py"
    file_a.write_text("print('repo A')", encoding="utf-8")

    # repo B
    repo_b = safe_dir / "arena"
    repo_b.mkdir()
    file_b = repo_b / "test.tex"
    file_b.write_text("Hello from Repo B", encoding="utf-8")

    result = read_files([str(file_a), str(file_b)])

    assert f"=== FILE: {str(file_a)} ===" in result
    assert "print('repo A')" in result
    assert f"=== FILE: {str(file_b)} ===" in result
    assert "Hello from Repo B" in result


def test_read_files_success_single_file(setup_workspaces):
    """Green Path: Seamlessly handles reading a single file via the batch tool."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "single.txt"
    test_file.write_text("Just one file.", encoding="utf-8")

    result = read_files([str(test_file)])
    assert f"=== FILE: {str(test_file)} ===" in result
    assert "Just one file." in result


def test_read_files_fallback_partial_failure(setup_workspaces):
    """
    Edge Path: If one file fails (e.g., doesn't exist), the tool must still
    successfully return the contents of the valid files.
    """
    safe_dir = setup_workspaces["safe"]

    valid_file = safe_dir / "good.txt"
    valid_file.write_text("This is good.", encoding="utf-8")

    missing_file = safe_dir / "missing.txt"

    result = read_files([str(valid_file), str(missing_file)])

    # verify the good file succeeded
    assert "This is good." in result

    # verify the missing file threw a targeted error, not a total crash
    assert "Error: File not found" in result
    assert str(missing_file) in result


def test_read_files_error_unauthorized_trap(setup_workspaces):
    """Red Path: Proves that batch processing enforces security boundaries per-file."""
    safe_dir = setup_workspaces["safe"]
    forbidden_dir = setup_workspaces["forbidden"]

    valid_file = safe_dir / "good.txt"
    valid_file.write_text("Valid", encoding="utf-8")

    malicious_file = forbidden_dir / "secret.txt"

    result = read_files([str(valid_file), str(malicious_file)])

    assert "Valid" in result
    assert "Security Exception" in result
    assert str(malicious_file) in result


# ==========================================
# Component: get_workspace_tree
# ==========================================


def test_get_workspace_tree_success_standard(setup_workspaces):
    """Green Path: Verifies the tree formatting logic prints a clean hierarchy."""
    safe_dir = setup_workspaces["safe"]
    (safe_dir / "src" / "api").mkdir(parents=True)
    (safe_dir / "src" / "main.py").write_text("", encoding="utf-8")
    (safe_dir / "src" / "api" / "routes.py").write_text("", encoding="utf-8")
    (safe_dir / "README.md").write_text("", encoding="utf-8")

    # create a hidden directory to ensure it gets ignored
    (safe_dir / ".git").mkdir()

    result = get_workspace_tree(str(safe_dir))

    assert "git/" in result
    assert "├── src" in result
    assert "│   ├── api" in result
    assert "│   │   └── routes.py" in result
    assert "│   └── main.py" in result
    assert "└── README.md" in result
    assert ".git" not in result  # should be skipped


# ==========================================
# Component: get_code_skeleton
# ==========================================


def test_get_code_skeleton_success_standard(setup_workspaces):
    """Green Path: Uses AST to successfully extract class and method signatures."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "logic.py"

    code = """
import os
from typing import List

class Processor(BaseMixin):
    def process(self, data: str):
        print(data)
        return True

def standalone_func():
    pass
"""
    test_file.write_text(code, encoding="utf-8")

    result = get_code_skeleton(str(test_file))

    assert "import os" in result
    assert "from typing import List" in result
    assert "class Processor(BaseMixin):" in result
    assert "def process(...): pass" in result
    assert "def standalone_func(...): pass" in result
    # function bodies should NOT be present
    assert "print(data)" not in result


def test_get_code_skeleton_error_invalid_filetype(setup_workspaces):
    """Red Path: Safely rejects non-Python files since it relies on AST parsing."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "doc.txt"
    test_file.write_text("def something(): pass", encoding="utf-8")

    result = get_code_skeleton(str(test_file))
    assert "Error: get_code_skeleton currently only supports Python (.py) files" in result


# ==========================================
# Component: read_file_section
# ==========================================


def test_read_file_section_success_standard(setup_workspaces):
    """Green Path: Successfully extracts a targeted block of text."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "report.tex"

    content = "Ignore this\n\\section{Methods}\nCore content\n\\section{Results}\nIgnore this too"
    test_file.write_text(content, encoding="utf-8")

    result = read_file_section(str(test_file), "\\section{Methods}", "\\section{Results}")

    assert result.startswith("\\section{Methods}")
    assert result.endswith("\\section{Results}")
    assert "Core content" in result
    assert "Ignore this" not in result.replace("Ignore this too", "")


def test_read_file_section_error_missing_marker(setup_workspaces):
    """Red Path: Gracefully errors if the requested boundary markers are missing."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "report.tex"
    test_file.write_text("Just some normal text", encoding="utf-8")

    result = read_file_section(str(test_file), "START", "END")

    # Assert the new dynamic error string
    assert "The start_marker 'START' was not found" in result


# ==========================================
# Component: grep_workspace
# ==========================================


def test_grep_workspace_success_standard(setup_workspaces):
    """Green Path: Successfully finds exact strings and reports file/line numbers."""
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "target.py"
    test_file.write_text("def hello():\n    print('FIND_ME')\n", encoding="utf-8")

    result = grep_workspace("FIND_ME", str(safe_dir))

    assert "target.py:2:" in result
    assert "print('FIND_ME')" in result


def test_grep_workspace_fallback_not_found(setup_workspaces):
    """Edge Path: Safely reports when a string does not exist."""
    safe_dir = setup_workspaces["safe"]
    result = grep_workspace("NONEXISTENT_STRING_12345", str(safe_dir))
    assert "No matches found" in result


# ==========================================
# Component: rename_file
# ==========================================


def test_rename_file_success_standard(setup_workspaces):
    """Green Path: Successfully renames and moves a file."""
    safe_dir = setup_workspaces["safe"]
    old_file = safe_dir / "old_name.py"
    old_file.write_text("print('test')", encoding="utf-8")

    new_file = safe_dir / "new_folder" / "new_name.py"

    result = rename_file(str(old_file), str(new_file))

    assert "Success" in result
    assert not old_file.exists()
    assert new_file.exists()
    assert new_file.read_text(encoding="utf-8") == "print('test')"


def test_rename_file_error_exists_trap(setup_workspaces):
    """Red Path: Safely aborts if the destination already exists."""
    safe_dir = setup_workspaces["safe"]
    old_file = safe_dir / "source.txt"
    old_file.write_text("source", encoding="utf-8")

    existing_dest = safe_dir / "dest.txt"
    existing_dest.write_text("dest", encoding="utf-8")

    result = rename_file(str(old_file), str(existing_dest))

    assert "Error: Destination already exists" in result
    assert old_file.exists()  # source should remain untouched


# ==========================================
# Component: delete_file
# ==========================================


def test_delete_file_success_standard(setup_workspaces):
    """Green Path: Successfully deletes a target file."""
    safe_dir = setup_workspaces["safe"]
    target_file = safe_dir / "junk.txt"
    target_file.write_text("garbage", encoding="utf-8")

    result = delete_file(str(target_file))

    assert "Success" in result
    assert not target_file.exists()


def test_delete_file_error_directory_trap(setup_workspaces):
    """Red Path: Refuses to delete directories to prevent massive accidental loss."""
    safe_dir = setup_workspaces["safe"]
    target_dir = safe_dir / "important_folder"
    target_dir.mkdir()

    result = delete_file(str(target_dir))

    assert "Error: Path is a directory" in result
    assert target_dir.exists()
