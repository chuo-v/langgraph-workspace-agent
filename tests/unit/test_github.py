import json
from pathlib import Path

import git
import pytest

from src.workspace_agent.tools.github import (
    _get_target_remote,
    apply_git_patch,
    cleanup_local_branch,
    comment_on_pull_request,
    create_branch_and_commit,
    get_git_diff,
    get_git_diff_blueprint,
    is_diff_empty,
    open_pull_request,
    set_commit_status,
    sync_repository,
    update_pull_request,
)

# ==========================================
# Helper: Test Fixtures
# ==========================================


@pytest.fixture
def setup_git_workspace(setup_workspaces, monkeypatch, mocker):
    """
    Creates a temporary safe workspace, initializes a real Git repository,
    and sets up isolated mocks so production GitHub tools can execute locally.
    """
    safe_dir = setup_workspaces["safe"]

    # Inject the dummy token so production security checks pass
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # Initialize a local git repository
    repo = git.Repo.init(safe_dir)
    test_file = safe_dir / "README.md"
    test_file.write_text("# Initial Repository", encoding="utf-8")

    repo.index.add([str(test_file)])
    repo.index.commit("Initial commit")

    # Explicitly rename the default branch to 'main' to prevent CI failures
    # caused by legacy 'master' defaults in different environments
    repo.git.branch("-M", "main")

    # Create a local bare repository to act as 'origin' for ALL tests
    remote_dir = safe_dir.parent / "remote.git"
    git.Repo.init(remote_dir, bare=True)
    origin = repo.create_remote("origin", str(remote_dir))
    origin.push("main")

    # Mock the regex search so github.py accepts our local bare repo as a valid URL
    mock_match = mocker.MagicMock()
    mock_match.group.return_value = "owner/repo"
    mocker.patch("src.workspace_agent.tools.github.re.search", return_value=mock_match)

    # Mock set_url so production code doesn't overwrite our local bare repo route
    mocker.patch("git.remote.Remote.set_url")

    return safe_dir, repo


# ==========================================
# Workflow: Sync Repository Operations
# ==========================================


def test_sync_repository_success(setup_git_workspace):
    """Green Path: The agent successfully checks out a branch and pulls changes."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # 2. Execute
    result_json = sync_repository(str(safe_dir), "main")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["branch"] == "main"
    assert repo.active_branch.name == "main"


def test_sync_repository_fallback_dirty_tree_auto_stash(setup_git_workspace):
    """Edge Path: The agent encounters a dirty tree and successfully auto-stashes."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # create an untracked file to make the tree "dirty"
    untracked_file = safe_dir / "new_script.py"
    untracked_file.write_text("print('hello')", encoding="utf-8")

    # 2. Execute
    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    # 3. Assertions
    # sync should now succeed instead of failing
    assert result["status"] == "success"

    # working directory should be completely clean
    assert not repo.is_dirty(untracked_files=True)

    # untracked file should be safely in the stash, removed from the filesystem
    assert not untracked_file.exists()


def test_sync_repository_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Ensure sync_repository aborts cleanly if the GITHUB_TOKEN is missing."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_sync_repository_error_invalid_url(setup_git_workspace, mocker):
    """Red Path: Ensure sync_repository aborts cleanly if the remote URL format is unrecognized."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # mock re.search to return None, simulating a non-standard Git remote URL
    mocker.patch("src.workspace_agent.tools.github.re.search", return_value=None)

    # 2. Execute
    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "invalid_github_url_format"


def test_sync_repository_error_git_command(setup_git_workspace):
    """Red Path: Ensure GitCommandError is safely converted to a JSON error response."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # 2. Execute
    # trigger a natural GitCommandError by attempting to sync a branch that does not exist
    result_json = sync_repository(str(safe_dir), target_branch="non-existent-branch-name")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "git_command_failed"
    # the actual git stderr details will mention the branch it failed to match
    assert "non-existent-branch-name" in result["details"]


# ==========================================
# Workflow: Utility Functions
# ==========================================


@pytest.mark.parametrize(
    "diff_input, expected_result",
    [
        # Green Paths: Truly empty scenarios
        (None, True),
        ("", True),
        ("No uncommitted changes.", True),
        ("No files changed.", True),
        ("No changes compared to main.", True),
        ("No changes compared to feature/branch-name", True),
        # Green Paths: Tolerates messy whitespace
        (" No uncommitted changes. \n", True),
        ("\nNo changes compared to develop\n", True),
        # Red Paths: Legitimate diffs
        ("diff --git a/README.md b/README.md\n+hello", False),
        # Edge Paths (The "Inception" Bug):
        # The artificial fallback string exists, but it's buried INSIDE a valid diff payload.
        # This MUST evaluate to False so the diff is processed normally.
        ("diff --git a/script.py b/script.py\n+print('No uncommitted changes.')", False),
        (
            "diff --git a/script.py b/script.py\n+if diff == 'No changes compared to main': pass",
            False,
        ),
    ],
)
def test_is_diff_empty(diff_input, expected_result):
    """
    Validates that empty diffs are correctly identified and ensures
    the logic is immune to substring traps where the fallback text
    appears inside the actual modified source code.
    """
    assert is_diff_empty(diff_input) == expected_result


# ==========================================
# Workflow: Repository State & Diff Operations
# ==========================================


def test_get_git_diff_success_with_changes(setup_git_workspace):
    """Green Path: Returns the actual diff text when uncommitted changes exist."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # modify an already tracked file to generate a standard git diff
    test_file = safe_dir / "README.md"
    test_file.write_text("This is a new line of code.", encoding="utf-8")

    # 2. Execute
    result = get_git_diff(str(safe_dir))

    # 3. Assertions
    assert "diff --git" in result
    assert "+This is a new line of code." in result


def test_get_git_diff_success_with_target_branch(setup_git_workspace):
    """Green Path: Compares the working tree against a specific target branch."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # create a new branch and modify a file
    repo.create_head("feature-branch").checkout()
    test_file = safe_dir / "README.md"
    test_file.write_text("Feature content.", encoding="utf-8")
    repo.index.add([str(test_file)])
    repo.index.commit("Feature commit")

    # 2. Execute
    # diff feature-branch against main
    result = get_git_diff(str(safe_dir), target_branch="main")

    # 3. Assertions
    assert "diff --git" in result
    assert "+Feature content." in result


def test_get_git_diff_blueprint_success(setup_git_workspace):
    """Green Path: Returns a clean name-status blueprint of the repository changes."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # mimic the orchestrator: create and checkout a feature branch
    repo.create_head("feature-branch").checkout()

    # modify an existing file
    test_file = safe_dir / "README.md"
    test_file.write_text("Modified content.", encoding="utf-8")
    repo.index.add([str(test_file)])

    # create a new file
    new_file = safe_dir / "new_file.py"
    new_file.write_text("print('hello')", encoding="utf-8")
    repo.index.add([str(new_file)])

    # commit the changes to the feature branch
    repo.index.commit("Add new file and modify README")

    # 2. Execute
    # diff the feature branch against main, exactly as review_pr_node does
    result = get_git_diff_blueprint(str(safe_dir), target_branch="main")

    # 3. Assertions
    # verify the git status flags (M for Modified, A for Added)
    assert "M\tREADME.md" in result
    assert "A\tnew_file.py" in result


def test_get_git_diff_fallback_no_changes(setup_git_workspace):
    """Edge Path: Returns a clean message when the working tree is clean."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # 2. Execute
    result = get_git_diff(str(safe_dir))

    # 3. Assertions
    assert "No uncommitted changes" in result


def test_get_git_diff_error_not_a_repo(tmp_path, monkeypatch):
    """Red Path: Safely traps the execution if the directory is not a git repo."""
    # 1. Setup Mock Environment
    # use the built-in tmp_path to create a completely fresh, non-git directory
    not_a_repo_dir = tmp_path / "empty_dir"
    not_a_repo_dir.mkdir()

    # temporarily authorize this directory so the security check passes
    monkeypatch.setenv("ALLOWED_PATHS", str(not_a_repo_dir))

    # 2. Execute
    result = get_git_diff(str(not_a_repo_dir))

    # 3. Assertions
    assert "Error: The specified directory is not a valid git repository" in result


def test_get_git_diff_blueprint_error_git_failure(setup_git_workspace):
    """Red Path: Ensure GitCommandError is safely converted to a string in blueprint extraction."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # 2. Execute
    # Attempting to diff against a nonexistent branch will raise GitCommandError
    result = get_git_diff_blueprint(str(safe_dir), target_branch="non-existent-branch")

    # 3. Assertions
    assert "Git error:" in result
    assert "non-existent-branch" in result


def test_get_git_diff_blueprint_error_not_a_repo(tmp_path, monkeypatch):
    """Red Path: Safely traps execution if directory is not a git repo for blueprint extraction."""
    # 1. Setup Mock Environment
    # use the built-in tmp_path to create a completely fresh, non-git directory
    not_a_repo_dir = tmp_path / "empty_dir"
    not_a_repo_dir.mkdir()
    monkeypatch.setenv("ALLOWED_PATHS", str(not_a_repo_dir))

    # 2. Execute
    result = get_git_diff_blueprint(str(not_a_repo_dir))

    # 3. Assertions
    assert "Error: The specified directory is not a valid git repository" in result


# ==========================================
# Workflow: Apply Git Patch
# ==========================================


def test_apply_git_patch_success(setup_git_workspace):
    """Green Path: Successfully applies a clean, valid unified diff patch."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace
    test_file = safe_dir / "README.md"

    # Generate a perfectly valid patch dynamically using Git
    test_file.write_text("# Patched Repository", encoding="utf-8")
    patch_content = repo.git.diff()

    # Revert the working tree so the file goes back to "# Initial Repository"
    repo.git.checkout("--", str(test_file))

    # 2. Execute
    # Apply the patch using the agent tool
    result = apply_git_patch(str(safe_dir), patch_content)

    # 3. Assertions
    assert "Success: Patch applied cleanly." in result
    assert test_file.read_text(encoding="utf-8") == "# Patched Repository"


def test_apply_git_patch_error_malformed_patch(setup_git_workspace):
    """Red Path: Catches GitCommandError when attempting to apply garbage data."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace

    # 2. Execute
    # Send plain text instead of a valid unified diff patch
    result = apply_git_patch(str(safe_dir), "This is definitely not a git patch.")

    # 3. Assertions
    assert "Error applying patch:" in result
    # Git usually complains about unrecognized input
    assert "unrecognized input" in result.lower() or "error" in result.lower()


def test_apply_git_patch_error_not_a_repo(tmp_path, monkeypatch):
    """Red Path: Gracefully fails if the target directory is not a git repository."""
    # 1. Setup Mock Environment
    not_a_repo_dir = tmp_path / "empty_dir"
    not_a_repo_dir.mkdir()

    # Temporarily authorize this directory so the security check passes
    monkeypatch.setenv("ALLOWED_PATHS", str(not_a_repo_dir))

    # 2. Execute
    result = apply_git_patch(str(not_a_repo_dir), "fake patch content")

    # 3. Assertions
    assert "Error: The specified directory is not a valid git repository." in result


def test_apply_git_patch_error_security_exception(setup_git_workspace):
    """Red Path: Ensures path traversal and unauthorized paths are securely blocked."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace

    # Try to apply a patch to a path explicitly outside the ALLOWED_PATHS safe_dir
    unauthorized_dir = safe_dir.parent / "secret_folder"

    # 2. Execute
    result = apply_git_patch(str(unauthorized_dir), "fake patch content")

    # 3. Assertions
    # The secure_resolve_path function should intercept this and raise a PermissionError
    assert "Security Exception" in result


# ==========================================
# Workflow: Branch & Commit Operations
# ==========================================


def test_create_branch_and_commit_success_new_branch(setup_git_workspace):
    """Green Path: Simulates staging files and committing to a new branch."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # modify the workspace
    test_file = safe_dir / "README.md"
    test_file.write_text("# Updated Repository", encoding="utf-8")

    # 2. Execute
    result_json = create_branch_and_commit(str(safe_dir), "feature-branch", "Update README")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["branch"] == "feature-branch"
    assert repo.active_branch.name == "feature-branch"


def test_create_branch_and_commit_success_existing_branch(setup_git_workspace):
    """Green Path: Simulates staging files and committing to an ALREADY EXISTING branch."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # create the branch beforehand to simulate an existing PR workflow
    repo.create_head("existing-feature")

    # modify the workspace
    test_file = safe_dir / "README.md"
    test_file.write_text("# Updated Existing Branch", encoding="utf-8")

    # 2. Execute
    result_json = create_branch_and_commit(str(safe_dir), "existing-feature", "Update README")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["branch"] == "existing-feature"
    assert repo.active_branch.name == "existing-feature"


def test_create_branch_and_commit_error_no_changes(setup_git_workspace):
    """Red Path: The agent attempts to commit without modifying any files."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace
    original_branch = repo.active_branch.name

    # 2. Execute
    result_json = create_branch_and_commit(str(safe_dir), "phantom-branch", "Empty commit")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "no_changes_to_commit"

    # Ensure the repo was reverted to the original branch
    assert repo.active_branch.name == original_branch
    # Ensure the empty branch was strictly deleted to prevent local git pollution
    assert "phantom-branch" not in repo.heads


def test_create_branch_and_commit_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: The agent attempts to push a branch without an authentication token."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = create_branch_and_commit(str(safe_dir), "new-branch", "Commit text")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_create_branch_and_commit_error_injection_prevention(setup_git_workspace):
    """Red Path: Ensures branch names resembling CLI flags are strictly rejected."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace

    # 2. Execute
    # Simulating a branch name designed to trigger an arbitrary file overwrite via Git
    result_json = create_branch_and_commit(str(safe_dir), "--upload-pack=evil.sh", "test")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert "Invalid branch name" in result["details"]
    assert "Cannot start with a hyphen" in result["details"]


# ==========================================
# Workflow: Open Pull Request Operations
# ==========================================


def test_open_pull_request_success(setup_git_workspace, monkeypatch, mocker):
    """
    Green Path: Mocks the httpx POST request to simulate a successful
    201 Created response from the GitHub API and verifies the URL extraction.
    """
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # mock the httpx client and its post method
    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    # simulate a successful 201 Created response from GitHub
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "id": 12345,
        "html_url": "https://github.com/owner/repo/pull/1",
        "state": "open",
    }
    mock_client_instance.post.return_value = mock_response

    # patch the Client context manager
    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = open_pull_request(str(safe_dir), "Add KAN-TabNet Evaluation", "feature-eval")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["pr_url"] == "https://github.com/owner/repo/pull/1"

    # verify the payload was sent to the correct endpoint
    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/pulls"
    assert kwargs["json"]["title"] == "Add KAN-TabNet Evaluation"


def test_open_pull_request_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is not in the environment."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = open_pull_request(str(safe_dir), "Test PR", "feature")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_open_pull_request_error_api_failure(setup_git_workspace, monkeypatch, mocker):
    """
    Red Path: Gracefully handles an error response (e.g., 422 Unprocessable Entity) from GitHub.
    """
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 422
    mock_response.text = "Validation Failed: Branch does not exist"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = open_pull_request(str(safe_dir), "Test PR", "feature-branch")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 422
    assert "Validation Failed" in result["details"]


def test_open_pull_request_error_unexpected_exception(setup_git_workspace, monkeypatch, mocker):
    """Red Path: Gracefully handles unexpected exceptions (like catastrophic network timeouts)."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # Mock the httpx Client context manager entry to instantly raise an Exception
    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Catastrophic Network Timeout"),
    )

    # 2. Execute
    result_json = open_pull_request(str(safe_dir), "Test PR", "feature-branch")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Catastrophic Network Timeout" in result["details"]


# ==========================================
# Workflow: Update Pull Request Operations
# ==========================================


def test_update_pull_request_success(setup_git_workspace, monkeypatch, mocker):
    """Green Path: Simulates a successful PATCH request to update an existing PR."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_response = mocker.MagicMock()
    mock_response.status_code = 200  # httpx.codes.OK
    mock_response.json.return_value = {"html_url": "https://github.com/owner/repo/pull/123"}

    mocker.patch("src.workspace_agent.tools.github.httpx.patch", return_value=mock_response)

    # 2. Execute
    result_json = update_pull_request(str(safe_dir), 123, title="New Title", body="New Body")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["pr_url"] == "https://github.com/owner/repo/pull/123"


def test_update_pull_request_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing during an update."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = update_pull_request(str(safe_dir), 123, title="New Title")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "Missing GITHUB_TOKEN environment variable."


def test_update_pull_request_error_no_payload(setup_git_workspace, monkeypatch):
    """Red Path: Fails gracefully if the orchestrator tries to update without a title or body."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # 2. Execute
    result_json = update_pull_request(str(safe_dir), 123)
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert "No title or body provided" in result["reason"]


def test_update_pull_request_error_api_failure(setup_git_workspace, monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection during an update (e.g., 422 Unprocessable Entity)."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_response = mocker.MagicMock()
    mock_response.status_code = 422
    mock_response.text = "Validation Failed"
    mocker.patch("src.workspace_agent.tools.github.httpx.patch", return_value=mock_response)

    # 2. Execute
    result_json = update_pull_request(str(safe_dir), 123, title="Bad Title")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "GitHub API HTTP 422"
    assert "Validation Failed" in result["details"]


# ==========================================
# Workflow: PR Comment Operations
# ==========================================


def test_comment_on_pull_request_success(setup_git_workspace, monkeypatch, mocker):
    """Green Path: Simulates a successful POST request to add a PR comment."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 201
    mock_response.json.return_value = {
        "html_url": "https://github.com/owner/repo/issues/1/comments/123"
    }
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = comment_on_pull_request(str(safe_dir), 1, "Agent evaluation completed.")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["comment_url"] == "https://github.com/owner/repo/issues/1/comments/123"

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/issues/1/comments"
    assert kwargs["json"]["body"] == "Agent evaluation completed."


def test_comment_on_pull_request_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = comment_on_pull_request(str(safe_dir), 1, "Test comment")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_comment_on_pull_request_error_api_failure(setup_git_workspace, monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 403
    mock_response.text = "Forbidden"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = comment_on_pull_request(str(safe_dir), 1, "Test comment")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 403
    assert "Forbidden" in result["details"]


def test_comment_on_pull_request_error_unexpected_exception(
    setup_git_workspace, monkeypatch, mocker
):
    """Red Path: Handles unexpected exceptions during API call."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Network failure"),
    )

    # 2. Execute
    result_json = comment_on_pull_request(str(safe_dir), 1, "Test comment")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Network failure" in result["details"]


# ==========================================
# Workflow: Commit Status Operations
# ==========================================


def test_set_commit_status_success(setup_git_workspace, monkeypatch, mocker):
    """Green Path: Simulates a successful POST request to update a commit status."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 201
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = set_commit_status(
        str(safe_dir), "sha12345", "success", "Agentic CI", "Tests passed"
    )
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "success"
    assert result["state"] == "success"
    assert result["context"] == "Agentic CI"

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/statuses/sha12345"
    assert kwargs["json"]["state"] == "success"
    assert kwargs["json"]["context"] == "Agentic CI"
    assert kwargs["json"]["description"] == "Tests passed"


def test_set_commit_status_error_invalid_state(setup_git_workspace, monkeypatch):
    """Red Path: Rejects an invalid commit status state locally before calling the API."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # 2. Execute
    result_json = set_commit_status(str(safe_dir), "sha12345", "invalid_state", "Context")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "invalid_status_state"
    assert "invalid_state" in result["details"]


def test_set_commit_status_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # 2. Execute
    result_json = set_commit_status(str(safe_dir), "sha12345", "success", "Context")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_set_commit_status_error_api_failure(setup_git_workspace, monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    # 2. Execute
    result_json = set_commit_status(str(safe_dir), "sha12345", "success", "Context")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 404
    assert "Not Found" in result["details"]


def test_set_commit_status_error_unexpected_exception(setup_git_workspace, monkeypatch, mocker):
    """Red Path: Handles unexpected exceptions during API call."""
    # 1. Setup Mock Environment
    safe_dir, _ = setup_git_workspace
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Timeout"),
    )

    # 2. Execute
    result_json = set_commit_status(str(safe_dir), "sha12345", "success", "Context")
    result = json.loads(result_json)

    # 3. Assertions
    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Timeout" in result["details"]


# ==========================================
# Workflow: Internal Hygiene & Cleanup
# ==========================================


def test_cleanup_local_branch_success(setup_git_workspace):
    """Green Path: Verifies local branch cleanup safely auto-stashes and deletes."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # create a branch and check it out
    new_head = repo.create_head("agent/test-branch")
    new_head.checkout()

    # create a dirty file to test the auto-stash hygiene feature
    dirty_file = safe_dir / "dirty_script.py"
    dirty_file.write_text("print('dirty')", encoding="utf-8")

    # 2. Execute
    # execute cleanup
    cleanup_local_branch(str(safe_dir), "main", "agent/test-branch")

    # 3. Assertions
    # verify repository was physically switched back to main
    assert repo.active_branch.name == "main"

    # verify the dirty file was safely stashed and removed from working tree
    assert not dirty_file.exists()

    # verify the temporary branch was completely deleted
    assert "agent/test-branch" not in repo.heads


def test_cleanup_local_branch_fallback_already_deleted(setup_git_workspace):
    """Edge Path: Ensures the cleanup utility does not crash if the branch was already deleted."""
    # 1. Setup Mock Environment
    safe_dir, repo = setup_git_workspace

    # ensure the branch does not exist
    assert "agent/phantom-branch" not in repo.heads

    # 2. Execute
    # invoke cleanup
    cleanup_local_branch(str(safe_dir), "main", "agent/phantom-branch")

    # 3. Assertions
    # verify repository remains stable
    assert repo.active_branch.name == "main"


# ==========================================
# Workflow: Git Remote Resolution
# ==========================================


def test_get_target_remote_global_default(mocker):
    """Green Path: Returns the global target_remote when no workspace matches."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.tools.github.settings.agent.target_remote", "origin")
    mocker.patch("src.workspace_agent.tools.github.settings.workspaces", {})

    mocker.patch(
        "src.workspace_agent.tools.github.secure_resolve_path", return_value=Path("/tmp/some_repo")
    )
    mocker.patch("src.workspace_agent.tools.github.get_allowed_paths", return_value=["/tmp"])

    # 2. Execute
    remote = _get_target_remote("/tmp/some_repo")

    # 3. Assertions
    assert remote == "origin"


def test_get_target_remote_workspace_override(mocker):
    """Green Path: Returns a workspace-specific target_remote override when matched."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.tools.github.settings.agent.target_remote", "origin")

    # Mock a workspace config with a specific remote override
    mock_ws = mocker.Mock()
    mock_ws.path = "/Users/username/git/example-project"
    mock_ws.target_remote = "upstream"

    mocker.patch("src.workspace_agent.tools.github.settings.workspaces", {"example": mock_ws})
    mocker.patch(
        "src.workspace_agent.tools.github.secure_resolve_path",
        # Simulate testing a directory deeply nested inside the workspace
        return_value=Path("/Users/username/git/example-project/src/nested"),
    )
    mocker.patch(
        "src.workspace_agent.tools.github.get_allowed_paths", return_value=["/Users/username/git"]
    )

    # 2. Execute
    remote = _get_target_remote("/Users/username/git/example-project/src/nested")

    # 3. Assertions
    assert remote == "upstream"


def test_get_target_remote_fallback_on_error(mocker):
    """Edge Path: Gracefully falls back to the global default if path resolution crashes."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.tools.github.settings.agent.target_remote", "origin")

    # Force the security resolution to crash
    mocker.patch(
        "src.workspace_agent.tools.github.get_allowed_paths",
        side_effect=Exception("Simulated Security Exception"),
    )

    # 2. Execute
    remote = _get_target_remote("/tmp/restricted_dir")

    # 3. Assertions
    # It must trap the exception and safely return the fallback
    assert remote == "origin"
