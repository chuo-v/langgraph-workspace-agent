import json

import git
import pytest

from src.workspace_agent.tools.github import (
    cleanup_local_branch,
    comment_on_pull_request,
    create_branch_and_commit,
    get_git_diff,
    get_git_diff_blueprint,
    open_pull_request,
    set_commit_status,
    sync_repository,
    update_pull_request,
)

# ==========================================
# Helper: Test Fixtures
# ==========================================


@pytest.fixture
def setup_git_workspace(tmp_path, monkeypatch, mocker):
    """
    Creates a temporary safe workspace, initializes a real Git repository,
    and sets up isolated mocks so production GitHub tools can execute locally.
    """
    safe_dir = tmp_path / "git"
    safe_dir.mkdir()
    monkeypatch.setenv("ALLOWED_PATHS", str(safe_dir))

    # inject the dummy token so production security checks pass
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # initialize a local git repository
    repo = git.Repo.init(safe_dir)
    test_file = safe_dir / "README.md"
    test_file.write_text("# Initial Repository", encoding="utf-8")

    repo.index.add([str(test_file)])
    repo.index.commit("Initial commit")

    # explicitly rename the default branch to 'main' to prevent CI failures
    # caused by legacy 'master' defaults in different environments
    repo.git.branch("-M", "main")

    # create a local bare repository to act as 'origin' for ALL tests
    remote_dir = safe_dir.parent / "remote.git"
    git.Repo.init(remote_dir, bare=True)
    origin = repo.create_remote("origin", str(remote_dir))
    origin.push("main")

    # mock the regex search so github.py accepts our local bare repo as a valid URL
    mock_match = mocker.MagicMock()
    mock_match.group.return_value = "owner/repo"
    mocker.patch("src.workspace_agent.tools.github.re.search", return_value=mock_match)

    # mock set_url so production code doesn't overwrite our local bare repo route
    mocker.patch("git.remote.Remote.set_url")

    return safe_dir, repo


# ==========================================
# Workflow: Sync Repository Operations
# ==========================================


def test_sync_repository_success(setup_git_workspace):
    """Green Path: The agent successfully checks out a branch and pulls changes."""
    safe_dir, repo = setup_git_workspace

    result_json = sync_repository(str(safe_dir), "main")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["branch"] == "main"
    assert repo.active_branch.name == "main"


def test_sync_repository_fallback_dirty_tree_auto_stash(setup_git_workspace):
    """Edge Path: The agent encounters a dirty tree and successfully auto-stashes."""
    safe_dir, repo = setup_git_workspace

    # create an untracked file to make the tree "dirty"
    untracked_file = safe_dir / "new_script.py"
    untracked_file.write_text("print('hello')", encoding="utf-8")

    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    # sync should now succeed instead of failing
    assert result["status"] == "success"

    # working directory should be completely clean
    assert not repo.is_dirty(untracked_files=True)

    # untracked file should be safely in the stash, removed from the filesystem
    assert not untracked_file.exists()


def test_sync_repository_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: Ensure sync_repository aborts cleanly if the GITHUB_TOKEN is missing."""
    safe_dir, repo = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_sync_repository_error_invalid_url(setup_git_workspace, mocker):
    """Red Path: Ensure sync_repository aborts cleanly if the remote URL format is unrecognized."""
    safe_dir, repo = setup_git_workspace

    # mock re.search to return None, simulating a non-standard Git remote URL
    mocker.patch("src.workspace_agent.tools.github.re.search", return_value=None)

    result_json = sync_repository(str(safe_dir))
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "invalid_github_url_format"


def test_sync_repository_error_git_command(setup_git_workspace):
    """Red Path: Ensure GitCommandError is safely converted to a JSON error response."""
    safe_dir, repo = setup_git_workspace

    # trigger a natural GitCommandError by attempting to sync a branch that does not exist
    result_json = sync_repository(str(safe_dir), target_branch="non-existent-branch-name")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "git_command_failed"
    # the actual git stderr details will mention the branch it failed to match
    assert "non-existent-branch-name" in result["details"]


# ==========================================
# Workflow: Repository State & Diff Operations
# ==========================================


def test_get_git_diff_success_with_changes(setup_git_workspace):
    """Green Path: Returns the actual diff text when uncommitted changes exist."""
    safe_dir, repo = setup_git_workspace

    # modify an already tracked file to generate a standard git diff
    test_file = safe_dir / "README.md"
    test_file.write_text("This is a new line of code.", encoding="utf-8")

    result = get_git_diff(str(safe_dir))

    assert "diff --git" in result
    assert "+This is a new line of code." in result


def test_get_git_diff_success_with_target_branch(setup_git_workspace):
    """Green Path: Compares the working tree against a specific target branch."""
    safe_dir, repo = setup_git_workspace

    # create a new branch and modify a file
    repo.create_head("feature-branch").checkout()
    test_file = safe_dir / "README.md"
    test_file.write_text("Feature content.", encoding="utf-8")
    repo.index.add([str(test_file)])
    repo.index.commit("Feature commit")

    # diff feature-branch against main
    result = get_git_diff(str(safe_dir), target_branch="main")

    assert "diff --git" in result
    assert "+Feature content." in result


def test_get_git_diff_blueprint_success(setup_git_workspace):
    """Green Path: Returns a clean name-status blueprint of the repository changes."""
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

    # diff the feature branch against main, exactly as review_pr_node does
    result = get_git_diff_blueprint(str(safe_dir), target_branch="main")

    # verify the git status flags (M for Modified, A for Added)
    assert "M\tREADME.md" in result
    assert "A\tnew_file.py" in result


def test_get_git_diff_fallback_no_changes(setup_git_workspace):
    """Edge Path: Returns a clean message when the working tree is clean."""
    safe_dir, repo = setup_git_workspace

    result = get_git_diff(str(safe_dir))

    assert "No uncommitted changes" in result


def test_get_git_diff_error_not_a_repo(tmp_path, monkeypatch):
    """Red Path: Safely traps the execution if the directory is not a git repo."""
    # use the built-in tmp_path to create a completely fresh, non-git directory
    not_a_repo_dir = tmp_path / "empty_dir"
    not_a_repo_dir.mkdir()

    # temporarily authorize this directory so the security check passes
    monkeypatch.setenv("ALLOWED_PATHS", str(not_a_repo_dir))

    result = get_git_diff(str(not_a_repo_dir))

    assert "Error: The specified directory is not a valid git repository" in result


def test_get_git_diff_blueprint_error_git_failure(setup_git_workspace):
    """Red Path: Ensure GitCommandError is safely converted to a string in blueprint extraction."""
    safe_dir, repo = setup_git_workspace

    # Attempting to diff against a nonexistent branch will raise GitCommandError
    result = get_git_diff_blueprint(str(safe_dir), target_branch="non-existent-branch")

    assert "Git error:" in result
    assert "non-existent-branch" in result


def test_get_git_diff_blueprint_error_not_a_repo(tmp_path, monkeypatch):
    """Red Path: Safely traps execution if directory is not a git repo for blueprint extraction."""
    # use the built-in tmp_path to create a completely fresh, non-git directory
    not_a_repo_dir = tmp_path / "empty_dir"
    not_a_repo_dir.mkdir()
    monkeypatch.setenv("ALLOWED_PATHS", str(not_a_repo_dir))

    result = get_git_diff_blueprint(str(not_a_repo_dir))

    assert "Error: The specified directory is not a valid git repository" in result


# ==========================================
# Workflow: Branch & Commit Operations
# ==========================================


def test_create_branch_and_commit_success_new_branch(setup_git_workspace):
    """Green Path: Simulates staging files and committing to a new branch."""
    safe_dir, repo = setup_git_workspace

    # modify the workspace
    test_file = safe_dir / "README.md"
    test_file.write_text("# Updated Repository", encoding="utf-8")

    result_json = create_branch_and_commit(str(safe_dir), "feature-branch", "Update README")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["branch"] == "feature-branch"
    assert repo.active_branch.name == "feature-branch"


def test_create_branch_and_commit_success_existing_branch(setup_git_workspace):
    """Green Path: Simulates staging files and committing to an ALREADY EXISTING branch."""
    safe_dir, repo = setup_git_workspace

    # create the branch beforehand to simulate an existing PR workflow
    repo.create_head("existing-feature")

    # modify the workspace
    test_file = safe_dir / "README.md"
    test_file.write_text("# Updated Existing Branch", encoding="utf-8")

    result_json = create_branch_and_commit(str(safe_dir), "existing-feature", "Update README")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["branch"] == "existing-feature"
    assert repo.active_branch.name == "existing-feature"


def test_create_branch_and_commit_error_no_changes(setup_git_workspace):
    """Red Path: The agent attempts to commit without modifying any files."""
    safe_dir, repo = setup_git_workspace
    original_branch = repo.active_branch.name

    result_json = create_branch_and_commit(str(safe_dir), "phantom-branch", "Empty commit")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "no_changes_to_commit"

    # Ensure the repo was reverted to the original branch
    assert repo.active_branch.name == original_branch
    # Ensure the empty branch was strictly deleted to prevent local git pollution
    assert "phantom-branch" not in repo.heads


def test_create_branch_and_commit_error_missing_token(setup_git_workspace, monkeypatch):
    """Red Path: The agent attempts to push a branch without an authentication token."""
    safe_dir, _ = setup_git_workspace
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = create_branch_and_commit(str(safe_dir), "new-branch", "Commit text")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_create_branch_and_commit_error_injection_prevention(setup_git_workspace):
    """Red Path: Ensures branch names resembling CLI flags are strictly rejected."""
    safe_dir, _ = setup_git_workspace

    # Simulating a branch name designed to trigger an arbitrary file overwrite via Git
    result_json = create_branch_and_commit(str(safe_dir), "--upload-pack=evil.sh", "test")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert "Invalid branch name" in result["details"]
    assert "Cannot start with a hyphen" in result["details"]


# ==========================================
# Workflow: Open Pull Request Operations
# ==========================================


def test_open_pull_request_success(monkeypatch, mocker):
    """
    Green Path: Mocks the httpx POST request to simulate a successful
    201 Created response from the GitHub API and verifies the URL extraction.
    """
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

    result_json = open_pull_request("owner/repo", "Add KAN-TabNet Evaluation", "feature-eval")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["pr_url"] == "https://github.com/owner/repo/pull/1"

    # verify the payload was sent to the correct endpoint
    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/pulls"
    assert kwargs["json"]["title"] == "Add KAN-TabNet Evaluation"


def test_open_pull_request_error_missing_token(monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is not in the environment."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = open_pull_request("owner/repo", "Test PR", "feature")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_open_pull_request_error_api_failure(monkeypatch, mocker):
    """
    Red Path: Gracefully handles an error response (e.g., 422 Unprocessable Entity) from GitHub.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 422
    mock_response.text = "Validation Failed: Branch does not exist"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    result_json = open_pull_request("owner/repo", "Test PR", "feature-branch")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 422
    assert "Validation Failed" in result["details"]


def test_open_pull_request_error_unexpected_exception(monkeypatch, mocker):
    """Red Path: Gracefully handles unexpected exceptions (like catastrophic network timeouts)."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    # Mock the httpx Client context manager entry to instantly raise an Exception
    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Catastrophic Network Timeout"),
    )

    result_json = open_pull_request("owner/repo", "Test PR", "feature-branch")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Catastrophic Network Timeout" in result["details"]


# ==========================================
# Workflow: Update Pull Request Operations
# ==========================================


def test_update_pull_request_success(monkeypatch, mocker):
    """Green Path: Simulates a successful PATCH request to update an existing PR."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_response = mocker.MagicMock()
    mock_response.status_code = 200  # httpx.codes.OK
    mock_response.json.return_value = {"html_url": "https://github.com/owner/repo/pull/123"}

    mocker.patch("src.workspace_agent.tools.github.httpx.patch", return_value=mock_response)

    result_json = update_pull_request("owner/repo", 123, title="New Title", body="New Body")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["pr_url"] == "https://github.com/owner/repo/pull/123"


def test_update_pull_request_error_missing_token(monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing during an update."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = update_pull_request("owner/repo", 123, title="New Title")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "Missing GITHUB_TOKEN environment variable."


def test_update_pull_request_error_no_payload(monkeypatch):
    """Red Path: Fails gracefully if the orchestrator tries to update without a title or body."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    result_json = update_pull_request("owner/repo", 123)
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert "No title or body provided" in result["reason"]


def test_update_pull_request_error_api_failure(monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection during an update (e.g., 422 Unprocessable Entity)."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_response = mocker.MagicMock()
    mock_response.status_code = 422
    mock_response.text = "Validation Failed"
    mocker.patch("src.workspace_agent.tools.github.httpx.patch", return_value=mock_response)

    result_json = update_pull_request("owner/repo", 123, title="Bad Title")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "GitHub API HTTP 422"
    assert "Validation Failed" in result["details"]


# ==========================================
# Workflow: PR Comment Operations
# ==========================================


def test_comment_on_pull_request_success(monkeypatch, mocker):
    """Green Path: Simulates a successful POST request to add a PR comment."""
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

    result_json = comment_on_pull_request("owner/repo", 1, "Agent evaluation completed.")
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["comment_url"] == "https://github.com/owner/repo/issues/1/comments/123"

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/issues/1/comments"
    assert kwargs["json"]["body"] == "Agent evaluation completed."


def test_comment_on_pull_request_error_missing_token(monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = comment_on_pull_request("owner/repo", 1, "Test comment")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_comment_on_pull_request_error_api_failure(monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 403
    mock_response.text = "Forbidden"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    result_json = comment_on_pull_request("owner/repo", 1, "Test comment")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 403
    assert "Forbidden" in result["details"]


def test_comment_on_pull_request_error_unexpected_exception(monkeypatch, mocker):
    """Red Path: Handles unexpected exceptions during API call."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Network failure"),
    )

    result_json = comment_on_pull_request("owner/repo", 1, "Test comment")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Network failure" in result["details"]


# ==========================================
# Workflow: Commit Status Operations
# ==========================================


def test_set_commit_status_success(monkeypatch, mocker):
    """Green Path: Simulates a successful POST request to update a commit status."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 201
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    result_json = set_commit_status(
        "owner/repo", "sha12345", "success", "Agentic CI", "Tests passed"
    )
    result = json.loads(result_json)

    assert result["status"] == "success"
    assert result["state"] == "success"
    assert result["context"] == "Agentic CI"

    mock_client_instance.post.assert_called_once()
    args, kwargs = mock_client_instance.post.call_args
    assert args[0] == "https://api.github.com/repos/owner/repo/statuses/sha12345"
    assert kwargs["json"]["state"] == "success"
    assert kwargs["json"]["context"] == "Agentic CI"
    assert kwargs["json"]["description"] == "Tests passed"


def test_set_commit_status_error_invalid_state(monkeypatch):
    """Red Path: Rejects an invalid commit status state locally before calling the API."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    result_json = set_commit_status("owner/repo", "sha12345", "invalid_state", "Context")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "invalid_status_state"
    assert "invalid_state" in result["details"]


def test_set_commit_status_error_missing_token(monkeypatch):
    """Red Path: Fails gracefully if GITHUB_TOKEN is missing."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result_json = set_commit_status("owner/repo", "sha12345", "success", "Context")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "missing_github_token"


def test_set_commit_status_error_api_failure(monkeypatch, mocker):
    """Red Path: Handles GitHub API rejection."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mock_client_instance = mocker.MagicMock()
    mock_response = mocker.MagicMock()

    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_client_instance.post.return_value = mock_response

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__", return_value=mock_client_instance
    )

    result_json = set_commit_status("owner/repo", "sha12345", "success", "Context")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "api_request_failed"
    assert result["status_code"] == 404
    assert "Not Found" in result["details"]


def test_set_commit_status_error_unexpected_exception(monkeypatch, mocker):
    """Red Path: Handles unexpected exceptions during API call."""
    monkeypatch.setenv("GITHUB_TOKEN", "mock_secure_token_123")

    mocker.patch(
        "src.workspace_agent.tools.github.httpx.Client.__enter__",
        side_effect=Exception("Timeout"),
    )

    result_json = set_commit_status("owner/repo", "sha12345", "success", "Context")
    result = json.loads(result_json)

    assert result["status"] == "error"
    assert result["reason"] == "unexpected_error"
    assert "Timeout" in result["details"]


# ==========================================
# Workflow: Internal Hygiene & Cleanup
# ==========================================


def test_cleanup_local_branch_success(setup_git_workspace):
    """Green Path: Verifies local branch cleanup safely auto-stashes and deletes."""
    safe_dir, repo = setup_git_workspace

    # create a branch and check it out
    new_head = repo.create_head("agent/test-branch")
    new_head.checkout()

    # create a dirty file to test the auto-stash hygiene feature
    dirty_file = safe_dir / "dirty_script.py"
    dirty_file.write_text("print('dirty')", encoding="utf-8")

    # execute cleanup
    cleanup_local_branch(str(safe_dir), "main", "agent/test-branch")

    # verify repository was physically switched back to main
    assert repo.active_branch.name == "main"

    # verify the dirty file was safely stashed and removed from working tree
    assert not dirty_file.exists()

    # verify the temporary branch was completely deleted
    assert "agent/test-branch" not in repo.heads


def test_cleanup_local_branch_fallback_already_deleted(setup_git_workspace):
    """Edge Path: Ensures the cleanup utility does not crash if the branch was already deleted."""
    safe_dir, repo = setup_git_workspace

    # ensure the branch does not exist
    assert "agent/phantom-branch" not in repo.heads

    # invoke cleanup
    cleanup_local_branch(str(safe_dir), "main", "agent/phantom-branch")

    # verify repository remains stable
    assert repo.active_branch.name == "main"
