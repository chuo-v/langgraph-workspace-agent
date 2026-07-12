import json
import os
import re
import tempfile
from pathlib import Path

import httpx
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.workspace_agent.core.config import settings
from src.workspace_agent.tools.filesystem import get_allowed_paths, secure_resolve_path

# ==========================================
# Module Configuration & Constants
# ==========================================

__all__ = [
    "sync_repository",
    "sync_to_commit",
    "create_branch_and_commit",
    "cleanup_local_branch",
    "get_git_diff",
    "get_git_diff_blueprint",
    "is_diff_empty",
    "apply_git_patch",
    "open_pull_request",
    "update_pull_request",
    "comment_on_pull_request",
    "set_commit_status",
]

_GIT_CREDENTIAL_HELPER = (
    '!f() { test "$1" = get && echo "username=x-access-token" '
    '&& echo "password=$GITHUB_TOKEN"; }; f'
)


# ==========================================
# Shared Validation & Parsing Helpers
# ==========================================


def _sanitize_branch_name(branch_name: str | None) -> None:
    """Prevents Git Argument Injection by rejecting flags."""
    if branch_name and str(branch_name).strip().startswith("-"):
        raise ValueError(
            f"Security Exception: Invalid branch name '{branch_name}'. Cannot start with a hyphen."
        )


def _get_repo_full_name(directory: str) -> str | None:
    """Extracts the 'owner/repo' string dynamically from the local git remote."""
    try:
        allowed = get_allowed_paths()
        repo_path = secure_resolve_path(directory, allowed)
        repo = Repo(repo_path)

        remote_name = _get_target_remote(directory)
        current_url = next(repo.remote(name=remote_name).urls)

        match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def _get_target_remote(directory: str) -> str:
    """Helper to determine the target remote for a given directory, checking workspace overrides."""
    try:
        allowed = get_allowed_paths()
        safe_path = secure_resolve_path(directory, allowed)

        longest_match = None
        matched_remote = None

        for ws_config in settings.workspaces.values():
            ws_path = Path(ws_config.path).resolve()
            if safe_path.is_relative_to(ws_path):
                if not longest_match or len(ws_path.parts) > len(longest_match.parts):
                    longest_match = ws_path
                    matched_remote = ws_config.target_remote

        return matched_remote or settings.agent.target_remote
    except Exception:
        # Fallback to the global default if path resolution fails
        return settings.agent.target_remote


# ==========================================
# Git Repository & Workspace Tools
# ==========================================

# === Repository Synchronization ===


def sync_repository(directory: str, target_branch: str = "main") -> str:
    """
    Checks out the target branch and performs a fast-forward pull.
    Automatically stashes any dirty/untracked files to prevent data loss.
    """
    try:
        _sanitize_branch_name(target_branch)

        allowed = get_allowed_paths()
        repo_path = secure_resolve_path(directory, allowed)

        repo = Repo(repo_path)

        remote_name = _get_target_remote(directory)

        error_reason = None
        token = os.getenv("GITHUB_TOKEN")

        try:
            current_url = next(repo.remote(name=remote_name).urls)
            match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)

            if not token:
                error_reason = "missing_github_token"
            elif not match:
                error_reason = "invalid_github_url_format"
        except ValueError:
            error_reason = f"remote '{remote_name}' not found"

        if error_reason:
            return json.dumps({"status": "error", "reason": error_reason})

        if repo.is_dirty(untracked_files=True):
            repo.git.stash(
                "push", "--include-untracked", "-m", "Auto-stashed by Workspace Agent before sync"
            )

        # Suppress terminal prompts entirely, and use an inline credential helper guarded to only
        # answer 'get' requests
        with repo.git.custom_environment(
            GITHUB_TOKEN=token, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="echo"
        ):
            c_flags = [
                "-c",
                "credential.helper=",
                "-c",
                f"credential.helper={_GIT_CREDENTIAL_HELPER}",
            ]
            git_bin = repo.git.GIT_PYTHON_GIT_EXECUTABLE

            repo.git.checkout(target_branch)
            repo.git.execute(
                [git_bin] + c_flags + ["pull", "--ff-only", remote_name, target_branch]
            )

        return json.dumps({"status": "success", "branch": target_branch})

    except InvalidGitRepositoryError:
        return json.dumps({"status": "error", "reason": "not_a_git_repository"})
    except GitCommandError as e:
        return json.dumps({"status": "error", "reason": "git_command_failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


def sync_to_commit(
    directory: str, commit_sha: str | None = None, pr_number: int | None = None
) -> str:
    """
    Fetches from origin and checks out a specific commit or Pull Request safely.
    Used primarily by the CI node to guarantee the tests run against the correct code.
    """
    if commit_sha and not re.match(r"^[0-9a-fA-F]+$", commit_sha):
        return json.dumps({"status": "error", "reason": "invalid_commit_sha"})

    try:
        allowed = get_allowed_paths()
        repo_path = secure_resolve_path(directory, allowed)
        repo = Repo(repo_path)

        remote_name = _get_target_remote(directory)

        error_reason = None
        token = os.getenv("GITHUB_TOKEN")

        try:
            current_url = next(repo.remote(name=remote_name).urls)
            match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)

            if not token:
                error_reason = "missing_github_token"
            elif not match:
                error_reason = "invalid_github_url_format"
        except ValueError:
            error_reason = f"remote '{remote_name}' not found"

        if error_reason:
            return json.dumps({"status": "error", "reason": error_reason})

        # Stash any dirty agent files to prevent accidental loss
        if repo.is_dirty(untracked_files=True):
            repo.git.stash(
                "push",
                "--include-untracked",
                "-m",
                "Auto-stashed by Workspace Agent before CI sync",
            )

        with repo.git.custom_environment(
            GITHUB_TOKEN=token, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="echo"
        ):
            c_flags = [
                "-c",
                "credential.helper=",
                "-c",
                f"credential.helper={_GIT_CREDENTIAL_HELPER}",
            ]

            res = _perform_git_checkout(repo, commit_sha, pr_number, c_flags, remote_name)

        return json.dumps(res)

    except InvalidGitRepositoryError:
        return json.dumps({"status": "error", "reason": "not_a_git_repository"})
    except GitCommandError as e:
        return json.dumps({"status": "error", "reason": "git_command_failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


def _perform_git_checkout(
    repo: Repo, commit_sha: str | None, pr_number: int | None, c_flags: list[str], remote_name: str
) -> dict[str, str | None]:
    """Helper to isolate the git fetch and checkout branching logic."""
    git_bin = repo.git.GIT_PYTHON_GIT_EXECUTABLE

    # 1. Prefer fetching via pr_number if available (universally safe for Forks)
    if pr_number:
        temp_branch = f"agent/ci-pr-{pr_number}"

        repo.git.execute(
            [git_bin]
            + c_flags
            + ["fetch", remote_name, f"+refs/pull/{pr_number}/head:refs/heads/{temp_branch}"]
        )

        target_checkout = commit_sha if commit_sha else temp_branch
        repo.git.checkout(target_checkout)

        return {"status": "success", "commit": commit_sha or repo.head.commit.hexsha}

    # 2. Fallback to origin fetch for triggers that lack a PR number
    if commit_sha:
        repo.git.execute([git_bin] + c_flags + ["fetch", remote_name])
        repo.git.checkout(commit_sha)
        return {"status": "success", "commit": commit_sha}

    return {"status": "error", "reason": "missing_target"}


# === Branch & Commit Operations ===


def create_branch_and_commit(directory: str, new_branch: str, commit_message: str) -> str:
    """
    Creates a new branch, stages all changes, commits, and pushes to origin.
    Returns a structured JSON string detailing the result.
    """
    try:
        _sanitize_branch_name(new_branch)

        allowed = get_allowed_paths()
        repo_path = secure_resolve_path(directory, allowed)
        repo = Repo(repo_path)

        remote_name = _get_target_remote(directory)

        error_reason = None
        token = os.getenv("GITHUB_TOKEN")

        if not token:
            error_reason = "missing_github_token"
        else:
            try:
                current_url = next(repo.remote(name=remote_name).urls)
                match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)
                if not match:
                    error_reason = "invalid_github_url_format"
            except ValueError:
                error_reason = f"remote '{remote_name}' not found"

        if error_reason:
            return json.dumps({"status": "error", "reason": error_reason})

        original_branch = repo.active_branch.name

        # create or checkout the branch
        branch_existed = new_branch in repo.heads
        if branch_existed:
            new_head = repo.heads[new_branch]
        else:
            new_head = repo.create_head(new_branch)
        new_head.checkout()

        # stage all untracked and modified files
        repo.git.add(A=True)

        # check if there is actually anything to commit
        if not repo.index.diff("HEAD"):
            repo.git.checkout(original_branch)
            # only delete the branch if we just created it (don't delete existing PR branches)
            if not branch_existed:
                repo.delete_head(new_branch)
            return json.dumps({"status": "error", "reason": "no_changes_to_commit"})

        # commit the changes
        repo.index.commit(commit_message)

        # Inject token dynamically and apply configuration directly to the push command
        with repo.git.custom_environment(
            GITHUB_TOKEN=token, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="echo"
        ):
            c_flags = [
                "-c",
                "credential.helper=",
                "-c",
                f"credential.helper={_GIT_CREDENTIAL_HELPER}",
            ]
            git_bin = repo.git.GIT_PYTHON_GIT_EXECUTABLE

            remote_obj = repo.remote(name=remote_name)
            repo.git.execute(
                [git_bin] + c_flags + ["push", "--set-upstream", remote_obj.name, new_branch]
            )

        return json.dumps({"status": "success", "branch": new_branch, "message": commit_message})

    except GitCommandError as e:
        return json.dumps({"status": "error", "reason": "git_command_failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


def cleanup_local_branch(directory: str, target_branch: str, branch_to_delete: str) -> None:
    """
    Safely switches to the target branch and forcefully deletes the specified branch.
    Used internally by the orchestrator for state hygiene.
    """
    _sanitize_branch_name(target_branch)
    _sanitize_branch_name(branch_to_delete)

    allowed = get_allowed_paths()
    repo_path = secure_resolve_path(directory, allowed)
    repo = Repo(repo_path)

    if repo.is_dirty(untracked_files=True):
        repo.git.stash(
            "push", "--include-untracked", "-m", "Auto-stashed by Workspace Agent during cleanup"
        )

    # physically checkout the target branch
    repo.git.checkout(target_branch)

    # safely delete the branch only if it exists locally
    if branch_to_delete in repo.heads:
        repo.delete_head(branch_to_delete, force=True)


# === Diff & Patch Inspection ===


def get_git_diff(directory: str, target_branch: str | None = None) -> str:
    """
    Returns the git diff. If target_branch is provided, compares the working tree against it.
    Otherwise, returns only uncommitted changes.
    """
    try:
        _sanitize_branch_name(target_branch)

        repo = Repo(directory)

        # Check if the repository is bare
        if repo.bare:
            return "Error: The specified directory is not a valid git repository."

        # Stage all changes (including untracked files) so they appear in the diff
        repo.git.add(A=True)

        if target_branch:
            # Diff working tree against the target branch to capture the cumulative PR changes
            diff_output = repo.git.diff(target_branch, cached=True)
        else:
            # Diff against HEAD to see what was just added
            # Handle empty repo (initial commit) scenarios gracefully
            try:
                diff_output = repo.git.diff("HEAD", cached=True)
            except GitCommandError:
                diff_output = repo.git.diff(cached=True)

        if not diff_output.strip():
            return (
                "No uncommitted changes."
                if not target_branch
                else f"No changes compared to {target_branch}."
            )

        return diff_output

    except InvalidGitRepositoryError:
        return "Error: The specified directory is not a valid git repository."
    except GitCommandError as e:
        return f"Git error: {str(e)}"
    except Exception as e:
        return f"Error retrieving git diff: {e}"


def get_git_diff_blueprint(directory: str, target_branch: str | None = None) -> str:
    """
    Returns a dense blueprint of the PR topological impact (e.g., added, modified, deleted files)
    using git diff --name-status.
    """
    try:
        _sanitize_branch_name(target_branch)

        repo = Repo(directory)

        if repo.bare:
            return "Error: The specified directory is not a valid git repository."

        # Stage all changes to include untracked files in the blueprint
        repo.git.add(A=True)

        if target_branch:
            blueprint_output = repo.git.diff(target_branch, name_status=True, cached=True)
        else:
            try:
                blueprint_output = repo.git.diff("HEAD", name_status=True, cached=True)
            except GitCommandError:
                blueprint_output = repo.git.diff(name_status=True, cached=True)

        if not blueprint_output.strip():
            return "No files changed."

        return blueprint_output

    except InvalidGitRepositoryError:
        return "Error: The specified directory is not a valid git repository."
    except GitCommandError as e:
        return f"Git error: {str(e)}"
    except Exception as e:
        return f"Error retrieving git blueprint: {e}"


def is_diff_empty(diff_str: str | None) -> bool:
    """
    Determines if a git diff or blueprint string represents an empty diff,
    matching the specific artificial fallback strings generated by the git tools.
    """
    if not diff_str:
        return True

    clean_diff = diff_str.strip()
    return (
        clean_diff == "No uncommitted changes."
        or clean_diff.startswith("No changes compared to")
        or clean_diff == "No files changed."
    )


def apply_git_patch(directory: str, patch_content: str) -> str:
    """
    Applies a standard unified diff patch file directly to the workspace.
    """
    try:
        allowed = get_allowed_paths()
        repo_path = secure_resolve_path(directory, allowed)
        repo = Repo(repo_path)

        if repo.bare:
            return "Error: The specified directory is not a valid git repository."

        # Create a temporary file to hold the patch content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(patch_content)
            tmp_name = tmp.name

        try:
            # Perform a dry run first using GitPython to check if it applies cleanly
            repo.git.apply("--check", tmp_name)

            # If the check passes, apply it for real
            repo.git.apply(tmp_name)
            return "Success: Patch applied cleanly."
        except GitCommandError as e:
            # GitCommandError captures stdout and stderr from git
            return f"Error applying patch:\n{str(e)}"
        finally:
            os.remove(tmp_name)

    except InvalidGitRepositoryError:
        return "Error: The specified directory is not a valid git repository."
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error applying patch: {str(e)}"


# ==========================================
# GitHub REST API Tools
# ==========================================

# === Pull Request Operations ===


def open_pull_request(
    directory: str,
    title: str,
    head_branch: str,
    base_branch: str = "main",
    body: str = "",
    repo_full_name: str | None = None,
) -> str:
    """
    Opens a Pull Request via the GitHub REST API using the GITHUB_TOKEN environment variable.
    Extracts the repo_full_name dynamically from the local git remote.
    Returns a structured JSON string containing the PR URL.
    """
    _sanitize_branch_name(head_branch)
    _sanitize_branch_name(base_branch)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    repo_name = repo_full_name or _get_repo_full_name(directory)
    if not repo_name:
        return json.dumps({"status": "error", "reason": "invalid_github_url_format"})

    url = f"https://api.github.com/repos/{repo_name}/pulls"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"title": title, "head": head_branch, "base": base_branch, "body": body}

    try:
        # utilizing httpx synchronously with a strict timeout
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload, timeout=15.0)

        if response.status_code == httpx.codes.CREATED:
            pr_data = response.json()
            return json.dumps({"status": "success", "pr_url": pr_data.get("html_url")})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "api_request_failed",
                    "status_code": response.status_code,
                    "details": response.text,
                }
            )

    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


def update_pull_request(
    directory: str,
    pr_number: int,
    title: str | None = None,
    body: str | None = None,
    repo_full_name: str | None = None,
) -> str:
    """
    Updates the title and/or body of an existing GitHub Pull Request.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps(
            {"status": "error", "reason": "Missing GITHUB_TOKEN environment variable."}
        )

    repo_name = repo_full_name or _get_repo_full_name(directory)
    if not repo_name:
        return json.dumps({"status": "error", "reason": "invalid_github_url_format"})

    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    payload = {}
    if title:
        payload["title"] = title
    if body is not None:
        payload["body"] = body

    if not payload:
        return json.dumps({"status": "error", "reason": "No title or body provided to update."})

    try:
        response = httpx.patch(url, headers=headers, json=payload, timeout=10.0)
        if response.status_code == httpx.codes.OK:
            return json.dumps({"status": "success", "pr_url": response.json().get("html_url")})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "reason": f"GitHub API HTTP {response.status_code}",
                    "details": response.text,
                }
            )
    except Exception as e:
        return json.dumps(
            {"status": "error", "reason": "Exception during API call", "details": str(e)}
        )


def comment_on_pull_request(
    directory: str, pr_number: int, body: str, repo_full_name: str | None = None
) -> str:
    """
    Posts a Markdown comment on an existing GitHub Pull Request.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    repo_name = repo_full_name or _get_repo_full_name(directory)
    if not repo_name:
        return json.dumps({"status": "error", "reason": "invalid_github_url_format"})

    # PR comments use the issues endpoint in the GitHub REST API
    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"body": body}

    try:
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload, timeout=15.0)

        if response.status_code == httpx.codes.CREATED:
            comment_data = response.json()
            return json.dumps({"status": "success", "comment_url": comment_data.get("html_url")})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "api_request_failed",
                    "status_code": response.status_code,
                    "details": response.text,
                }
            )

    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


# === Commit Status & CI Integration ===


def set_commit_status(
    directory: str,
    commit_sha: str,
    state: str,
    context_str: str,
    description: str = "",
    repo_full_name: str | None = None,
) -> str:
    """
    Sets the CI status (pending, success, error, failure) of a commit via the GitHub REST API.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    repo_name = repo_full_name or _get_repo_full_name(directory)
    if not repo_name:
        return json.dumps({"status": "error", "reason": "invalid_github_url_format"})

    url = f"https://api.github.com/repos/{repo_name}/statuses/{commit_sha}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Validate the state string
    if state not in ["error", "failure", "pending", "success"]:
        return json.dumps({"status": "error", "reason": "invalid_status_state", "details": state})

    payload = {
        "state": state,
        "context": context_str,
        # GitHub limits the description field to 140 characters
        "description": description[:140],
    }

    try:
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload, timeout=15.0)

        if response.status_code == httpx.codes.CREATED:
            return json.dumps({"status": "success", "state": state, "context": context_str})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "reason": "api_request_failed",
                    "status_code": response.status_code,
                    "details": response.text,
                }
            )

    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})
