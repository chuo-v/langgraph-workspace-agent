import json
import os
import re

import httpx
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.workspace_agent.tools.filesystem import get_allowed_paths, secure_resolve_path


def _sanitize_branch_name(branch_name: str | None) -> None:
    """Prevents Git Argument Injection by rejecting flags."""
    if branch_name and str(branch_name).strip().startswith("-"):
        raise ValueError(
            f"Security Exception: Invalid branch name '{branch_name}'. Cannot start with a hyphen."
        )


# ==========================================
# Core Git & GitHub Functions (Native Tools)
# ==========================================


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

        error_reason = None
        token = os.getenv("GITHUB_TOKEN")
        current_url = next(repo.remotes.origin.urls)
        match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)

        if not token:
            error_reason = "missing_github_token"
        elif not match:
            error_reason = "invalid_github_url_format"

        if error_reason:
            return json.dumps({"status": "error", "reason": error_reason})

        if repo.is_dirty(untracked_files=True):
            repo.git.stash(
                "push", "--include-untracked", "-m", "Auto-stashed by Workspace Agent before sync"
            )

        repo_full_name = match.group(1)

        # configure the environment to use the token
        with repo.git.custom_environment(GIT_ASKPASS="echo", GIT_TOKEN=token):
            authenticated_url = f"https://{token}@github.com/{repo_full_name}.git"
            repo.remotes.origin.set_url(authenticated_url)

            try:
                repo.git.checkout(target_branch)
                repo.git.pull("--ff-only", "origin", target_branch)
            finally:
                # guarantee the URL is reset back to safe HTTPS even if the pull fails
                safe_url = f"https://github.com/{repo_full_name}.git"
                repo.remotes.origin.set_url(safe_url)

        return json.dumps({"status": "success", "branch": target_branch})

    except InvalidGitRepositoryError:
        return json.dumps({"status": "error", "reason": "not_a_git_repository"})
    except GitCommandError as e:
        return json.dumps({"status": "error", "reason": "git_command_failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


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

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            return json.dumps({"status": "error", "reason": "missing_github_token"})

        # dynamically extract 'owner/repo'
        current_url = next(repo.remotes.origin.urls)
        match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", current_url)
        if not match:
            return json.dumps({"status": "error", "reason": "invalid_github_url_format"})
        repo_full_name = match.group(1)

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

        # inject token and push
        with repo.git.custom_environment(GIT_ASKPASS="echo", GIT_TOKEN=token):
            authenticated_url = f"https://{token}@github.com/{repo_full_name}.git"
            repo.remotes.origin.set_url(authenticated_url)

            try:
                origin = repo.remote(name="origin")
                repo.git.push("--set-upstream", origin, new_branch)
            finally:
                # guarantee the URL is scrubbed after pushing
                safe_url = f"https://github.com/{repo_full_name}.git"
                repo.remotes.origin.set_url(safe_url)

        return json.dumps({"status": "success", "branch": new_branch, "message": commit_message})

    except GitCommandError as e:
        return json.dumps({"status": "error", "reason": "git_command_failed", "details": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "reason": "unexpected_error", "details": str(e)})


def open_pull_request(
    repo_full_name: str, title: str, head_branch: str, base_branch: str = "main", body: str = ""
) -> str:
    """
    Opens a Pull Request via the GitHub REST API using the GITHUB_TOKEN environment variable.
    repo_full_name must be formatted as 'owner/repo'.
    Returns a structured JSON string containing the PR URL.
    """
    _sanitize_branch_name(head_branch)
    _sanitize_branch_name(base_branch)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    url = f"https://api.github.com/repos/{repo_full_name}/pulls"
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
    repo_full_name: str, pr_number: int, title: str = None, body: str = None
) -> str:
    """
    Updates the title and/or body of an existing GitHub Pull Request.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps(
            {"status": "error", "reason": "Missing GITHUB_TOKEN environment variable."}
        )

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
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


def get_git_diff(directory: str, target_branch: str = None) -> str:
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


def get_git_diff_blueprint(directory: str, target_branch: str = None) -> str:
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


def comment_on_pull_request(repo_full_name: str, pr_number: int, body: str) -> str:
    """
    Posts a Markdown comment on an existing GitHub Pull Request.
    repo_full_name must be formatted as 'owner/repo'.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    # PR comments use the issues endpoint in the GitHub REST API
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
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


def set_commit_status(
    repo_full_name: str, commit_sha: str, state: str, context_str: str, description: str = ""
) -> str:
    """
    Sets the CI status (pending, success, error, failure) of a commit via the GitHub REST API.
    repo_full_name must be formatted as 'owner/repo'.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return json.dumps({"status": "error", "reason": "missing_github_token"})

    url = f"https://api.github.com/repos/{repo_full_name}/statuses/{commit_sha}"
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


# ==========================================
# Internal Utility & Hygiene Functions
# ==========================================


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
