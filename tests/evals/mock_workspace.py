import json
import os
import shutil
import tempfile
from unittest.mock import patch

from git import Repo

from src.workspace_agent.core.config import settings
from src.workspace_agent.tools.registry import execute_tool_call as original_execute_tool_call


class MockWorkspaceTracker:
    """Context manager to intercept network boundaries and provide ephemeral filesystems."""

    def __init__(self):
        self.invocation_history = []
        self.temp_dirs = []
        self.sandbox_paths = {}
        self.original_workspace_paths = {}
        self.original_allowed_paths = []

    def _setup_sandboxes(self):
        """Mirrors real workspaces into ephemeral /tmp directories for safe testing."""
        for ws_name, ws_config in settings.workspaces.items():
            # Create a secure temporary directory
            temp_dir = tempfile.TemporaryDirectory(prefix=f"agent_eval_{ws_name}_")
            self.temp_dirs.append(temp_dir)
            temp_path = temp_dir.name
            self.sandbox_paths[ws_name] = temp_path

            real_path = ws_config.path
            if os.path.exists(real_path):
                # Copy files over, IGNORING the .git folder so we can make a fresh test repo
                shutil.copytree(
                    real_path,
                    temp_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git"),
                )

            # Initialize a fresh git repository in the sandbox
            repo = Repo.init(temp_path)

            # Force local git config so commits don't crash in headless CI pipelines
            with repo.config_writer() as git_config:
                git_config.set_value("user", "name", "Test Agent Bot")
                git_config.set_value("user", "email", "bot@agent.local")

            # Ensure at least one commit exists so diffs against HEAD don't crash
            readme_path = os.path.join(temp_path, "TEST_ARENA_README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(f"# Ephemeral Sandbox for {ws_name}")

            # Seed faulty Python script for eval self-correction test
            calc_path = os.path.join(temp_path, "calculator.py")
            with open(calc_path, "w") as f:
                f.write("def divide(a, b):\n    return a / b\n\nprint(divide(10, 0))\n")

            repo.git.add(A=True)
            repo.git.commit("-m", "Initial sandbox commit")

            if "main" not in [h.name for h in repo.heads]:
                repo.git.branch("-M", "main")

    def __enter__(self):
        # 1. Spin up the ephemeral sandboxes
        self._setup_sandboxes()

        # 2. Reroute the agent's global config to point to the temporary sandboxes
        self.original_allowed_paths = list(settings.allowed_paths)
        settings.allowed_paths.clear()

        allowed_paths = []
        for ws_name, ws_config in settings.workspaces.items():
            self.original_workspace_paths[ws_name] = ws_config.path
            ws_config.path = self.sandbox_paths[ws_name]
            settings.allowed_paths.append(self.sandbox_paths[ws_name])
            allowed_paths.append(self.sandbox_paths[ws_name])

        # 3. Patch env vars for permission logic and PR nodes
        self.env_patcher = patch.dict(
            os.environ,
            {
                "GITHUB_USERNAME": "test-agent-bot",
                "ALLOWED_PATHS": ",".join(allowed_paths),
            },
        )
        self.env_patcher.start()

        # 4. Mock only remote / network boundary tools
        self.patcher_pr = patch("src.workspace_agent.orchestrator.nodes.open_pull_request")
        self.mock_pr = self.patcher_pr.start()
        self.mock_pr.side_effect = self._mock_create_pr

        self.patcher_update_pr = patch("src.workspace_agent.orchestrator.nodes.update_pull_request")
        self.mock_update_pr = self.patcher_update_pr.start()
        self.mock_update_pr.side_effect = self._mock_update_pr

        self.patcher_branch = patch(
            "src.workspace_agent.orchestrator.nodes.create_branch_and_commit"
        )
        self.mock_branch = self.patcher_branch.start()
        self.mock_branch.side_effect = self._mock_create_branch

        self.patcher_sync = patch("src.workspace_agent.orchestrator.nodes.sync_repository")
        self.mock_sync = self.patcher_sync.start()
        self.mock_sync.side_effect = self._mock_sync

        self.patcher_delete = patch("src.workspace_agent.orchestrator.nodes.cleanup_local_branch")
        self.mock_delete = self.patcher_delete.start()
        self.mock_delete.side_effect = self._mock_delete_branch

        self.patcher_python = patch("src.workspace_agent.tools.sandbox.run_python_script")
        self.mock_python = self.patcher_python.start()
        self.mock_python.side_effect = self._mock_run_python_script

        # Acts as an inescapable choke point for native tool execution observability
        self.patcher_exec = patch("src.workspace_agent.orchestrator.nodes.execute_tool_call")
        self.mock_exec = self.patcher_exec.start()

        def _mock_execute(tool_call, config=None):
            self.invocation_history.append(
                {"tool": tool_call.get("name"), "kwargs": tool_call.get("args")}
            )
            return original_execute_tool_call(tool_call, config)

        self.mock_exec.side_effect = _mock_execute

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        patch.stopall()

        # Restore original paths to the singleton config so the state isn't poisoned
        for ws_name, original_path in self.original_workspace_paths.items():
            settings.workspaces[ws_name].path = original_path

        settings.allowed_paths.clear()
        settings.allowed_paths.extend(self.original_allowed_paths)

        # Nuke the ephemeral sandboxes from the system
        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                print(f"      [!] Sandbox cleanup warning: {e}")

    def _mock_run_python_script(self, *args, **kwargs) -> str:
        """Reads the actual state of the file from the ephemeral disk to verify fixes."""
        script_path = kwargs.get("script_path") or (args[0] if args else "")

        self.invocation_history.append(
            {"tool": "run_python_script", "kwargs": {"script_path": script_path}}
        )

        target_path = script_path
        if not os.path.isabs(script_path):
            # Resolve the path relative to the active sandboxes if passed relatively
            for base_path in self.sandbox_paths.values():
                possible_path = os.path.join(base_path, script_path)
                if os.path.exists(possible_path):
                    target_path = possible_path
                    break

        if os.path.exists(target_path):
            with open(target_path) as f:
                content = f.read()

            # Check if the bug is still present. We check if they added a return 0 safety.
            if "a / b" in content and "return 0" not in content:
                return (
                    "Traceback (most recent call last):\n"
                    '  File "calculator.py", line 4, in <module>\n'
                    "    print(divide(10, 0))\n"
                    '  File "calculator.py", line 2, in divide\n'
                    "    return a / b\n"
                    "ZeroDivisionError: division by zero"
                )

            return "Execution successful.\nOutput:\n0"

        return f"Error: File {script_path} not found."

    def _mock_create_pr(self, *args, **kwargs):
        self.invocation_history.append({"tool": "open_pull_request", "kwargs": kwargs})
        return json.dumps(
            {
                "status": "success",
                "pr_url": "https://github.com/mock-repo/pull/123",
                "branch": "agent/mock-branch-name",
            }
        )

    def _mock_update_pr(self, *args, **kwargs):
        self.invocation_history.append({"tool": "update_pull_request", "kwargs": kwargs})
        return json.dumps({"status": "success", "pr_url": "https://github.com/mock-repo/pull/123"})

    def _mock_create_branch(self, *args, **kwargs):
        self.invocation_history.append({"tool": "create_branch_and_commit", "kwargs": kwargs})

        # Extract variables safely from positional or keyword arguments
        directory = kwargs.get("directory") or (args[0] if len(args) > 0 else None)
        new_branch = kwargs.get("new_branch") or (
            args[1] if len(args) > 1 else "agent/mock-branch-name"
        )
        commit_message = kwargs.get("commit_message") or (
            args[2] if len(args) > 2 else "Mocked commit message"
        )

        # Actually perform the local commit in the ephemeral sandbox.
        # This advances the local Git HEAD so the next turn starts with a clean slate for its diffs.
        if directory:
            try:
                repo = Repo(directory)
                if new_branch in repo.heads:
                    new_head = repo.heads[new_branch]
                else:
                    new_head = repo.create_head(new_branch)
                new_head.checkout()

                repo.git.add(A=True)
                # Only commit if there are actually changes to prevent GitCommandErrors
                if repo.index.diff("HEAD") or repo.untracked_files:
                    repo.index.commit(commit_message)
            except Exception as e:
                print(f"      [!] Sandbox local commit warning: {e}")

        return json.dumps({"status": "success", "branch": new_branch, "message": commit_message})

    def _mock_sync(self, *args, **kwargs):
        self.invocation_history.append({"tool": "sync_repository", "kwargs": kwargs})
        branch = kwargs.get("target_branch") or (args[1] if len(args) > 1 else "main")
        return json.dumps({"status": "success", "branch": branch})

    def _mock_delete_branch(self, *args, **kwargs):
        # Mock branch deletion to prevent tests from accidentally deleting real unpushed branches
        self.invocation_history.append({"tool": "cleanup_local_branch", "kwargs": kwargs})
