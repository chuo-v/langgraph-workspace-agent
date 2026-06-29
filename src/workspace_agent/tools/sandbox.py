import os
from pathlib import Path

import docker
from langchain_core.runnables import RunnableConfig

from src.workspace_agent.core.config import settings
from src.workspace_agent.tools.filesystem import get_allowed_paths, secure_resolve_path


def _get_docker_client(config: RunnableConfig):
    """
    Retrieves the Docker client, now routing through the socket proxy
    for hardened container orchestration.
    """
    # Check if a client was explicitly injected (useful for testing)
    client = config.get("configurable", {}).get("docker_client")
    if client:
        return client

    # Fallback to initializing the client pointed at the proxy
    # In docker-compose, this resolves to the proxy container
    try:
        return docker.from_env()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Docker Daemon/Proxy: {e}") from e


def _get_secure_mounts(host_mount_dir: Path) -> tuple[list, dict]:
    """
    Creates secure volume and tmpfs mappings to explicitly mask sensitive host files.
    Uses RAM disks and null devices to avoid Docker-in-Docker path translation issues.
    Recursively protects nested secrets and git submodules.
    """
    volumes = [f"{host_mount_dir.absolute()}:/workspace:rw"]
    tmpfs = {}

    if not host_mount_dir.exists():
        return volumes, tmpfs

    blocked_dirs = {".git", ".secrets", ".ssh", ".aws", ".venv", "venv", "env"}

    # Use os.walk to recursively catch nested .env files and submodules
    for root, dirs, files in os.walk(host_mount_dir):
        current_root = Path(root)

        # 1. Mask Directories
        for d in list(dirs):  # Iterate over a copy so we can safely modify the original list
            if d in blocked_dirs:
                # Calculate the exact path inside the container
                rel_path = (current_root / d).relative_to(host_mount_dir)
                sandbox_path = f"/workspace/{rel_path}"

                # Mask with an empty, read-only RAM disk
                tmpfs[sandbox_path] = "ro"

                # Prevent os.walk from descending into this directory
                # (saves time and prevents mount bloat)
                dirs.remove(d)

        # 2. Mask Files
        for f in files:
            if f.startswith(".env") or f.endswith((".pem", ".key", ".cert")):
                rel_path = (current_root / f).relative_to(host_mount_dir)
                sandbox_path = f"/workspace/{rel_path}"
                volumes.append(f"/dev/null:{sandbox_path}:ro")

    return volumes, tmpfs


def _truncate_logs(
    logs: str, max_lines: int = 500, head_lines: int = 50, tail_lines: int = 450
) -> str:
    """
    Safely truncates massive terminal outputs to prevent LLM context window blowouts.
    Keeps the beginning (setup) and end (tracebacks/summaries) of the logs.
    """
    # Safeguard: Ensure head and tail don't exceed max_lines
    if head_lines + tail_lines > max_lines:
        # Prioritize the tail (which contains the actual errors/tracebacks)
        tail_lines = min(tail_lines, max_lines)
        head_lines = max_lines - tail_lines

    lines = logs.splitlines()
    if len(lines) <= max_lines:
        return logs

    omitted = len(lines) - head_lines - tail_lines
    placeholder = f"\n\n... [ {omitted} lines of logs truncated to preserve context ] ...\n\n"

    top = "\n".join(lines[:head_lines]) if head_lines > 0 else ""
    bottom = "\n".join(lines[-tail_lines:]) if tail_lines > 0 else ""

    return top + placeholder + bottom


def run_python_script(script_path: str, config: RunnableConfig) -> str:
    """
    Spins up an ephemeral Docker container to execute a Python script safely.
    The directory containing the script is bind-mounted into the container.
    """
    try:
        allowed = get_allowed_paths()
        safe_path = secure_resolve_path(script_path, allowed)

        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: Python script not found at {safe_path}"

        # Mount the parent directory so the script can access adjacent files
        mount_dir = safe_path.parent
        filename = safe_path.name

        client = _get_docker_client(config)

        # Get our dynamically masked volumes
        secure_vols, secure_tmpfs = _get_secure_mounts(mount_dir)

        # Spin up the ephemeral container
        container = client.containers.run(
            "agent-sandbox:latest",
            command=["python", filename],
            volumes=secure_vols,
            tmpfs=secure_tmpfs,
            working_dir="/workspace",
            detach=True,
            remove=False,
            network="sandbox_net",
            mem_limit="4g",
            nano_cpus=4000000000,
        )

        try:
            result = container.wait(timeout=30)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = result.get("StatusCode", "UNKNOWN")
        except Exception:
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = "TIMEOUT"
        finally:
            # Guarantee container destruction to release file locks
            container.remove(force=True)

        # Prevent context blowout
        logs = _truncate_logs(logs)

        if status_code == "TIMEOUT":
            return f"Error: Script Execution Timed Out after 30s.\n\nPartial Logs:\n{logs}"

        return f"Execution Finished (Exit Code: {status_code})\n\nOutput:\n{logs}"

    except docker.errors.APIError as e:
        return f"Docker API Error: {str(e)}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected Error executing Python script: {str(e)}"


def compile_latex_document(tex_file_path: str, config: RunnableConfig) -> str:
    """
    Compiles a LaTeX document using latexmk inside the Docker sandbox.
    Resulting PDF persists in the mounted directory.
    """
    try:
        allowed = get_allowed_paths()
        safe_path = secure_resolve_path(tex_file_path, allowed)

        if not safe_path.exists() or not safe_path.is_file():
            return f"Error: LaTeX file not found at {safe_path}"

        mount_dir = safe_path.parent
        filename = safe_path.name

        client = _get_docker_client(config)

        # Get our dynamically masked volumes
        secure_vols, secure_tmpfs = _get_secure_mounts(mount_dir)

        # Run latexmk to compile the PDF
        container = client.containers.run(
            "agent-sandbox:latest",
            # nonstopmode prevents the compiler from hanging on syntax errors
            command=["latexmk", "-pdf", "-interaction=nonstopmode", filename],
            volumes=secure_vols,
            tmpfs=secure_tmpfs,
            working_dir="/workspace",
            detach=True,
            remove=False,
            network="sandbox_net",
            mem_limit="4g",
            nano_cpus=4000000000,
        )

        try:
            result = container.wait(timeout=60)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = result.get("StatusCode", "UNKNOWN")
        except Exception:
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = "TIMEOUT"
        finally:
            container.remove(force=True)

        # Prevent context blowout (LaTeX tracebacks are notoriously long, keep more tail lines)
        logs = _truncate_logs(logs)

        if status_code == "TIMEOUT":
            return f"Error: Compilation Timed Out after 60s.\n\nPartial Logs:\n{logs}"

        return f"Compilation Finished (Exit Code: {status_code})\n\nCompiler Logs:\n{logs}"

    except docker.errors.APIError as e:
        return f"Docker API Error: {str(e)}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected Error compiling LaTeX: {str(e)}"


def _resolve_workspace_and_path(safe_path: Path) -> tuple[Path, str]:
    """Helper to determine the strict workspace boundary and relative test path."""
    workspace_root = None
    for ws_config in settings.workspaces.values():
        ws_path = Path(ws_config.path).resolve()
        if safe_path.is_relative_to(ws_path):
            # Use the longest (most specific) match in case of nested workspaces
            if not workspace_root or len(ws_path.parts) > len(workspace_root.parts):
                workspace_root = ws_path

    # Fallback Boundary: tightly restrict the mount to its immediate parent
    if not workspace_root:
        return safe_path.parent, safe_path.name

    return workspace_root, str(safe_path.relative_to(workspace_root))


def _build_pytest_command(workspace_root: Path, rel_path: str) -> str:
    """Helper to dynamically detect and build dependency setup and pytest commands."""
    setup_cmds = []
    if (workspace_root / "requirements.txt").exists():
        setup_cmds.append("uv pip install -v --system -r requirements.txt")
    if (workspace_root / "requirements-dev.txt").exists():
        setup_cmds.append("uv pip install -v --system -r requirements-dev.txt")

    # Fallback for standard Python packages if no requirements.txt exists
    if not setup_cmds and (workspace_root / "pyproject.toml").exists():
        setup_cmds.append("uv pip install -v --system .")

    # Combine setup commands with the pytest execution
    if setup_cmds:
        chained_setup = " && ".join(setup_cmds)
        return f"{chained_setup} && pytest {rel_path} -v --tb=short"

    return f"pytest {rel_path} -v --tb=short"


def run_pytest(test_file_path: str, config: RunnableConfig) -> str:
    """
    Runs pytest on a specific test file or directory inside the Docker sandbox.
    Mounts the root of the workspace to ensure module imports resolve correctly.
    """
    try:
        allowed = get_allowed_paths()
        safe_path = secure_resolve_path(test_file_path, allowed)

        if not safe_path.exists():
            return f"Error: Test path not found at {safe_path}"

        # 1. Resolve workspace boundaries and relative execution paths
        workspace_root, rel_path = _resolve_workspace_and_path(safe_path)

        # 2. Build the exact bash command chaining pip installs and pytest
        full_command = _build_pytest_command(workspace_root, rel_path)

        client = _get_docker_client(config)

        # 3. Get our dynamically masked volumes
        secure_vols, secure_tmpfs = _get_secure_mounts(workspace_root)

        # 4. Spin up the ephemeral container
        container = client.containers.run(
            "agent-sandbox:latest",
            command=["/bin/bash", "-c", full_command],
            volumes=secure_vols,
            tmpfs=secure_tmpfs,
            working_dir="/workspace",
            detach=True,
            remove=False,
            network="sandbox_net",
            mem_limit="4g",
            nano_cpus=4000000000,
        )

        try:
            result = container.wait(timeout=180)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = result.get("StatusCode", "UNKNOWN")
        except Exception:
            # Trap the timeout exception to explicitly grab the logs before killing it
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            status_code = "TIMEOUT"
        finally:
            container.remove(force=True)

        # Prevent context blowout
        logs = _truncate_logs(logs)

        if status_code == "TIMEOUT":
            return f"Error: Pytest Execution Timed Out after 180s.\n\nPartial Test Logs:\n{logs}"

        return f"Pytest Execution Finished (Exit Code: {status_code})\n\nTest Logs:\n{logs}"

    except docker.errors.APIError as e:
        return f"Docker API Error: {str(e)}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected Error executing pytest: {str(e)}"
