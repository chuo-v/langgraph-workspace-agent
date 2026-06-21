import os
from pathlib import Path

import docker
from langchain_core.runnables import RunnableConfig

from src.workspace_agent.core.config import settings
from src.workspace_agent.tools.filesystem import get_allowed_paths, secure_resolve_path


def _get_docker_client(config: RunnableConfig):
    """Retrieves the injected Docker client from the RunnableConfig."""
    client = config.get("configurable", {}).get("docker_client")
    if not client:
        raise RuntimeError("Docker client not found in injected config.")
    return client


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
            remove=False,  # Remove manually after grabbing the logs
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1000000000,
        )

        # Wait for execution to finish (with a strict timeout)
        result = container.wait(timeout=30)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")

        # Prevent context blowout
        logs = _truncate_logs(logs)

        # Cleanup
        container.remove(force=True)

        return f"Execution Finished (Exit Code: {result['StatusCode']})\n\nOutput:\n{logs}"

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
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1000000000,
        )

        # LaTeX compilation can take slightly longer, so timeout is 60s
        result = container.wait(timeout=60)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")

        # Prevent context blowout (LaTeX tracebacks are notoriously long, keep more tail lines)
        logs = _truncate_logs(logs)

        container.remove(force=True)

        return f"Compilation Finished (Exit Code: {result['StatusCode']})\n\nCompiler Logs:\n{logs}"

    except docker.errors.APIError as e:
        return f"Docker API Error: {str(e)}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected Error compiling LaTeX: {str(e)}"


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

        # 1. Strict Boundary: Match against explicitly configured workspaces in config.yaml
        workspace_root = None
        for ws_config in settings.workspaces.values():
            ws_path = Path(ws_config.path).resolve()
            if safe_path.is_relative_to(ws_path):
                # Use the longest (most specific) match in case of nested workspaces
                if not workspace_root or len(ws_path.parts) > len(workspace_root.parts):
                    workspace_root = ws_path

        # 2. Fallback Boundary: If it's a loose file outside configured workspaces
        # but inside ALLOWED_PATHS, tightly restrict the mount to its immediate parent.
        if not workspace_root:
            workspace_root = safe_path.parent
            rel_path = safe_path.name
        else:
            rel_path = str(safe_path.relative_to(workspace_root))

        client = _get_docker_client(config)

        # Get our dynamically masked volumes
        secure_vols, secure_tmpfs = _get_secure_mounts(workspace_root)

        # Spin up the ephemeral container
        container = client.containers.run(
            "agent-sandbox:latest",
            # Use -v for verbose output and --tb=short to keep error logs from blowing up
            # the context window
            command=["pytest", rel_path, "-v", "--tb=short"],
            volumes=secure_vols,
            tmpfs=secure_tmpfs,
            working_dir="/workspace",
            detach=True,
            remove=False,
            network_mode="none",
            mem_limit="512m",
            nano_cpus=1000000000,
        )

        # Wait for execution to finish (tests can take longer, so 60s timeout)
        result = container.wait(timeout=60)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")

        # Prevent context blowout
        logs = _truncate_logs(logs)

        # Cleanup
        container.remove(force=True)

        return (
            f"Pytest Execution Finished (Exit Code: {result['StatusCode']})\n\nTest Logs:\n{logs}"
        )

    except docker.errors.APIError as e:
        return f"Docker API Error: {str(e)}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected Error executing pytest: {str(e)}"
