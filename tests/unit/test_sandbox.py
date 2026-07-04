import docker
import pytest

from src.workspace_agent.tools.sandbox import (
    _get_secure_mounts,
    _truncate_logs,
    compile_latex_document,
    run_pytest,
    run_python_script,
)

# ==========================================
# Helper: Test Fixtures
# ==========================================


@pytest.fixture
def mock_docker_client(mocker):
    """
    Mocks the Docker daemon and container lifecycle. Does NOT globally patch it,
    so tests must dynamically inject it via the config dictionary.
    """
    mock_client = mocker.MagicMock()
    mock_container = mocker.MagicMock()

    # default behavior: simulate a successful execution
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"Standard Output Log"

    mock_client.containers.run.return_value = mock_container

    return mock_client, mock_container


# ==========================================
# Component: _truncate_logs
# ==========================================


def test_truncate_logs_success_preserves_short_logs():
    """Green Path: Logs under the limit are returned untouched."""
    short_log = "Line 1\nLine 2\nLine 3"
    assert _truncate_logs(short_log, max_lines=5) == short_log


def test_truncate_logs_success_slices_massive_logs():
    """Green Path: Massive logs are safely truncated in the middle."""
    # 1. Setup Mock Environment
    massive_log = "\n".join([f"Line {i}" for i in range(1, 301)])

    # 2. Execute
    truncated = _truncate_logs(massive_log, max_lines=50, head_lines=10, tail_lines=40)

    # 3. Assertions
    assert "Line 10" in truncated
    assert "Line 11" not in truncated  # Sliced out
    assert "250 lines of logs truncated" in truncated
    assert "Line 261" in truncated


def test_truncate_logs_fallback_parameter_safeguard():
    """Edge Path: Safeguards against invalid head/tail configurations by prioritizing the tail."""
    # 1. Setup Mock Environment
    massive_log = "\n".join([f"Line {i}" for i in range(1, 301)])

    # 2. Execute
    # Requesting 100 total lines, but the max is constrained to 50.
    # The function should prioritize the 50 tail lines and reduce head lines to 0.
    truncated = _truncate_logs(massive_log, max_lines=50, head_lines=50, tail_lines=50)

    # 3. Assertions
    assert "Line 251" in truncated  # The tail is preserved
    assert "Line 1\n" not in truncated  # The head is safely sacrificed
    assert "250 lines of logs truncated" in truncated


# ==========================================
# Component: _get_secure_mounts
# ==========================================


def test_get_secure_mounts_success_masking(tmp_path):
    """
    Green Path: Verifies that _get_secure_mounts correctly masks sensitive files and submodules.
    """
    # 1. Setup Mock Environment
    host_dir = tmp_path / "repo"
    host_dir.mkdir()

    # Create nested sensitive structure
    (host_dir / ".git").mkdir()
    (host_dir / "src").mkdir()
    (host_dir / "src" / ".env.local").touch()
    (host_dir / "secrets.pem").touch()

    # 2. Execute
    volumes, tmpfs = _get_secure_mounts(host_dir)

    # 3. Assertions
    # Check that the base mount is present
    assert any(f"{host_dir.absolute()}:/workspace:rw" in str(v) for v in volumes)

    # Check /dev/null masking for MULTIPLE files (Verifies the list collision fix)
    assert "/dev/null:/workspace/src/.env.local:ro" in volumes
    assert "/dev/null:/workspace/secrets.pem:ro" in volumes

    # Check tmpfs masking for directories
    assert "/workspace/.git" in tmpfs
    assert tmpfs["/workspace/.git"] == "ro"


# ==========================================
# Workflow: Python Script Execution
# ==========================================


def test_run_python_script_success_execution(setup_workspaces, mock_docker_client):
    """Green Path: Simulates a successful Python script execution."""
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "01-08-eval-pipeline.py"
    test_file.write_text("print('Evaluations complete')", encoding="utf-8")

    client, container = mock_docker_client
    container.logs.return_value = b"Evaluations complete"

    config = {"configurable": {"docker_client": client}}

    # 2. Execute
    result = run_python_script(str(test_file), config=config)

    # 3. Assertions
    assert "Execution Finished" in result
    assert "Exit Code: 0" in result
    assert "Evaluations complete" in result

    # verify the correct bind mount was requested
    client.containers.run.assert_called_once()
    _, kwargs = client.containers.run.call_args
    assert any(str(safe_dir) in v for v in kwargs["volumes"])

    # verify the ephemeral container was cleaned up
    container.remove.assert_called_once_with(force=True)


def test_run_python_script_fallback_docker_api_error(setup_workspaces, mocker):
    """
    Edge Path: Verifies the tool safely catches and returns internal Docker
    daemon errors (like out of memory, or missing image).
    """
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "01-08-eval-pipeline.py"
    test_file.write_text("pass", encoding="utf-8")

    # mock the client to throw an APIError when trying to run the container
    mock_client = mocker.MagicMock()
    mock_client.containers.run.side_effect = docker.errors.APIError("Insufficient memory")

    config = {"configurable": {"docker_client": mock_client}}

    # 2. Execute
    result = run_python_script(str(test_file), config=config)

    # 3. Assertions
    assert "Docker API Error" in result
    assert "Insufficient memory" in result


def test_run_python_script_error_docker_daemon_offline(setup_workspaces, mocker):
    """
    Red Path: Verifies that if `docker_client` is not successfully injected into the config
    at startup (because the daemon is offline), the tools gracefully abort.
    """
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "01-08-eval-pipeline.py"
    test_file.write_text("pass", encoding="utf-8")

    # Simulate daemon offline by not loading it into the config dict
    config = {"configurable": {"docker_client": None}}

    # Force the fallback to fail, simulating a completely unreachable daemon
    mocker.patch(
        "src.workspace_agent.tools.sandbox.docker.from_env",
        side_effect=Exception("Connection refused"),
    )

    # 2. Execute
    result = run_python_script(str(test_file), config=config)

    # 3. Assertions
    assert "Unexpected Error" in result
    assert "Failed to connect to Docker Daemon/Proxy" in result


def test_run_python_script_error_file_not_found(setup_workspaces):
    """
    Red Path: Verifies the tool aborts gracefully if the requested
    script does not physically exist in the workspace.
    """
    safe_dir = setup_workspaces["safe"]
    missing_file = safe_dir / "does_not_exist.py"
    result = run_python_script(str(missing_file), config={})
    assert "Error: Python script not found" in result


def test_run_python_script_error_traceback_capture(setup_workspaces, mock_docker_client):
    """
    Red Path: Traceback Capture.
    Simulates a script crashing to ensure STDERR is properly captured and
    returned to the LLM so it can debug its own code.
    """
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "01-08-eval-pipeline.py"
    test_file.write_text("1 / 0", encoding="utf-8")

    client, container = mock_docker_client
    container.wait.return_value = {"StatusCode": 1}
    container.logs.return_value = (
        b"Traceback (most recent call last):\nZeroDivisionError: division by zero"
    )

    config = {"configurable": {"docker_client": client}}

    # 2. Execute
    result = run_python_script(str(test_file), config=config)

    # 3. Assertions
    assert "Exit Code: 1" in result
    assert "ZeroDivisionError" in result


def test_run_python_script_error_unauthorized_path(setup_workspaces):
    """
    Red Path: Proves that the path verification circuit breaker
    triggers before the Docker SDK is ever invoked.
    """
    forbidden_file = setup_workspaces["safe"] / ".." / "system" / "secret.py"

    result = run_python_script(str(forbidden_file), config={})
    assert "Security Exception" in result


# ==========================================
# Workflow: Pytest Operations
# ==========================================


def test_run_pytest_success_execution(setup_workspaces, mocker):
    """Green Path: Successfully executes pytest inside the container and returns logs."""
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "test_logic.py"
    test_file.write_text("def test_ok(): pass", encoding="utf-8")

    # mock the Docker client and container behavior
    mock_client = mocker.MagicMock()
    mock_container = mocker.MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"1 passed in 0.01s"
    mock_client.containers.run.return_value = mock_container

    config = {"configurable": {"docker_client": mock_client}}

    # 2. Execute
    result = run_pytest(str(test_file), config=config)

    # 3. Assertions
    assert "Exit Code: 0" in result
    assert "1 passed" in result

    # verify the command was constructed correctly
    mock_client.containers.run.assert_called_once()
    args, kwargs = mock_client.containers.run.call_args

    # Verify it uses the bash wrapper and the pytest command is in the executable string
    assert kwargs["command"][0] == "/bin/bash"
    assert kwargs["command"][1] == "-c"
    assert "pytest" in kwargs["command"][2]
    assert "test_logic.py" in kwargs["command"][2]


def test_run_pytest_success_workspace_boundary_resolution(mocker, tmp_path):
    """
    Green Path: Verifies that run_pytest strictly limits volume mounts to config.yaml boundaries.
    """
    # 1. Setup Mock Environment
    # Create a mock monorepo structure
    repo_root = tmp_path / "monorepo"
    backend_root = repo_root / "backend"
    backend_root.mkdir(parents=True)
    test_file = backend_root / "test_api.py"
    test_file.touch()

    mocker.patch(
        "src.workspace_agent.tools.sandbox.get_allowed_paths", return_value=[str(repo_root)]
    )

    # Mock settings.workspaces to specifically register ONLY the backend sub-folder
    mock_ws = mocker.Mock()
    mock_ws.path = str(backend_root)
    mocker.patch("src.workspace_agent.tools.sandbox.settings.workspaces", {"backend": mock_ws})

    # Mock Docker client
    mock_client = mocker.Mock()
    mock_container = mocker.Mock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b"Passed"
    mock_client.containers.run.return_value = mock_container

    config = {"configurable": {"docker_client": mock_client}}

    # 2. Execute
    run_pytest(str(test_file), config=config)

    # 3. Assertions
    # Verify that docker run was called with the backend_root mounted, NOT the entire monorepo root
    run_kwargs = mock_client.containers.run.call_args[1]
    volumes = run_kwargs["volumes"]

    # Verify the backend boundary was chosen
    assert any(str(backend_root) in v for v in volumes)
    assert not any(str(repo_root) + ":/workspace" in v for v in volumes)


def test_run_pytest_fallback_test_failure(setup_workspaces, mocker):
    """Fallback Path: Successfully runs pytest, but correctly reports a test failure exit code."""
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "test_logic.py"
    test_file.write_text("def test_fail(): assert False", encoding="utf-8")

    mock_client = mocker.MagicMock()
    mock_container = mocker.MagicMock()
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.return_value = b"1 failed in 0.02s"
    mock_client.containers.run.return_value = mock_container

    config = {"configurable": {"docker_client": mock_client}}

    # 2. Execute
    result = run_pytest(str(test_file), config=config)

    # 3. Assertions
    assert "Exit Code: 1" in result
    assert "1 failed" in result


def test_run_pytest_error_not_found(setup_workspaces):
    """Red Path: Bypasses Docker entirely if the requested test file does not exist locally."""
    safe_dir = setup_workspaces["safe"]
    missing_file = safe_dir / "test_missing.py"
    result = run_pytest(str(missing_file), config={})
    assert "Error: Test path not found" in result


# ==========================================
# Workflow: LaTeX Compilation
# ==========================================


def test_compile_latex_document_success_compilation(setup_workspaces, mock_docker_client):
    """Green Path: Simulates a successful latexmk compilation."""
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "KAN-TabNet-report.tex"
    test_file.write_text(
        r"\documentclass{article}\begin{document}Test\end{document}", encoding="utf-8"
    )

    client, container = mock_docker_client
    container.logs.return_value = b"Output written on KAN-TabNet-report.pdf"

    config = {"configurable": {"docker_client": client}}

    # 2. Execute
    result = compile_latex_document(str(test_file), config=config)

    # 3. Assertions
    assert "Compilation Finished" in result
    assert "Exit Code: 0" in result
    assert "Output written on KAN-TabNet-report.pdf" in result


def test_compile_latex_document_error_compilation_failure(setup_workspaces, mock_docker_client):
    """
    Red Path: Simulates a LaTeX compilation failure (e.g. syntax error)
    and ensures the compiler logs are returned to the agent.
    """
    # 1. Setup Mock Environment
    safe_dir = setup_workspaces["safe"]
    test_file = safe_dir / "broken-report.tex"
    test_file.write_text(r"\documentclass{article}", encoding="utf-8")

    client, container = mock_docker_client
    container.wait.return_value = {"StatusCode": 1}
    container.logs.return_value = b"Runaway argument? Emergency stop."

    config = {"configurable": {"docker_client": client}}

    # 2. Execute
    result = compile_latex_document(str(test_file), config=config)

    # 3. Assertions
    assert "Compilation Finished" in result
    assert "Exit Code: 1" in result
    assert "Emergency stop." in result
