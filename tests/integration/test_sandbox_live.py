import os

import docker
import pytest

from src.workspace_agent.tools.sandbox import run_pytest

# Skip these tests in CI environments where Docker might not be available
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Requires a live Docker daemon"
)

# ==========================================
# Workflow: Live Sandbox Integration
# ==========================================


def test_run_pytest_success_live_execution(tmp_path, monkeypatch):
    """
    Green Path: Physically spins up the agent-sandbox container and executes a real
    pytest suite against the mounted filesystem.
    """
    # 1. Setup Live Environment
    safe_dir = tmp_path / "git"
    safe_dir.mkdir()
    monkeypatch.setenv("ALLOWED_PATHS", str(safe_dir))

    test_file = safe_dir / "test_live_logic.py"
    test_file.write_text("def test_addition():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    try:
        real_client = docker.from_env()
        real_client.ping()
    except docker.errors.DockerException:
        pytest.skip("Docker daemon is not running locally.")

    config = {"configurable": {"docker_client": real_client}}

    # 2. Execute
    result = run_pytest(str(test_file), config=config)

    # 3. Assertions
    assert "Exit Code: 0" in result
    assert "1 passed" in result
    assert "test_live_logic.py" in result
