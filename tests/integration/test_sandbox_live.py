import os

import docker
import pytest

from src.workspace_agent.tools.sandbox import run_pytest

# Skip these tests in CI environments where Docker might not be available
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Requires a live Docker daemon"
)


def test_live_run_pytest_success(tmp_path, monkeypatch):
    """
    Physically spins up the agent-sandbox container and executes a real
    pytest suite against the mounted filesystem.
    """
    # 1. Setup a real physical workspace
    safe_dir = tmp_path / "git"
    safe_dir.mkdir()
    monkeypatch.setenv("ALLOWED_PATHS", str(safe_dir))

    # 2. Write a real Python test file to the host disk
    test_file = safe_dir / "test_live_logic.py"
    test_file.write_text("def test_addition():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    # 3. Connect to the real local Docker daemon
    try:
        real_client = docker.from_env()
        real_client.ping()
    except docker.errors.DockerException:
        pytest.skip("Docker daemon is not running locally.")

    # 4. Inject the real client, bypassing mocks
    config = {"configurable": {"docker_client": real_client}}

    # 5. Execute the tool
    # This will take a few seconds as it physically boots the container
    result = run_pytest(str(test_file), config=config)

    # 6. Assert against the physical reality
    assert "Exit Code: 0" in result
    assert "1 passed" in result
    assert "test_live_logic.py" in result
