import pytest

from src.workspace_agent.core.prompt_manager import PromptManager


# Reset the singleton before each test to ensure a clean state
@pytest.fixture(autouse=True)
def reset_prompt_manager():
    PromptManager._prompts = None
    yield
    PromptManager._prompts = None


# ==========================================
# Component: PromptManager.get
# ==========================================


def test_get_success_dynamic_formatting(mocker):
    """Green Path: Retrieves and correctly formats a prompt with injected kwargs."""
    # 1. Setup Mock Environment
    mock_yaml = {"execution": {"cross_workspace_prompt": "Target: {target_ws} at {target_path}"}}
    mocker.patch("src.workspace_agent.core.prompt_manager.yaml.safe_load", return_value=mock_yaml)
    mocker.patch("src.workspace_agent.core.prompt_manager.open", mocker.mock_open())

    # 2. Execute
    result = PromptManager.get(
        "execution", "cross_workspace_prompt", target_ws="my_app", target_path="/tmp"
    )

    # 3. Assertions
    assert result == "Target: my_app at /tmp"


def test_get_success_static_prompt(mocker):
    """Green Path: Retrieves a static prompt without any dynamic kwargs formatting."""
    # 1. Setup Mock Environment
    mock_yaml = {"router": {"system_prompt": "You are a router."}}
    mocker.patch("src.workspace_agent.core.prompt_manager.yaml.safe_load", return_value=mock_yaml)
    mocker.patch("src.workspace_agent.core.prompt_manager.open", mocker.mock_open())

    # 2. Execute
    result = PromptManager.get("router", "system_prompt")

    # 3. Assertions
    assert result == "You are a router."


def test_get_fallback_missing_key(mocker):
    """
    Edge Path: Returns an empty string safely if the key does not exist within a valid section.
    """
    # 1. Setup Mock Environment
    mock_yaml = {"router": {"system_prompt": "You are a router."}}
    mocker.patch("src.workspace_agent.core.prompt_manager.yaml.safe_load", return_value=mock_yaml)
    mocker.patch("src.workspace_agent.core.prompt_manager.open", mocker.mock_open())

    # 2. Execute
    result = PromptManager.get("router", "nonexistent_key")

    # 3. Assertions
    assert result == ""


def test_get_fallback_missing_section(mocker):
    """Edge Path: Returns an empty string safely if the YAML section does not exist."""
    # 1. Setup Mock Environment
    mock_yaml = {"router": {"system_prompt": "You are a router."}}
    mocker.patch("src.workspace_agent.core.prompt_manager.yaml.safe_load", return_value=mock_yaml)
    mocker.patch("src.workspace_agent.core.prompt_manager.open", mocker.mock_open())

    # 2. Execute
    result = PromptManager.get("nonexistent_section", "system_prompt")

    # 3. Assertions
    assert result == ""


# ==========================================
# Component: PromptManager.get_prompts
# ==========================================


def test_get_prompts_success_singleton_behavior(mocker):
    """Green Path: Ensures yaml.safe_load is strictly called only once and cached in memory."""
    # 1. Setup Mock Environment
    mock_yaml = {"test": {"key": "value"}}
    mock_load = mocker.patch(
        "src.workspace_agent.core.prompt_manager.yaml.safe_load", return_value=mock_yaml
    )
    mocker.patch("src.workspace_agent.core.prompt_manager.open", mocker.mock_open())

    # 2. Execute
    PromptManager.get("test", "key")
    PromptManager.get("test", "key")
    PromptManager.get_prompts()

    # 3. Assertions
    # Verify the disk operation was bypassed on subsequent calls
    assert mock_load.call_count == 1
