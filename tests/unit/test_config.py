from unittest.mock import mock_open

import pytest
from pydantic import ValidationError

from src.workspace_agent.core.config import WorkspaceAgentConfig, load_configuration

# ==========================================
# Helper: Reusable Mock LLM Configurations
# ==========================================

MOCK_LLM_DICT = {
    "base_tier": {
        "default_model": "test",
        "available_models": {"test": {"provider": "ollama", "model_name": "test"}},
    },
    "standard_tier": {
        "default_model": "test",
        "available_models": {"test": {"provider": "deepseek", "model_name": "test"}},
    },
    "frontier_tier": {
        "default_model": "test",
        "available_models": {"test": {"provider": "gemini", "model_name": "test"}},
    },
}

MOCK_LLM_YAML = """
llm:
  base_tier:
    default_model: "test"
    available_models:
      test:
        provider: "ollama"
        model_name: "test"
  standard_tier:
    default_model: "test"
    available_models:
      test:
        provider: "deepseek"
        model_name: "test"
  frontier_tier:
    default_model: "test"
    available_models:
      test:
        provider: "gemini"
        model_name: "test"
"""


# ==========================================
# Component: WorkspaceAgentConfig (Schema Validation)
# ==========================================


def test_workspace_agent_config_success_complete():
    """
    Green Path: A perfectly formed configuration dictionary should parse without errors,
    and default values should populate correctly.
    """
    # 1. Setup Mock Environment
    valid_data = {
        "agent": {
            "target_branch": "develop",
            "target_remote": "upstream",
            "max_sandbox_retries": 2,
        },
        "allowed_paths": ["/Users/username/git/langgraph-workspace-agent"],
        "workspaces": {
            "workspace_agent": {
                "description": "The core orchestration architecture.",
                "path": "/Users/username/git/langgraph-workspace-agent",
                "ci_suites": [
                    {
                        "name": "E2E Pipeline Evaluator",
                        "command": "python -m tests.evals.run_e2e_evals",
                        "timeout_seconds": 180,
                    }
                ],
            }
        },
        "llm": MOCK_LLM_DICT,
    }

    # 2. Execute
    config = WorkspaceAgentConfig(**valid_data)

    # 3. Assertions
    assert config.agent.target_branch == "develop"
    assert config.agent.target_remote == "upstream"
    assert config.agent.max_sandbox_retries == 2
    assert len(config.allowed_paths) == 1
    assert "workspace_agent" in config.workspaces
    assert (
        config.workspaces["workspace_agent"].path == "/Users/username/git/langgraph-workspace-agent"
    )

    # Verify the new CI Suite whitelist parsed correctly
    ci_suites = config.workspaces["workspace_agent"].ci_suites
    assert len(ci_suites) == 1
    assert ci_suites[0].name == "E2E Pipeline Evaluator"
    assert ci_suites[0].command == "python -m tests.evals.run_e2e_evals"
    assert ci_suites[0].timeout_seconds == 180


def test_workspace_agent_config_success_defaults():
    """
    Edge Path: Verifies that omitted optional fields (like the 'agent' block,
    workspace 'target_branch', and 'ci_suites') gracefully fallback to their default factories.
    """
    # 1. Setup Mock Environment
    minimal_data = {
        "allowed_paths": ["/Users/username/git"],
        "workspaces": {
            "minimal_workspace": {
                "description": "A workspace with just the bare minimum.",
                "path": "/Users/username/git/minimal",
            }
        },
        "llm": MOCK_LLM_DICT,  # injected
    }

    # 2. Execute
    config = WorkspaceAgentConfig(**minimal_data)

    # 3. Assertions
    # Verify the agent block was auto-generated with defaults
    assert config.agent.target_branch == "main"
    assert config.agent.target_remote == "origin"
    assert config.agent.max_sandbox_retries == 3
    assert config.agent.show_telemetry

    # Verify workspace defaults
    assert config.workspaces["minimal_workspace"].target_branch is None
    assert config.workspaces["minimal_workspace"].target_remote is None
    # Verify ci_suites falls back to an empty list securely
    assert config.workspaces["minimal_workspace"].ci_suites == []


def test_workspace_agent_config_error_invalid_ci_suite():
    """
    Red Path: The system must fail if a CI suite is missing required fields like 'command'.
    """
    # 1. Setup Mock Environment
    invalid_data = {
        "allowed_paths": ["/Users/username/git"],
        "workspaces": {
            "broken_workspace": {
                "description": "Workspace with invalid CI suite.",
                "path": "/tmp/workspace",
                "ci_suites": [
                    {
                        "name": "Missing Command Test"
                        # 'command' is intentionally omitted
                    }
                ],
            }
        },
        "llm": MOCK_LLM_DICT,
    }

    # 2. Execute
    with pytest.raises(ValidationError) as exc_info:
        WorkspaceAgentConfig(**invalid_data)

    # 3. Assertions
    assert "command\n  Field required" in str(exc_info.value)


def test_workspace_agent_config_error_missing_paths():
    """
    Red Path: The system must fail if the allowed_paths list is empty.
    An empty sandbox is a severe security risk.
    """
    # 1. Setup Mock Environment
    invalid_data = {
        "agent": {"target_branch": "main", "max_sandbox_retries": 3},
        "allowed_paths": [],  # intentionally empty to trigger the min_length=1 failure
        "workspaces": {},
        "llm": MOCK_LLM_DICT,  # injected
    }

    # 2. Execute
    with pytest.raises(ValidationError) as exc_info:
        WorkspaceAgentConfig(**invalid_data)

    # 3. Assertions
    assert "allowed_paths" in str(exc_info.value)
    assert "at least 1" in str(exc_info.value).lower()


def test_workspace_agent_config_error_missing_workspace_keys():
    """
    Red Path: The system must fail if a workspace is missing required keys like 'path'.
    """
    # 1. Setup Mock Environment
    invalid_data = {
        "agent": {},
        "allowed_paths": ["/Users/username/git"],
        "workspaces": {
            "broken_workspace": {
                "description": "Missing the path key."
                # 'path' is intentionally omitted
            }
        },
        "llm": MOCK_LLM_DICT,  # injected
    }

    # 2. Execute
    with pytest.raises(ValidationError) as exc_info:
        WorkspaceAgentConfig(**invalid_data)

    # 3. Assertions
    assert "path\n  Field required" in str(exc_info.value)


def test_workspace_agent_config_error_telegram_model_keys():
    """
    Red Path: The system must fail if a model alias in the YAML contains
    spaces, hyphens, or uppercase letters that break Telegram slash commands.
    """
    # 1. Setup Mock Environment
    invalid_llm_dict = {
        "base_tier": MOCK_LLM_DICT["base_tier"],
        "standard_tier": MOCK_LLM_DICT["standard_tier"],
        "frontier_tier": {
            "default_model": "bad-key",
            "available_models": {
                "bad-key-with-hyphens": {"provider": "gemini", "model_name": "gemini-1.5"}
            },
        },
    }

    invalid_data = {
        "agent": {"target_branch": "main"},
        "allowed_paths": ["/tmp"],
        "workspaces": {},
        "llm": invalid_llm_dict,
    }

    # 2. Execute
    with pytest.raises(ValidationError) as exc_info:
        WorkspaceAgentConfig(**invalid_data)

    # 3. Assertions
    assert "Telegram-safe" in str(exc_info.value)
    assert "bad-key-with-hyphens" in str(exc_info.value)


# ==========================================
# Component: load_configuration (YAML File Loader)
# ==========================================


def test_load_configuration_success_from_file(mocker):
    """
    Green Path: Verifies that load_configuration correctly reads
    the YAML from disk and returns a valid WorkspaceAgentConfig instance.
    """
    # 1. Setup Mock Environment
    mock_yaml_content = (
        """
agent:
  target_branch: "main"
  max_sandbox_retries: 3
allowed_paths:
  - "/Users/vernon/git"
workspaces:
  workspace_agent:
    description: "Core architecture and orchestration logic."
    path: "/Users/vernon/git/langgraph-workspace-agent"
"""
        + MOCK_LLM_YAML
    )

    # mock os.path.exists to simulate finding the live config.yaml
    mocker.patch("os.path.exists", return_value=True)

    # mock builtins.open to return our dummy YAML content
    mocker.patch("builtins.open", mock_open(read_data=mock_yaml_content))

    # 2. Execute
    config = load_configuration()

    # 3. Assertions
    assert isinstance(config, WorkspaceAgentConfig)
    assert config.agent.target_branch == "main"
    assert (
        config.workspaces["workspace_agent"].path == "/Users/vernon/git/langgraph-workspace-agent"
    )


def test_load_configuration_error_schema_validation(mocker):
    """
    Red Path: If the YAML is valid but fails Pydantic schema validation
    (e.g., completely missing 'allowed_paths'), the system must abort.
    """
    # 1. Setup Mock Environment
    mocker.patch("os.path.exists", return_value=True)

    # valid YAML syntax, but violates our strict Pydantic rules (missing allowed_paths)
    mock_yaml = f"agent: {{}}\nworkspaces: {{}}\n{MOCK_LLM_YAML}"
    mocker.patch("builtins.open", mock_open(read_data=mock_yaml))

    # 2. Execute
    with pytest.raises(SystemExit) as exc_info:
        load_configuration()

    # 3. Assertions
    assert exc_info.value.code == 1


def test_load_configuration_error_malformed_yaml(mocker):
    """
    Red Path: If the file exists but contains invalid YAML syntax,
    the system must catch the YAMLError and abort via sys.exit(1).
    """
    # 1. Setup Mock Environment
    mocker.patch("os.path.exists", return_value=True)

    # broken YAML (unclosed string/list)
    mocker.patch("builtins.open", mock_open(read_data="agent: ["))

    # 2. Execute
    with pytest.raises(SystemExit) as exc_info:
        load_configuration()

    # 3. Assertions
    assert exc_info.value.code == 1


def test_load_configuration_error_missing_file(mocker):
    """
    Red Path: If the live config.yaml is missing, the system must
    strictly enforce the fail-fast security check and abort via sys.exit(1)
    rather than silently booting with dummy example paths.
    """
    # 1. Setup Mock Environment
    # simulate the missing config.yaml
    mocker.patch("os.path.exists", return_value=False)

    # 2. Execute
    with pytest.raises(SystemExit) as exc_info:
        load_configuration()

    # 3. Assertions
    assert exc_info.value.code == 1
