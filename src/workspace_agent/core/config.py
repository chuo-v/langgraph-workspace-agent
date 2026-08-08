import os
import re
import sys

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.workspace_agent.core.provider import Provider

__all__ = [
    "OrchestrationConfig",
    "CISuiteConfig",
    "WorkspaceConfig",
    "ModelDefinition",
    "TierConfig",
    "LLMConfig",
    "WorkspaceAgentConfig",
    "load_configuration",
    "validate_environment_secrets",
    "settings",
]

# resolve paths up to the repository root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))


# ==========================================
# Configuration Schemas
# ==========================================


class OrchestrationConfig(BaseModel):
    """Configuration for agent orchestration limits, defaults, and ChatOps settings."""

    target_branch: str = Field(
        default="main", description="The default Git branch for agent operations."
    )
    target_remote: str = Field(
        default="origin", description="The default Git remote for agent operations."
    )
    allowed_github_users: list[str] = Field(
        default_factory=list,
        description="List of GitHub usernames authorized to trigger agent workflows via webhooks.",
    )
    max_sandbox_retries: int = Field(
        default=3, description="Max autonomous retry attempts if code execution fails."
    )
    max_concurrent_ci_jobs: int = Field(
        default=1,
        description="Maximum number of Agentic CI test suites permitted to run concurrently.",
    )
    router_timeout_seconds: int = Field(
        default=90,
        description=(
            "Max time in seconds to wait for the intent router before escalating or timing out."
        ),
    )
    max_consecutive_tool_steps: int = Field(
        default=60,
        description=(
            "Max number of consecutive tool/AI steps before the circuit breaker aborts execution."
        ),
    )
    show_telemetry: bool = Field(
        default=True, description="Toggle to display LLM API call counts in Telegram."
    )
    agent_prefix: str = Field(
        default="🤖", description="Text prepended to PR titles, PR comments, and commit messages."
    )
    chatops_name: str = Field(
        default="agent", description="The base name the agent listens to for ChatOps mentions."
    )


class CISuiteConfig(BaseModel):
    """Configuration definition for a specific Agentic CI/CD test suite."""

    name: str = Field(..., description="Display name for the GitHub Status Check context.")
    command: str = Field(..., description="The strictly whitelisted shell command to execute.")
    timeout_seconds: int = Field(
        default=60, description="OS-level hard timeout to prevent the agent from hanging."
    )


class WorkspaceConfig(BaseModel):
    """Configuration overrides and specific checks assigned to a single managed workspace."""

    description: str = Field(..., min_length=10)
    path: str = Field(...)
    target_branch: str | None = Field(default=None, description="Overrides global target_branch")
    target_remote: str | None = Field(default=None, description="Overrides global target_remote")
    pre_commit_suites: list[CISuiteConfig] = Field(
        default_factory=list,
        description=(
            "Local checks and auto-fixes to run before committing "
            "(e.g., formatters, type-checkers)."
        ),
    )
    ci_suites: list[CISuiteConfig] = Field(
        default_factory=list, description="Whitelisted Agentic CI/CD test suites."
    )


class ModelDefinition(BaseModel):
    """Maps a specific LLM model identifier to its executing provider."""

    provider: Provider
    model_name: str = Field(description="The exact string expected by the provider API")


class TierConfig(BaseModel):
    """Defines a tiered grouping of available models (e.g., base, standard, frontier)."""

    default_model: str
    available_models: dict[str, ModelDefinition]

    @field_validator("available_models")
    @classmethod
    def _validate_telegram_safe_keys(
        cls, v: dict[str, ModelDefinition]
    ) -> dict[str, ModelDefinition]:
        """Ensures all model dictionary keys strictly conform to Telegram-safe formatting."""
        for key in v.keys():
            if not re.match(r"^[a-zA-Z0-9_]+$", key):
                raise ValueError(
                    f"Model alias '{key}' is invalid. Keys must be Telegram-safe "
                    "(alphanumeric and underscores only)."
                )
        return v


class LLMConfig(BaseModel):
    """Consolidated configuration mapping tiers to their respective models."""

    base_tier: TierConfig
    standard_tier: TierConfig
    frontier_tier: TierConfig


class WorkspaceAgentConfig(BaseModel):
    """The root configuration object holding the complete operational state of the agent."""

    agent: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    llm: LLMConfig
    # enforce that at least one path is whitelisted for the sandbox
    allowed_paths: list[str] = Field(..., min_length=1)
    workspaces: dict[str, WorkspaceConfig]


# ==========================================
# Initialization & Validation Logic
# ==========================================


def load_configuration() -> WorkspaceAgentConfig:
    """Loads and validates the YAML configuration schema.

    State Transitions:
    - Ingests data from `config.yaml` or the environment's config path override.
    - Mutates `allowed_paths` directly if the `ALLOWED_PATHS` environment variable is provided.

    Exceptions:
    - Triggers sys.exit(1) if the file is missing, contains malformed YAML,
      or fails schema validation.
    """
    config_path = os.getenv("WORKSPACE_AGENT_CONFIG_PATH") or os.path.join(
        PROJECT_ROOT, "config.yaml"
    )

    if not os.path.exists(config_path):
        print(f"FATAL: No configuration file found at {config_path}")
        print(
            "Please copy 'config.example.yaml' to 'config.yaml' and configure your allowed paths "
            "and LLM settings."
        )
        sys.exit(1)

    try:
        with open(config_path) as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"FATAL: Malformed YAML syntax in {config_path}:\n{e}")
        sys.exit(1)

    # Catch Docker Compose environment overrides
    env_allowed_paths = os.getenv("ALLOWED_PATHS")
    if env_allowed_paths:
        # Split by comma to support multiple paths if needed,
        # and strip whitespace to prevent silent matching errors
        raw_data["allowed_paths"] = [
            path.strip() for path in env_allowed_paths.split(",") if path.strip()
        ]

    try:
        # Pydantic unpacks the dict and enforces the types/constraints
        return WorkspaceAgentConfig(**raw_data)
    except ValidationError as e:
        print(f"FATAL: Configuration schema validation failed in {config_path}:")
        print(e)
        sys.exit(1)


def validate_environment_secrets(config: WorkspaceAgentConfig) -> None:
    """Cross-references the active YAML configuration with system environment variables.

    State Transitions:
    - Scans for required core integrations and dynamically computes required LLM keys.

    Exceptions:
    - Triggers sys.exit(1) and aborts the boot sequence if any referenced integration
      lacks a secret key.
    """
    missing_secrets: list[str] = []

    # 1. Core Application Integrations
    core_secrets = [
        "GITHUB_USERNAME",
        "GITHUB_TOKEN",
        "GITHUB_WEBHOOK_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_SECRET_TOKEN",
        "AUTHORIZED_OWNER_CHAT_ID",
    ]
    for secret in core_secrets:
        if not os.getenv(secret):
            missing_secrets.append(secret)

    # 2. Dynamic LLM Providers (Only ask for keys of models actually configured)
    _check_provider_key(config.llm.base_tier, missing_secrets)
    _check_provider_key(config.llm.standard_tier, missing_secrets)
    _check_provider_key(config.llm.frontier_tier, missing_secrets)

    # 3. Fast-Fail if anything is missing
    if missing_secrets:
        # Deduplicate list just in case
        missing_secrets = sorted(list(set(missing_secrets)))
        print("======================================================")
        print("FATAL BOOT ERROR: Missing required environment secrets")
        print("======================================================")
        print("Please ensure the following variables are set in your .env file")
        print("or injected by your deployment environment:\n")
        for secret in missing_secrets:
            print(f"  ❌ {secret}")
        print("\nBoot sequence aborted.")
        sys.exit(1)


def _check_provider_key(tier_config: TierConfig, missing_secrets: list[str]) -> None:
    """Inspects a specific model tier to determine if its default provider requires an API key,
    appending any missing keys to the tracking array.
    """
    model_def = tier_config.available_models.get(tier_config.default_model)
    if model_def:
        # Handle enum string value or raw string depending on how Provider is defined
        provider_val = getattr(model_def.provider, "value", str(model_def.provider)).lower()

        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": None,  # No key required for local execution
        }

        expected_key = provider_keys.get(provider_val)
        if expected_key and not os.getenv(expected_key):
            missing_secrets.append(expected_key)


# ==========================================
# Boot Sequence Execution
# ==========================================

# Load the .env file early so os.getenv works during instantiation
load_dotenv()

# Instantiate the singleton so other modules can import `settings`
settings: WorkspaceAgentConfig = load_configuration()

# Validate the environment immediately after loading the YAML config,
# UNLESS we are running automated tests.
if "pytest" not in sys.modules:
    validate_environment_secrets(settings)
