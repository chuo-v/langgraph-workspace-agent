import os
import re
import sys

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.workspace_agent.core.provider import Provider

# resolve paths up to the repository root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))


class OrchestrationConfig(BaseModel):
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


class CISuiteConfig(BaseModel):
    name: str = Field(..., description="Display name for the GitHub Status Check context.")
    command: str = Field(..., description="The strictly whitelisted shell command to execute.")
    timeout_seconds: int = Field(
        default=60, description="OS-level hard timeout to prevent the agent from hanging."
    )


class WorkspaceConfig(BaseModel):
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
    provider: Provider
    model_name: str = Field(description="The exact string expected by the provider API")


class TierConfig(BaseModel):
    default_model: str
    available_models: dict[str, ModelDefinition]

    @field_validator("available_models")
    @classmethod
    def validate_telegram_safe_keys(cls, v):
        for key in v.keys():
            if not re.match(r"^[a-zA-Z0-9_]+$", key):
                raise ValueError(
                    f"Model alias '{key}' is invalid. Keys must be Telegram-safe "
                    "(alphanumeric and underscores only)."
                )
        return v


class LLMConfig(BaseModel):
    base_tier: TierConfig
    standard_tier: TierConfig
    frontier_tier: TierConfig


class WorkspaceAgentConfig(BaseModel):
    agent: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    llm: LLMConfig
    # enforce that at least one path is whitelisted for the sandbox
    allowed_paths: list[str] = Field(..., min_length=1)
    workspaces: dict[str, WorkspaceConfig]


def load_configuration() -> WorkspaceAgentConfig:
    """
    Loads and validates the YAML configuration.
    Requires a live config.yaml to be present to prevent the daemon from booting
    with invalid dummy paths from the example template.
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


def validate_environment_secrets(config: WorkspaceAgentConfig):
    """
    Cross-references the active YAML configuration with the environment variables
    to ensure all required secrets are present before the application boots.
    """
    missing_secrets = []

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
    def _check_provider_key(tier_config: TierConfig):
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

    # Check all tiers for their default models
    _check_provider_key(config.llm.base_tier)
    _check_provider_key(config.llm.standard_tier)
    _check_provider_key(config.llm.frontier_tier)

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


# Load the .env file early so os.getenv works during instantiation
load_dotenv()

# instantiate the singleton so other modules can import `settings`
settings = load_configuration()

# Validate the environment immediately after loading the YAML config,
# UNLESS we are running automated tests.
if "pytest" not in sys.modules:
    validate_environment_secrets(settings)
