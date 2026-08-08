import os
from typing import Any

import yaml

__all__ = ["PromptManager"]

# ==========================================
# Module Configuration & Constants
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PROMPTS_FILE = os.getenv("WORKSPACE_AGENT_PROMPTS_PATH") or os.path.join(
    PROJECT_ROOT, "prompts.yaml"
)


# ==========================================
# Prompt Manager
# ==========================================


class PromptManager:
    """Centralized registry and loader for YAML-based system prompts.

    Manages the lifecycle of loading, caching, and dynamically formatting prompt
    templates from the local filesystem for agent execution.
    """

    _prompts: dict[str, Any] | None = None

    @classmethod
    def get(cls, section: str, key: str, **kwargs: Any) -> str:
        """Retrieves and optionally formats a prompt template from the registry.

        Args:
            section: The top-level category or domain in the YAML prompt file.
            key: The specific identifier for the prompt template within the section.
            **kwargs: Dynamic variables used to format the prompt template string.

        Returns:
            The raw prompt string if no kwargs are provided, or the formatted string
            with keyword arguments injected into placeholders.

        Raises:
            FileNotFoundError: If the underlying prompt file cannot be located during
                initial load.
            KeyError: If formatting keyword arguments do not match placeholders in
                the template string.
        """
        prompts = cls.get_prompts()
        template = prompts.get(section, {}).get(key, "")
        if kwargs:
            return template.format(**kwargs)
        return template

    @classmethod
    def get_prompts(cls) -> dict[str, Any]:
        """Loads and caches the prompt definitions from the local filesystem.

        On initial invocation, reads the YAML file specified by PROMPTS_FILE and stores
        the parsed dictionary in class state for subsequent requests.

        Returns:
            A dictionary containing the parsed hierarchical structure of prompt templates.

        Raises:
            FileNotFoundError: If the configured PROMPTS_FILE does not exist on disk.
        """
        if cls._prompts is None:
            if not os.path.exists(PROMPTS_FILE):
                raise FileNotFoundError(f"Prompts file not found at {PROMPTS_FILE}")

            with open(PROMPTS_FILE, encoding="utf-8") as f:
                cls._prompts = yaml.safe_load(f)
        return cls._prompts
