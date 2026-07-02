import os

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
PROMPTS_FILE = os.getenv("WORKSPACE_AGENT_PROMPTS_PATH") or os.path.join(
    PROJECT_ROOT, "prompts.yaml"
)


class PromptManager:
    _prompts = None

    @classmethod
    def get_prompts(cls) -> dict:
        if cls._prompts is None:
            if not os.path.exists(PROMPTS_FILE):
                raise FileNotFoundError(f"Prompts file not found at {PROMPTS_FILE}")

            with open(PROMPTS_FILE, encoding="utf-8") as f:
                cls._prompts = yaml.safe_load(f)
        return cls._prompts

    @classmethod
    def get(cls, section: str, key: str, **kwargs) -> str:
        """
        Retrieves a prompt from the registry.
        If kwargs are provided, it dynamically formats the string.
        """
        prompts = cls.get_prompts()
        template = prompts.get(section, {}).get(key, "")
        if kwargs:
            return template.format(**kwargs)
        return template
