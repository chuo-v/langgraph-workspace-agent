from enum import StrEnum

__all__ = ["Provider"]


class Provider(StrEnum):
    """Enumeration of supported LLM service providers.

    This enum serves as the authoritative contract for specifying, configuring,
    and routing model requests to the appropriate underlying language model
    provider integrations across the workspace agent.
    """

    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
