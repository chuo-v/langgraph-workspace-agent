import os

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.provider import Provider
from src.workspace_agent.llm.callbacks import get_langfuse_callback


def get_llm(
    tier_name: str,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
):
    """
    Dynamically initializes an LLM based on the YAML configuration.
    `tier_name` should be 'base', 'standard', or 'frontier'.
    """
    tier_config = getattr(settings.llm, f"{tier_name}_tier")
    model_key = requested_model_key or tier_config.default_model

    if model_key not in tier_config.available_models:
        raise ValueError(f"Model '{model_key}' is not configured for the {tier_name} tier.")

    model_def = tier_config.available_models[model_key]
    provider = model_def.provider

    cb = get_langfuse_callback()
    callbacks = [cb] if cb else []

    llm = None

    try:
        # -----------------------------------------
        # Ollama (Local)
        # -----------------------------------------
        if provider == Provider.OLLAMA:
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            ollama_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            llm = ChatOllama(
                base_url=ollama_url,
                model=model_def.model_name,
                temperature=temperature,
                max_retries=max_retries,
                callbacks=callbacks,
            )

        # -----------------------------------------
        # Anthropic Claude
        # -----------------------------------------
        elif provider == Provider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

                llm = ChatAnthropic(
                    model=model_def.model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_retries=max_retries,
                    timeout=timeout,
                    callbacks=callbacks,
                )

        # -----------------------------------------
        # Google Gemini
        # -----------------------------------------
        elif provider == Provider.GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

                gemini_temp = max(temperature, 0.1)
                llm = ChatGoogleGenerativeAI(
                    model=model_def.model_name,
                    google_api_key=api_key,
                    temperature=gemini_temp,
                    max_retries=max_retries,
                    timeout=timeout,
                    callbacks=callbacks,
                )

        # -----------------------------------------
        # OpenAI
        # -----------------------------------------
        elif provider == Provider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from langchain_openai import ChatOpenAI  # noqa: PLC0415

                llm = ChatOpenAI(
                    model=model_def.model_name,
                    openai_api_key=api_key,
                    temperature=temperature,
                    max_retries=max_retries,
                    timeout=timeout,
                    callbacks=callbacks,
                )

        # -----------------------------------------
        # DeepSeek
        # -----------------------------------------
        elif provider == Provider.DEEPSEEK:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                from langchain_openai import ChatOpenAI  # noqa: PLC0415

                llm = ChatOpenAI(
                    model=model_def.model_name,
                    openai_api_key=api_key,
                    openai_api_base="https://api.deepseek.com/v1",
                    temperature=temperature,
                    max_retries=max_retries,
                    timeout=timeout,
                    callbacks=callbacks,
                )

        return llm

    except Exception as e:
        print(f"Warning: Failed to initialize {tier_name} tier LLM ({model_key}): {e}")
        return None
