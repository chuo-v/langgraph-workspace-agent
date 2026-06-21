import pytest

from src.workspace_agent.core.provider import Provider
from src.workspace_agent.llm.factory import get_llm

# ==========================================
# Helper: Mocking Configurations
# ==========================================


@pytest.fixture
def mock_settings(mocker):
    """Helper to dynamically inject available models into the factory settings."""

    def _setup_mock(provider: Provider, model_name: str, tier: str = "standard"):
        mock_model_def = mocker.Mock()
        mock_model_def.provider = provider
        mock_model_def.model_name = model_name

        mock_tier_config = mocker.Mock()
        mock_tier_config.default_model = "test_model"
        mock_tier_config.available_models = {"test_model": mock_model_def}

        mocker.patch("src.workspace_agent.llm.factory.getattr", return_value=mock_tier_config)

    return _setup_mock


# ==========================================
# Component: get_llm (LLM Factory)
# ==========================================


def test_get_llm_success_ollama_base(mocker, mock_settings):
    """Green Path: Successfully initializes an Ollama local client."""
    mock_settings(Provider.OLLAMA, "qwen2.5:32b", "base")
    mock_chat = mocker.patch("langchain_ollama.ChatOllama")

    llm = get_llm("base", "test_model")

    assert llm is not None
    mock_chat.assert_called_once()
    assert mock_chat.call_args[1]["model"] == "qwen2.5:32b"


def test_get_llm_success_deepseek_standard(mocker, mock_settings):
    """Green Path: Successfully initializes DeepSeek via the OpenAI client."""
    mock_settings(Provider.DEEPSEEK, "deepseek-chat", "standard")
    mocker.patch("os.getenv", return_value="sk-deepseek")
    mock_chat = mocker.patch("langchain_openai.ChatOpenAI")

    llm = get_llm("standard", "test_model")

    assert llm is not None
    mock_chat.assert_called_once()
    assert mock_chat.call_args[1]["openai_api_base"] == "https://api.deepseek.com/v1"


def test_get_llm_success_openai_standard(mocker, mock_settings):
    """Green Path: Successfully initializes an OpenAI client."""
    mock_settings(Provider.OPENAI, "gpt-4o", "standard")
    mocker.patch("os.getenv", return_value="sk-12345")
    mock_chat = mocker.patch("langchain_openai.ChatOpenAI")

    llm = get_llm("standard", "test_model")

    assert llm is not None
    mock_chat.assert_called_once()
    assert mock_chat.call_args[1]["openai_api_key"] == "sk-12345"


def test_get_llm_success_anthropic_frontier(mocker, mock_settings):
    """Green Path: Successfully initializes Anthropic."""
    mock_settings(Provider.ANTHROPIC, "claude-3-7", "frontier")
    mocker.patch("os.getenv", return_value="sk-ant-123")
    mock_chat = mocker.patch("langchain_anthropic.ChatAnthropic")

    llm = get_llm("frontier", "test_model")

    assert llm is not None
    mock_chat.assert_called_once()
    assert mock_chat.call_args[1]["model"] == "claude-3-7"


def test_get_llm_success_gemini_frontier(mocker, mock_settings):
    """Green Path: Successfully initializes a Gemini client with temperature constraints."""
    mock_settings(Provider.GEMINI, "gemini-2.5-pro", "frontier")
    mocker.patch("os.getenv", return_value="AIza-12345")
    mock_chat = mocker.patch("langchain_google_genai.ChatGoogleGenerativeAI")

    # Pass exactly 0.0 to test the gemini_temp = max(temperature, 0.1) safeguard
    llm = get_llm("frontier", "test_model", temperature=0.0)

    assert llm is not None
    mock_chat.assert_called_once()
    # Verify the temperature floor successfully overrode the 0.0 request
    assert mock_chat.call_args[1]["temperature"] == 0.1


def test_get_llm_fallback_missing_api_key(mocker, mock_settings):
    """Edge Path: Returns None gracefully if the API key is missing."""
    mock_settings(Provider.OPENAI, "gpt-4o", "standard")
    mocker.patch("os.getenv", return_value=None)  # Simulate missing key

    llm = get_llm("standard", "test_model")

    assert llm is None


def test_get_llm_fallback_initialization_crash(mocker, mock_settings):
    """Edge Path: Returns None gracefully if the underlying LangChain client crashes."""
    mock_settings(Provider.OLLAMA, "qwen", "base")
    # Simulate an unexpected crash from the library or socket
    mocker.patch(
        "langchain_ollama.ChatOllama", side_effect=Exception("Library missing or socket dead")
    )

    llm = get_llm("base", "test_model")

    assert llm is None


def test_get_llm_error_invalid_model(mocker):
    """Red Path: Raises ValueError if the requested model key does not exist."""
    mock_tier_config = mocker.Mock()
    mock_tier_config.default_model = "default_key"
    mock_tier_config.available_models = {}

    mocker.patch("src.workspace_agent.llm.factory.getattr", return_value=mock_tier_config)

    with pytest.raises(ValueError) as exc:
        get_llm("standard", requested_model_key="missing_key")

    assert "not configured" in str(exc.value)
