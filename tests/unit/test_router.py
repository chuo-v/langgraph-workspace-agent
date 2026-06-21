from unittest.mock import call

import pytest

from src.workspace_agent.llm.schemas import RoutingDecision
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    TIER_FRONTIER,
    TIER_STANDARD,
    TerminalEscalationError,
    get_execution_llm,
    get_intent_router,
    get_tier_for_model,
)

# ==========================================
# Component: get_tier_for_model
# ==========================================


def test_get_tier_for_model_success_standard(mocker):
    """Green Path: Resolves model keys to their correct configured tiers."""
    # Mock settings.llm hierarchy to avoid dependency on the local yaml file
    mock_settings = mocker.patch("src.workspace_agent.orchestrator.router.settings")
    mock_settings.llm.base_tier.available_models = {"qwen_local": {}}
    mock_settings.llm.standard_tier.available_models = {"deepseek_fast": {}}
    mock_settings.llm.frontier_tier.available_models = {"gemini_pro": {}}

    assert get_tier_for_model("qwen_local") == TIER_BASE
    assert get_tier_for_model("deepseek_fast") == TIER_STANDARD
    assert get_tier_for_model("gemini_pro") == TIER_FRONTIER
    assert get_tier_for_model("unknown_model") is None


# ==========================================
# Component: get_execution_llm
# ==========================================


def test_get_execution_llm_success_inferred_tier(mocker):
    """Green Path: Explicit model request correctly infers and overrides the base tier."""
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=TIER_FRONTIER
    )
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_frontier_llm"
    )

    # We requested TIER_BASE, but provided a model key that belongs to TIER_FRONTIER
    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="gemini_pro")

    assert llm == "mock_frontier_llm"
    mock_factory.assert_called_once_with(
        tier_name="frontier",
        requested_model_key="gemini_pro",
        temperature=0.0,
        max_retries=0,
        timeout=120.0,
    )


def test_get_execution_llm_success_parameter_passing(mocker):
    """Green Path: Verifies configuration kwargs are successfully passed to the LLM builder."""
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_1"
    )

    get_execution_llm(requested_tier=TIER_BASE, temperature=0.7)

    # verify the temperature and default connection parameters were passed down
    mock_factory.assert_called_once_with(
        tier_name="base", requested_model_key=None, temperature=0.7, max_retries=0, timeout=120.0
    )


def test_get_execution_llm_success_tier_1(mocker):
    """Green Path: Requests Tier 1 and successfully receives it."""
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_1")
    llm = get_execution_llm(requested_tier=TIER_BASE)
    assert llm == "mock_tier_1"


def test_get_execution_llm_success_tier_2(mocker):
    """Green Path: Directly requests Tier 2 and successfully receives it."""
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_2")
    llm = get_execution_llm(requested_tier=TIER_STANDARD)
    assert llm == "mock_tier_2"


def test_get_execution_llm_success_tier_3(mocker):
    """Green Path: Directly requests Tier 3 and successfully receives it."""
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_3")
    llm = get_execution_llm(requested_tier=TIER_FRONTIER)
    assert llm == "mock_tier_3"


def test_get_execution_llm_fallback_escalate_t1_to_t2(mocker):
    """
    Edge Path: Requests Tier 1, but it returns None.
    Verifies the router cleanly escalates and returns Tier 2.
    """
    # Simulate Base failing (None), then Standard succeeding
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, "mock_tier_2"]
    )

    llm = get_execution_llm(requested_tier=TIER_BASE)

    assert llm == "mock_tier_2"
    assert mock_factory.call_count == 2
    mock_factory.assert_has_calls(
        [
            call(
                tier_name="base",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
            call(
                tier_name="standard",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
        ]
    )


def test_get_execution_llm_fallback_escalate_t1_to_t3(mocker):
    """
    Edge Path: Requests Tier 1, but both T1 and T2 are offline.
    Verifies the router escalates all the way to Tier 3.
    """
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, None, "mock_tier_3"]
    )

    llm = get_execution_llm(requested_tier=TIER_BASE)

    assert llm == "mock_tier_3"
    assert mock_factory.call_count == 3
    mock_factory.assert_has_calls(
        [
            call(
                tier_name="base",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
            call(
                tier_name="standard",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
            call(
                tier_name="frontier",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
        ]
    )


def test_get_execution_llm_fallback_escalate_t2_to_t3(mocker):
    """
    Edge Path: Requests Tier 2 directly, but it returns None.
    Verifies the router cleanly escalates to Tier 3.
    """
    # Simulate Standard failing (None), then Frontier succeeding
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, "mock_tier_3"]
    )

    llm = get_execution_llm(requested_tier=TIER_STANDARD)

    assert llm == "mock_tier_3"
    assert mock_factory.call_count == 2
    mock_factory.assert_has_calls(
        [
            call(
                tier_name="standard",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
            call(
                tier_name="frontier",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
        ]
    )


def test_get_execution_llm_fallback_escalation_drops_model_key(mocker):
    """Edge Path: When escalating due to failure, explicit model constraints MUST be dropped."""
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=TIER_BASE
    )

    # Base fails, so it escalates to standard
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, "mock_tier_2"]
    )

    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="qwen_local")

    assert llm == "mock_tier_2"
    mock_factory.assert_has_calls(
        [
            call(
                tier_name="base",
                requested_model_key="qwen_local",
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
            # CRITICAL: standard tier must NOT receive "qwen_local",
            # otherwise the provider will crash
            call(
                tier_name="standard",
                requested_model_key=None,
                temperature=0.0,
                max_retries=0,
                timeout=120.0,
            ),
        ]
    )


def test_get_execution_llm_fallback_invalid_model(mocker):
    """
    Edge Path: Invalid explicit model request prints warning and
    falls back to requested tier.
    """
    mocker.patch("src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=None)
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_base_llm"
    )

    # Try to request a model not in config.yaml
    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="hallucinated_model")

    assert llm == "mock_base_llm"
    # It should drop the invalid key and instantiate the default base tier normally
    mock_factory.assert_called_once_with(
        tier_name="base", requested_model_key=None, temperature=0.0, max_retries=0, timeout=120.0
    )


def test_get_execution_llm_error_terminal_escalation(mocker):
    """
    Red Path: Requests Tier 3 (or escalates to it),
    but no Frontier model is configured. Verifies it raises the Terminal error.
    """
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value=None)

    with pytest.raises(TerminalEscalationError) as exc_info:
        get_execution_llm(requested_tier=TIER_FRONTIER)

    assert "CRITICAL: Frontier model is required" in str(exc_info.value)


# ==========================================
# Component: get_intent_router
# ==========================================


def test_get_intent_router_success_standard_provider(mocker):
    """
    Green Path: Standard providers (OpenAI, Anthropic) use the default schema configuration.
    """
    mock_llm = mocker.Mock()
    mock_llm.model_name = "gpt-4o"

    mocker.patch("src.workspace_agent.orchestrator.router.get_execution_llm", return_value=mock_llm)
    mocker.patch(
        "src.workspace_agent.orchestrator.router.PromptManager.get", return_value="mock prompt"
    )

    get_intent_router(TIER_BASE)

    # Verify standard schema calling without the method override
    mock_llm.with_structured_output.assert_called_once_with(RoutingDecision)


def test_get_intent_router_fallback_deepseek_gemini_workaround(mocker):
    """
    Edge Path: DeepSeek and Gemini APIs reject/deadlock on LangChain's default
    json_schema mode for complex schemas. Verify the router correctly forces function_calling.
    """
    mock_llm = mocker.Mock()
    mock_llm.model_name = "gemini-1.5-pro"

    mocker.patch("src.workspace_agent.orchestrator.router.get_execution_llm", return_value=mock_llm)
    mocker.patch(
        "src.workspace_agent.orchestrator.router.PromptManager.get", return_value="mock prompt"
    )

    get_intent_router(TIER_BASE)

    # Verify the explicit method override was applied
    mock_llm.with_structured_output.assert_called_once_with(
        RoutingDecision, method="function_calling"
    )


def test_get_intent_router_fallback_terminal_escalation(mocker):
    """Edge Path: Returns None if no LLM providers are available for the requested tier."""
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_execution_llm",
        side_effect=TerminalEscalationError("No models available"),
    )

    router = get_intent_router(TIER_BASE)
    assert router is None
