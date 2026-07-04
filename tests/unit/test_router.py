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
    # 1. Setup Mock Environment
    # Mock settings.llm hierarchy to avoid dependency on the local yaml file
    mock_settings = mocker.patch("src.workspace_agent.orchestrator.router.settings")
    mock_settings.llm.base_tier.available_models = {"qwen_local": {}}
    mock_settings.llm.standard_tier.available_models = {"deepseek_fast": {}}
    mock_settings.llm.frontier_tier.available_models = {"gemini_pro": {}}

    # 2. Execute
    tier_qwen = get_tier_for_model("qwen_local")
    tier_deepseek = get_tier_for_model("deepseek_fast")
    tier_gemini = get_tier_for_model("gemini_pro")
    tier_unknown = get_tier_for_model("unknown_model")

    # 3. Assertions
    assert tier_qwen == TIER_BASE
    assert tier_deepseek == TIER_STANDARD
    assert tier_gemini == TIER_FRONTIER
    assert tier_unknown is None


# ==========================================
# Component: get_execution_llm
# ==========================================


def test_get_execution_llm_success_inferred_tier(mocker):
    """Green Path: Explicit model request correctly infers and overrides the base tier."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=TIER_FRONTIER
    )
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_frontier_llm"
    )

    # 2. Execute
    # We requested TIER_BASE, but provided a model key that belongs to TIER_FRONTIER
    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="gemini_pro")

    # 3. Assertions
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
    # 1. Setup Mock Environment
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_1"
    )

    # 2. Execute
    get_execution_llm(requested_tier=TIER_BASE, temperature=0.7)

    # 3. Assertions
    # verify the temperature and default connection parameters were passed down to base
    mock_factory.assert_any_call(
        tier_name="base", requested_model_key=None, temperature=0.7, max_retries=0, timeout=120.0
    )


def test_get_execution_llm_success_tier_1(mocker):
    """Green Path: Requests Tier 1 and successfully receives it."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_1")

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_BASE)

    # 3. Assertions
    assert llm == "mock_tier_1"


def test_get_execution_llm_success_tier_2(mocker):
    """Green Path: Directly requests Tier 2 and successfully receives it."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_2")

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_STANDARD)

    # 3. Assertions
    assert llm == "mock_tier_2"


def test_get_execution_llm_success_tier_3(mocker):
    """Green Path: Directly requests Tier 3 and successfully receives it."""
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value="mock_tier_3")

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_FRONTIER)

    # 3. Assertions
    assert llm == "mock_tier_3"


def test_get_execution_llm_fallback_escalate_t1_to_t2(mocker):
    """
    Edge Path: Requests Tier 1, but it returns None.
    Verifies the router cleanly escalates and returns Tier 2.
    """
    # 1. Setup Mock Environment
    # Simulate Base failing (None), Standard succeeding, Frontier failing (None)
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, "mock_tier_2", None]
    )

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_BASE)

    # 3. Assertions
    assert llm == "mock_tier_2"
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


def test_get_execution_llm_fallback_escalate_t1_to_t3(mocker):
    """
    Edge Path: Requests Tier 1, but both T1 and T2 are offline.
    Verifies the router escalates all the way to Tier 3.
    """
    # 1. Setup Mock Environment
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, None, "mock_tier_3"]
    )

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_BASE)

    # 3. Assertions
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
    # 1. Setup Mock Environment
    # Simulate Standard failing (None), then Frontier succeeding
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", side_effect=[None, "mock_tier_3"]
    )

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_STANDARD)

    # 3. Assertions
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
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=TIER_BASE
    )

    # Base fails, Standard succeeds, Frontier succeeds
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm",
        side_effect=[None, "mock_tier_2", "mock_tier_3"],
    )

    # 2. Execute
    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="qwen_local")

    # 3. Assertions
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
            call(
                tier_name="frontier",
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
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.orchestrator.router.get_tier_for_model", return_value=None)
    mock_factory = mocker.patch(
        "src.workspace_agent.orchestrator.router.get_llm", return_value="mock_base_llm"
    )

    # 2. Execute
    # Try to request a model not in config.yaml
    llm = get_execution_llm(requested_tier=TIER_BASE, requested_model_key="hallucinated_model")

    # 3. Assertions
    assert llm == "mock_base_llm"
    # It should drop the invalid key and instantiate the default base tier normally
    mock_factory.assert_any_call(
        tier_name="base", requested_model_key=None, temperature=0.0, max_retries=0, timeout=120.0
    )


def test_get_execution_llm_error_terminal_escalation(mocker):
    """
    Red Path: Requests Tier 3 (or escalates to it),
    but no Frontier model is configured. Verifies it raises the Terminal error.
    """
    # 1. Setup Mock Environment
    mocker.patch("src.workspace_agent.orchestrator.router.get_llm", return_value=None)

    # 2. Execute
    with pytest.raises(TerminalEscalationError) as exc_info:
        get_execution_llm(requested_tier=TIER_FRONTIER)

    # 3. Assertions
    assert "CRITICAL: Frontier model is required" in str(exc_info.value)


# ==========================================
# Component: get_intent_router
# ==========================================


def test_get_intent_router_success_standard_provider(mocker):
    """
    Green Path: Standard providers (OpenAI, Anthropic) use the default schema configuration.
    """
    # 1. Setup Mock Environment
    mock_llm = mocker.Mock()
    mock_llm.model_name = "gpt-4o"

    # We now mock the sequence list generator instead of the single execution fetcher
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_execution_llm_sequence",
        return_value=[mock_llm],
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.router.PromptManager.get", return_value="mock prompt"
    )

    # 2. Execute
    get_intent_router(TIER_BASE)

    # 3. Assertions
    # Verify standard schema calling without the method override
    mock_llm.with_structured_output.assert_called_once_with(RoutingDecision)


def test_get_intent_router_fallback_deepseek_gemini_workaround(mocker):
    """
    Edge Path: DeepSeek and Gemini APIs reject/deadlock on LangChain's default
    json_schema mode for complex schemas. Verify the router correctly forces function_calling.
    """
    # 1. Setup Mock Environment
    mock_llm = mocker.Mock()
    mock_llm.model_name = "gemini-1.5-pro"

    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_execution_llm_sequence",
        return_value=[mock_llm],
    )
    mocker.patch(
        "src.workspace_agent.orchestrator.router.PromptManager.get", return_value="mock prompt"
    )

    # 2. Execute
    get_intent_router(TIER_BASE)

    # 3. Assertions
    # Verify the explicit method override was applied
    mock_llm.with_structured_output.assert_called_once_with(
        RoutingDecision, method="function_calling"
    )


def test_get_intent_router_fallback_terminal_escalation(mocker):
    """Edge Path: Returns None if no LLM providers are available for the requested tier."""
    # 1. Setup Mock Environment
    mocker.patch(
        "src.workspace_agent.orchestrator.router.get_execution_llm_sequence",
        side_effect=TerminalEscalationError("No models available"),
    )

    # 2. Execute
    router = get_intent_router(TIER_BASE)

    # 3. Assertions
    assert router is None
