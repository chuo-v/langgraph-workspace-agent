from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.llm.factory import get_llm
from src.workspace_agent.llm.schemas import RoutingDecision

# Define the explicit public API contract to prevent namespace pollution
__all__ = [
    "TIER_BASE",
    "TIER_STANDARD",
    "TIER_FRONTIER",
    "TerminalEscalationError",
    "get_intent_router",
    "get_execution_llm",
    "get_execution_llm_sequence",
    "get_tier_for_model",
]

# ==========================================
# Constants & Exceptions
# ==========================================

TIER_BASE: int = 1
TIER_STANDARD: int = 2
TIER_FRONTIER: int = 3


class TerminalEscalationError(Exception):
    """Raised when a task requires Tier 3 reasoning (directly or via escalation)
    but no Frontier model is configured or available. Expected to transition
    the workflow into a failure state or alert the human operator.
    """

    pass


# ==========================================
# Routing Orchestration
# ==========================================


def get_intent_router(requested_tier: int = TIER_BASE, temperature: float = 0.0) -> Runnable | None:
    """Returns a unified LCEL Chain combining the System Prompt, LLM, and Output Parser.
    Includes built-in runtime fallbacks to seamlessly escalate to higher tiers if a
    lower-tier model (e.g., local Ollama) crashes or times out.

    Expected State Transitions: Instantiates and returns an executable Runnable sequence
    that yields a RoutingDecision. Returns None if all LLM tiers fail to initialize.
    Exceptions: Catches TerminalEscalationError internally and safely returns None.
    """
    try:
        # Dynamically align socket timeout with the orchestrator thread pool
        timeout_val: float = float(settings.agent.router_timeout_seconds) - 2.0
        if timeout_val <= 0:
            timeout_val = 60.0

        # Note: Max retries is set low (1) so it fails fast and triggers the fallback
        llms: list[Runnable] = get_execution_llm_sequence(
            requested_tier=requested_tier,
            temperature=temperature,
            max_retries=1,
            timeout=timeout_val,
        )
    except TerminalEscalationError:
        return None

    if not llms:
        return None

    # 1. Dynamically load the prompts from the registry
    valid_workspaces: list[str] = list(settings.workspaces.keys())

    sys_prompt: str = PromptManager.get(
        "router", "system_prompt", valid_workspaces=valid_workspaces
    ).strip()

    # We do not pass kwargs here so the `{instruction}` tokens remain intact for LangChain to format
    human_prompt: str = PromptManager.get("router", "human_prompt").strip()

    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            ("human", human_prompt),
        ]
    )

    # 2. Build the fallback chain sequence
    chains: list[Runnable] = []
    for llm in llms:
        structured_llm: Runnable = _apply_routing_structured_output(llm)
        chains.append(prompt | structured_llm)

    # The primary chain is the requested tier (e.g., Base Qwen)
    primary_chain: Runnable = chains[0]

    # The fallbacks are the subsequent tiers (e.g., Standard DeepSeek, Frontier Gemini)
    if len(chains) > 1:
        return primary_chain.with_fallbacks(chains[1:])

    return primary_chain


def _apply_routing_structured_output(llm: Runnable) -> Runnable:
    """Safely applies the RoutingDecision structured output schema based on the provider.
    Architectural Intent: Abstracts provider-specific function calling paradigms
    (e.g., DeepSeek/Gemini vs standard API) away from the main orchestration logic.
    """
    model_name: str = getattr(llm, "model_name", "")
    if "deepseek" in str(model_name).lower() or "gemini" in str(model_name).lower():
        return llm.with_structured_output(RoutingDecision, method="function_calling")
    return llm.with_structured_output(RoutingDecision)


# ==========================================
# LLM Execution & Provisioning
# ==========================================


def get_execution_llm(
    requested_tier: int = TIER_BASE,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
) -> Runnable | None:
    """Legacy wrapper for nodes that expect a single LLM to execute tools on.
    Returns the highest priority LLM available based on tier sequences.

    Expected State Transitions: Extracts and returns the primary LLM configuration from a sequence.
    Exceptions: May propagate TerminalEscalationError if the requested tier requires a
    Frontier model and none are available.
    """
    llms: list[Runnable] = get_execution_llm_sequence(
        requested_tier=requested_tier,
        requested_model_key=requested_model_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )
    return llms[0] if llms else None


def get_execution_llm_sequence(
    requested_tier: int = TIER_BASE,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
) -> list[Runnable]:
    """Returns a sequence of initialized LLM clients starting from the requested tier,
    escalating up to Tier 3. Used to construct robust runtime fallback chains.

    Expected State Transitions: Aggregates a list of ordered LLM Runnables for fallbacks.
    Exceptions: Raises TerminalEscalationError if the frontier tier is explicitly
    required but no models are successfully loaded across any tier.
    """
    llms: list[Runnable] = []

    # 1. Automatic Tier Inference for Explicit Models
    if requested_model_key:
        inferred_tier: int | None = get_tier_for_model(requested_model_key)
        if inferred_tier:
            requested_tier = inferred_tier
        else:
            print(
                f"Warning: Model '{requested_model_key}' not found in any tier. "
                "Ignoring explicit override."
            )
            requested_model_key = None

    # ---------------------------------------------------------
    # Tier 1 (base/local)
    # ---------------------------------------------------------
    if requested_tier <= TIER_BASE:
        llm = get_llm(
            tier_name="base",
            requested_model_key=(requested_model_key if requested_tier == TIER_BASE else None),
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
        if llm:
            llms.append(llm)
        else:
            print("Warning: Tier 1 (Base) initialization failed. Escalating standard tier.")
            requested_tier = TIER_STANDARD  # trigger escalation
            requested_model_key = None

    # ---------------------------------------------------------
    # Tier 2 (standard/throughput)
    # ---------------------------------------------------------
    if requested_tier <= TIER_STANDARD:
        llm = get_llm(
            tier_name="standard",
            requested_model_key=(requested_model_key if requested_tier == TIER_STANDARD else None),
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
        if llm:
            llms.append(llm)
        else:
            print("Warning: Tier 2 (Standard) initialization failed. Escalating frontier tier.")
            requested_tier = TIER_FRONTIER
            requested_model_key = None

    # ---------------------------------------------------------
    # Tier 3 (frontier/premium)
    # ---------------------------------------------------------
    if requested_tier <= TIER_FRONTIER:
        llm = get_llm(
            tier_name="frontier",
            requested_model_key=(requested_model_key if requested_tier == TIER_FRONTIER else None),
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
        if llm:
            llms.append(llm)
        elif not llms:
            # circuit breaker only if NO models were loaded at all
            raise TerminalEscalationError(
                "CRITICAL: Frontier model is required but not configured, "
                "and no lower tiers are available."
            )

    return llms


def get_tier_for_model(model_key: str) -> int | None:
    """Dynamically resolves which tier a specific model is configured in.
    Architectural Intent: Serves as a mapping helper to ensure explicit string
    overrides properly sync with internal integer orchestration tiers.
    """
    if model_key in settings.llm.base_tier.available_models:
        return TIER_BASE
    if model_key in settings.llm.standard_tier.available_models:
        return TIER_STANDARD
    if model_key in settings.llm.frontier_tier.available_models:
        return TIER_FRONTIER
    return None
