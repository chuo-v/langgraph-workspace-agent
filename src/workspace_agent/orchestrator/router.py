from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.llm.factory import get_llm
from src.workspace_agent.llm.schemas import RoutingDecision

# ==========================================
# Tier Constants
# ==========================================
TIER_BASE = 1
TIER_STANDARD = 2
TIER_FRONTIER = 3


class TerminalEscalationError(Exception):
    """
    Raised when a task requires Tier 3 reasoning (directly or via escalation)
    but no Frontier model is configured or available.
    """

    pass


def get_tier_for_model(model_key: str) -> int | None:
    """Helper to dynamically resolve which tier a specific model is configured in."""
    if model_key in settings.llm.base_tier.available_models:
        return TIER_BASE
    if model_key in settings.llm.standard_tier.available_models:
        return TIER_STANDARD
    if model_key in settings.llm.frontier_tier.available_models:
        return TIER_FRONTIER
    return None


def get_execution_llm_sequence(
    requested_tier: int = TIER_BASE,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
) -> list:
    """
    Returns a sequence of initialized LLM clients starting from the requested tier,
    escalating up to Tier 3. Used to construct robust runtime fallback chains.
    """
    llms = []

    # 1. Automatic Tier Inference for Explicit Models
    if requested_model_key:
        inferred_tier = get_tier_for_model(requested_model_key)
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


def get_execution_llm(
    requested_tier: int = TIER_BASE,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
):
    """
    Legacy wrapper for nodes that expect a single LLM to execute tools on.
    Returns the highest priority LLM available.
    """
    llms = get_execution_llm_sequence(
        requested_tier=requested_tier,
        requested_model_key=requested_model_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
    )
    return llms[0] if llms else None


def _apply_routing_structured_output(llm):
    """Helper to safely apply the structured output schema based on the provider."""
    model_name = getattr(llm, "model_name", "")
    if "deepseek" in str(model_name).lower() or "gemini" in str(model_name).lower():
        return llm.with_structured_output(RoutingDecision, method="function_calling")
    return llm.with_structured_output(RoutingDecision)


def get_intent_router(requested_tier: int = TIER_BASE, temperature: float = 0.0) -> Runnable | None:
    """
    Returns a unified LCEL Chain combining the System Prompt, LLM, and Output Parser.
    Includes built-in runtime fallbacks to seamlessly escalate to higher tiers if a
    lower-tier model (e.g., local Ollama) crashes or times out.
    """
    try:
        # Dynamically align socket timeout with the orchestrator thread pool
        timeout_val = float(settings.agent.router_timeout_seconds) - 2.0
        if timeout_val <= 0:
            timeout_val = 60.0

        # Note: Max retries is set low (1) so it fails fast and triggers the fallback
        llms = get_execution_llm_sequence(
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
    valid_workspaces = list(settings.workspaces.keys())

    sys_prompt = PromptManager.get(
        "router", "system_prompt", valid_workspaces=valid_workspaces
    ).strip()

    # We do not pass kwargs here so the `{instruction}` tokens remain intact for LangChain to format
    human_prompt = PromptManager.get("router", "human_prompt").strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            ("human", human_prompt),
        ]
    )

    # 2. Build the fallback chain sequence
    chains = []
    for llm in llms:
        structured_llm = _apply_routing_structured_output(llm)
        chains.append(prompt | structured_llm)

    # The primary chain is the requested tier (e.g., Base Qwen)
    primary_chain = chains[0]

    # The fallbacks are the subsequent tiers (e.g., Standard DeepSeek, Frontier Gemini)
    if len(chains) > 1:
        return primary_chain.with_fallbacks(chains[1:])

    return primary_chain
