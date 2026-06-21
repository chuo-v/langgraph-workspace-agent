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


def get_execution_llm(
    requested_tier: int = TIER_BASE,
    requested_model_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 0,
    timeout: float = 120.0,
):
    """
    Returns the appropriate LLM client based on the requested tier.
    Implements the 'Upward Escalation Rule' to guarantee fault tolerance.
    """
    llm = None

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
        if not llm:
            print("Warning: Tier 1 (Base) unavailable. Escalating to Tier 2.")
            requested_tier = TIER_STANDARD  # trigger escalation
            requested_model_key = None  # drop explicit model constraint on escalation

    # ---------------------------------------------------------
    # Tier 2 (standard/throughput)
    # ---------------------------------------------------------
    if requested_tier == TIER_STANDARD:
        llm = get_llm(
            tier_name="standard",
            requested_model_key=(requested_model_key if requested_tier == TIER_STANDARD else None),
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
        if not llm:
            print("Warning: Tier 2 (Standard) unavailable. Escalating to Tier 3.")
            requested_tier = TIER_FRONTIER  # trigger escalation
            requested_model_key = None

    # ---------------------------------------------------------
    # Tier 3 (frontier/premium)
    # ---------------------------------------------------------
    if requested_tier == TIER_FRONTIER:
        llm = get_llm(
            tier_name="frontier",
            requested_model_key=(requested_model_key if requested_tier == TIER_FRONTIER else None),
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
        if not llm:
            # circuit breaker
            raise TerminalEscalationError(
                "CRITICAL: Frontier model is required but not configured. Task execution aborted."
            )

    return llm


def get_intent_router(requested_tier: int = TIER_BASE, temperature: float = 0.0) -> Runnable | None:
    """
    Returns a unified LCEL Chain combining the System Prompt, LLM, and Output Parser.
    Used to route and classify user intents.
    """
    try:
        # Dynamically align socket timeout with the orchestrator thread pool
        timeout_val = float(settings.agent.router_timeout_seconds) - 2.0
        if timeout_val <= 0:
            timeout_val = 60.0

        llm = get_execution_llm(
            requested_tier=requested_tier,
            temperature=temperature,
            max_retries=2,
            timeout=timeout_val,
        )
    except TerminalEscalationError:
        return None

    if not llm:
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

    # 2. Configure structured output safely
    # DeepSeek and Gemini APIs strictly reject or deadlock on LangChain's default
    # 'json_schema' format for complex schemas. We must force 'function_calling'.
    model_name = getattr(llm, "model_name", "")
    if "deepseek" in str(model_name).lower() or "gemini" in str(model_name).lower():
        structured_llm = llm.with_structured_output(RoutingDecision, method="function_calling")
    else:
        structured_llm = llm.with_structured_output(RoutingDecision)

    # 3. Return the unified chain
    return prompt | structured_llm
