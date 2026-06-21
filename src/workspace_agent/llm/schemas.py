from typing import Literal

from pydantic import BaseModel, Field

# ==========================================
# Structured Output Schemas
# ==========================================


class RoutingDecision(BaseModel):
    """
    Defines the exact JSON structure the Tier 1 model must return
    during the parse_intent_node phase. Uses Structured Chain-of-Thought
    to prevent cognitive overload during disambiguation.
    """

    # force the model to extract entities first
    explicit_workspace_mentioned: str | None = Field(
        description="The exact workspace named in the prompt, or null if none was specified."
    )
    explicit_files_mentioned: list[str] = Field(
        default_factory=list,
        description="A list of any specific files or scripts mentioned.",
    )

    # force the model to evaluate the context
    is_context_missing: bool = Field(
        description=(
            "True if the user wants to edit or read a file but didn't provide enough "
            "information to locate it (e.g., missing workspace or filename)."
        )
    )
    clarification_question_to_ask: str | None = Field(
        description=(
            "If is_context_missing is True, write the exact question to ask the user to "
            "disambiguate."
        ),
    )

    # final routing decision
    intent_category: Literal["conversational", "workspace_operation", "workspace_read_only"] = (
        Field(
            description=(
                "'conversational' if the user is asking a general knowledge question or chatting. "
                "'workspace_operation' if the user is asking to modify code, edit files, or manage "
                "git. "
                "'workspace_read_only' if the user is asking to read files, run scripts, or search "
                "the workspace without modifying anything."
            )
        )
    )
    inferred_workspace: str | None = Field(
        description=(
            "The registered friendly name of the TARGET workspace the user intends to modify or "
            "focus on (e.g., 'langgraph-workspace-agent', 'kan-tabnet-experiments'). "
            "If the user asks to read from Workspace A to update Workspace B, return Workspace B. "
            "Must be null if the intent_category is 'conversational' or if the target is ambiguous."
        ),
    )
    task_complexity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Low: Explicit edits in known files. Medium: Standard tasks. "
            "High: Architectural, vague, or repo-wide understanding."
        ),
    )
    router_confidence: float = Field(
        description=(
            "A confidence score between 0.0 and 1.0 indicating how certain you are "
            "about the user's objective and the inferred workspace."
        ),
        ge=0.0,
        le=1.0,
    )
