"""
DSPy-facing data interfaces for the MoA pipeline.

This module adapts the Shark Tank structured pitch dataset to the declarative
Mixture-of-Agents (MoA) program, allowing consistent validation and formatting.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from pitchLLM.utils import PitchInput, format_pitch_input


class MoAAgentInstruction(BaseModel):
    """Instruction bundle assigned to a single specialized agent."""

    agent_name: str = Field(..., description="Identifier used for logging and prompts.")
    system_message: str = Field(
        ..., description="System guidance describing the agent's perspective."
    )
    user_prompt: str = Field(..., description="Task-specific instructions for the agent.")


class MoAAgentContribution(BaseModel):
    """Captured output from an agent after completing its assignment."""

    agent_name: str = Field(..., description="Identifier of the contributing agent.")
    focus_area: str = Field(..., description="Brief description of the agent's focus.")
    content: str = Field(..., description="Agent's generated narrative segment.")


class MoAPitchPrediction(BaseModel):
    """Final MoA prediction containing the synthesized pitch and supporting context."""

    pitch: str = Field(..., description="Synthesized investor pitch.")
    agent_contributions: List[MoAAgentContribution] = Field(
        default_factory=list,
        description="List of agent outputs included for interpretability.",
    )


def parse_pitch_input(raw_input: Dict[str, Any]) -> PitchInput:
    """
    Validate and convert raw dataset dictionaries into PitchInput models.

    Args:
        raw_input: Dictionary retrieved from the structured pitch dataset.

    Returns:
        Validated PitchInput instance.

    Raises:
        ValueError: When validation fails due to missing or malformed fields.
    """
    if raw_input is None:
        raise ValueError("Pitch input payload is missing.")

    try:
        return PitchInput.model_validate(raw_input)
    except ValidationError as validation_error:
        raise ValueError(
            "Pitch input payload failed validation."
        ) from validation_error


def build_agent_context(pitch_input: PitchInput) -> str:
    """
    Generate the shared context string distributed to all agents.

    Args:
        pitch_input: Validated structured pitch description.

    Returns:
        String describing the company, problem, and solution for prompting.
    """
    return format_pitch_input(pitch_input)


def sanitise_agent_contribution(contribution: str) -> str:
    """
    Normalise agent outputs before aggregation.

    Args:
        contribution: Raw textual segment provided by an agent.

    Returns:
        Cleaned contribution suitable for downstream synthesis.
    """
    if contribution is None:
        return ""

    stripped = contribution.strip()
    return stripped


def example_to_pitch_input(example: Any) -> PitchInput:
    """
    Convert generic dataset records into validated PitchInput instances.

    Args:
        example: Either a DSPy Example or a plain dictionary.

    Returns:
        Validated PitchInput instance suitable for downstream modules.
    """
    if example is None:
        raise ValueError("Example payload is missing.")

    if hasattr(example, "input"):
        raw_payload = example.input
    elif isinstance(example, dict):
        raw_payload = example
    else:
        raise ValueError(
            "Example payload must expose an 'input' attribute or be a dictionary."
        )

    return parse_pitch_input(raw_payload)


