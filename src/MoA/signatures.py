"""
DSPy signatures describing the MoA multi-agent prompting workflow.
"""

import dspy


class PlanMoATasks(dspy.Signature):
    """
    Generate specialised agent roles and prompts for a Shark Tank pitch.

    Output must be a JSON array named `agents`, where each entry contains:
    - agent_name: string identifier
    - system_message: system instruction for the agent
    - user_prompt: user task prompt
    """

    context: str = dspy.InputField(
        desc="Structured facts describing the company, problem, and solution."
    )
    agent_plan: str = dspy.OutputField(
        desc=(
            "JSON with key 'agents', each containing agent_name, system_message, "
            "and user_prompt."
        )
    )


class WriteMoASection(dspy.Signature):
    """Draft a pitch segment given an agent role, prompt, and shared context."""

    context: str = dspy.InputField(desc="Structured Shark Tank facts.")
    system_message: str = dspy.InputField(
        desc="Short description of the agent's persona and responsibilities."
    )
    user_prompt: str = dspy.InputField(
        desc="Specific writing goals for this agent's contribution."
    )
    contribution: str = dspy.OutputField(
        desc="Concise, fact-grounded narrative segment supporting the final pitch."
    )


class AggregateMoAPitch(dspy.Signature):
    """
    Synthesize agent sections into a cohesive investor pitch.

    Optimisation should focus on this module to improve final pitch quality.
    """

    context: str = dspy.InputField(desc="Structured facts for factual grounding.")
    agent_contributions: str = dspy.InputField(
        desc="Serialised list of labelled agent sections."
    )
    pitch: str = dspy.OutputField(
        desc="Final MoA investor pitch ready for presentation."
    )

