"""
DSPy signatures describing the MoA multi-agent prompting workflow.
"""

import dspy


class PlanMoATasks(dspy.Signature):
    """
    Generate specialised agent roles and prompts for a Shark Tank pitch.

    Output must be a JSON object with keys:
    - agents: list of entries containing
    - agent_name: string identifier
    - system_message: system instruction for the agent
    - user_prompt: user task prompt
    - model_id: canonical model identifier routed by the runtime
    - persona: concise persona label used for focused prompting
    - aggregator: optional object describing the final combiner persona/prompt.
    """

    context: str = dspy.InputField(
        desc="Structured facts describing the company, problem, and solution."
    )
    available_models: str = dspy.InputField(
        desc="JSON list or description of the candidate model identifiers the planner may assign."
    )
    agent_plan: str = dspy.OutputField(
        desc=(
            "JSON object with keys 'agents' and optional 'aggregator'. Each agent entry "
            "must include agent_name, system_message, user_prompt, model_id, and persona."
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
    persona_style: str = dspy.InputField(
        desc="Persona label or high-level style cue reinforcing differentiation."
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
    aggregator_guidance: str = dspy.InputField(
        desc="Persona-specific guidance for the aggregator combiner."
    )
    pitch: str = dspy.OutputField(
        desc="Final MoA investor pitch ready for presentation."
    )

