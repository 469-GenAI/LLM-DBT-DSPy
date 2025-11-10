"""
MoA DSPy program assembling planning, agent writing, and aggregation modules.
"""

import json
from typing import Dict, List

import dspy

from MoA.data_interfaces import (
    MoAAgentContribution,
    MoAAgentInstruction,
    build_agent_context,
    example_to_pitch_input,
    sanitise_agent_contribution,
)
from MoA.signatures import AggregateMoAPitch, PlanMoATasks, WriteMoASection


def _load_agent_plan(agent_plan_payload: str) -> List[MoAAgentInstruction]:
    """
    Parse task planner JSON into validated agent instructions.

    Args:
        agent_plan_payload: JSON document returned by the task planner.

    Returns:
        List of MoAAgentInstruction objects.
    """
    if agent_plan_payload is None:
        raise ValueError("Task planner returned no payload.")

    try:
        parsed = json.loads(agent_plan_payload)
    except json.JSONDecodeError as decode_error:
        raise ValueError("Task planner payload is not valid JSON.") from decode_error

    agents = parsed.get("agents", [])
    if not isinstance(agents, list) or not agents:
        raise ValueError("Task planner produced an empty agent plan.")

    instructions = []
    for raw_agent in agents:
        if not isinstance(raw_agent, dict):
            continue

        if not {"agent_name", "system_message", "user_prompt"} <= set(raw_agent):
            continue

        instructions.append(
            MoAAgentInstruction(
                agent_name=str(raw_agent["agent_name"]).strip(),
                system_message=str(raw_agent["system_message"]).strip(),
                user_prompt=str(raw_agent["user_prompt"]).strip(),
            )
        )

    if not instructions:
        raise ValueError("Task planner payload contained no valid agent entries.")

    return instructions


class MoAMixtureProgram(dspy.Module):
    """
    Declarative Mixture-of-Agents program optimised for the final aggregation step.
    """

    def __init__(self, num_agents: int = 3):
        super().__init__()
        self.num_agents = num_agents
        self.task_planner = dspy.ChainOfThought(PlanMoATasks)
        self.agent_writer = dspy.ChainOfThought(WriteMoASection)
        self.pitch_aggregator = dspy.ChainOfThought(AggregateMoAPitch)

    def forward(self, input: Dict) -> dspy.Prediction:
        """
        Execute the planner, agent writer, and aggregator to produce a pitch.

        Args:
            input: Structured Shark Tank facts dictionary.

        Returns:
            MoAPitchPrediction containing the final pitch and agent contributions.
        """
        pitch_input = example_to_pitch_input(input)
        context = build_agent_context(pitch_input)

        plan_prediction = self.task_planner(context=context)
        instructions = _load_agent_plan(plan_prediction.agent_plan)[: self.num_agents]

        contributions: List[MoAAgentContribution] = []
        serialized_sections: List[str] = []

        for instruction in instructions:
            writer_prediction = self.agent_writer(
                context=context,
                system_message=instruction.system_message,
                user_prompt=instruction.user_prompt,
            )
            cleaned_content = sanitise_agent_contribution(writer_prediction.contribution)

            if not cleaned_content:
                continue

            contribution = MoAAgentContribution(
                agent_name=instruction.agent_name,
                focus_area=instruction.user_prompt,
                content=cleaned_content,
            )
            contributions.append(contribution)
            serialized_sections.append(
                f"{instruction.agent_name}\nROLE: {instruction.system_message}\nPROMPT: {instruction.user_prompt}\nCONTENT:\n{cleaned_content}"
            )

        if not contributions:
            raise ValueError("No agent contributions were generated.")

        aggregated_prediction = self.pitch_aggregator(
            context=context,
            agent_contributions="\n\n---\n\n".join(serialized_sections),
        )

        final_pitch = sanitise_agent_contribution(aggregated_prediction.pitch)
        if not final_pitch:
            raise ValueError("Pitch aggregation produced an empty result.")

        return dspy.Prediction(
            pitch=final_pitch,
            agent_contributions=[contribution.model_dump() for contribution in contributions],
        )


class MoAGenerator:
    """
    Factory wiring the MoA DSPy modules with a configured language model.
    """

    def __init__(self, lm: dspy.LM, num_agents: int = 3):
        """
        Args:
            lm: Configured dspy.LM for generation.
            num_agents: Number of agents to plan for each pitch.
        """
        if lm is None:
            raise ValueError("Language model is required for MoAGenerator.")

        self.lm = lm
        self.num_agents = num_agents
        self.program = MoAMixtureProgram(num_agents=num_agents)

        dspy.configure(lm=lm)

