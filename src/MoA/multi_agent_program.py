"""
Declarative multi-agent pitch generation program powered by DSPy.

This module translates the procedural Mixture-of-Agents workflow from `src/MoA/MoA.py`
into a declarative pipeline that keeps the original agent diversity while enabling
targeted optimization of the final aggregation step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import dspy

# Constants reused from the procedural implementation for consistency.
NUM_AGENTS_DEFAULT = 3


@dataclass
class AgentPlan:
    """
    Structured task assignment for a single agent.
    """

    system_prompt: str
    user_prompt: str


@dataclass
class PlannerOutput:
    """
    Structured task plan covering all agents.
    """

    tasks: List[AgentPlan]
    raw_plan: str


@dataclass
class AgentRunResult:
    """
    Captured output from an individual agent pass.
    """

    response: str
    plan: AgentPlan


class TaskPlannerSignature(dspy.Signature):
    """
    Produce detailed instructions for each agent from pitch facts.
    """

    pitch_facts: str = dspy.InputField(desc="Facts describing the product and pitch context")
    agent_count: str = dspy.InputField(desc="Number of agents that must be coordinated")
    plan_text: str = dspy.OutputField(
        desc=(
            "Structured text mirroring the format in the original pipeline. "
            "Must include AGENT sections, SYSTEM and USER lines."
        )
    )


class AgentPitchSignature(dspy.Signature):
    """
    Generate a pitch section for a specific agent role.
    """

    system_prompt: str = dspy.InputField(
        desc="Agent-specific system prompt describing responsibilities and tone"
    )
    user_prompt: str = dspy.InputField(desc="Task details supplied to the agent")
    pitch_section: str = dspy.OutputField(
        desc="Completed pitch section tailored to the assigned responsibilities"
    )


class AggregatePitchSignature(dspy.Signature):
    """
    Synthesize the final pitch from all agent sections.
    """

    original_prompt: str = dspy.InputField(desc="Original user prompt passed to the taskmaster")
    agent_sections: str = dspy.InputField(
        desc="Combined content from all agent responses, clearly segmented"
    )
    final_pitch: str = dspy.OutputField(
        desc="Polished Shark Tank pitch incorporating the strongest ideas from every agent"
    )


class FeedbackSignature(dspy.Signature):
    """
    Provide constructive feedback on a generated pitch.
    """

    editor_prompt: str = dspy.InputField(desc="Evaluation framing with quality criteria")
    pitch_text: str = dspy.InputField(desc="Pitch that requires feedback")
    feedback: str = dspy.OutputField(desc="Actionable feedback that can drive improvements")


def parse_structured_plan(plan_text: str, agent_total: int) -> PlannerOutput:
    """
    Parse structured taskmaster output into per-agent assignments.

    The parser preserves backwards compatibility with the procedural implementation.
    """
    if not plan_text.strip():
        raise ValueError("Task planner returned an empty plan_text payload")

    sections = re.split(r"AGENT \d+:", plan_text, flags=re.IGNORECASE)
    tasks: List[AgentPlan] = []

    for section in sections[1:]:
        lines = [line.strip() for line in section.strip().split("\n") if line.strip()]
        system_prompt = ""
        user_prompt = ""
        for line in lines:
            if line.upper().startswith("SYSTEM:"):
                system_prompt = line.split(":", 1)[1].strip()
            elif line.upper().startswith("USER:"):
                user_prompt = line.split(":", 1)[1].strip()

        if not system_prompt:
            system_prompt = "You are a pitch expert focused on a critical section of the presentation."
        if not user_prompt:
            user_prompt = "Deliver a compelling pitch section aligned with your focus area."
        tasks.append(AgentPlan(system_prompt=system_prompt, user_prompt=user_prompt))

    while len(tasks) < agent_total:
        tasks.append(
            AgentPlan(
                system_prompt="You are a pitch expert focused on strengthening narrative cohesion.",
                user_prompt="Add a compelling section that reinforces the central pitch theme.",
            )
        )

    if len(tasks) > agent_total:
        tasks = tasks[:agent_total]

    return PlannerOutput(tasks=tasks, raw_plan=plan_text)


class TaskPlannerModule(dspy.Module):
    """
    Declarative replacement for the procedural taskmaster call.
    """

    def __init__(self, *, agent_total: int):
        super().__init__()
        self.agent_total = agent_total
        self.generate_plan = dspy.ChainOfThought(TaskPlannerSignature)

    def forward(self, *, pitch_facts: str) -> PlannerOutput:
        plan_prediction = self.generate_plan(
            pitch_facts=pitch_facts,
            agent_count=str(self.agent_total),
        )
        return parse_structured_plan(plan_prediction.plan_text, self.agent_total)


class AgentWorkerModule(dspy.Module):
    """
    Executes agent-specific prompts against dedicated models.
    """

    def __init__(self, *, agent_lms: Sequence[dspy.LM]):
        super().__init__()
        self.agent_lms = list(agent_lms)
        self.generate_section = dspy.ChainOfThought(AgentPitchSignature)

    def forward(self, *, planner_output: PlannerOutput) -> List[AgentRunResult]:
        if not planner_output.tasks:
            raise ValueError("Planner output must contain at least one agent assignment")

        responses: List[AgentRunResult] = []
        for index, plan in enumerate(planner_output.tasks):
            lm_index = min(index, len(self.agent_lms) - 1)
            agent_lm = self.agent_lms[lm_index]
            with dspy.context(lm=agent_lm):
                prediction = self.generate_section(
                    system_prompt=plan.system_prompt,
                    user_prompt=plan.user_prompt,
                )
            responses.append(AgentRunResult(response=prediction.pitch_section, plan=plan))

        return responses


class AggregatorModule(dspy.Module):
    """
    Aggregates agent responses into a single high-quality pitch.
    """

    def __init__(self):
        super().__init__()
        self.aggregate_pitch = dspy.ChainOfThought(AggregatePitchSignature)

    def forward(self, *, original_prompt: str, agent_runs: List[AgentRunResult]) -> dspy.Prediction:
        if not agent_runs:
            raise ValueError("Cannot aggregate pitch without agent responses")

        sections_payload = json.dumps(
            [
                {
                    "index": index + 1,
                    "system_prompt": run.plan.system_prompt,
                    "user_prompt": run.plan.user_prompt,
                    "response": run.response,
                }
                for index, run in enumerate(agent_runs)
            ],
            indent=2,
        )

        return self.aggregate_pitch(
            original_prompt=original_prompt,
            agent_sections=sections_payload,
        )


class FeedbackModule(dspy.Module):
    """
    Mirrors the evaluator/editor loop from the procedural implementation.
    """

    def __init__(self):
        super().__init__()
        self.generate_feedback = dspy.ChainOfThought(FeedbackSignature)

    def forward(self, *, editor_prompt: str, pitch_text: str) -> str:
        feedback_prediction = self.generate_feedback(
            editor_prompt=editor_prompt,
            pitch_text=pitch_text,
        )
        return feedback_prediction.feedback


class MultiAgentPitchProgram(dspy.Module):
    """
    Declarative Mixture-of-Agents program exposing the same high-level contract
    as the procedural `generate_pitch` function from `MoA.py`.
    """

    def __init__(
        self,
        *,
        planner: TaskPlannerModule,
        workers: AgentWorkerModule,
        aggregator: AggregatorModule,
        feedback_module: Optional[FeedbackModule] = None,
    ):
        super().__init__()
        self.planner = planner
        self.workers = workers
        self.aggregator = aggregator
        self.feedback_module = feedback_module

    def forward(
        self,
        *,
        pitch_facts: str,
        original_prompt: str,
        editor_prompt: Optional[str] = None,
        apply_feedback: bool = False,
        feedback_loops: int = 1,
    ) -> Dict[str, str]:
        if apply_feedback and not self.feedback_module:
            raise ValueError("Feedback module required when apply_feedback is True")

        if feedback_loops < 1:
            raise ValueError("feedback_loops must be at least 1")

        latest_pitch = ""
        latest_feedback = ""
        aggregated_plan: Optional[PlannerOutput] = None
        agent_runs: List[AgentRunResult] = []

        for loop_index in range(feedback_loops):
            loop_prompt = original_prompt
            if latest_pitch:
                loop_prompt = (
                    f"{original_prompt}\n\nPrevious pitch attempt:\n{latest_pitch}\n"
                )
            if latest_feedback:
                loop_prompt = (
                    f"{loop_prompt}\n\nFeedback to address:\n{latest_feedback}\n"
                )

            aggregated_plan = self.planner(pitch_facts=pitch_facts)
            agent_runs = self.workers(planner_output=aggregated_plan)
            aggregation_prediction = self.aggregator(
                original_prompt=loop_prompt,
                agent_runs=agent_runs,
            )
            latest_pitch = aggregation_prediction.final_pitch

            if apply_feedback and loop_index < feedback_loops - 1 and editor_prompt:
                latest_feedback = self.feedback_module(
                    editor_prompt=editor_prompt,
                    pitch_text=latest_pitch,
                )

        plan_summary = aggregated_plan.raw_plan if aggregated_plan else ""
        agent_payload = json.dumps(
            [
                {
                    "system_prompt": run.plan.system_prompt,
                    "user_prompt": run.plan.user_prompt,
                    "response": run.response,
                }
                for run in agent_runs
            ],
            indent=2,
        )

        return {
            "pitch": latest_pitch,
            "task_plan": plan_summary,
            "agent_payload": agent_payload,
            "feedback": latest_feedback,
        }


def build_multi_agent_program(
    *,
    agent_lms: Sequence[dspy.LM],
    aggregator_lm: dspy.LM,
    planner_lm: Optional[dspy.LM] = None,
    feedback_lm: Optional[dspy.LM] = None,
    agent_total: int = NUM_AGENTS_DEFAULT,
    enable_feedback: bool = True,
) -> MultiAgentPitchProgram:
    """
    Factory that mirrors the procedural wiring in `MoA.py`.
    """
    if not agent_lms:
        raise ValueError("At least one agent language model must be provided")

    planner_module = TaskPlannerModule(agent_total=agent_total)
    workers_module = AgentWorkerModule(agent_lms=agent_lms)
    aggregator_module = AggregatorModule()
    feedback_module = FeedbackModule() if enable_feedback and feedback_lm else None

    context_overrides: Dict[str, dspy.LM] = {}
    if planner_lm:
        context_overrides["planner"] = planner_lm
    if feedback_module and feedback_lm:
        context_overrides["feedback"] = feedback_lm

    # Ensure aggregator uses the desired LM during forward passes.
    aggregator_module.aggregate_pitch = dspy.ChainOfThought(
        AggregatePitchSignature,
        lm=aggregator_lm,
    )

    if planner_lm:
        planner_module.generate_plan = dspy.ChainOfThought(
            TaskPlannerSignature,
            lm=planner_lm,
        )

    workers_module.generate_section = dspy.ChainOfThought(
        AgentPitchSignature,
    )

    if feedback_module and feedback_lm:
        feedback_module.generate_feedback = dspy.ChainOfThought(
            FeedbackSignature,
            lm=feedback_lm,
        )

    return MultiAgentPitchProgram(
        planner=planner_module,
        workers=workers_module,
        aggregator=aggregator_module,
        feedback_module=feedback_module,
    )


def aggregate_training_examples(examples: Sequence[Dict[str, str]]) -> List[dspy.Example]:
    """
    Transform historical pitches into DSPy training examples for aggregator fine-tuning.
    """
    training_examples: List[dspy.Example] = []
    for example in examples:
        source_prompt = example.get("original_prompt", "")
        agent_payload = example.get("agent_payload", "")
        expected_pitch = example.get("pitch", "")

        if not source_prompt or not agent_payload or not expected_pitch:
            continue

        training_examples.append(
            dspy.Example(
                original_prompt=source_prompt,
                agent_sections=agent_payload,
                final_pitch=expected_pitch,
            )
        )

    return training_examples


def compile_aggregator(
    *,
    program: MultiAgentPitchProgram,
    training_examples: Sequence[dspy.Example],
    optimization_method: str = "bootstrap",
) -> AggregatorModule:
    """
    Optimize only the aggregation step using DSPy's compilation utilities.
    """
    if not training_examples:
        raise ValueError("At least one training example is required to compile the aggregator")

    optimizer: Optional[dspy.Optimizer] = None
    if optimization_method == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=lambda example, pred, trace=None: int(bool(pred.final_pitch)),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
    elif optimization_method == "bootstrap_random":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=lambda example, pred, trace=None: int(bool(pred.final_pitch)),
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=10,
        )
    else:
        raise ValueError(f"Unsupported optimization method: {optimization_method}")

    compiled_aggregator = optimizer.compile(
        program.aggregator,
        trainset=list(training_examples),
    )
    program.aggregator = compiled_aggregator
    return compiled_aggregator


