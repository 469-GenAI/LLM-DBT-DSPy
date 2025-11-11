"""
Command-line runner for the DSPy Mixture-of-Agents program.

The runner keeps the multi-agent design from `MoA.py` while delegating execution
to the declarative modules defined in `multi_agent_program.py`. Only the final
aggregation step is compiled/optimized, matching the design goal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import dspy
from dotenv import load_dotenv

from .multi_agent_program import (
    NUM_AGENTS_DEFAULT,
    MultiAgentPitchProgram,
    aggregate_training_examples,
    build_multi_agent_program,
    compile_aggregator,
)
from .sharktank_utils import load_facts

# Default prompts lifted from the procedural implementation for continuity.
TASKMASTER_SYSTEM_PROMPT = """
You are a sharktank pitch director coordinating {agent_total} agents.
Break down the pitch creation into {agent_total} distinct tasks and assign each to an agent.

For each agent, provide:
1. A system role description
2. A specific user task prompt

Format your response as structured text (NOT JSON):

AGENT 1:
SYSTEM: [role description for agent 1]
USER: [specific task prompt for agent 1]

AGENT 2:
SYSTEM: [role description for agent 2]
USER: [specific task prompt for agent 2]

AGENT 3:
SYSTEM: [role description for agent 3]
USER: [specific task prompt for agent 3]

Be specific about what each agent should focus on. No JSON, no code blocks, just structured text.
"""

AGGREGATOR_SYSTEM_PROMPT = """
You have been provided with a set of responses from various open-source models to the latest user query. 
Your task is to synthesize these responses into a single, high-quality response. Output a script for prospective entrepreneurs to use.
It is crucial to critically evaluate the information provided in these responses,
recognizing that some of it may be biased or incorrect. 
Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:
"""

EDITOR_PROMPT = """
You are a pitch editor. You will be given a pitch. Evaluate its strength. 
Give constructive feedback on how to improve the pitch a for shark tank pitch.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Here is the pitch:
"""

# Default Groq-backed model selections mirroring the procedural script.
DEFAULT_PLANNER_MODEL = "groq/llama-3.1-8b-instant"
DEFAULT_AGGREGATOR_MODEL = "groq/llama-3.1-8b-instant"
DEFAULT_FEEDBACK_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_AGENT_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/qwen3-32b",
]

# Respect guidance on API rate limits by default.
RATE_LIMIT_SECONDS = 2.5


def parse_arguments() -> argparse.Namespace:
    """
    Parse CLI arguments controlling the pipeline execution.
    """
    parser = argparse.ArgumentParser(description="Run the DSPy Mixture-of-Agents pipeline.")
    parser.add_argument(
        "--num-agents",
        type=int,
        default=NUM_AGENTS_DEFAULT,
        help="Number of agents to coordinate (default: 3).",
    )
    parser.add_argument(
        "--agent-models",
        type=str,
        nargs="*",
        default=DEFAULT_AGENT_MODELS,
        help="Groq model identifiers for agent workers.",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default=DEFAULT_PLANNER_MODEL,
        help="Groq model identifier for the task planner.",
    )
    parser.add_argument(
        "--aggregator-model",
        type=str,
        default=DEFAULT_AGGREGATOR_MODEL,
        help="Groq model identifier for the aggregator (optimized step).",
    )
    parser.add_argument(
        "--feedback-model",
        type=str,
        default=DEFAULT_FEEDBACK_MODEL,
        help="Groq model identifier for the evaluator/editor.",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable feedback loop between iterations.",
    )
    parser.add_argument(
        "--feedback-loops",
        type=int,
        default=1,
        help="Number of self-improvement loops to run (default: 1).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_SECONDS,
        help="Delay between major API calls (seconds).",
    )
    parser.add_argument(
        "--scenario-key",
        type=str,
        default=None,
        help="Specific fact key to run. Defaults to a random selection.",
    )
    parser.add_argument(
        "--facts-path",
        type=str,
        default=None,
        help="Override path to facts JSON.",
    )
    parser.add_argument(
        "--aggregator-training",
        type=str,
        default=None,
        help="Path to JSONL/JSON file with historical agent payloads for aggregator compilation.",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "bootstrap_random"],
        help="Aggregator optimization strategy.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to persist the generated pitch payload.",
    )
    return parser.parse_args()


def load_training_payload(path: Path) -> List[Dict[str, str]]:
    """
    Load training examples used for aggregator compilation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training payload not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = file.read()

    try:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON content in {path}") from exc


def build_lm(model_name: str, temperature: float = 1.0) -> dspy.LM:
    """
    Helper to instantiate Groq chat models.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is required to run the DSPy Mixture-of-Agents pipeline.")

    return dspy.LM(
        model_name,
        model_type="chat",
        api_key=api_key,
        temperature=temperature,
    )


def build_agent_lms(model_names: Sequence[str]) -> List[dspy.LM]:
    """
    Build per-agent LM instances.
    """
    return [build_lm(model_name) for model_name in model_names]


def select_fact_key(facts_store: Dict[str, Dict[str, str]], desired_key: Optional[str]) -> str:
    """
    Pick a fact key deterministically or randomly.
    """
    if desired_key:
        if desired_key not in facts_store:
            raise KeyError(f"Requested scenario key '{desired_key}' not found in facts store.")
        return desired_key

    if not facts_store:
        raise ValueError("Facts store is empty. Provide a valid facts dataset.")

    return random.choice(list(facts_store.keys()))


def build_original_prompt(facts: Dict[str, str]) -> str:
    """
    Mirror the procedural user prompt structure.
    """
    return (
        "My name is Paul McCarthney.\n"
        f"Here are the facts of my product: {json.dumps(facts, indent=2)}"
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, str]:
    """
    Execute the DSPy Mixture-of-Agents pipeline based on parsed arguments.
    """
    load_dotenv()

    agent_models = args.agent_models or DEFAULT_AGENT_MODELS
    if len(agent_models) < args.num_agents:
        raise ValueError("Provide at least as many agent models as requested agents.")

    agent_lms = build_agent_lms(agent_models[: args.num_agents])
    planner_lm = build_lm(args.planner_model, temperature=0.7)
    aggregator_lm = build_lm(args.aggregator_model, temperature=0.7)
    feedback_lm = None if args.no_feedback else build_lm(args.feedback_model, temperature=0.3)

    dspy.configure(lm=aggregator_lm, track_usage=True)

    program: MultiAgentPitchProgram = build_multi_agent_program(
        agent_lms=agent_lms,
        aggregator_lm=aggregator_lm,
        planner_lm=planner_lm,
        feedback_lm=feedback_lm,
        agent_total=args.num_agents,
        enable_feedback=not args.no_feedback,
    )

    facts_store = load_facts(args.facts_path)
    scenario_key = select_fact_key(facts_store, args.scenario_key)
    scenario_facts = facts_store[scenario_key]

    original_prompt = build_original_prompt(scenario_facts)

    if args.aggregator_training:
        training_path = Path(args.aggregator_training)
        training_payload = load_training_payload(training_path)
        training_examples = aggregate_training_examples(training_payload)
        compile_aggregator(
            program=program,
            training_examples=training_examples,
            optimization_method=args.optimization,
        )

    pitch_payload = program(
        pitch_facts=json.dumps(scenario_facts, indent=2),
        original_prompt=(
            TASKMASTER_SYSTEM_PROMPT.format(agent_total=args.num_agents)
            + "\n"
            + original_prompt
        ),
        editor_prompt=None if args.no_feedback else EDITOR_PROMPT,
        apply_feedback=not args.no_feedback,
        feedback_loops=args.feedback_loops,
    )

    if args.rate_limit > 0:
        time.sleep(args.rate_limit)

    pitch_payload["scenario_key"] = scenario_key
    pitch_payload["agent_models"] = agent_models[: args.num_agents]
    pitch_payload["planner_model"] = args.planner_model
    pitch_payload["aggregator_model"] = args.aggregator_model
    pitch_payload["feedback_model"] = None if args.no_feedback else args.feedback_model

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(pitch_payload, file, indent=2)

    return pitch_payload


def main() -> None:
    """
    CLI entry-point.
    """
    args = parse_arguments()
    payload = run_pipeline(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

