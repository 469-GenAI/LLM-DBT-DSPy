# MoA_DSPy.py
# ------------------------------------------------------------
# Declarative Self-Improving Python (DSPy) version of your Mixture-of-Agents pipeline
#
# Run:
#   python MoA_DSPy.py --optimization mipro --train-size 30 --test-size 10 --save-program
#
# Env:
#   - GROQ_API_KEY (for "groq/..." via LiteLLM in dspy.LM)
#   - OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY if you switch models
#
# Data:
#   - HuggingFace dataset `isaidchia/sharktank_pitches_modified`
#     (structured rows with input dict + gold pitch string).
#
# This file imports evaluator and signature from the same directory "DSPy":
#   DSPy/
#     ├─ MoA_DSPy.py        (this file)
#     ├─ AssessPitch.py     (provides AssessPitchQuality signature)
#     └─ generator.py       (provides PitchEvaluator module wrapper)
# ------------------------------------------------------------
import os
import json
import time
from pathlib import Path
from datetime import datetime
import json
import argparse
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm


import dspy
from dspy.teleprompt import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    KNNFewShot,
    MIPROv2,
)
from pydantic import ValidationError

# Import your local modules from the same directory package "DSPy"
from AssessPitch import AssessPitchQuality, PitchAssessment  # noqa: F401 (imported for type clarity)
from data_loader import load_and_prepare_data
from evaluator import PitchEvaluator
from generator import PitchGenerator
from utils import (
    PitchInput,
    capture_mlflow_run_id,
    create_pitch_vectorizer,
    format_pitch_input,
    print_evaluation_summary,
    results_to_dataframe,
    save_program_with_metadata,
    save_results_csv,
)
from dotenv import load_dotenv
import mlflow
import warnings
import logging

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABRICKS_PATH = os.getenv("DATABRICKS_PATH")
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")

RESULTS_DIR = Path("MoA/results")
PROGRAMS_DIR = Path("MoA/optimised_programs")
# Suppress MLflow trace ID collision warnings
warnings.filterwarnings("ignore", message="Failed to send trace to MLflow backend")
logging.getLogger("mlflow.tracing.export.mlflow_v3").setLevel(logging.ERROR)

# comment out if you want to stop tracking
if DATABRICKS_PATH:
    run_name = f"MoA_DSPy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(DATABRICKS_PATH + run_name)
    mlflow.dspy.autolog(
        log_compiles=True,    # Track optimization process
        log_evals=True,       # Track evaluation results
        log_traces_from_compile=True  # Track program traces during optimization
    )
else:
    print("No DATABRICKS_PATH found in .env")


def create_lm(model_id: str, temperature: float, max_tokens: int) -> dspy.LM:
    """
    Create a DSPy language model configured for the requested provider.

    Args:
        model_id: Provider-prefixed model identifier, e.g. "groq/..." or "bedrock/..."
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens for generation.

    Returns:
        Configured DSPy language model instance.
    """
    if model_id.startswith("bedrock/"):
        if not BEDROCK_API_KEY:
            raise ValueError(
                "BEDROCK_API_KEY missing in environment while a Bedrock model was requested"
            )
        return dspy.LM(
            model=model_id,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=BEDROCK_API_KEY,
        )

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing in environment for Groq-backed model configuration")
    return dspy.LM(
        model=model_id,
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=GROQ_API_KEY,
    )

def convert_examples_to_moa_inputs(examples: List[dspy.Example]) -> List[dspy.Example]:
    """
    Convert structured dataset examples into MoA-friendly DSPy examples.

    Each HuggingFace-derived example exposes a structured `input` dictionary and a
    ground-truth `output` pitch. We flatten the structured payload into a textual
    `facts` representation while retaining the original dictionary for downstream
    logging and synthesis with PitchGenerator.
    """
    converted: List[dspy.Example] = []

    for example in examples:
        structured_input = getattr(example, "input", {})
        try:
            # Validate and format the structured payload into a deterministic prompt.
            pitch_input = PitchInput(**structured_input)
            facts_text = format_pitch_input(pitch_input)
        except ValidationError as error:
            example_id = getattr(example, "id", "unknown")
            print(f"Warning: Validation failed for example {example_id}: {error}")
            facts_text = json.dumps(structured_input, ensure_ascii=False)

        converted.append(
            dspy.Example(
                id=getattr(example, "id", ""),
                facts=facts_text,
                structured_input=structured_input,
                ground_truth_pitch=getattr(example, "output", ""),
            ).with_inputs("facts", "structured_input")
        )

    random.shuffle(converted)
    return converted


def slice_examples(examples: List[dspy.Example], size: int) -> List[dspy.Example]:
    """
    Select a bounded number of examples while guarding against empty slices.
    """
    if size <= 0:
        return examples
    return examples[: min(size, len(examples))]


def prepare_train_test_sets(
    dataset_splits: Dict[str, List[dspy.Example]],
    train_size: int,
    test_size: int,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Convert HuggingFace dataset splits into MoA-compatible train and test sets.

    Returns train/test lists populated with `facts`, `structured_input`, and
    `ground_truth_pitch` fields ready for DSPy programs and evaluator metrics.
    """
    train_split = convert_examples_to_moa_inputs(dataset_splits.get("train", []))
    test_split = convert_examples_to_moa_inputs(dataset_splits.get("test", []))

    if not train_split:
        raise ValueError("No training examples found in dataset. Ensure the dataset is populated.")

    if not test_split:
        print("Warning: Test split empty; falling back to tail of training examples.")
        test_split = train_split.copy()

    trainset = slice_examples(train_split, train_size)
    testset = slice_examples(test_split, test_size)

    if not testset:
        testset = train_split[-min(test_size, len(train_split)) :]

    print(f"[data] Train examples: {len(trainset)} | Test examples: {len(testset)}")
    return trainset, testset


# ------------- Signatures -------------

class PlanTasks(dspy.Signature):
    """Decompose facts into a small set of agent roles and concrete subtasks (JSON)."""

    facts: str = dspy.InputField(desc="Product/company facts for a Shark Tank style pitch.")
    plan_json: str = dspy.OutputField(desc=textwrap.dedent("""
        STRICT JSON object:
        {
          "agents": [
            {"role": "Problem Framer", "task": "Summarize pain points with proof"},
            {"role": "Solution Architect", "task": "Position product with differentiators"},
            {"role": "Numbers & Terms", "task": "Initial terms (amount, equity) + valuation logic"}
          ]
        }
        No markdown fencing. Keep 2-3 agents only.
    """))


class AgentWrite(dspy.Signature):
    """Each agent writes a succinct section grounded in facts."""

    facts: str = dspy.InputField()
    role: str = dspy.InputField()
    task: str = dspy.InputField()
    draft: str = dspy.OutputField(desc="Concise, factual section. 150-220 words. No fluff.")


class SynthesizePitch(dspy.Signature):
    """Aggregate agent drafts into a single, well-structured pitch JSON."""

    facts: str = dspy.InputField()
    drafts: str = dspy.InputField(desc="Concatenated agent drafts with role labels.")
    pitch_json: str = dspy.OutputField(desc=textwrap.dedent("""
        STRICT JSON:
        {
          "Pitch": "Coherent pitch script (<= 350 words)",
          "Initial_Offer": {
            "Funding_Amount": "e.g., $300,000",
            "Equity_Offered": "e.g., 10%",
            "Valuation": "e.g., $3M",
            "Key_Terms": "short optional notes"
          }
        }
        No markdown/backticks.
    """))


# ------------- Modules -------------

class TaskPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict(PlanTasks)

    def forward(self, facts: str) -> Dict[str, Any]:
        plan_out = self.plan(facts=facts).plan_json
        try:
            plan = json.loads(plan_out)
            assert isinstance(plan.get("agents"), list) and 1 <= len(plan["agents"]) <= 4
        except Exception:
            plan = {
                "agents": [
                    {"role": "Problem Framer", "task": "Summarize pain points with evidence from facts"},
                    {"role": "Solution Architect", "task": "Map product features to pain points; highlight differentiation"},
                    {"role": "Numbers & Terms", "task": "Propose funding amount, equity, valuation logic grounded in facts"}
                ]
            }
        return plan


class AgentEnsemble(dspy.Module):
    def __init__(self, max_agents: int = 3):
        super().__init__()
        self.max_agents = max_agents
        self.writer = dspy.Predict(AgentWrite)

    def forward(self, facts: str, plan: Dict[str, Any]) -> List[Dict[str, str]]:
        agents = plan.get("agents", [])[: self.max_agents]
        outputs: List[Dict[str, str]] = []
        for a in agents:
            role = a.get("role", "Writer")
            task = a.get("task", "Write a short, factual section")
            draft = self.writer(facts=facts, role=role, task=task).draft
            outputs.append({"role": role, "draft": draft})
        return outputs


class PitchSynthesizer(dspy.Module):
    def __init__(self, pitch_generator: PitchGenerator):
        super().__init__()
        self.synth = dspy.Predict(SynthesizePitch)
        self.pitch_generator = pitch_generator

    def forward(
        self,
        facts: str,
        agent_outputs: List[Dict[str, str]],
        structured_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        """
        Combine agent drafts into JSON and synthesize a final pitch script.

        The JSON synthesis maintains the MoA guidance, while the dedicated
        `PitchGenerator` produces a conversational pitch grounded in the original
        structured input. We fall back to the JSON pitch when the generator cannot
        validate the payload.
        """
        joined = "\n\n".join([f"[{output['role']}]\n{output['draft']}" for output in agent_outputs])
        pitch_json = self.synth(facts=facts, drafts=joined, config=config or {}).pitch_json

        generated_pitch = ""
        if structured_input:
            try:
                generator_prediction = self.pitch_generator.generate(
                    input_data=structured_input,
                    config=config or {},
                )
                generated_pitch = getattr(generator_prediction, "pitch", "").strip()
            except Exception as error:
                print(f"Warning: PitchGenerator failed with structured input: {error}")

        if not generated_pitch:
            generated_pitch = extract_pitch_from_json(pitch_json)

        return dspy.Prediction(pitch_json=pitch_json, pitch=generated_pitch)


class PitchMoAProgram(dspy.Module):
    """
    End-to-end declarative MoA:
      TaskPlanner -> AgentEnsemble (N agents) -> PitchSynthesizer
    """
    def __init__(self, pitch_generator: PitchGenerator, num_agents: int = 3):
        super().__init__()
        self.planner = TaskPlanner()
        self.agents = AgentEnsemble(max_agents=num_agents)
        self.synth = PitchSynthesizer(pitch_generator=pitch_generator)

    def forward(
        self,
        facts: str,
        structured_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> dspy.Prediction:
        plan = self.planner(facts=facts)
        drafts = self.agents(facts=facts, plan=plan)
        with dspy.context(**(config or {})):
            return self.synth(
                facts=facts,
                agent_outputs=drafts,
                structured_input=structured_input or {},
                config=config or {},
            )


# ------------- Metric / Evaluator wiring -------------

def extract_pitch_from_json(pitch_json: str) -> str:
    """
    Extract the `Pitch` field from structured JSON output.

    Falls back to returning the raw JSON string when the payload cannot be parsed
    or the expected key is missing.
    """
    if not pitch_json:
        return ""

    try:
        parsed = json.loads(pitch_json)
    except json.JSONDecodeError:
        return pitch_json

    if isinstance(parsed, dict):
        pitch_value = parsed.get("Pitch", "")
        if isinstance(pitch_value, str):
            return pitch_value
    return pitch_json


def resolve_optimizer(name: str, metric_fn, trainset: List[dspy.Example]):
    """
    Resolve the requested DSPy teleprompter optimizer based on CLI flag.
    """
    optimizer_name = (name or "none").lower()
    if optimizer_name == "none":
        return None

    if metric_fn is None:
        raise ValueError("resolve_optimizer requires metric_fn when optimization is enabled")

    if not trainset:
        raise ValueError("resolve_optimizer requires a non-empty trainset")

    if optimizer_name == "mipro":
        return MIPROv2(metric=metric_fn, init_temperature=1.0)

    if optimizer_name in ("bootstrap", "bootstrapfewshot", "bfs"):
        # BootstrapFewShot optimizes demonstrations via iterative selection.
        return BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=4, max_labeled_demos=4)

    if optimizer_name in ("bootstrap_random", "bootstrapfewshot_random"):
        # Randomized search extends BootstrapFewShot with candidate exploration.
        return BootstrapFewShotWithRandomSearch(
            metric=metric_fn,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=10,
        )

    if optimizer_name == "knn":
        # KNNFewShot reuses nearest neighbors at inference time.
        vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
        return KNNFewShot(k=3, trainset=trainset, vectorizer=vectorizer)

    raise ValueError(f"Unknown optimization method: {optimizer_name}")


def make_metric(pitch_evaluator: PitchEvaluator):
    """
    Metric(example, pred, trace) -> float in [0,1] to MAXIMIZE.
    Uses your PitchEvaluator (AssessPitchQuality under the hood).
    """
    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        facts = example.facts
        gt_pitch = getattr(example, "ground_truth_pitch", "") or ""
        generated_pitch = getattr(pred, "pitch", "") or ""
        if not generated_pitch:
            generated_pitch = extract_pitch_from_json(getattr(pred, "pitch_json", "") or "")

        # Evaluator returns 0.0-1.0
        score = pitch_evaluator.get_score(
            pitch_facts=facts,
            ground_truth_pitch=gt_pitch,
            generated_pitch=generated_pitch
        )
        return float(score)
    return metric


# ------------- CLI -------------

def main():
    parser = argparse.ArgumentParser(description="DSPy MoA Pitch Generator")
    parser.add_argument("--dataset-name", type=str, default="isaidchia/sharktank_pitches_modified",
                        help="HuggingFace dataset containing structured Shark Tank pitches")
    parser.add_argument("--optimization", type=str, default="none", choices=["none", "mipro", "bootstrap"])
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling examples")

    parser.add_argument("--generator-model", type=str, default="groq/llama-3.3-70b-versatile",
                        help="Generator LM (LiteLLM-style id), e.g., groq/llama-3.3-70b-versatile")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)

    parser.add_argument("--evaluator-model", type=str, default="groq/openai/gpt-oss-120b",
                        help="Evaluator LM (for AssessPitchQuality)")

    parser.add_argument("--run-name", type=str, default="moa_dspy_run", help="Label for persisted artifacts")
    parser.add_argument("--save-program", action="store_true",
                        help="If set, save the compiled/tuned program (full program) to --save-dir.")
    parser.add_argument("--save-dir", type=str, default=str(PROGRAMS_DIR),
                        help="Directory to save the full program when --save-program is set.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    program_save_dir = Path(args.save_dir)
    program_save_dir.mkdir(parents=True, exist_ok=True)

    # Configure generator LM
    gen_lm = create_lm(
        model_id=args.generator_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=gen_lm)
    pitch_generator = PitchGenerator(lm=gen_lm)

    # Build separate evaluator (with its own LM to avoid interference)
    eval_lm = create_lm(
        model_id=args.evaluator_model,
        temperature=0.1,
        max_tokens=2048,
    )
    pitch_evaluator = PitchEvaluator(lm=eval_lm)

    # Load structured dataset and convert to MoA-ready examples
    dataset_splits = load_and_prepare_data(args.dataset_name)
    trainset, testset = prepare_train_test_sets(
        dataset_splits,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    # Program
    program = PitchMoAProgram(pitch_generator=pitch_generator, num_agents=args.num_agents)

    # Metric for optimization/eval
    metric_fn = make_metric(pitch_evaluator)

    # Optimize if requested
    teleprompter = resolve_optimizer(args.optimization, metric_fn, trainset)
    mlflow_run_id = None
    if teleprompter is not None:
        print(f"[compile] Optimizing with {teleprompter.__class__.__name__} ...")
        if isinstance(teleprompter, KNNFewShot):
            # KNNFewShot compiles lazily for runtime neighbor lookup.
            program = teleprompter.compile(program)
        else:
            program = teleprompter.compile(program, trainset=trainset)
        print("[compile] Done.")
        mlflow_run_id = capture_mlflow_run_id(DATABRICKS_PATH, run_name)

    # Evaluate on test set
    scores: List[float] = []
    results: List[Dict[str, Any]] = []
    # Track artifact outputs for downstream metadata enrichment
    results_csv_path: Optional[str] = None
    results_timestamp: Optional[str] = None
    print("[eval] Running on test set...")
    run_timestamp = int(time.time())

    for index, example in enumerate(
        tqdm(testset, desc="Evaluating pitches", unit="pitch", total=len(testset))
    ):
        structured_input = getattr(example, "structured_input", {}) or {}
        rollout_config: Dict[str, Any] = {
            "rollout_id": f"moa_eval_{run_timestamp}_{index}",
            "temperature": max(args.temperature, 0.1),
        }
        prediction = program(
            facts=example.facts,
            structured_input=structured_input,
            config=rollout_config,
        )
        score = metric_fn(example, prediction)
        scores.append(score)

        generated_pitch = getattr(prediction, "pitch", "") or extract_pitch_from_json(
            getattr(prediction, "pitch_json", "")
        )

        with dspy.context(
            lm=pitch_evaluator.lm,
            rollout_id=f"{rollout_config['rollout_id']}_eval",
            temperature=0.1,
        ):
            assessment = pitch_evaluator.get_full_assessment(
                pitch_facts=example.facts,
                ground_truth_pitch=getattr(example, "ground_truth_pitch", "") or "",
                generated_pitch=generated_pitch,
            )

        results.append(
            {
                "id": getattr(example, "id", ""),
                "input": structured_input,
                "facts": example.facts,
                "ground_truth": getattr(example, "ground_truth_pitch", ""),
                "generated_pitch": generated_pitch,
                "pitch_json": getattr(prediction, "pitch_json", ""),
                "assessment": assessment,
            }
        )

    avg_score = sum(scores) / max(1, len(scores))
    print(f"[eval] Test size={len(testset)} | Avg Judge Score (0-1) = {avg_score:.3f}")

    if results:
        df = results_to_dataframe(
            results=results,
            optimization_method=args.optimization,
            generator_model=args.generator_model,
            evaluator_model=args.evaluator_model,
        )
        results_csv_path, results_timestamp = save_results_csv(
            df=df,
            optimization_method=args.optimization,
            run_name=args.run_name,
            mlflow_run_id=mlflow_run_id,
            output_dir=results_dir,
        )
        print_evaluation_summary(
            df=df,
            generator_model=args.generator_model,
            evaluator_model=args.evaluator_model,
            optimization_method=args.optimization,
        )
        print(f"[save] Results CSV: {Path(results_csv_path).resolve()}")

    # Save compiled/tuned program
    if args.save_program:
        saved_program_path = save_program_with_metadata(
            program=program,
            save_dir=str(program_save_dir),
            optimization_method=args.optimization,
            generator_model=args.generator_model,
            evaluator_model=args.evaluator_model,
            trainset_size=len(trainset),
            testset_size=len(testset),
            run_name=args.run_name,
            mlflow_run_id=mlflow_run_id,
            results_csv_filename=Path(results_csv_path).name if results_csv_path else None,
            results_timestamp=results_timestamp,
        )
        print(f"[save] Program saved to {Path(saved_program_path).resolve()}")


if __name__ == "__main__":
    main()