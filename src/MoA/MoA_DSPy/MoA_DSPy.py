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
#   - ./all_processed_facts.txt  (JSON dict of {fact_name: facts_dict_or_text})
#     If dict, ground truth can be under keys like "Full_Pitch" / "output" / "ground_truth".
#
# This file imports evaluator and signature from the same directory "DSPy":
#   DSPy/
#     ├─ MoA_DSPy.py        (this file)
#     ├─ AssessPitch.py     (provides AssessPitchQuality signature)
#     └─ generator.py       (provides PitchEvaluator module wrapper)
# ------------------------------------------------------------

import os
import json
import argparse
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot

# Import your local modules from the same directory package "DSPy"
from AssessPitch import AssessPitchQuality, PitchAssessment  # noqa: F401 (imported for type clarity)
from generator import PitchEvaluator


# ------------- Data I/O -------------

def load_facts(relative_file_path: Optional[str] = None) -> Dict[str, Any]:
    """Load your facts JSON; default path: ./all_processed_facts.txt."""
    if relative_file_path is None:
        script_dir = Path(__file__).parent
        relative_file_path = script_dir / "all_processed_facts.txt"
    p = Path(relative_file_path)
    if not p.exists():
        raise FileNotFoundError(f"Facts file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.loads(f.read())


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
    def __init__(self):
        super().__init__()
        self.synth = dspy.Predict(SynthesizePitch)

    def forward(self, facts: str, agent_outputs: List[Dict[str, str]]) -> str:
        joined = "\n\n".join([f"[{o['role']}]\n{o['draft']}" for o in agent_outputs])
        return self.synth(facts=facts, drafts=joined).pitch_json


class PitchMoAProgram(dspy.Module):
    """
    End-to-end declarative MoA:
      TaskPlanner -> AgentEnsemble (N agents) -> PitchSynthesizer
    """
    def __init__(self, num_agents: int = 3):
        super().__init__()
        self.planner = TaskPlanner()
        self.agents = AgentEnsemble(max_agents=num_agents)
        self.synth = PitchSynthesizer()

    def forward(self, facts: str) -> dspy.Prediction:
        plan = self.planner(facts=facts)
        drafts = self.agents(facts=facts, plan=plan)
        pitch_json = self.synth(facts=facts, agent_outputs=drafts)
        return dspy.Prediction(pitch_json=pitch_json)


# ------------- Metric / Evaluator wiring -------------

def extract_ground_truth_pitch(facts_value: Any) -> str:
    """Try common keys to find a human-written pitch."""
    if isinstance(facts_value, dict):
        for key in ("Full_Pitch", "ground_truth", "gold_pitch", "output", "pitch", "Ground_Truth", "GoldPitch"):
            if key in facts_value and isinstance(facts_value[key], str) and facts_value[key].strip():
                return facts_value[key]
    return ""


def make_examples_from_facts(facts_store: Dict[str, Any]) -> List[dspy.Example]:
    """Unlabeled examples for evaluator-based optimization (MiPRO)."""
    examples: List[dspy.Example] = []
    for _, facts in facts_store.items():
        facts_str = facts if isinstance(facts, str) else json.dumps(facts, ensure_ascii=False)
        gt_pitch = extract_ground_truth_pitch(facts) if isinstance(facts, dict) else ""
        examples.append(
            dspy.Example(facts=facts_str, ground_truth_pitch=gt_pitch).with_inputs("facts", "ground_truth_pitch")
        )
    random.shuffle(examples)
    return examples


def train_test_split(examples: List[dspy.Example], train_size: int, test_size: int):
    n = len(examples)
    if n == 0:
        raise ValueError("No examples found. Is all_processed_facts.txt populated?")
    train = examples[: min(train_size, n)]
    test = examples[min(train_size, n): min(train_size + test_size, n)]
    if len(test) == 0:
        test = examples[-min(test_size, len(examples)) :]
    return train, test


def resolve_optimizer(name: str, metric_fn, trainset: List[dspy.Example]):
    name = (name or "none").lower()
    if name in ("mipro", "miprov2", "mipro_v2"):
        return MIPROv2(
            metric=metric_fn,
            max_bootstrapped_demos=6,
            num_candidates=3,
            max_train_iters=2,
            verbose=True,
        )
    if name in ("bootstrap", "bootstrapfewshot", "bfs"):
        return BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=6,
            num_candidates=2,
            max_train_iters=2,
            verbose=True,
        )
    return None


def make_metric(pitch_evaluator: PitchEvaluator):
    """
    Metric(example, pred, trace) -> float in [0,1] to MAXIMIZE.
    Uses your PitchEvaluator (AssessPitchQuality under the hood).
    """
    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        facts = example.facts
        gt_pitch = getattr(example, "ground_truth_pitch", "") or ""
        # Parse generated pitch JSON, extract "Pitch" if available
        gen = getattr(pred, "pitch_json", "") or ""
        try:
            pj = json.loads(gen)
            generated_pitch = pj.get("Pitch", gen) if isinstance(pj, dict) else gen
        except Exception:
            generated_pitch = gen
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
    parser.add_argument("--facts-path", type=str, default=None, help="Path to all_processed_facts.txt")
    parser.add_argument("--optimization", type=str, default="none", choices=["none", "mipro", "bootstrap"])
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=3)

    parser.add_argument("--lm-model", type=str, default="groq/llama-3.3-70b-versatile",
                        help="Generator LM (LiteLLM-style id), e.g., groq/llama-3.3-70b-versatile")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2048)

    parser.add_argument("--eval-lm-model", type=str, default="groq/llama-3.3-70b-versatile",
                        help="Evaluator LM (for AssessPitchQuality)")

    parser.add_argument("--save-program", action="store_true",
                        help="If set, save the compiled/tuned program (full program) to --save-dir.")
    parser.add_argument("--save-dir", type=str, default="compiled_pitch_program",
                        help="Directory to save the full program when --save-program is set.")

    parser.add_argument("--dump-predictions", type=str, default="predictions.jsonl",
                        help="Where to dump test predictions as JSONL")

    args = parser.parse_args()

    # Configure generator LM
    gen_lm = dspy.LM(model=args.lm_model, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.configure(lm=gen_lm)

    # Build separate evaluator (with its own LM to avoid interference)
    eval_lm = dspy.LM(model=args.eval_lm_model, temperature=0.2, max_tokens=1024)
    pitch_evaluator = PitchEvaluator(lm=eval_lm)

    # Load data
    facts_store = load_facts(args.facts_path)
    examples = make_examples_from_facts(facts_store)
    trainset, testset = train_test_split(examples, train_size=args.train_size, test_size=args.test_size)

    # Program
    program = PitchMoAProgram(num_agents=args.num_agents)

    # Metric for optimization/eval
    metric_fn = make_metric(pitch_evaluator)

    # Optimize if requested
    teleprompter = resolve_optimizer(args.optimization, metric_fn, trainset)
    if teleprompter is not None:
        print(f"[compile] Optimizing with {teleprompter.__class__.__name__} ...")
        program = teleprompter.compile(program, trainset=trainset)
        print("[compile] Done.")

    # Evaluate on test set
    scores = []
    print("[eval] Running on test set...")
    for ex in testset:
        pred = program(facts=ex.facts)
        score = metric_fn(ex, pred)
        scores.append(score)
    avg_score = sum(scores) / max(1, len(scores))
    print(f"[eval] Test size={len(testset)} | Avg Judge Score (0-1) = {avg_score:.3f}")

    # Dump predictions (+ score and assessment)
    out_path = Path(args.dump_predictions)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in testset:
            pred = program(facts=ex.facts)
            pitch_json = getattr(pred, "pitch_json", "")
            # Extract pitch text for richer report
            try:
                pj = json.loads(pitch_json)
                generated_pitch = pj.get("Pitch", pitch_json)
            except Exception:
                generated_pitch = pitch_json

            assessment = pitch_evaluator.get_full_assessment(
                pitch_facts=ex.facts,
                ground_truth_pitch=getattr(ex, "ground_truth_pitch", "") or "",
                generated_pitch=generated_pitch
            )
            row = {
                "facts": ex.facts,
                "ground_truth_pitch": getattr(ex, "ground_truth_pitch", ""),
                "pitch_json": pitch_json,
                "assessment": assessment
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[save] Wrote predictions to {out_path.resolve()}")

    # Save compiled/tuned program
    if args.save_program:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Full-program save (prompts + learned settings)
        program.save(str(save_dir), save_program=True)
        print(f"[save] Program saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()