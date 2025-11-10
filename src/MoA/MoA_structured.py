"""
DSPy implementation of the Mixture-of-Agents Shark Tank pitch generator.
"""

import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy
import mlflow
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from MoA.services import compile_moa_program, create_moa_evaluator, create_moa_generator
from pitchLLM.data_loader import load_and_prepare_data
from pitchLLM.models.evaluator import PitchEvaluator
from pitchLLM.utils import (
    capture_mlflow_run_id,
    print_evaluation_summary,
    results_to_dataframe,
    save_program_with_metadata,
    save_results_csv,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")
DATABRICKS_PATH = os.getenv("DATABRICKS_PATH")

DATASET_SHUFFLE_SEED = 42
RATE_LIMIT_DELAY = 2.5

_GLOBAL_EVALUATOR: Optional[PitchEvaluator] = None


def configure_global_evaluator(evaluator_lm: dspy.LM) -> PitchEvaluator:
    """
    Initialise the global evaluator used for optimisation and evaluation.
    """
    global _GLOBAL_EVALUATOR
    _GLOBAL_EVALUATOR = create_moa_evaluator(evaluator_lm)
    return _GLOBAL_EVALUATOR


def get_global_evaluator() -> PitchEvaluator:
    """
    Retrieve the global evaluator instance, ensuring it is configured.
    """
    if _GLOBAL_EVALUATOR is None:
        raise ValueError("Evaluator has not been configured. Call configure_global_evaluator first.")
    return _GLOBAL_EVALUATOR


def pitch_quality_metric(example: dspy.Example, pred: Any, trace: Optional[Any] = None) -> float:
    """
    Score generated pitches using the GPT-OSS evaluator for optimisation.
    """
    del trace

    generated_pitch = getattr(pred, "pitch", "")
    if not generated_pitch:
        return 0.0

    evaluator = get_global_evaluator()
    pitch_facts = json.dumps(example.input, indent=2)
    return evaluator.get_score(
        pitch_facts=pitch_facts,
        ground_truth_pitch=example.output,
        generated_pitch=generated_pitch,
    )


def evaluate_moa_program(
    program: Any,
    testset: List[dspy.Example],
    evaluator: Optional[PitchEvaluator],
    rate_limit: bool,
    use_evaluator: bool,
) -> List[Dict[str, Any]]:
    """
    Execute evaluation runs across the test set and optionally compute scores.
    """
    results: List[Dict[str, Any]] = []
    run_timestamp = int(time.time())

    for idx, example in enumerate(tqdm(testset, desc="Evaluating", unit="pitch")):
        try:
            prediction = program(
                input=example.input,
                config={
                    "rollout_id": f"{run_timestamp}_{idx}",
                    "temperature": 1.0,
                },
            )

            generated_pitch = getattr(prediction, "pitch", "")
            agent_contributions = getattr(prediction, "agent_contributions", [])

            if use_evaluator and evaluator:
                pitch_facts = json.dumps(example.input, indent=2)
                assessment = evaluator.get_full_assessment(
                    pitch_facts=pitch_facts,
                    ground_truth_pitch=example.output,
                    generated_pitch=generated_pitch,
                )
            else:
                assessment = None

            results.append(
                {
                    "id": example.id,
                    "input": example.input,
                    "ground_truth": example.output,
                    "generated_pitch": generated_pitch,
                    "agent_contributions": agent_contributions,
                    "assessment": assessment,
                }
            )

            if rate_limit and idx < len(testset) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        except Exception as error:
            results.append(
                {
                    "id": example.id,
                    "input": example.input,
                    "ground_truth": example.output,
                    "generated_pitch": "",
                    "agent_contributions": [],
                    "assessment": {
                        "factual_score": 0.0,
                        "narrative_score": 0.0,
                        "style_score": 0.0,
                        "reasoning": f"Error: {error}",
                        "final_score": 0.0,
                    },
                }
            )

    return results


def configure_language_model(model_name: str, api_key: Optional[str], temperature: float) -> dspy.LM:
    """
    Configure a DSPy language model using the provided credentials.
    """
    if model_name.startswith("bedrock/"):
        return dspy.LM(
            model_name,
            model_type="chat",
            api_key=api_key,
            temperature=temperature,
        )

    return dspy.LM(
        model_name,
        model_type="chat",
        api_key=api_key,
        temperature=temperature,
    )


def add_agent_contributions(df: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extend the base results dataframe with agent contribution metadata.
    """
    contributions = [
        json.dumps(result.get("agent_contributions", []), ensure_ascii=False)
        for result in results
    ]
    df = df.copy()
    df["agent_contributions"] = contributions
    return df


def main() -> None:
    """
    Entry point for running the MoA DSPy structured pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run MoA structured pitch generation with DSPy")
    parser.add_argument(
        "--optimization",
        type=str,
        default="none",
        choices=["none", "bootstrap", "bootstrap_random", "knn", "mipro"],
        help="Optimization method to use",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="groq/llama-3.3-70b-versatile",
        help="Model to use for pitch generation",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="groq/openai/gpt-oss-120b",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training examples to use",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of test examples to use",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use detailed assessment with the evaluator model",
    )
    parser.add_argument(
        "--save-program",
        action="store_true",
        help="Persist the compiled program and metadata",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable rate limiting between requests",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of agents to instantiate in the MoA program",
    )

    args = parser.parse_args()

    run_name = f"moa_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if DATABRICKS_PATH:
        mlflow.set_experiment(DATABRICKS_PATH + run_name)
        mlflow.dspy.autolog(
            log_compiles=True,
            log_evals=True,
            log_traces_from_compile=True,
        )

    print("=" * 80)
    print("DSPy MoA Structured Pitch Generation")
    print("=" * 80)
    print("\nModel configuration:")
    print(f"  Generator: {args.generator_model}")
    print(f"  Evaluator: {args.evaluator_model}")

    generator_lm = configure_language_model(
        model_name=args.generator_model,
        api_key=BEDROCK_API_KEY if args.generator_model.startswith("bedrock/") else GROQ_API_KEY,
        temperature=1.0,
    )
    evaluator_lm = configure_language_model(
        model_name=args.evaluator_model,
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )

    configure_global_evaluator(evaluator_lm)

    dspy.configure(lm=generator_lm, track_usage=True)

    print("\n1. Loading dataset...")
    dataset = load_and_prepare_data()
    trainset = dataset["train"]
    testset = dataset["test"]

    random.seed(DATASET_SHUFFLE_SEED)
    random.shuffle(trainset)

    if args.train_size:
        trainset = trainset[: args.train_size]
        print(f"   Using {len(trainset)} training examples")

    if args.test_size:
        testset = testset[: args.test_size]
        print(f"   Using {len(testset)} test examples")

    print("\n2. Creating MoA program...")
    program = create_moa_generator(generator_lm, num_agents=args.num_agents)

    print("\n3. Compiling program...")
    program = compile_moa_program(
        program=program,
        trainset=trainset,
        optimization_method=args.optimization,
        metric=pitch_quality_metric if args.optimization != "none" else None,
    )

    mlflow_run_id = None

    if DATABRICKS_PATH:
        mlflow_run_id = capture_mlflow_run_id(DATABRICKS_PATH, run_name)

    if args.save_program:
        save_program_with_metadata(
            program=program,
            save_dir="optimized_programs",
            optimization_method=args.optimization,
            generator_model=generator_lm.model,
            evaluator_model=evaluator_lm.model,
            trainset_size=len(trainset),
            testset_size=len(testset),
            run_name=run_name,
            mlflow_run_id=mlflow_run_id,
        )

    print("\n4. Evaluating program...")
    if args.no_rate_limit:
        print("   Rate limiting disabled.")
    else:
        print(f"   Rate limiting enabled: {RATE_LIMIT_DELAY}s between requests")

    evaluator_instance = get_global_evaluator() if args.evaluate else None

    results = evaluate_moa_program(
        program=program,
        testset=testset,
        evaluator=evaluator_instance,
        rate_limit=not args.no_rate_limit,
        use_evaluator=args.evaluate,
    )

    print("\n5. Saving results...")
    df = results_to_dataframe(
        results=results,
        optimization_method=args.optimization,
        generator_model=generator_lm.model,
        evaluator_model=evaluator_lm.model,
    )
    df = add_agent_contributions(df, results)

    results_path = save_results_csv(
        df=df,
        optimization_method=args.optimization,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id,
    )

    if args.evaluate:
        print_evaluation_summary(
            df=df,
            generator_model=generator_lm.model,
            evaluator_model=evaluator_lm.model,
            optimization_method=args.optimization,
        )

    print("\n✓ Done!")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

