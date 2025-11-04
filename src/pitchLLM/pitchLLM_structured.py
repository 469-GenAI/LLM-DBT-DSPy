# pitchLLM_structured.py - DSPy implementation for structured pitch generation
"""
DSPy-based pitch generation system that takes structured input and generates
narrative pitches similar to Shark Tank pitches.

This implementation supports optimization via:
- BootstrapFewShot: Standard few-shot learning with bootstrapped demonstrations
- BootstrapFewShotWithRandomSearch: Random search over bootstrapped demonstrations
- KNNFewShot: K-Nearest Neighbors for finding relevant training examples
- MIPROv2: Multi-stage instruction/prompt optimization
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import random

import dspy
from pydantic import BaseModel, Field
from tqdm import tqdm
import pandas as pd
import mlflow
import warnings
import logging

# Import local utilities
from utils import (
    PitchInput,
    format_pitch_input,
    capture_mlflow_run_id,
    save_program_with_metadata,
    results_to_dataframe,
    save_results_csv,
    print_evaluation_summary
)
from data_loader import load_and_prepare_data
from eval.AssessPitch import AssessPitchQuality
from models import PitchGenerator, PitchEvaluator
from models.generator import StructuredPitchProgram

# Suppress MLflow trace ID collision warnings
warnings.filterwarnings("ignore", message="Failed to send trace to MLflow backend")
logging.getLogger("mlflow.tracing.export.mlflow_v3").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABRICKS_PATH = os.getenv("DATABRICKS_PATH")

# comment out if you want to stop tracking
if DATABRICKS_PATH:
    run_name = f"pitchLLM_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(DATABRICKS_PATH + run_name)
    mlflow.dspy.autolog(
        log_compiles=True,    # Track optimization process
        log_evals=True,       # Track evaluation results
        log_traces_from_compile=True  # Track program traces during optimization
    )
else:
    print("No DATABRICKS_PATH found in .env")

# Configure DSPy language models
# Generator: Llama 3.3 70B (fast, cost-effective for generation)
generator_lm = dspy.LM(
    "groq/llama-3.3-70b-versatile", 
    model_type="chat", 
    api_key=GROQ_API_KEY,
    temperature=1.0
)

# Evaluator: GPT OSS 120B (more powerful, objective evaluation - different architecture)
evaluator_lm = dspy.LM(
    "groq/openai/gpt-oss-120b", 
    model_type="chat", 
    api_key=GROQ_API_KEY,
    temperature=0.7
)

# Set default LM to generator (for pitch generation and optimization)
dspy.configure(lm=generator_lm, track_usage=True)

# Rate limiting configuration for Groq (30 requests per minute limit)
RATE_LIMIT_DELAY = 2.5  # Seconds between API calls (safe margin: 24 calls/min)

DATASET_SHUFFLE_SEED = 42

# ---------- 1) MODEL INSTANCES ----------
# Note: PitchGenerator and PitchEvaluator are now in models/ package
# This provides clear separation between generator and evaluator models

# Create global evaluator instance for metrics
_global_evaluator = None

def get_evaluator():
    """Get or create global evaluator instance."""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = PitchEvaluator(evaluator_lm)
    return _global_evaluator


def pitch_quality_metric(example, pred, trace=None):
    """
    Metric function for evaluating pitch quality during optimization.
    Uses the evaluator model for objective assessment.
    
    Args:
        example: DSPy Example with input and output (ground truth)
        pred: Prediction from the model
        trace: Optional trace information
        
    Returns:
        Float score between 0.0 and 1.0
    """
    try:
        # Extract the generated pitch
        generated_pitch = pred.pitch if hasattr(pred, "pitch") else str(pred)
        
        # Extract ground truth
        ground_truth = example.output
        
        # Extract pitch facts from input
        pitch_facts = json.dumps(example.input, indent=2)
        
        # Use the evaluator to assess quality
        evaluator = get_evaluator()
        final_score = evaluator.get_score(
            pitch_facts=pitch_facts,
            ground_truth_pitch=ground_truth,
            generated_pitch=generated_pitch
        )
        
        return final_score
            
    except Exception as e:
        print(f"Error in pitch_quality_metric: {e}")
        return 0.0


def simple_pitch_metric(example, pred, trace=None):
    """
    Simplified metric that checks if a pitch was generated.
    Useful for faster iteration during development.
    
    Args:
        example: DSPy Example with input and output
        pred: Prediction from the model
        trace: Optional trace information
        
    Returns:
        Boolean indicating if pitch was generated
    """
    try:
        generated_pitch = pred.pitch if hasattr(pred, "pitch") else str(pred)
        # Check if pitch is non-empty and has reasonable length
        return bool(generated_pitch and len(generated_pitch) > 100)
    except Exception:
        return False


# ---------- 4) OPTIMIZATION ----------
def compile_program(
    program: StructuredPitchProgram,
    trainset: list,
    optimization_method: str = "none",
    metric=None
):
    """
    Compile the program with optional optimization.
    
    Args:
        program: The StructuredPitchProgram to optimize
        trainset: Training examples
        optimization_method: "none", "bootstrap", "bootstrap_random", "knn", or "mipro"
        metric: Evaluation metric function
        
    Returns:
        Compiled (possibly optimized) program
    """
    if optimization_method == "none":
        print("Running without optimization (baseline)")
        return program
    
    if metric is None:
        metric = simple_pitch_metric
    
    if optimization_method == "bootstrap":
        print("Compiling with BootstrapFewShot...")
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        compiled_program = optimizer.compile(program, trainset=trainset)
        return compiled_program

    elif optimization_method == "bootstrap_random":
        print("Compiling with BootstrapFewShotWithRandomSearch...")
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=10  # Number of random programs to evaluate
        )
        compiled_program = optimizer.compile(program, trainset=trainset)
        return compiled_program
    
    elif optimization_method == "knn":
        print("Compiling with KNNFewShot...")
        optimizer = dspy.KNNFewShot(
            k=3,  # Number of nearest neighbors to use
            trainset=trainset
        )
        # KNNFewShot works differently - it wraps the program
        compiled_program = optimizer.compile(program, trainset=trainset)
        return compiled_program

    elif optimization_method == "mipro":
        print("Compiling with MIPROv2...")
        optimizer = dspy.MIPROv2(
            metric=metric,
            # num_candidates=10,
            init_temperature=1.0
        )
        compiled_program = optimizer.compile(
            program,
            trainset=trainset,
            # num_trials=10,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        return compiled_program
    
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


# ---------- 5) EVALUATION ----------
def evaluate_program(program, testset, use_evaluator=False, rate_limit=True):
    """
    Evaluate the program on a test set with rate limiting.
    
    Args:
        program: The program to evaluate
        testset: List of test examples
        use_evaluator: Whether to use detailed AssessPitch evaluation
        rate_limit: Whether to enforce rate limiting (default: True for Groq)
        
    Returns:
        List of results with predictions and scores
        
    Note:
        With rate limiting enabled, respects Groq's 30 requests/min limit.
        Each example makes 2 API calls (generation + evaluation), so we process
        ~12 examples per minute safely.
    """
    results = []
    # Create dedicated evaluator
    evaluator = PitchEvaluator(evaluator_lm) if use_evaluator else None
    
    # Create dedicated generator
    generator = PitchGenerator(generator_lm)
    
    # Generate unique run identifier for cache busting
    run_timestamp = int(time.time())
    
    for idx, example in enumerate(tqdm(testset, desc="Evaluating", unit="pitch")):
        try:
            # Generate pitch using dedicated generator model
            prediction = generator.generate(
                example.input,
                config={
                    "rollout_id": f"{run_timestamp}_{idx}",  # Unique per example
                    "temperature": 1.0  # Bypass cache
                }
            )
            generated_pitch = prediction.pitch if hasattr(prediction, "pitch") else str(prediction)
            
            # Rate limiting after generation
            if rate_limit and idx < len(testset) - 1:
                time.sleep(RATE_LIMIT_DELAY)
            
            # Evaluate if requested using dedicated evaluator model
            if use_evaluator:
                pitch_facts = json.dumps(example.input, indent=2)
                # Use convenience method that returns dict directly
                assessment_data = evaluator.get_full_assessment(
                    pitch_facts=pitch_facts,
                    ground_truth_pitch=example.output,
                    generated_pitch=generated_pitch
                )
                
                # Rate limiting after evaluation
                if rate_limit and idx < len(testset) - 1:
                    time.sleep(RATE_LIMIT_DELAY)
            else:
                assessment_data = None
            
            results.append({
                "id": example.id,
                "input": example.input,
                "ground_truth": example.output,
                "generated_pitch": generated_pitch,
                "assessment": assessment_data
            })
            
        except Exception as e:
            print(f"\nError processing example {example.id}: {e}")
            results.append({
                "id": example.id,
                "input": example.input,
                "ground_truth": example.output,
                "generated_pitch": "",
                "assessment": {
                    "factual_score": 0.0,
                    "narrative_score": 0.0,
                    "style_score": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "final_score": 0.0
                }
            })
    
    return results


# ---------- 6) MAIN EXECUTION ----------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run structured pitch generation with DSPy")
    parser.add_argument(
        "--optimization",
        type=str,
        default="none",
        choices=["none", "bootstrap", "bootstrap_random", "knn", "mipro"],
        help="Optimization method to use"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training examples to use (default: all)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of test examples to use (default: 10)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use detailed AssessPitch evaluation with GPT-OSS-120B"
    )
    parser.add_argument(
    "--save-program",
    action="store_true",
    help="Save the optimized program after compilation"
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable rate limiting (use with caution - may hit API limits)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DSPy Structured Pitch Generation")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data from HuggingFace...")
    data = load_and_prepare_data()
    trainset = data["train"]
    testset = data["test"]

    # Shuffle with seed for reproducibility
    random.seed(DATASET_SHUFFLE_SEED)
    random.shuffle(trainset)
    
    # Limit sizes if requested
    if args.train_size:
        trainset = trainset[:args.train_size]
        print(f"   Using {len(trainset)} training examples")
    
    if args.test_size:
        testset = testset[:args.test_size]
        print(f"   Using {len(testset)} test examples")
    
    # Create program using dedicated generator
    print("\n2. Creating pitch generation program...")
    generator = PitchGenerator(generator_lm)
    program = generator.program  # Get the underlying DSPy module for optimization
    
    # Compile with optimization if requested
    if args.optimization != "none":
        print(f"\n3. Compiling with {args.optimization}...")
        program = compile_program(
            program,
            trainset,
            optimization_method=args.optimization,
            # metric=simple_pitch_metric # for testing
            metric=pitch_quality_metric # for evaluation
        )
        
        # Capture MLflow run_id (refactored)
        mlflow_run_id = capture_mlflow_run_id(DATABRICKS_PATH, run_name)
        # mlflow_run_id = None
        
        # Save program with metadata (refactored)
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
                mlflow_run_id=mlflow_run_id
            )
    else:
        print("\n3. Running baseline (no optimization)...")
        # For baseline, no MLflow run
        mlflow_run_id = None
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    if not args.no_rate_limit:
        print(f"   Rate limiting enabled: {RATE_LIMIT_DELAY}s delay between API calls")
        print(f"   Estimated time: ~{len(testset) * RATE_LIMIT_DELAY * (2 if args.evaluate else 1) / 60:.1f} minutes")
    results = evaluate_program(program, testset, use_evaluator=args.evaluate, rate_limit=not args.no_rate_limit)
    
    # Save results (refactored)
    print("\n5. Saving results...")
    df = results_to_dataframe(
        results=results,
        optimization_method=args.optimization,
        generator_model=generator_lm.model,
        evaluator_model=evaluator_lm.model
    )
    
    save_results_csv(
        df=df,
        optimization_method=args.optimization,
        run_name=run_name,
        mlflow_run_id=mlflow_run_id
    )
    
    # Print summary statistics (refactored)
    if args.evaluate:
        print_evaluation_summary(
            df=df,
            generator_model=generator_lm.model,
            evaluator_model=evaluator_lm.model,
            optimization_method=args.optimization
        )
    
    print("\n✓ Done!")

