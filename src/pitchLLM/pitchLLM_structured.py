# pitchLLM_structured.py - DSPy implementation for structured pitch generation
"""
DSPy-based pitch generation system that takes structured input and generates
narrative pitches similar to Shark Tank pitches.

This implementation supports optimization via BootstrapFewShot and MIPROv2.
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

# Import local utilities
from utils import PitchInput, format_pitch_input
from data_loader import load_and_prepare_data
from eval.AssessPitch import AssessPitchQuality

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABRICKS_PATH = os.getenv("DATABRICKS_PATH")

# comment out if you want to stop tracking
if DATABRICKS_PATH:
    run_name = f"pitchLLM_structured_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(DATABRICKS_PATH + run_name)
    mlflow.dspy.autolog()
else:
    print("No DATABRICKS_PATH found in .env")

# Configure DSPy language models
# Generator: Llama 3.3 70B (fast, cost-effective for generation)
generator_lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)

# Evaluator: GPT OSS 120B (more powerful, objective evaluation - different architecture)
evaluator_lm = dspy.LM("groq/openai/gpt-oss-120b", model_type="chat", api_key=GROQ_API_KEY)

# Set default LM to generator (for pitch generation and optimization)
dspy.configure(lm=generator_lm)

# Rate limiting configuration for Groq (30 requests per minute limit)
RATE_LIMIT_DELAY = 2.5  # Seconds between API calls (safe margin: 24 calls/min)

DATASET_SHUFFLE_SEED = 42

# ---------- 1) DSPy SIGNATURES ----------
class PitchGenerationSig(dspy.Signature):
    """
    Generate a compelling Shark Tank pitch from structured input.
    
    The pitch should:
    - Start with an engaging introduction of the founders and company
    - Present the investment ask clearly
    - Tell a story about the problem from the customer's perspective
    - Introduce the solution with compelling details
    - End with a strong call to action for the Sharks
    """
    
    pitch_data: str = dspy.InputField(
        desc="Structured pitch data including company, founders, problem story, solution, and investment ask"
    )
    
    pitch: str = dspy.OutputField(
        desc="A compelling, narrative pitch in the style of Shark Tank presentations. "
             "Should be conversational, engaging, and tell a complete story from problem to solution."
    )


# ---------- 2) DSPy MODULE ----------
class StructuredPitchProgram(dspy.Module):
    """
    Main DSPy module for generating structured pitches.
    
    Takes structured pitch input and generates a narrative pitch string
    using chain-of-thought reasoning.
    """
    
    def __init__(self):
        super().__init__()
        self.generate_pitch = dspy.ChainOfThought(PitchGenerationSig)
    
    def forward(self, input: dict):
        """
        Generate a pitch from structured input.
        
        Args:
            input: Dictionary containing structured pitch data (from HF dataset)
            
        Returns:
            dspy.Prediction with pitch field
        """
        # Convert the input dict to PitchInput model for validation and formatting
        try:
            pitch_input = PitchInput(**input)
            formatted_input = format_pitch_input(pitch_input)
        except Exception as e:
            print(f"Warning: Could not parse input as PitchInput: {e}")
            # Fallback: use JSON representation
            formatted_input = json.dumps(input, indent=2)
        
        # Generate the pitch using chain-of-thought
        prediction = self.generate_pitch(pitch_data=formatted_input)
        
        return prediction


# ---------- 3) EVALUATION METRIC ----------
class PitchEvaluator(dspy.Module):
    """
    Module for evaluating pitch quality using AssessPitchQuality signature.
    Uses GPT-OSS-120B for objective, powerful evaluation (different from 70B generator).
    """
    
    def __init__(self):
        super().__init__()
        # Initialize without passing lm parameter
        self.assess = dspy.ChainOfThought(AssessPitchQuality)
    
    def forward(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str):
        """
        Assess the quality of a generated pitch using GPT-OSS-120B.
        
        Args:
            pitch_facts: The structured facts the pitch was based on
            ground_truth_pitch: The gold-standard human-written pitch
            generated_pitch: The AI-generated pitch
            
        Returns:
            Assessment with Pydantic PitchAssessment model (scores + reasoning)
        """
        # Use context manager to temporarily switch to evaluator model
        with dspy.context(lm=evaluator_lm):
            assessment = self.assess(
                pitch_facts=pitch_facts,
                ground_truth_pitch=ground_truth_pitch,
                generated_pitch=generated_pitch
            )
        return assessment


def pitch_quality_metric(example, pred, trace=None):
    """
    Metric function for evaluating pitch quality during optimization.
    
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
        evaluator = PitchEvaluator()
        assessment = evaluator(
            pitch_facts=pitch_facts,
            ground_truth_pitch=ground_truth,
            generated_pitch=generated_pitch
        )
        
        # Extract final score from Pydantic assessment
        try:
            # assessment.assessment is a PitchAssessment Pydantic model
            final_score = float(assessment.assessment.final_score)
            return final_score
        except (AttributeError, ValueError) as e:
            print(f"Warning: Could not extract final score: {e}")
            # Fallback: basic comparison
            return 0.5 if generated_pitch else 0.0
            
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
        optimization_method: "none", "bootstrap", or "mipro"
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
    
    elif optimization_method == "mipro":
        print("Compiling with MIPROv2...")
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=10,
            init_temperature=1.0
        )
        compiled_program = optimizer.compile(
            program,
            trainset=trainset,
            num_trials=10,
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
    evaluator = PitchEvaluator() if use_evaluator else None
    
    for idx, example in enumerate(tqdm(testset, desc="Evaluating", unit="pitch")):
        try:
            # Generate pitch (uses generator_lm: 70B)
            prediction = program(input=example.input)
            generated_pitch = prediction.pitch if hasattr(prediction, "pitch") else str(prediction)
            
            # Rate limiting after generation
            if rate_limit and idx < len(testset) - 1:
                time.sleep(RATE_LIMIT_DELAY)
            
            # Evaluate if requested (uses evaluator_lm: 120B)
            if use_evaluator:
                pitch_facts = json.dumps(example.input, indent=2)
                assessment = evaluator(
                    pitch_facts=pitch_facts,
                    ground_truth_pitch=example.output,
                    generated_pitch=generated_pitch
                )
                
                try:
                    # assessment.assessment is a Pydantic PitchAssessment model
                    pitch_assessment = assessment.assessment
                    assessment_data = {
                        "factual_score": float(pitch_assessment.factual_score),
                        "narrative_score": float(pitch_assessment.narrative_score),
                        "style_score": float(pitch_assessment.style_score),
                        "reasoning": str(pitch_assessment.reasoning),
                        "final_score": float(pitch_assessment.final_score)
                    }
                except (AttributeError, ValueError, TypeError) as e:
                    print(f"\n  Warning: Failed to extract assessment: {e}")
                    assessment_data = {
                        "factual_score": 0.0,
                        "narrative_score": 0.0,
                        "style_score": 0.0,
                        "reasoning": f"Parse error: {str(e)}",
                        "final_score": 0.0
                    }
                
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
        choices=["none", "bootstrap", "mipro"],
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
    
    # Create program
    print("\n2. Creating pitch generation program...")
    program = StructuredPitchProgram()
    
    # Compile with optimization if requested
    if args.optimization != "none":
        print(f"\n3. Compiling with {args.optimization}...")
        program = compile_program(
            program,
            trainset,
            optimization_method=args.optimization,
            metric=simple_pitch_metric
        )
        if args.save_program:
            save_dir = "optimized_programs"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/pitch_{args.optimization}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            program.save(save_path)
            print(f"   ✓ Saved optimized program to: {save_path}")
    else:
        print("\n3. Running baseline (no optimization)...")
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    if not args.no_rate_limit:
        print(f"   Rate limiting enabled: {RATE_LIMIT_DELAY}s delay between API calls")
        print(f"   Estimated time: ~{len(testset) * RATE_LIMIT_DELAY * (2 if args.evaluate else 1) / 60:.1f} minutes")
    results = evaluate_program(program, testset, use_evaluator=args.evaluate, rate_limit=not args.no_rate_limit)
    
    # Save results
    print("\n5. Saving results...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create results dataframe
    rows = []
    for result in results:
        row = {
            "id": result["id"],
            "company_name": result["input"].get("company_name", "Unknown"),
            "ground_truth": result["ground_truth"],
            "generated_pitch": result["generated_pitch"],
            "optimization_method": args.optimization,
            "model_name": generator_lm.model,
            "evaluator_model_name": evaluator_lm.model,
            "timestamp": timestamp
        }
        
        if result["assessment"]:
            row["final_score"] = result["assessment"].get("final_score", 0.0)
            row["factual_score"] = result["assessment"].get("factual_score", 0.0)
            row["narrative_score"] = result["assessment"].get("narrative_score", 0.0)
            row["style_score"] = result["assessment"].get("style_score", 0.0)
            row["reasoning"] = result["assessment"].get("reasoning", "")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_file = f"structured_pitch_results_{args.optimization}_{run_name}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary statistics
    if args.evaluate and "final_score" in df.columns:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Generator Model: groq/llama-3.3-70b-versatile")
        print(f"Evaluator Model: groq/openai/gpt-oss-120b")
        print(f"Optimization Method: {args.optimization}")
        print(f"\nScores (0.0-1.0):")
        print(f"  Average Final Score: {df['final_score'].mean():.3f} ± {df['final_score'].std():.3f}")
        print(f"  Average Factual Score: {df['factual_score'].mean():.3f} ± {df['factual_score'].std():.3f}")
        print(f"  Average Narrative Score: {df['narrative_score'].mean():.3f} ± {df['narrative_score'].std():.3f}")
        print(f"  Average Style Score: {df['style_score'].mean():.3f} ± {df['style_score'].std():.3f}")
    
    print("\n✓ Done!")

