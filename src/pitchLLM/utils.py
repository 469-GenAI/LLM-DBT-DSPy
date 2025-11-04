# utils.py - Pydantic models for structured pitch input
"""
Utility models and helper functions for the structured pitch generation system.
These models match the input format from the HuggingFace sharktank_pitches dataset.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import json
import os
from datetime import datetime


class ProblemStory(BaseModel):
    """Model representing the problem narrative in a pitch."""
    persona: str = Field(..., description="The target customer persona")
    routine: List[str] = Field(..., description="List of routine behaviors or actions")
    core_problem: str = Field(..., description="The central problem being addressed")
    hygiene_gap: str = Field(..., description="The gap in current solutions")
    problem_keywords: List[str] = Field(..., description="Key problem descriptors")


class ProductSolution(BaseModel):
    """Model representing the product solution in a pitch."""
    name: str = Field(..., description="Product or company name")
    product_category: str = Field(..., description="Category of the product")
    key_differentiator: str = Field(..., description="What makes this product unique")
    application: str = Field(..., description="How the product is used")
    features_keywords: List[str] = Field(..., description="Key features of the product")
    benefits_keywords: List[str] = Field(..., description="Key benefits to customers")


class ClosingTheme(BaseModel):
    """Model representing the closing theme of a pitch."""
    call_to_action: str = Field(..., description="The call to action for investors")
    mission: str = Field(..., description="The company's mission statement")
    target_audience: str = Field(..., description="Target audience description")


class InitialOfferInput(BaseModel):
    """Model representing the investment offer details."""
    amount: str = Field(..., description="Funding amount requested (e.g., '$400k')")
    equity: str = Field(..., description="Equity percentage offered (e.g., '5%')")


class PitchInput(BaseModel):
    """Complete structured input for pitch generation."""
    founders: List[str] = Field(..., description="List of founder names")
    company_name: str = Field(..., description="Name of the company")
    initial_offer: InitialOfferInput = Field(..., description="Investment offer details")
    problem_story: ProblemStory = Field(..., description="The problem narrative")
    product_solution: ProductSolution = Field(..., description="The product solution")
    closing_theme: ClosingTheme = Field(..., description="Closing theme and call to action")


def format_pitch_input(pitch_input: PitchInput) -> str:
    """
    Format a PitchInput object into a structured string for DSPy processing.
    
    Args:
        pitch_input: Structured pitch input data
        
    Returns:
        Formatted string representation of the pitch input
    """
    return f"""
Company: {pitch_input.company_name}
Founders: {', '.join(pitch_input.founders)}
Investment Ask: {pitch_input.initial_offer.amount} for {pitch_input.initial_offer.equity} equity

PROBLEM:
Persona: {pitch_input.problem_story.persona}
Core Problem: {pitch_input.problem_story.core_problem}
Gap: {pitch_input.problem_story.hygiene_gap}

SOLUTION:
Product: {pitch_input.product_solution.name}
Category: {pitch_input.product_solution.product_category}
Differentiator: {pitch_input.product_solution.key_differentiator}
Application: {pitch_input.product_solution.application}
Key Features: {', '.join(pitch_input.product_solution.features_keywords)}
Key Benefits: {', '.join(pitch_input.product_solution.benefits_keywords)}

CLOSING:
Mission: {pitch_input.closing_theme.mission}
Call to Action: {pitch_input.closing_theme.call_to_action}
""".strip()


# ---------- LOGGING AND SAVING UTILITIES ----------

def capture_mlflow_run_id(
    databricks_path: Optional[str],
    run_name: str
) -> Optional[str]:
    """
    Capture the MLflow run_id after optimization completes.
    
    Args:
        databricks_path: Path to Databricks MLflow tracking
        run_name: Name of the current run
        
    Returns:
        MLflow run_id if available, None otherwise
        
    Note:
        active_run() often returns None with autolog, so we search for the most
        recent run in the experiment. Uses timeout to prevent hanging on network issues.
    """
    if not databricks_path:
        return None
    
    try:
        # First try active_run (fast, but usually returns None with autolog)
        current_run = mlflow.active_run()
        if current_run:
            mlflow_run_id = current_run.info.run_id
            print(f"   ✓ MLflow Run ID: {mlflow_run_id}")
            return mlflow_run_id
        
        # Fall back to searching for the most recent run (needed for autolog)
        # Use threading with timeout to prevent hanging
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        def search_mlflow():
            """Search for the most recent run in this experiment."""
            experiment = mlflow.get_experiment_by_name(databricks_path + run_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if not runs.empty:
                    return runs.iloc[0]["run_id"]
            return None
        
        # Execute search with 10 second timeout (Databricks can be slow)
        print(f"   Searching for MLflow run...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_mlflow)
            try:
                mlflow_run_id = future.result(timeout=10.0)
                if mlflow_run_id:
                    print(f"   ✓ MLflow Run ID: {mlflow_run_id}")
                    return mlflow_run_id
                else:
                    print(f"   ℹ No MLflow run found yet (may appear in next attempt)")
                    return None
            except FutureTimeoutError:
                print(f"   ⚠ MLflow search timed out (network issue - continuing without run_id)")
                return None
        
    except Exception as e:
        print(f"   ⚠ Could not capture MLflow run_id: {e}")
        return None

def generate_output_filename(
    optimization_method: str,
    run_name: str,
    mlflow_run_id: Optional[str] = None,
    file_type: str = "csv"
) -> str:
    """
    Generate a consistent filename for results or programs.
    
    Args:
        optimization_method: The optimization method used
        run_name: The run name
        mlflow_run_id: Optional MLflow run_id to include
        file_type: File extension (csv, json, etc.)
        
    Returns:
        Formatted filename
    """
    filename = f"structured_pitch_results_{optimization_method}_{run_name}"
    if mlflow_run_id:
        filename += f"_{mlflow_run_id[:8]}"  # Add first 8 chars of run_id
    filename += f".{file_type}"
    return filename


def save_program_with_metadata(
    program,
    save_dir: str,
    optimization_method: str,
    generator_model: str,
    evaluator_model: str,
    trainset_size: int,
    testset_size: int,
    run_name: str,
    mlflow_run_id: Optional[str] = None
) -> str:
    """
    Save a DSPy program with comprehensive metadata.
    
    Args:
        program: The DSPy program to save
        save_dir: Directory to save to
        optimization_method: Optimization method used
        generator_model: Generator model name
        evaluator_model: Evaluator model name
        trainset_size: Size of training set
        testset_size: Size of test set
        run_name: The run name
        mlflow_run_id: Optional MLflow run_id
        
    Returns:
        Path to the saved program
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/pitch_{optimization_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save the program first
    program.save(save_path)
    
    # Load and enhance with metadata
    with open(save_path, "r") as f:
        saved_data = json.load(f)
    
    # Ensure metadata section exists
    if "metadata" not in saved_data:
        saved_data["metadata"] = {}
    
    # Add models used
    saved_data["metadata"]["models_used"] = {
        "generator": generator_model,
        "evaluator": evaluator_model,
        "optimization_method": optimization_method
    }
    
    # Add training info
    saved_data["metadata"]["training_info"] = {
        "train_size": trainset_size,
        "test_size": testset_size,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Add tracking info (MLflow run_id and CSV filename reference)
    csv_filename = generate_output_filename(
        optimization_method,
        run_name,
        mlflow_run_id,
        "csv"
    )
    
    saved_data["metadata"]["tracking"] = {
        "mlflow_run_id": mlflow_run_id,
        "results_csv_file": csv_filename,
        "run_name": run_name
    }
    
    # Save back with enhanced metadata
    with open(save_path, "w") as f:
        json.dump(saved_data, f, indent=2)
    
    print(f"   ✓ Saved optimized program to: {save_path}")
    print(f"   ✓ Model info: {generator_model}")
    
    return save_path


def results_to_dataframe(
    results: List[Dict[str, Any]],
    optimization_method: str,
    generator_model: str,
    evaluator_model: str
) -> pd.DataFrame:
    """
    Convert evaluation results to a pandas DataFrame.
    
    Args:
        results: List of result dictionaries
        optimization_method: Optimization method used
        generator_model: Generator model name
        evaluator_model: Evaluator model name
        
    Returns:
        DataFrame with results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    rows = []
    for result in results:
        row = {
            "id": result["id"],
            "company_name": result["input"].get("company_name", "Unknown"),
            "ground_truth": result["ground_truth"],
            "generated_pitch": result["generated_pitch"],
            "optimization_method": optimization_method,
            "model_name": generator_model,
            "evaluator_model_name": evaluator_model,
            "timestamp": timestamp
        }
        
        # Add assessment scores if available
        if result["assessment"]:
            row["final_score"] = result["assessment"].get("final_score", 0.0)
            row["factual_score"] = result["assessment"].get("factual_score", 0.0)
            row["narrative_score"] = result["assessment"].get("narrative_score", 0.0)
            row["style_score"] = result["assessment"].get("style_score", 0.0)
            row["reasoning"] = result["assessment"].get("reasoning", "")
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_results_csv(
    df: pd.DataFrame,
    optimization_method: str,
    run_name: str,
    mlflow_run_id: Optional[str] = None
) -> str:
    """
    Save results DataFrame to CSV with consistent naming.
    
    Args:
        df: DataFrame with results
        optimization_method: Optimization method used
        run_name: The run name
        mlflow_run_id: Optional MLflow run_id
        
    Returns:
        Path to saved CSV file
    """
    output_file = generate_output_filename(
        optimization_method,
        run_name,
        mlflow_run_id,
        "csv"
    )
    
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    if mlflow_run_id:
        print(f"✓ MLflow Run ID: {mlflow_run_id}")
    
    return output_file


def print_evaluation_summary(
    df: pd.DataFrame,
    generator_model: str,
    evaluator_model: str,
    optimization_method: str
) -> None:
    """
    Print evaluation summary statistics.
    
    Args:
        df: DataFrame with evaluation results
        generator_model: Generator model name
        evaluator_model: Evaluator model name
        optimization_method: Optimization method used
    """
    if "final_score" not in df.columns:
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Generator Model: {generator_model}")
    print(f"Evaluator Model: {evaluator_model}")
    print(f"Optimization Method: {optimization_method}")
    print(f"\nScores (0.0-1.0):")
    print(f"  Average Final Score: {df['final_score'].mean():.3f} ± {df['final_score'].std():.3f}")
    print(f"  Average Factual Score: {df['factual_score'].mean():.3f} ± {df['factual_score'].std():.3f}")
    print(f"  Average Narrative Score: {df['narrative_score'].mean():.3f} ± {df['narrative_score'].std():.3f}")
    print(f"  Average Style Score: {df['style_score'].mean():.3f} ± {df['style_score'].std():.3f}")

