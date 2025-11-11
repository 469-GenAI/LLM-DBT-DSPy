# utils.py - Pydantic models for structured pitch input
"""
Utility models and helper functions for the structured pitch generation system.
These models match the input format from the HuggingFace sharktank_pitches dataset.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path


class PitchInput(BaseModel):
    """Complete structured input for pitch generation."""
    company: str = Field(..., description="Name of the company")
    founder: List[str] = Field(..., description="List of founder names")
    offer: str = Field(..., description="Investment offer (e.g., '125000 for 20%')")
    problem_summary: str = Field(..., description="Summary of the problem being addressed")
    solution_summary: str = Field(..., description="Summary of the solution provided")


def format_pitch_input(pitch_input: PitchInput) -> str:
    """
    Format a PitchInput object into a structured string for DSPy processing.
    
    Args:
        pitch_input: Structured pitch input data
        
    Returns:
        Formatted string representation of the pitch input
    """
    return f"""
Company: {pitch_input.company}
Founders: {', '.join(pitch_input.founder)}
Investment Offer: {pitch_input.offer}

PROBLEM:
{pitch_input.problem_summary}

SOLUTION:
{pitch_input.solution_summary}
""".strip()


# ---------- EMBEDDING AND VECTORIZATION UTILITIES ----------

def flatten_dict_to_text(d: Dict[str, Any], parent_key: str = "", sep: str = ": ") -> List[str]:
    """
    Recursively flatten a nested dictionary into 'key: value' text pairs.
    
    This preserves semantic context by keeping keys alongside their values,
    which improves embedding quality for semantic similarity matching.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested structures (used in recursion)
        sep: Separator between key and value
        
    Returns:
        List of 'key: value' strings
        
    Example:
        >>> data = {
        ...     "company": "PDX Pet Design",
        ...     "founder": ["Founder 1", "Founder 2"],
        ...     "offer": "300,000 for 15%",
        ...     "problem_summary": "Disposable cat toys are wasteful",
        ...     "solution_summary": "Sustainable cat toys"
        ... }
        >>> flatten_dict_to_text(data)
        [
            "company: PDX Pet Design",
            "founder: Founder 1, Founder 2",
            "offer: 300,000 for 15%",
            "problem_summary: Disposable cat toys are wasteful",
            "solution_summary: Sustainable cat toys"
        ]
    """
    items = []
    
    for key, value in d.items():
        # Create hierarchical key for nested dicts (optional, can be simplified)
        new_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict_to_text(value, parent_key="", sep=sep))
        elif isinstance(value, list):
            # Join list items with commas
            list_str = ", ".join(str(item) for item in value)
            items.append(f"{key}{sep}{list_str}")
        elif value is not None:  # Skip None values
            items.append(f"{key}{sep}{value}")
    
    return items


def pitch_input_to_embedding_text(pitch_input_dict: Dict[str, Any]) -> str:
    """
    Convert a pitch input dictionary to a flattened text string for embedding.
    
    This function is specifically designed for use with KNNFewShot vectorizers,
    transforming complex nested pitch structures into semantically rich text
    that preserves context through key-value pairs.
    
    Args:
        pitch_input_dict: Dictionary containing pitch structure (from HF dataset)
        
    Returns:
        Space-separated string of 'key: value' pairs suitable for embedding
        
    Example:
        >>> pitch_dict = {
        ...     "company": "PDX Pet Design",
        ...     "founder": ["Founder 1"],
        ...     "offer": "300,000 for 15%",
        ...     "problem_summary": "Cat toys are wasteful",
        ...     "solution_summary": "Sustainable cat toys"
        ... }
        >>> text = pitch_input_to_embedding_text(pitch_dict)
        >>> # Returns: "company: PDX Pet Design founder: Founder 1 offer: 300,000 for 15% ..."
    """
    flattened_pairs = flatten_dict_to_text(pitch_input_dict)
    return " ".join(flattened_pairs)


def create_pitch_vectorizer(model_name: str = "all-MiniLM-L6-v2"):
    """
    Create a vectorizer function for KNNFewShot that handles complex pitch input dictionaries.
    
    This vectorizer converts nested pitch structures into embeddings by:
    1. Flattening the input dict with keys (preserving semantic context)
    2. Converting to a single text string
    3. Encoding with SentenceTransformer
    
    Args:
        model_name: SentenceTransformer model to use for embeddings
                   Default: "all-MiniLM-L6-v2" (lightweight, 384 dimensions)
                   Alternatives:
                   - "all-mpnet-base-v2" (better quality, 768 dimensions)
                   - "BAAI/bge-small-en-v1.5" (good for retrieval, 384 dimensions)
        
    Returns:
        Callable vectorizer function that takes a dspy.Example and returns embeddings
        
    Example:
        >>> vectorizer = create_pitch_vectorizer()
        >>> from dspy import KNNFewShot
        >>> optimizer = KNNFewShot(k=3, trainset=trainset, vectorizer=vectorizer)
        >>> compiled_program = optimizer.compile(program, trainset=trainset)
    
    Note:
        The vectorizer is created as a closure to avoid reloading the embedding
        model for each example. The SentenceTransformer model is loaded once
        and reused for all vectorization calls.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for KNNFewShot vectorization. "
            "Install it with: pip install sentence-transformers"
        )
    
    # Load the embedding model once (closure captures this)
    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)
    print(f"✓ Loaded embedding model with {embedding_model.get_sentence_embedding_dimension()} dimensions")
    
    def vectorize_example(example) -> Any:
        """
        Convert a DSPy Example with complex input dict to an embedding vector.
        
        Args:
            example: dspy.Example with 'input' field containing pitch structure dict
            
        Returns:
            numpy array of embeddings (shape: [embedding_dim])
        """
        try:
            if hasattr(example, "input") and isinstance(example.input, dict):
                # Convert complex input dict to flattened text
                input_text = pitch_input_to_embedding_text(example.input)
            else:
                # Fallback for simple string inputs or unexpected formats
                input_text = str(example)
            
            # Generate embedding
            embedding = embedding_model.encode(input_text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            print(f"Warning: Error vectorizing example: {e}")
            # Return zero vector as fallback to avoid breaking the optimization
            import numpy as np
            return np.zeros(embedding_model.get_sentence_embedding_dimension())
    
    return vectorize_example


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
            "company_name": result["input"].get("company", "Unknown"),
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
    mlflow_run_id: Optional[str] = None,
    output_dir: Union[str, Path] = "MoA/results"
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
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_file = output_dir_path / generate_output_filename(
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

