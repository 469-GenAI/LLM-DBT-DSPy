# data_loader.py - Data loading utilities for DSPy
"""
Data loading utilities for the structured pitch generation system.
Loads data from HuggingFace datasets and local JSONL files, converts to DSPy Examples.

Purpose:
- Loads test.jsonl for evaluation (converts to DSPy Examples)
- Different from data_indexer.py which loads train.jsonl for indexing into vector DB
"""
import dspy
import json
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any, Optional


def load_jsonl_file(jsonl_path: str) -> List[Dict]:
    """
    Load data from a local JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def prepare_dspy_examples_from_dicts(data_dicts: List[Dict]) -> List[dspy.Example]:
    """
    Convert list of dictionaries to DSPy Example objects.
    
    Args:
        data_dicts: List of dicts with 'id', 'input', 'output' keys
        
    Returns:
        List of DSPy Example objects
    """
    examples = []
    
    for row in data_dicts:
        example = dspy.Example(
            id=row.get("id", ""),
            input=row.get("input", {}),
            output=row.get("output", "")
        )
        example = example.with_inputs("input")
        examples.append(example)
    
    return examples


def load_hf_dataset(dataset_name: str = "isaidchia/sharktank_pitches_modified") -> Dict[str, Any]:
    """
    Load the HuggingFace sharktank_pitches dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        
    Returns:
        Dataset object with train and test splits
        
    Note:
        Assumes user has authenticated via `huggingface-cli login`
    """
    print(f"Loading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Train split size: {len(dataset['train'])}")
    print(f"Test split size: {len(dataset['test'])}")
    return dataset


def prepare_dspy_examples(hf_dataset_split) -> List[dspy.Example]:
    """
    Convert HuggingFace dataset split to DSPy Example objects.
    
    Args:
        hf_dataset_split: A split from the HuggingFace dataset (e.g., dataset['train'])
        
    Returns:
        List of DSPy Example objects with input_keys set correctly
        
    Example:
        Each row in the HF dataset has:
        - id: unique identifier
        - input: dict with pitch structure (company, founder, offer, problem_summary, solution_summary)
        - output: string with the actual pitch text
        
        We convert this to dspy.Example and mark 'input' as the input field
    """
    examples = []
    
    for row in hf_dataset_split:
        # Create a DSPy Example with all fields from the row
        # This includes: id,input (dict), output (string)
        example = dspy.Example(
            id=row["id"],
            input=row["input"],
            output=row["output"]
        )
        
        # Mark 'input' as the input field and 'output' as the ground truth
        # This tells DSPy which field to pass to the model and which to compare against
        example = example.with_inputs("input")
        
        examples.append(example)
    
    print(f"Prepared {len(examples)} DSPy examples")
    return examples


def load_and_prepare_data(
    dataset_name: Optional[str] = None,
    train_jsonl_path: Optional[str] = None,
    test_jsonl_path: Optional[str] = None
) -> Dict[str, List[dspy.Example]]:
    """
    Convenience function to load dataset and prepare both train and test splits.
    
    Supports two modes:
    1. HuggingFace dataset: Pass dataset_name
    2. Local JSONL files: Pass train_jsonl_path and test_jsonl_path
    
    Args:
        dataset_name: Name of the HuggingFace dataset (if using HF)
        train_jsonl_path: Path to local train.jsonl file (if using local files)
        test_jsonl_path: Path to local test.jsonl file (if using local files)
        
    Returns:
        Dictionary with 'train' and 'test' keys containing DSPy Examples
    """
    # Check if using local JSONL files
    if train_jsonl_path or test_jsonl_path:
        print("Loading from local JSONL files:")
        print(f"  Train: {train_jsonl_path}")
        print(f"  Test: {test_jsonl_path}")
        
        train_examples = []
        test_examples = []
        
        if train_jsonl_path:
            train_data = load_jsonl_file(train_jsonl_path)
            train_examples = prepare_dspy_examples_from_dicts(train_data)
            print(f"Prepared {len(train_examples)} DSPy examples from local files")
        
        if test_jsonl_path:
            test_data = load_jsonl_file(test_jsonl_path)
            test_examples = prepare_dspy_examples_from_dicts(test_data)
            print(f"Prepared {len(test_examples)} DSPy examples from local files")
        
        return {
            "train": train_examples,
            "test": test_examples
        }
    
    # Default: Use HuggingFace dataset
    if dataset_name is None:
        dataset_name = "isaidchia/sharktank_pitches_modified"
    
    # Load the dataset from HuggingFace
    dataset = load_hf_dataset(dataset_name)
    
    # Convert both splits to DSPy Examples
    train_examples = prepare_dspy_examples(dataset["train"])
    test_examples = prepare_dspy_examples(dataset["test"])
    
    return {
        "train": train_examples,
        "test": test_examples
    }

