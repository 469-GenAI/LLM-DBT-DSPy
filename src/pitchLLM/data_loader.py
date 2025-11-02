# data_loader.py - Data loading utilities for DSPy
"""
Data loading utilities for the structured pitch generation system.
Loads data from HuggingFace datasets and converts to DSPy Examples.
"""
import dspy
from datasets import load_dataset
from typing import List, Dict, Any


def load_hf_dataset(dataset_name: str = "isaidchia/sharktank_pitches") -> Dict[str, Any]:
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
        - _meta_index: index metadata
        - input: dict with pitch structure (founders, company_name, etc.)
        - output: string with the actual pitch text
        
        We convert this to dspy.Example and mark 'input' as the input field
    """
    examples = []
    
    for row in hf_dataset_split:
        # Create a DSPy Example with all fields from the row
        # This includes: id, _meta_index, input (dict), output (string)
        example = dspy.Example(
            id=row["id"],
            _meta_index=row["_meta_index"],
            input=row["input"],
            output=row["output"]
        )
        
        # Mark 'input' as the input field and 'output' as the ground truth
        # This tells DSPy which field to pass to the model and which to compare against
        example = example.with_inputs("input")
        
        examples.append(example)
    
    print(f"Prepared {len(examples)} DSPy examples")
    return examples


def load_and_prepare_data(dataset_name: str = "isaidchia/sharktank_pitches") -> Dict[str, List[dspy.Example]]:
    """
    Convenience function to load dataset and prepare both train and test splits.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        
    Returns:
        Dictionary with 'train' and 'test' keys containing DSPy Examples
    """
    # Load the dataset from HuggingFace
    dataset = load_hf_dataset(dataset_name)
    
    # Convert both splits to DSPy Examples
    train_examples = prepare_dspy_examples(dataset["train"])
    test_examples = prepare_dspy_examples(dataset["test"])
    
    return {
        "train": train_examples,
        "test": test_examples
    }

