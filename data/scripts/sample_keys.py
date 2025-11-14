#!/usr/bin/env python3
"""
Simple script to sample 10 random keys from the JSON file with a specific seed.
"""

import json
import random
import sys
from pathlib import Path

def sample_keys(json_file_path: str, seed: int = 42, num_samples: int = 10) -> list[str]:
    """
    Sample random keys from a JSON file with a specific seed.
    
    Args:
        json_file_path: Path to the JSON file
        seed: Random seed for reproducibility
        num_samples: Number of keys to sample
        
    Returns:
        List of sampled keys
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Load the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    
    # Get all keys from the JSON
    all_keys = list(data.keys())
    print("Number of keys in the JSON file: ", len(all_keys))
    # print("Keys: ", all_keys)
    
    if len(all_keys) < num_samples:
        print(f"Warning: Only {len(all_keys)} keys available, returning all keys.")
        return all_keys
    
    # Sample random keys
    sampled_keys = random.sample(all_keys, num_samples)
    
    return sampled_keys

def main():
    """Main function to run the key sampling script."""
    # Default file path
    json_file_path = "data/all_processed_facts.json"
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"Error: File '{json_file_path}' not found.")
        print("Please make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Set seed and number of samples
    seed = 42
    num_samples = 10
    
    print(f"Sampling {num_samples} random keys from '{json_file_path}' with seed {seed}")
    print("-" * 60)
    
    # Sample the keys
    sampled_keys = sample_keys(json_file_path, seed, num_samples)
    
    # Print the results
    for i, key in enumerate(sampled_keys, 1):
        print(f"{i:2d}. \"{key}\"")
    
    print("-" * 60)
    print(f"Total keys sampled: {len(sampled_keys)}")

if __name__ == "__main__":
    main()
