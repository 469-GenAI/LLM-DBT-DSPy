"""
Dataset preparation for DSPy optimization
Prepares training, validation, and test sets from all_processed_facts.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import dspy
from dataclasses import dataclass
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_paths import get_all_processed_facts


@dataclass
class PitchExample:
    """Structure for a single training example"""
    product_data: dict
    expected_has_pitch: bool = True
    expected_has_offer: bool = True
    expected_has_valuation: bool = True
    

class SharkTankDataset:
    """Dataset manager for SharkTank pitch data"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize dataset from all_processed_facts.json
        
        Args:
            data_path: Path to all_processed_facts.json file. If None, uses data/all_processed_facts.json
        """
        if data_path is None:
            # Use centralized data path utility
            self.data_path = get_all_processed_facts()
        else:
            self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Could not find all_processed_facts.json at: {self.data_path}. "
                f"Please specify data_path or ensure file exists in data/ directory."
            )
        self.raw_data = self._load_data()
        self.examples = self._create_examples()
        
    def _load_data(self) -> dict:
        """Load raw JSON data"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_examples(self) -> List[PitchExample]:
        """Convert raw data to structured examples"""
        examples = []
        
        for product_key, product_data in self.raw_data.items():
            # Create a proper example with input/output structure
            example = PitchExample(
                product_data=product_data,
                expected_has_pitch=True,
                expected_has_offer=True,
                expected_has_valuation=True
            )
            examples.append(example)
        
        return examples
    
    def to_dspy_examples(self, examples: List[PitchExample]) -> List[dspy.Example]:
        """Convert PitchExample objects to dspy.Example format with inputs"""
        dspy_examples = []
        
        for ex in examples:
            # Convert product_data to a JSON string for the model
            product_json = json.dumps(ex.product_data, indent=2)
            
            # Create dspy.Example with proper input/output separation
            dspy_ex = dspy.Example(
                product_data=ex.product_data,
                product_json=product_json,
                expected_has_pitch=ex.expected_has_pitch,
                expected_has_offer=ex.expected_has_offer,
                expected_has_valuation=ex.expected_has_valuation
            ).with_inputs('product_data', 'product_json')
            
            dspy_examples.append(dspy_ex)
        
        return dspy_examples
    
    def split(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """
        Split dataset into train/val/test sets
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_examples, val_examples, test_examples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle examples
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)
        
        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_raw = shuffled[:train_end]
        val_raw = shuffled[train_end:val_end]
        test_raw = shuffled[val_end:]
        
        # Convert to dspy.Example format
        train_set = self.to_dspy_examples(train_raw)
        val_set = self.to_dspy_examples(val_raw)
        test_set = self.to_dspy_examples(test_raw)
        
        print(f"Dataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        return train_set, val_set, test_set
    
    def get_subset(self, n: int, seed: int = 42) -> List[dspy.Example]:
        """
        Get a random subset of n examples for quick testing
        
        Args:
            n: Number of examples to return
            seed: Random seed
            
        Returns:
            List of dspy.Example objects
        """
        random.seed(seed)
        subset = random.sample(self.examples, min(n, len(self.examples)))
        return self.to_dspy_examples(subset)


def load_dataset(
    data_path: str = None,
    train_size: int = None,
    val_size: int = None,
    test_size: int = None,
    seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Convenience function to load and split the dataset
    
    Args:
        data_path: Path to all_processed_facts.json
        train_size: If specified, limit training set to this size
        val_size: If specified, limit validation set to this size
        test_size: If specified, limit test set to this size
        seed: Random seed
        
    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    dataset = SharkTankDataset(data_path)
    train_set, val_set, test_set = dataset.split(seed=seed)
    
    # Optionally limit sizes
    if train_size is not None:
        train_set = train_set[:train_size]
    if val_size is not None:
        val_set = val_set[:val_size]
    if test_size is not None:
        test_set = test_set[:test_size]
    
    return train_set, val_set, test_set


if __name__ == "__main__":
    # Test dataset loading
    print("Loading SharkTank dataset...")
    dataset = SharkTankDataset()
    
    print(f"\nTotal examples: {len(dataset.examples)}")
    
    # Test split
    train, val, test = dataset.split()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Show example structure
    if len(train) > 0:
        print("\nExample structure:")
        example = train[0]
        print(f"Input keys: {example._input_keys}")
        print(f"Product data keys: {list(example.product_data.keys())}")

