"""
Category classification for pitches using preset taxonomy + LLM.

Stores categories in a separate mapping file (doesn't modify original data).
"""

import json
import dspy
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

PRESET_CATEGORIES = [
    "Health & Fitness",
    "Food & Beverage", 
    "Beauty & Personal Care",
    "Technology & Software",
    "Home & Lifestyle",
    "Fashion & Apparel",
    "Pet Products",
    "Children & Family",
    "Automotive & Transportation",
    "Business & Productivity",
    "Entertainment & Media",
    "Other"
]


class CategoryClassificationSig(dspy.Signature):
    """Classify a product into a category."""
    
    product_info: str = dspy.InputField(
        desc="Product information including company, problem, solution"
    )
    
    category: str = dspy.OutputField(
        desc=f"One of these categories: {', '.join(PRESET_CATEGORIES)}"
    )


class CategoryClassifier:
    """LLM-based category classifier."""
    
    def __init__(self, lm):
        self.lm = lm
        self.classifier = dspy.ChainOfThought(CategoryClassificationSig)
        self.categories = PRESET_CATEGORIES
    
    def classify(self, pitch_data: Dict) -> str:
        """Classify a pitch into a preset category."""
        # Extract info
        input_data = pitch_data.get('input', {})
        company = input_data.get('company', '')
        problem = input_data.get('problem_summary', '')
        solution = input_data.get('solution_summary', '')
        
        product_info = f"""
Company: {company}
Problem: {problem}
Solution: {solution}
"""
        
        # Classify
        with dspy.context(lm=self.lm):
            result = self.classifier(product_info=product_info)
        
        category = result.category.strip()
        
        # Normalize to preset
        return self._normalize(category)
    
    def _normalize(self, category: str) -> str:
        """Normalize to preset category."""
        category = category.strip()
        
        # Exact match
        if category in self.categories:
            return category
        
        # Fuzzy match
        cat_lower = category.lower()
        for preset in self.categories:
            preset_lower = preset.lower()
            if (preset_lower in cat_lower or 
                cat_lower in preset_lower or
                any(word in preset_lower for word in cat_lower.split() if len(word) > 3)):
                return preset
        
        return "Other"


def load_category_mapping(mapping_path: Path) -> Dict[str, str]:
    """
    Load category mapping from file.
    
    Format: {"pitch_id": "category", ...}
    
    Args:
        mapping_path: Path to category mapping JSON file
        
    Returns:
        Dict mapping pitch_id -> category
    """
    if not mapping_path.exists():
        logger.warning(f"Category mapping file not found: {mapping_path}")
        return {}
    
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded {len(mapping)} category mappings from {mapping_path}")
        return mapping
    except Exception as e:
        logger.error(f"Error loading category mapping: {e}")
        return {}


def save_category_mapping(mapping: Dict[str, str], mapping_path: Path):
    """Save category mapping to file."""
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    logger.info(f"Saved {len(mapping)} category mappings to {mapping_path}")


def classify_train_data(
    input_path: Path,
    mapping_path: Path,
    classifier: CategoryClassifier,
    overwrite_existing: bool = False
) -> Dict[str, str]:
    """
    Classify all pitches and save to mapping file.
    
    Args:
        input_path: Path to train.jsonl
        mapping_path: Path to save category mapping JSON
        classifier: CategoryClassifier instance
        overwrite_existing: If True, reclassify existing categories
        
    Returns:
        Dict mapping pitch_id -> category
    """
    # Load existing mapping if it exists
    existing_mapping = {}
    if mapping_path.exists() and not overwrite_existing:
        existing_mapping = load_category_mapping(mapping_path)
        logger.info(f"Loaded {len(existing_mapping)} existing categories")
    
    # Classify all pitches
    category_mapping = existing_mapping.copy()
    new_count = 0
    updated_count = 0
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            pitch_data = json.loads(line)
            pitch_id = pitch_data.get('id', f'line_{line_num}')
            
            # Skip if already classified (unless overwriting)
            if pitch_id in category_mapping and not overwrite_existing:
                continue
            
            # Classify
            try:
                category = classifier.classify(pitch_data)
                category_mapping[pitch_id] = category
                
                if pitch_id in existing_mapping:
                    updated_count += 1
                else:
                    new_count += 1
                
                if line_num % 10 == 0:
                    logger.info(f"Classified {line_num} pitches... ({new_count} new, {updated_count} updated)")
            except Exception as e:
                logger.error(f"Error classifying pitch {pitch_id}: {e}")
                category_mapping[pitch_id] = "Other"
    
    # Save mapping
    save_category_mapping(category_mapping, mapping_path)
    
    logger.info(f"✓ Classification complete: {new_count} new, {updated_count} updated, {len(category_mapping)} total")
    
    # Print category distribution
    from collections import Counter
    category_counts = Counter(category_mapping.values())
    logger.info("\nCategory distribution:")
    for cat, count in category_counts.most_common():
        logger.info(f"  {cat}: {count}")
    
    return category_mapping


if __name__ == "__main__":
    """Run classification on train.jsonl."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Setup LM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found")
    
    lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    # Setup paths
    input_path = Path("data/hf (new)/train.jsonl")
    mapping_path = Path("data/hf (new)/category_mapping.json")
    
    # Classify
    classifier = CategoryClassifier(lm)
    mapping = classify_train_data(
        input_path=input_path,
        mapping_path=mapping_path,
        classifier=classifier,
        overwrite_existing=False  # Set to True to reclassify all
    )
    
    print(f"\n✓ Category mapping saved to: {mapping_path}")

