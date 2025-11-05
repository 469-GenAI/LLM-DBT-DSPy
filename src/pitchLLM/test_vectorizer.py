#!/usr/bin/env python3
"""
Test script for the pitch vectorizer implementation.
Verifies that the KNNFewShot vectorizer correctly handles complex pitch inputs.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    flatten_dict_to_text,
    pitch_input_to_embedding_text,
    create_pitch_vectorizer
)
import dspy
import json


def test_flatten_dict():
    """Test the flatten_dict_to_text function."""
    print("=" * 80)
    print("TEST 1: Flatten Dictionary")
    print("=" * 80)
    
    test_data = {
        "company_name": "PDX Pet Design",
        "initial_offer": {
            "amount": "300,000",
            "equity": "15%"
        },
        "problem_story": {
            "persona": "cat owners",
            "problem_keywords": ["disposable", "wasteful"]
        }
    }
    
    flattened = flatten_dict_to_text(test_data)
    print("\nInput:")
    print(json.dumps(test_data, indent=2))
    print("\nFlattened output:")
    for item in flattened:
        print(f"  - {item}")
    
    assert len(flattened) > 0, "Should produce flattened output"
    assert any("company_name" in item for item in flattened), "Should include company_name"
    print("\n✓ Test passed!")


def test_pitch_to_text():
    """Test the pitch_input_to_embedding_text function."""
    print("\n" + "=" * 80)
    print("TEST 2: Convert Pitch to Embedding Text")
    print("=" * 80)
    
    pitch_dict = {
        "founders": ["Founder 1", "Founder 2"],
        "company_name": "PDX Pet Design",
        "initial_offer": {"amount": "300,000", "equity": "15%"},
        "problem_story": {
            "persona": "cat owners",
            "core_problem": "cat toys are disposable"
        }
    }
    
    embedding_text = pitch_input_to_embedding_text(pitch_dict)
    print("\nInput:")
    print(json.dumps(pitch_dict, indent=2))
    print("\nEmbedding text (first 200 chars):")
    print(f"  {embedding_text[:200]}...")
    print(f"\nFull length: {len(embedding_text)} characters")
    
    assert len(embedding_text) > 0, "Should produce text output"
    assert "PDX Pet Design" in embedding_text, "Should contain company name"
    assert "cat owners" in embedding_text, "Should contain persona"
    print("\n✓ Test passed!")


def test_vectorizer():
    """Test the create_pitch_vectorizer function."""
    print("\n" + "=" * 80)
    print("TEST 3: Create and Use Vectorizer")
    print("=" * 80)
    
    # Create a test example
    pitch_input = {
        "founders": ["Founder 1", "Founder 2"],
        "company_name": "PDX Pet Design",
        "initial_offer": {"amount": "300,000", "equity": "15%"},
        "problem_story": {
            "persona": "cat owners",
            "core_problem": "cat toys are disposable and quickly lose interest"
        },
        "product_solution": {
            "name": "Shrew",
            "product_category": "cat wellness products",
            "key_differentiator": "intelligent cat companion"
        },
        "closing_theme": {
            "mission": "revolutionize pet products",
            "target_audience": "cat owners"
        }
    }
    
    example = dspy.Example(
        id="test-1",
        input=pitch_input,
        output="Test pitch output"
    ).with_inputs("input")
    
    print("\nCreating vectorizer...")
    vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
    
    print("\nVectorizing example...")
    embedding = vectorizer(example)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding type: {type(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    assert embedding is not None, "Should produce embedding"
    assert len(embedding) == 384, "all-MiniLM-L6-v2 should produce 384-dim embeddings"
    print("\n✓ Test passed!")


def test_similarity():
    """Test that similar pitches produce similar embeddings."""
    print("\n" + "=" * 80)
    print("TEST 4: Semantic Similarity")
    print("=" * 80)
    
    # Two similar pitches (pet products)
    pitch1 = {
        "company_name": "PDX Pet Design",
        "problem_story": {"persona": "cat owners", "core_problem": "disposable cat toys"},
        "product_solution": {"name": "Shrew", "product_category": "cat wellness products"}
    }
    
    pitch2 = {
        "company_name": "Dog Toys Inc",
        "problem_story": {"persona": "dog owners", "core_problem": "boring dog toys"},
        "product_solution": {"name": "FetchBot", "product_category": "dog wellness products"}
    }
    
    # One dissimilar pitch (food tech)
    pitch3 = {
        "company_name": "NuMilk",
        "problem_story": {"persona": "health conscious consumers", "core_problem": "processed dairy milk"},
        "product_solution": {"name": "NuMilk Machine", "product_category": "dairy-free milk"}
    }
    
    vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
    
    ex1 = dspy.Example(input=pitch1, output="").with_inputs("input")
    ex2 = dspy.Example(input=pitch2, output="").with_inputs("input")
    ex3 = dspy.Example(input=pitch3, output="").with_inputs("input")
    
    emb1 = vectorizer(ex1)
    emb2 = vectorizer(ex2)
    emb3 = vectorizer(ex3)
    
    # Calculate cosine similarity
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine_similarity(emb1, emb2)  # Similar (both pet products)
    sim_1_3 = cosine_similarity(emb1, emb3)  # Different (pet vs food)
    
    print(f"\nSimilarity between pet products (pitch1 vs pitch2): {sim_1_2:.4f}")
    print(f"Similarity between pet and food tech (pitch1 vs pitch3): {sim_1_3:.4f}")
    
    # Similar pitches should have higher similarity
    assert sim_1_2 > sim_1_3, "Similar pitches should have higher similarity scores"
    print(f"\n✓ Test passed! Pet products are more similar to each other ({sim_1_2:.4f}) than to food tech ({sim_1_3:.4f})")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PITCH VECTORIZER TEST SUITE" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_flatten_dict()
        test_pitch_to_text()
        test_vectorizer()
        test_similarity()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nThe vectorizer is ready to use with KNNFewShot optimization.")
        print("\nTo use it in training:")
        print("  python pitchLLM_structured.py --optimization knn --train-size 20 --test-size 5")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

