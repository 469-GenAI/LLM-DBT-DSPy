"""
Quick test script to compare RAG vs Non-RAG pitch generation.
Shows side-by-side output for inspection.

Loads a test product from test.jsonl instead of using hardcoded data.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import dspy
import random

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import PitchGenerator, RAGPitchGenerator
from data_loader import load_jsonl_file, prepare_dspy_examples_from_dicts

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Configure models
generator_lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    model_type="chat",
    api_key=GROQ_API_KEY
)

# Create generators
print("="*80)
print("Initializing generators...")
print("="*80)
non_rag_generator = PitchGenerator(generator_lm)
rag_generator = RAGPitchGenerator(generator_lm, top_k=5)

print("\n✓ Generators ready\n")

# Load test product from test.jsonl
test_jsonl_path = Path("data/hf (new)/test.jsonl")
if not test_jsonl_path.exists():
    raise FileNotFoundError(f"Test file not found: {test_jsonl_path}")

print(f"Loading test product from {test_jsonl_path}...")
test_data = load_jsonl_file(test_jsonl_path)
if not test_data:
    raise ValueError(f"No test data found in {test_jsonl_path}")

# Select a random test product (or use first one)
random.seed(42)
test_item = random.choice(test_data)
test_product = test_item.get('input', {})

print("="*80)
print("TEST PRODUCT (from test.jsonl):")
print("="*80)
print(f"ID: {test_item.get('id', 'N/A')}")
print(f"Company: {test_product.get('company', 'N/A')}")
print(f"Founder: {', '.join(test_product.get('founder', []))}")
print(f"Offer: {test_product.get('offer', 'N/A')}")
print(f"Problem: {test_product.get('problem_summary', 'N/A')[:100]}...")
print(f"Solution: {test_product.get('solution_summary', 'N/A')[:100]}...")

print("\n" + "="*80)
print("GENERATING NON-RAG PITCH...")
print("="*80)
non_rag_result = non_rag_generator.generate(test_product)
non_rag_pitch = non_rag_result.pitch if hasattr(non_rag_result, 'pitch') else str(non_rag_result)

print("\n" + "-"*80)
print("NON-RAG OUTPUT:")
print("-"*80)
print(non_rag_pitch)

print("\n" + "="*80)
print("GENERATING RAG PITCH...")
print("="*80)
rag_result = rag_generator.generate(test_product)
rag_pitch = rag_result.pitch if hasattr(rag_result, 'pitch') else str(rag_result)

# Show retrieved pitches
if hasattr(rag_result, '_metadata') and rag_result._metadata:
    print("\n" + "-"*80)
    print("RETRIEVED SIMILAR PITCHES:")
    print("-"*80)
    retrieved = rag_result._metadata.get('retrieved_pitches', [])
    for i, p in enumerate(retrieved, 1):
        print(f"{i}. {p.get('product', 'Unknown')} (similarity: {p.get('similarity', 0):.3f})")

print("\n" + "-"*80)
print("RAG OUTPUT:")
print("-"*80)
print(rag_pitch)

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"Non-RAG length: {len(non_rag_pitch)} chars")
print(f"RAG length: {len(rag_pitch)} chars")
print(f"\nRetrieved examples: {len(retrieved) if hasattr(rag_result, '_metadata') and rag_result._metadata else 0}")

print("\n" + "="*80)

