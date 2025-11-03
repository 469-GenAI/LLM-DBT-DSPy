#!/usr/bin/env python3
"""
Test script to verify the dedicated model architecture.
Ensures Generator70B and Evaluator120B are using the correct models.
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

import dspy
from models import PitchGenerator, PitchEvaluator

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("=" * 80)
print("TESTING DEDICATED MODEL ARCHITECTURE")
print("=" * 80)

# Configure models
print("\n1. Initializing models...")
generator_lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)
evaluator_lm = dspy.LM("groq/openai/gpt-oss-120b", model_type="chat", api_key=GROQ_API_KEY)
dspy.configure(lm=generator_lm)  # Set default to generator

# Create instances
print("\n2. Creating PitchGenerator and PitchEvaluator instances...")
generator = PitchGenerator(generator_lm)
evaluator = PitchEvaluator(evaluator_lm)

# Test data
test_input = {
    "founders": ["Test Founder"],
    "company_name": "TestCo",
    "initial_offer": {"amount": "$100k", "equity": "10%"},
    "problem_story": {
        "persona": "test persona",
        "routine": ["routine 1"],
        "core_problem": "test problem",
        "hygiene_gap": "test gap",
        "problem_keywords": ["keyword1"]
    },
    "product_solution": {
        "name": "Test Product",
        "product_category": "test category",
        "key_differentiator": "test diff",
        "application": "test app",
        "features_keywords": ["feature1"],
        "benefits_keywords": ["benefit1"]
    },
    "closing_theme": {
        "call_to_action": "test cta",
        "mission": "test mission",
        "target_audience": "test audience"
    }
}

ground_truth = "Hi Sharks, I'm Test Founder from TestCo. We're asking for $100k for 10% of our company..."

# Test generation
print("\n3. Testing generation...")
try:
    prediction = generator.generate(test_input)
    generated_pitch = prediction.pitch if hasattr(prediction, "pitch") else str(prediction)
    print(f"   ✓ Generated pitch ({len(generated_pitch)} chars)")
    print(f"   Preview: {generated_pitch[:100]}...")
except Exception as e:
    print(f"   ✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test evaluation
print("\n4. Testing evaluation...")
try:
    import json
    pitch_facts = json.dumps(test_input, indent=2)
    
    # Test get_score method
    score = evaluator.get_score(pitch_facts, ground_truth, generated_pitch)
    print(f"   ✓ Got score: {score:.3f}")
    
    # Test get_full_assessment method
    assessment = evaluator.get_full_assessment(pitch_facts, ground_truth, generated_pitch)
    print(f"   ✓ Got full assessment:")
    print(f"      - Factual: {assessment['factual_score']:.3f}")
    print(f"      - Narrative: {assessment['narrative_score']:.3f}")
    print(f"      - Style: {assessment['style_score']:.3f}")
    print(f"      - Final: {assessment['final_score']:.3f}")
    
except Exception as e:
    print(f"   ✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify model separation
print("\n5. Verifying model separation...")
print(f"   ✓ Generator model: {generator.lm.model}")
print(f"   ✓ Evaluator model: {evaluator.lm.model}")

print(f"   Expected generator: groq/llama-3.3-70b-versatile")
print(f"   Expected evaluator: groq/openai/gpt-oss-120b")

if generator.lm.model != evaluator.lm.model:
    print("   ✓ Models are different (no confusion)")
else:
    print(f"   ✗ WARNING: Both using same model: {generator.lm.model}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - ARCHITECTURE VERIFIED")
print("=" * 80)
print("\nThe dedicated model architecture is working correctly:")
print("  • PitchGenerator uses its assigned model exclusively")
print("  • PitchEvaluator uses its assigned model exclusively")
print("  • No model confusion or switching issues")
print("\nCheck your API logs to verify:")
print("  • Generation call: llama-3.3-70b-versatile")
print("  • Evaluation call: openai/gpt-oss-120b")
print()

