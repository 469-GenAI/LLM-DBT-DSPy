#!/usr/bin/env python3
"""
Quick verification script to ensure dual-model setup is working correctly.
"""
import os
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("=" * 80)
print("VERIFYING DUAL-MODEL SETUP")
print("=" * 80)

# Test 1: Check API key
print("\n1. Checking API Key...")
if GROQ_API_KEY:
    print(f"   ✓ GROQ_API_KEY found (length: {len(GROQ_API_KEY)})")
else:
    print("   ✗ GROQ_API_KEY not found in .env")
    exit(1)

# Test 2: Initialize models
print("\n2. Initializing models...")
try:
    generator_lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)
    print("   ✓ Generator model initialized (Llama 3.3 70B)")
except Exception as e:
    print(f"   ✗ Failed to initialize generator: {e}")
    exit(1)

try:
    evaluator_lm = dspy.LM("groq/openai/gpt-oss-120b", model_type="chat", api_key=GROQ_API_KEY)
    print("   ✓ Evaluator model initialized (GPT-OSS 120B)")
except Exception as e:
    print(f"   ✗ Failed to initialize evaluator: {e}")
    exit(1)

# Test 3: Test generator
print("\n3. Testing generator (70B)...")
dspy.configure(lm=generator_lm)
try:
    response = generator_lm("Say 'Generator working' in exactly those words.")
    print(f"   ✓ Generator response received")
    print(f"   Response preview: {str(response)[:100]}...")
except Exception as e:
    print(f"   ✗ Generator test failed: {e}")
    exit(1)

# Test 4: Test evaluator
print("\n4. Testing evaluator (120B)...")
try:
    # Temporarily configure evaluator
    dspy.configure(lm=evaluator_lm)
    response = evaluator_lm("Say 'Evaluator working' in exactly those words.")
    print(f"   ✓ Evaluator response received")
    print(f"   Response preview: {str(response)[:100]}...")
    # Restore generator as default
    dspy.configure(lm=generator_lm)
except Exception as e:
    print(f"   ✗ Evaluator test failed: {e}")
    exit(1)

# Test 5: Import local modules
print("\n5. Testing local module imports...")
try:
    from utils import PitchInput
    print("   ✓ utils.py imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import utils: {e}")
    exit(1)

try:
    from data_loader import load_and_prepare_data
    print("   ✓ data_loader.py imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import data_loader: {e}")
    exit(1)

try:
    from pitchLLM_structured import StructuredPitchProgram, PitchEvaluator
    print("   ✓ pitchLLM_structured.py imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import pitchLLM_structured: {e}")
    exit(1)

print("\n" + "=" * 80)
print("✓ ALL CHECKS PASSED - SYSTEM READY")
print("=" * 80)
print("\nYou can now run:")
print("  python pitchLLM_structured.py --optimization none --test-size 3 --evaluate")
print()

