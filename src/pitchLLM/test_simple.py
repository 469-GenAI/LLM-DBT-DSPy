# test_simple.py - Simple test script to verify the implementation
"""
Quick test script to verify the structured pitch generation system works.
Tests basic functionality without full optimization.
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import dspy
from utils import PitchInput, format_pitch_input
from data_loader import load_and_prepare_data
from pitchLLM_structured import StructuredPitchProgram

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure DSPy
lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)
dspy.configure(lm=lm)


def test_data_loading():
    """Test that data can be loaded from HuggingFace."""
    print("=" * 80)
    print("TEST 1: Data Loading")
    print("=" * 80)
    
    try:
        data = load_and_prepare_data()
        print(f"✓ Successfully loaded data")
        print(f"  - Train examples: {len(data['train'])}")
        print(f"  - Test examples: {len(data['test'])}")
        
        # Check first example structure
        first_example = data['train'][0]
        print(f"\n✓ First example structure:")
        print(f"  - Has 'input': {hasattr(first_example, 'input')}")
        print(f"  - Has 'output': {hasattr(first_example, 'output')}")
        print(f"  - Input keys set: {first_example.input_keys}")
        
        return data
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def test_utils():
    """Test that Pydantic models work correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Pydantic Models")
    print("=" * 80)
    
    try:
        # Create a sample input
        sample_input = {
            "founders": ["Test Founder"],
            "company_name": "Test Company",
            "initial_offer": {
                "amount": "$100k",
                "equity": "10%"
            },
            "problem_story": {
                "persona": "Test persona",
                "routine": ["routine 1", "routine 2"],
                "core_problem": "Test problem",
                "hygiene_gap": "Test gap",
                "problem_keywords": ["keyword1", "keyword2"]
            },
            "product_solution": {
                "name": "Test Product",
                "product_category": "Test Category",
                "key_differentiator": "Test differentiator",
                "application": "Test application",
                "features_keywords": ["feature1", "feature2"],
                "benefits_keywords": ["benefit1", "benefit2"]
            },
            "closing_theme": {
                "call_to_action": "Test CTA",
                "mission": "Test mission",
                "target_audience": "Test audience"
            }
        }
        
        pitch_input = PitchInput(**sample_input)
        print(f"✓ Successfully created PitchInput model")
        print(f"  - Company: {pitch_input.company_name}")
        print(f"  - Founders: {len(pitch_input.founders)}")
        
        formatted = format_pitch_input(pitch_input)
        print(f"\n✓ Successfully formatted input ({len(formatted)} chars)")
        
        return True
    except Exception as e:
        print(f"✗ Error with utils: {e}")
        return False


def test_pitch_generation(data):
    """Test that pitch generation works."""
    print("\n" + "=" * 80)
    print("TEST 3: Pitch Generation")
    print("=" * 80)
    
    if data is None:
        print("✗ Skipping (data not loaded)")
        return False
    
    try:
        # Create program
        program = StructuredPitchProgram()
        print("✓ Successfully created StructuredPitchProgram")
        
        # Generate pitch from first example
        example = data['test'][0]
        print(f"\nGenerating pitch for: {example.input.get('company_name', 'Unknown')}")
        
        prediction = program(input=example.input)
        generated_pitch = prediction.pitch if hasattr(prediction, 'pitch') else str(prediction)
        
        print(f"\n✓ Successfully generated pitch")
        print(f"  - Length: {len(generated_pitch)} characters")
        print(f"  - Preview: {generated_pitch[:200]}...")
        
        print(f"\n✓ Ground truth length: {len(example.output)} characters")
        print(f"  - Preview: {example.output[:200]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error generating pitch: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("STRUCTURED PITCH GENERATION - SIMPLE TESTS")
    print("=" * 80 + "\n")
    
    # Test 1: Data loading
    data = test_data_loading()
    
    # Test 2: Utils
    test_utils()
    
    # Test 3: Pitch generation
    test_pitch_generation(data)
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

