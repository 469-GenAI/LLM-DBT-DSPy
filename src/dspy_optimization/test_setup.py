"""
Test script to verify DSPy optimization setup
Run this to ensure all components are working before running full optimization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import dspy
        print("  ✓ dspy")
    except ImportError as e:
        print(f"  ✗ dspy: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError as e:
        print(f"  ✗ python-dotenv: {e}")
        return False
    
    try:
        import datasets
        print("  ✓ datasets")
    except ImportError as e:
        print(f"  ✗ datasets: {e}")
        return False
    
    try:
        from dspy_optimization.dataset_prep import load_dataset, SharkTankDataset
        print("  ✓ dataset_prep")
    except ImportError as e:
        print(f"  ✗ dataset_prep: {e}")
        return False
    
    try:
        from dspy_optimization.metrics import get_all_metrics
        print("  ✓ metrics")
    except ImportError as e:
        print(f"  ✗ metrics: {e}")
        return False
    
    try:
        from agents.pitchLLM import PitchProgram
        print("  ✓ pitchLLM")
    except ImportError as e:
        print(f"  ✗ pitchLLM: {e}")
        return False
    
    return True


def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    
    try:
        from dspy_optimization.dataset_prep import SharkTankDataset
        
        dataset = SharkTankDataset()
        print(f"  ✓ Loaded {len(dataset.examples)} examples")
        
        # Test split
        train, val, test = dataset.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        print(f"  ✓ Split: {len(train)} train, {len(val)} val, {len(test)} test")
        
        # Test example structure
        if len(train) > 0:
            example = train[0]
            print(f"  ✓ Example has inputs: {example._input_keys}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metric functions"""
    print("\nTesting metrics...")
    
    try:
        from dspy_optimization.metrics import get_all_metrics
        
        metrics = get_all_metrics()
        print(f"  ✓ Available metrics: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_program_creation():
    """Test creating a PitchProgram"""
    print("\nTesting program creation...")
    
    try:
        from agents.pitchLLM import PitchProgram
        
        program = PitchProgram()
        print("  ✓ PitchProgram created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("DSPy Optimization Setup Test")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Loading", test_dataset_loading),
        ("Metrics", test_metrics),
        ("Program Creation", test_program_creation)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run optimization.")
        print("\nQuick start command:")
        print("  python src/dspy_optimization/optimize_pitch.py --train-size 20 --test-size 5 --mipro-mode light")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running optimization.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

