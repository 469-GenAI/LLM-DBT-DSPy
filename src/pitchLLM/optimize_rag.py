"""
Optimize RAG program with DSPy MIPROv2.

This script:
1. Loads train/val/test sets
2. Creates RAGStructuredPitchProgram
3. Optimizes it with MIPROv2 (optimizes HOW to use retrieved examples)
4. Evaluates and compares: Optimized RAG vs Non-Optimized RAG vs Non-RAG
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import dspy
import logging
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_generator import RAGStructuredPitchProgram
from models.generator import StructuredPitchProgram
from models import PitchEvaluator
from rag.retriever import PitchRetriever, RetrievalStrategy
from data_loader import load_and_prepare_data
from eval.AssessPitch import AssessPitchQuality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")


def pitch_quality_metric(example, pred, trace=None):
    """
    Metric function for DSPy optimization.
    Evaluates pitch quality using LLM-as-a-Judge.
    """
    try:
        # Extract generated pitch
        generated_pitch = pred.pitch if hasattr(pred, "pitch") else str(pred)
        
        # Extract ground truth
        ground_truth = example.output
        
        # Format input facts
        pitch_facts = json.dumps(example.input, indent=2)
        
        # Use evaluator (will be set globally)
        evaluator = get_global_evaluator()
        if evaluator is None:
            logger.warning("Evaluator not set, using simple metric")
            return 1.0 if generated_pitch else 0.0
        
        final_score = evaluator.get_score(
            pitch_facts=pitch_facts,
            ground_truth_pitch=ground_truth,
            generated_pitch=generated_pitch
        )
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error in pitch_quality_metric: {e}")
        return 0.0


# Global evaluator instance
_global_evaluator = None

def set_global_evaluator(evaluator):
    """Set global evaluator for metric function."""
    global _global_evaluator
    _global_evaluator = evaluator

def get_global_evaluator():
    """Get global evaluator."""
    return _global_evaluator


def evaluate_program(program, test_set, evaluator, program_name="Program"):
    """
    Evaluate a program on test set.
    
    Returns:
        dict with metrics
    """
    logger.info(f"\nEvaluating {program_name}...")
    
    results = []
    total_score = 0.0
    
    for example in tqdm(test_set, desc=f"Evaluating {program_name}"):
        try:
            # Generate pitch
            prediction = program(input=example.input)
            generated_pitch = prediction.pitch if hasattr(prediction, "pitch") else str(prediction)
            
            # Evaluate quality
            score = evaluator.get_score(
                pitch_facts=json.dumps(example.input, indent=2),
                ground_truth_pitch=example.output,
                generated_pitch=generated_pitch
            )
            
            results.append({
                'input': example.input,
                'generated': generated_pitch,
                'ground_truth': example.output,
                'score': score
            })
            
            total_score += score
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            results.append({
                'input': example.input,
                'generated': '',
                'ground_truth': example.output,
                'score': 0.0
            })
    
    avg_score = total_score / len(test_set) if test_set else 0.0
    success_rate = sum(1 for r in results if r['score'] > 0) / len(results) if results else 0.0
    
    return {
        'avg_score': avg_score,
        'success_rate': success_rate,
        'results': results,
        'num_examples': len(test_set)
    }


def main():
    """Main optimization script."""
    
    print("="*80)
    print("RAG + DSPy Optimization")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load train/val/test sets")
    print("  2. Create RAG program")
    print("  3. Optimize RAG program with MIPROv2")
    print("  4. Compare: Optimized RAG vs Non-Optimized RAG vs Non-RAG")
    print("="*80)
    
    # Configuration - Use ALL available data
    # Train: 196 examples, Test: 49 examples
    # Split train into train (180) + val (16) for optimization
    train_size = None  # None = use all available
    val_size = 16  # ~10% of train for validation
    test_size = None  # None = use all available
    mipro_mode = "light"  # light/medium/heavy
    
    print(f"\nConfiguration:")
    print(f"  Train size: ALL available (~196)")
    print(f"  Val size: {val_size} (split from train)")
    print(f"  Test size: ALL available (~49)")
    print(f"  MIPRO mode: {mipro_mode}")
    
    # Setup models
    print("\n" + "="*80)
    print("SETTING UP MODELS")
    print("="*80)
    
    generator_lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    evaluator_lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    dspy.configure(lm=generator_lm)
    
    evaluator = PitchEvaluator(evaluator_lm)
    set_global_evaluator(evaluator)
    
    print("✓ Models configured")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    data = load_and_prepare_data(
        train_jsonl_path="data/hf (new)/train.jsonl",
        test_jsonl_path="data/hf (new)/test.jsonl"
    )
    
    # Use all available data
    train_set = data["train"] if train_size is None else data["train"][:train_size]
    test_set = data["test"] if test_size is None else data["test"][:test_size]
    
    # Create validation set from train (take last val_size examples)
    val_set = train_set[-val_size:] if len(train_set) > val_size else train_set[:val_size]
    train_set = train_set[:-val_size] if len(train_set) > val_size else train_set
    
    print(f"✓ Loaded {len(train_set)} train examples")
    print(f"✓ Loaded {len(val_set)} validation examples")
    print(f"✓ Loaded {len(test_set)} test examples")
    
    # Initialize retriever
    print("\n" + "="*80)
    print("INITIALIZING RAG RETRIEVER")
    print("="*80)
    
    retriever = PitchRetriever(auto_index=False)  # Already indexed
    stats = retriever.get_retriever_stats()
    print(f"✓ Vector store: {stats['total_documents']} pitches indexed")
    
    # Create RAG program
    print("\n" + "="*80)
    print("CREATING RAG PROGRAM")
    print("="*80)
    
    rag_program = RAGStructuredPitchProgram(
        retriever=retriever,
        top_k=5,
        retrieval_strategy=RetrievalStrategy.HYBRID_PRIORITIZE,
        prioritize_successful=True,
        prioritize_category=True
    )
    
    print("✓ RAG program created")
    
    # Evaluate baseline RAG (non-optimized)
    print("\n" + "="*80)
    print("EVALUATING BASELINE RAG (Non-Optimized)")
    print("="*80)
    
    baseline_rag_results = evaluate_program(
        rag_program,
        test_set,  # Use all test examples
        evaluator,
        "Baseline RAG"
    )
    
    print(f"\nBaseline RAG Results:")
    print(f"  Average Score: {baseline_rag_results['avg_score']:.3f}")
    print(f"  Success Rate: {baseline_rag_results['success_rate']:.1%}")
    
    # Optimize RAG program with MIPROv2
    print("\n" + "="*80)
    print("OPTIMIZING RAG PROGRAM WITH MIPROv2")
    print("="*80)
    print(f"\nThis will optimize HOW the RAG program uses retrieved examples.")
    print(f"MIPRO mode: {mipro_mode}")
    print(f"This may take 15-30 minutes and cost ~$2-5...")
    
    try:
        optimizer = dspy.MIPROv2(
            metric=pitch_quality_metric,
            auto=mipro_mode,
            num_threads=4,
            verbose=True
        )
        
        print("\nStarting optimization...")
        optimized_rag_program = optimizer.compile(
            rag_program,
            trainset=train_set,
            valset=val_set,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        
        print("\n✓ MIPROv2 optimization completed!")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Optimization failed. Using baseline RAG for comparison.")
        optimized_rag_program = rag_program
    
    # Evaluate optimized RAG
    print("\n" + "="*80)
    print("EVALUATING OPTIMIZED RAG")
    print("="*80)
    
    optimized_rag_results = evaluate_program(
        optimized_rag_program,
        test_set,  # Use all test examples
        evaluator,
        "Optimized RAG"
    )
    
    print(f"\nOptimized RAG Results:")
    print(f"  Average Score: {optimized_rag_results['avg_score']:.3f}")
    print(f"  Success Rate: {optimized_rag_results['success_rate']:.1%}")
    
    # Compare with Non-RAG baseline
    print("\n" + "="*80)
    print("EVALUATING NON-RAG BASELINE")
    print("="*80)
    
    non_rag_program = StructuredPitchProgram()
    non_rag_results = evaluate_program(
        non_rag_program,
        test_set,  # Use all test examples
        evaluator,
        "Non-RAG"
    )
    
    print(f"\nNon-RAG Results:")
    print(f"  Average Score: {non_rag_results['avg_score']:.3f}")
    print(f"  Success Rate: {non_rag_results['success_rate']:.1%}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Quality Score':<15} {'Success Rate':<15}")
    print("-" * 55)
    print(f"{'Non-RAG':<25} {non_rag_results['avg_score']:<15.3f} {non_rag_results['success_rate']:<15.1%}")
    print(f"{'Baseline RAG':<25} {baseline_rag_results['avg_score']:<15.3f} {baseline_rag_results['success_rate']:<15.1%}")
    print(f"{'Optimized RAG':<25} {optimized_rag_results['avg_score']:<15.3f} {optimized_rag_results['success_rate']:<15.1%}")
    
    # Calculate improvements
    if baseline_rag_results['avg_score'] > 0:
        rag_improvement = ((optimized_rag_results['avg_score'] - baseline_rag_results['avg_score']) / baseline_rag_results['avg_score']) * 100
        print(f"\nRAG Improvement: {rag_improvement:+.1f}% (Optimized vs Baseline)")
    
    if non_rag_results['avg_score'] > 0:
        vs_non_rag = ((optimized_rag_results['avg_score'] - non_rag_results['avg_score']) / non_rag_results['avg_score']) * 100
        print(f"vs Non-RAG: {vs_non_rag:+.1f}% improvement")
    
    # Save results
    output_dir = Path("rag_optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"optimization_results_{timestamp}.json"
    
    results_summary = {
        'timestamp': timestamp,
        'config': {
            'train_size': len(train_set),
            'val_size': len(val_set),
            'test_size': len(test_set),
            'mipro_mode': mipro_mode
        },
        'results': {
            'non_rag': {
                'avg_score': non_rag_results['avg_score'],
                'success_rate': non_rag_results['success_rate']
            },
            'baseline_rag': {
                'avg_score': baseline_rag_results['avg_score'],
                'success_rate': baseline_rag_results['success_rate']
            },
            'optimized_rag': {
                'avg_score': optimized_rag_results['avg_score'],
                'success_rate': optimized_rag_results['success_rate']
            }
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

