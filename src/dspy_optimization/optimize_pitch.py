"""
DSPy Prompt Optimization for Pitch Generation
Implements MIPROv2, BootstrapFewShot, and BootstrapRS optimizers
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import dspy

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agno_agents.pitchLLM import PitchProgram, FinancialsSig, PitchSig, Financials, DraftPitch
from agno_agents.utils import ValuationTools
from dspy_optimization.dataset_prep import load_dataset, SharkTankDataset
from dspy_optimization.metrics import get_composite_metric, get_structure_metric, get_all_metrics


class OptimizationConfig:
    """Configuration for DSPy optimization"""
    
    def __init__(
        self,
        optimizer_type: str = "mipro",
        model: str = "gpt-4o-mini",
        train_size: int = 50,
        val_size: int = 15,
        test_size: int = 20,
        metric_name: str = "composite",
        num_threads: int = 8,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 3,
        mipro_mode: str = "light",
        output_dir: str = "./optimized_models",
        seed: int = 42
    ):
        self.optimizer_type = optimizer_type
        self.model = model
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.metric_name = metric_name
        self.num_threads = num_threads
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.mipro_mode = mipro_mode
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def __str__(self):
        return f"""OptimizationConfig(
    optimizer: {self.optimizer_type}
    model: {self.model}
    train_size: {self.train_size}
    val_size: {self.val_size}
    test_size: {self.test_size}
    metric: {self.metric_name}
    threads: {self.num_threads}
    mipro_mode: {self.mipro_mode}
    output_dir: {self.output_dir}
)"""


def setup_dspy(api_key: str = None, model: str = "gpt-4o-mini"):
    """Configure DSPy with LLM"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Use OpenAI format for DSPy LM
    if "/" not in model:
        model = f"openai/{model}"
    
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)
    
    print(f"✓ DSPy configured with model: {model}")
    return lm


def prepare_datasets(config: OptimizationConfig):
    """Load and prepare train/val/test datasets"""
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    train_set, val_set, test_set = load_dataset(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed
    )
    
    print(f"✓ Train set: {len(train_set)} examples")
    print(f"✓ Val set: {len(val_set)} examples")
    print(f"✓ Test set: {len(test_set)} examples")
    
    return train_set, val_set, test_set


def get_metric_function(metric_name: str):
    """Get the metric function by name"""
    metrics = get_all_metrics()
    
    if metric_name == "structure":
        return get_structure_metric()
    elif metric_name == "composite":
        return get_composite_metric()
    elif metric_name in metrics:
        return metrics[metric_name]
    else:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metrics.keys())}")


def optimize_with_mipro(
    program: PitchProgram,
    train_set: list,
    val_set: list,
    config: OptimizationConfig,
    metric_fn: callable
):
    """Optimize program using MIPROv2"""
    print("\n" + "="*70)
    print(f"OPTIMIZING WITH MIPROv2 ({config.mipro_mode.upper()} MODE)")
    print("="*70)
    
    print(f"Training on {len(train_set)} examples...")
    print(f"Using {config.num_threads} threads")
    print(f"Metric: {config.metric_name}")
    
    try:
        # MIPROv2 optimizer with appropriate settings
        optimizer = dspy.MIPROv2(
            metric=metric_fn,
            auto=config.mipro_mode,  # "light", "medium", or "heavy"
            num_threads=config.num_threads,
            verbose=True
        )
        
        # Compile/optimize the program
        optimized_program = optimizer.compile(
            program,
            trainset=train_set,
            valset=val_set if val_set else None,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            max_labeled_demos=config.max_labeled_demos
        )
        
        print("\n✓ MIPROv2 optimization completed!")
        return optimized_program
        
    except Exception as e:
        print(f"\n✗ Error during MIPROv2 optimization: {e}")
        raise


def optimize_with_bootstrap_fewshot(
    program: PitchProgram,
    train_set: list,
    config: OptimizationConfig,
    metric_fn: callable
):
    """Optimize program using BootstrapFewShot"""
    print("\n" + "="*70)
    print("OPTIMIZING WITH BootstrapFewShot")
    print("="*70)
    
    print(f"Training on {len(train_set)} examples...")
    print(f"Using {config.num_threads} threads")
    print(f"Metric: {config.metric_name}")
    
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=metric_fn,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            max_labeled_demos=config.max_labeled_demos,
            num_threads=config.num_threads
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=train_set
        )
        
        print("\n✓ BootstrapFewShot optimization completed!")
        return optimized_program
        
    except Exception as e:
        print(f"\n✗ Error during BootstrapFewShot optimization: {e}")
        raise


def optimize_with_bootstrap_rs(
    program: PitchProgram,
    train_set: list,
    config: OptimizationConfig,
    metric_fn: callable
):
    """Optimize program using BootstrapRS (Random Search)"""
    print("\n" + "="*70)
    print("OPTIMIZING WITH BootstrapRS")
    print("="*70)
    
    print(f"Training on {len(train_set)} examples...")
    print(f"Using {config.num_threads} threads")
    print(f"Metric: {config.metric_name}")
    
    try:
        # BootstrapRS does random search over prompt variations
        optimizer = dspy.BootstrapRS(
            metric=metric_fn,
            max_bootstrapped_demos=config.max_bootstrapped_demos,
            num_threads=config.num_threads
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=train_set
        )
        
        print("\n✓ BootstrapRS optimization completed!")
        return optimized_program
        
    except Exception as e:
        print(f"\n✗ Error during BootstrapRS optimization: {e}")
        raise


def evaluate_program(
    program: PitchProgram,
    test_set: list,
    metric_fn: callable,
    program_name: str = "Program"
):
    """Evaluate program on test set"""
    print("\n" + "="*70)
    print(f"EVALUATING {program_name.upper()}")
    print("="*70)
    
    scores = []
    successes = 0
    
    for i, example in enumerate(test_set):
        try:
            # Run prediction
            pred = program(example.product_data)
            
            # Calculate metric
            score = metric_fn(example, pred)
            scores.append(score)
            
            if isinstance(score, bool):
                if score:
                    successes += 1
            else:
                if score >= 0.5:  # Threshold for "success"
                    successes += 1
            
            print(f"Example {i+1}/{len(test_set)}: Score = {score:.3f}" if isinstance(score, float) else f"Example {i+1}/{len(test_set)}: {'✓' if score else '✗'}")
            
        except Exception as e:
            print(f"Example {i+1}/{len(test_set)}: Error - {e}")
            scores.append(0.0)
    
    # Calculate final metrics
    if scores:
        avg_score = sum(scores) / len(scores)
        success_rate = successes / len(scores)
    else:
        avg_score = 0.0
        success_rate = 0.0
    
    print(f"\n{program_name} Results:")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Success Rate: {success_rate:.1%}")
    
    return {
        'avg_score': avg_score,
        'success_rate': success_rate,
        'scores': scores
    }


def save_optimized_program(program: PitchProgram, config: OptimizationConfig, results: dict):
    """Save optimized program and results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{config.optimizer_type}_{config.model.replace('/', '_')}_{timestamp}"
    
    # Save program
    program_path = config.output_dir / f"{filename}.json"
    program.save(str(program_path))
    print(f"\n✓ Saved optimized program to: {program_path}")
    
    # Save results
    results_path = config.output_dir / f"{filename}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_path}")
    
    return program_path, results_path


def main():
    parser = argparse.ArgumentParser(
        description="Optimize DSPy pitch generation with MIPROv2, BootstrapFewShot, or BootstrapRS"
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        default='mipro',
        choices=['mipro', 'bootstrap_fewshot', 'bootstrap_rs'],
        help='Optimizer to use (default: mipro)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='Model to use (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '--train-size',
        type=int,
        default=50,
        help='Number of training examples (default: 50)'
    )
    
    parser.add_argument(
        '--val-size',
        type=int,
        default=15,
        help='Number of validation examples (default: 15)'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        default=20,
        help='Number of test examples (default: 20)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='composite',
        choices=['structure', 'completeness', 'pitch_quality', 'financial_consistency', 'composite'],
        help='Metric to optimize for (default: composite)'
    )
    
    parser.add_argument(
        '--mipro-mode',
        type=str,
        default='light',
        choices=['light', 'medium', 'heavy'],
        help='MIPROv2 optimization mode (default: light). Light is faster and cheaper.'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads for parallel processing (default: 8)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./optimized_models',
        help='Directory to save optimized models (default: ./optimized_models)'
    )
    
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline evaluation'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = OptimizationConfig(
        optimizer_type=args.optimizer,
        model=args.model,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        metric_name=args.metric,
        num_threads=args.threads,
        mipro_mode=args.mipro_mode,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("DSPy PITCH OPTIMIZATION")
    print("="*70)
    print(config)
    
    # Setup DSPy
    lm = setup_dspy(model=config.model)
    
    # Prepare datasets
    train_set, val_set, test_set = prepare_datasets(config)
    
    # Get metric function
    metric_fn = get_metric_function(config.metric_name)
    print(f"✓ Using metric: {config.metric_name}")
    
    # Create baseline program
    print("\n" + "="*70)
    print("CREATING BASELINE PROGRAM")
    print("="*70)
    baseline_program = PitchProgram()
    print("✓ Baseline program created")
    
    # Evaluate baseline (optional)
    baseline_results = None
    if not args.skip_baseline and test_set:
        baseline_results = evaluate_program(
            baseline_program,
            test_set[:min(5, len(test_set))],  # Quick baseline on 5 examples
            metric_fn,
            program_name="Baseline"
        )
    
    # Optimize program
    if config.optimizer_type == 'mipro':
        optimized_program = optimize_with_mipro(
            baseline_program,
            train_set,
            val_set,
            config,
            metric_fn
        )
    elif config.optimizer_type == 'bootstrap_fewshot':
        optimized_program = optimize_with_bootstrap_fewshot(
            baseline_program,
            train_set,
            config,
            metric_fn
        )
    elif config.optimizer_type == 'bootstrap_rs':
        optimized_program = optimize_with_bootstrap_rs(
            baseline_program,
            train_set,
            config,
            metric_fn
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    
    # Evaluate optimized program
    optimized_results = evaluate_program(
        optimized_program,
        test_set,
        metric_fn,
        program_name="Optimized"
    )
    
    # Compare results
    if baseline_results:
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(f"Baseline:  {baseline_results['avg_score']:.3f} avg score, {baseline_results['success_rate']:.1%} success")
        print(f"Optimized: {optimized_results['avg_score']:.3f} avg score, {optimized_results['success_rate']:.1%} success")
        
        improvement = optimized_results['avg_score'] - baseline_results['avg_score']
        print(f"\nImprovement: {improvement:+.3f} ({improvement/baseline_results['avg_score']*100:+.1f}%)")
    
    # Save results
    results = {
        'config': {
            'optimizer': config.optimizer_type,
            'model': config.model,
            'train_size': len(train_set),
            'val_size': len(val_set),
            'test_size': len(test_set),
            'metric': config.metric_name
        },
        'baseline': baseline_results,
        'optimized': optimized_results,
        'timestamp': datetime.now().isoformat()
    }
    
    program_path, results_path = save_optimized_program(optimized_program, config, results)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Optimized program saved to: {program_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()




