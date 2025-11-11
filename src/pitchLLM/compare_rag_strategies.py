"""
Comprehensive comparison of RAG retrieval strategies.

Tests three approaches:
1. Pure Semantic: Pure similarity-based retrieval
2. Hybrid Filter: Semantic + metadata filtering
3. Hybrid Prioritize: Semantic + rule-based prioritization

Generates detailed performance metrics and comparison reports.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import dspy

# Setup paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitchLLM.models.rag_generator import RAGPitchGenerator
from pitchLLM.models.generator import PitchGenerator
from pitchLLM.models.evaluator import PitchEvaluator
from pitchLLM.data_loader import load_and_prepare_data
from pitchLLM.rag.retriever import RetrievalStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGStrategyComparator:
    """
    Compare different RAG retrieval strategies.
    """
    
    def __init__(
        self,
        generator_lm,
        evaluator_lm,
        test_size: int = 20,
        top_k: int = 5
    ):
        """
        Initialize comparator.
        
        Args:
            generator_lm: LM for pitch generation
            evaluator_lm: LM for evaluation
            test_size: Number of test examples to evaluate
            top_k: Number of retrieved pitches
        """
        self.generator_lm = generator_lm
        self.evaluator_lm = evaluator_lm
        self.test_size = test_size
        self.top_k = top_k
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        self.evaluator = PitchEvaluator(evaluator_lm)
        
        # Load test data
        logger.info(f"Loading {test_size} test examples from test.jsonl...")
        data = load_and_prepare_data(
            train_jsonl_path="data/hf (new)/train.jsonl",
            test_jsonl_path="data/hf (new)/test.jsonl"
        )
        import random
        random.seed(42)
        test_examples = data["test"]
        random.shuffle(test_examples)
        self.test_data = test_examples[:test_size]
        logger.info(f"Loaded {len(self.test_data)} test examples")
        
        # Initialize generators for each strategy
        logger.info("Initializing RAG generators...")
        self.generators = {
            "Pure_Semantic": RAGPitchGenerator(
                generator_lm,
                top_k=top_k,
                retrieval_strategy=RetrievalStrategy.PURE_SEMANTIC
            ),
            "Hybrid_Filter_Deals": RAGPitchGenerator(
                generator_lm,
                top_k=top_k,
                retrieval_strategy=RetrievalStrategy.HYBRID_FILTER,
                filter_successful_deals=True
            ),
            "Hybrid_Prioritize": RAGPitchGenerator(
                generator_lm,
                top_k=top_k,
                retrieval_strategy=RetrievalStrategy.HYBRID_PRIORITIZE,
                prioritize_successful=True,
                prioritize_category=True
            ),
            "Non_RAG": PitchGenerator(generator_lm)  # Baseline
        }
        
        # Note: Hybrid_Filter_Category will be created per-product since category varies
        
        logger.info("✓ Comparator initialized")
    
    def evaluate_strategy(
        self,
        generator,
        generator_name: str,
        example: dspy.Example
    ) -> Dict:
        """
        Evaluate a single example with a generator strategy.
        
        Returns:
            Dict with metrics
        """
        start_time = time.time()
        prediction = None  # Initialize to avoid UnboundLocalError
        
        # Generate pitch
        try:
            prediction = generator.generate(example.input)
            generated_pitch = prediction.pitch if hasattr(prediction, 'pitch') else str(prediction)
            success = True
        except Exception as e:
            logger.error(f"Generation failed ({generator_name}): {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            generated_pitch = ""
            success = False
        
        latency = time.time() - start_time
        
        # Get quality score
        quality_score = 0.0
        if success and generated_pitch:
            try:
                quality_score = self.evaluator.get_score(
                    pitch_facts=json.dumps(example.input),
                    ground_truth_pitch=example.output,
                    generated_pitch=generated_pitch
                )
            except Exception as e:
                logger.error(f"Evaluation failed ({generator_name}): {e}")
        
        # Estimate tokens
        tokens_used = len(generated_pitch.split()) * 1.3 if generated_pitch else 0
        
        # Get retrieval metadata if RAG (only if prediction exists and succeeded)
        retrieval_metrics = {}
        if success and prediction and hasattr(prediction, '_metadata') and prediction._metadata:
            retrieval_metrics = {
                'num_retrieved': prediction._metadata.get('num_retrieved', 0),
                'retrieved_pitches': prediction._metadata.get('retrieved_pitches', [])
            }
            
            # Calculate average similarity
            retrieved = retrieval_metrics.get('retrieved_pitches', [])
            if retrieved:
                # Handle both dict format and list format
                similarities = []
                for p in retrieved:
                    if isinstance(p, dict):
                        sim = p.get('similarity', p.get('similarity_score', 0))
                        similarities.append(sim)
                    else:
                        similarities.append(0)
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                retrieval_metrics['avg_similarity'] = avg_similarity
        
        return {
            'generator': generator_name,
            'success': success,
            'generated_pitch': generated_pitch,
            'quality_score': quality_score,
            'latency_seconds': latency,
            'tokens_estimated': tokens_used,
            'pitch_length': len(generated_pitch),
            **retrieval_metrics
        }
    
    def run_comparison(self) -> pd.DataFrame:
        """
        Run full comparison across all strategies.
        
        Returns:
            DataFrame with results
        """
        results = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Comparing {len(self.generators)} strategies on {len(self.test_data)} examples")
        logger.info(f"{'='*80}\n")
        
        for i, example in enumerate(tqdm(self.test_data, desc="Evaluating")):
            product_id = f"test_{i+1}"
            
            # Extract product category for category filtering
            product_category = None
            if isinstance(example.input, dict):
                if 'product_description' in example.input:
                    desc = example.input['product_description']
                    if isinstance(desc, dict):
                        product_category = desc.get('category')
                elif 'category' in example.input:
                    product_category = example.input['category']
            
            # Evaluate each strategy
            for strategy_name, generator in self.generators.items():
                # Skip category filter strategy if no category available
                if strategy_name == "Hybrid_Filter_Category" and not product_category:
                    continue
                
                # Create category filter generator on-the-fly if needed
                if strategy_name == "Hybrid_Filter_Category" and product_category:
                    generator = RAGPitchGenerator(
                        self.generator_lm,
                        top_k=self.top_k,
                        retrieval_strategy=RetrievalStrategy.HYBRID_FILTER,
                        filter_category=product_category
                    )
                
                result = self.evaluate_strategy(generator, strategy_name, example)
                result['product_id'] = product_id
                result['ground_truth'] = example.output
                result['product_category'] = product_category
                results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def generate_report(self, df: pd.DataFrame, output_dir: str = "rag_comparison_results") -> Dict:
        """
        Generate comprehensive comparison report.
        
        Args:
            df: Results DataFrame
            output_dir: Directory to save reports
            
        Returns:
            Summary statistics dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate summary statistics
        summary = {}
        for strategy in df['generator'].unique():
            strategy_df = df[df['generator'] == strategy]
            
            summary[strategy] = {
                'success_rate': strategy_df['success'].mean(),
                'avg_quality_score': strategy_df['quality_score'].mean(),
                'avg_latency': strategy_df['latency_seconds'].mean(),
                'avg_tokens': strategy_df['tokens_estimated'].mean(),
                'avg_pitch_length': strategy_df['pitch_length'].mean(),
                'num_examples': len(strategy_df)
            }
            
            # Add retrieval metrics if RAG
            if 'num_retrieved' in strategy_df.columns:
                summary[strategy]['avg_retrieved'] = strategy_df['num_retrieved'].mean()
                summary[strategy]['avg_similarity'] = strategy_df['avg_similarity'].mean()
        
        # Save detailed results
        results_file = output_path / f"detailed_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Saved detailed results to: {results_file}")
        
        # Save summary
        summary_file = output_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")
        
        # Generate comparison report
        report_file = output_path / f"comparison_report_{timestamp}.md"
        self._generate_markdown_report(df, summary, report_file)
        logger.info(f"Saved comparison report to: {report_file}")
        
        return summary
    
    def _generate_markdown_report(
        self,
        df: pd.DataFrame,
        summary: Dict,
        output_file: Path
    ):
        """Generate markdown comparison report."""
        
        with open(output_file, 'w') as f:
            f.write("# RAG Retrieval Strategy Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Test Examples: {len(df['product_id'].unique())}\n")
            f.write(f"Strategies Compared: {len(df['generator'].unique())}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write("| Strategy | Success Rate | Avg Quality | Avg Latency (s) | Avg Tokens | Avg Retrieved | Avg Similarity |\n")
            f.write("|----------|--------------|-------------|------------------|------------|---------------|----------------|\n")
            
            for strategy, stats in summary.items():
                success_rate = stats['success_rate'] * 100
                quality = stats['avg_quality_score']
                latency = stats['avg_latency']
                tokens = stats['avg_tokens']
                retrieved = stats.get('avg_retrieved', 'N/A')
                similarity = stats.get('avg_similarity', 'N/A')
                
                # Handle N/A values for formatting
                retrieved_str = f"{retrieved:.0f}" if isinstance(retrieved, (int, float)) else str(retrieved)
                similarity_str = f"{similarity:.3f}" if isinstance(similarity, (int, float)) else str(similarity)
                
                f.write(f"| {strategy} | {success_rate:.1f}% | {quality:.3f} | {latency:.2f} | {tokens:.0f} | {retrieved_str} | {similarity_str} |\n")
            
            f.write("\n## Strategy Descriptions\n\n")
            f.write("### 1. Pure_Semantic\n")
            f.write("- Pure similarity-based retrieval\n")
            f.write("- No filters or prioritization\n")
            f.write("- Baseline approach\n\n")
            
            f.write("### 2. Hybrid_Filter_Deals\n")
            f.write("- Semantic search + metadata filtering\n")
            f.write("- Only retrieves successful deals\n")
            f.write("- Filters at query time\n\n")
            
            f.write("### 3. Hybrid_Prioritize\n")
            f.write("- Semantic search + rule-based prioritization\n")
            f.write("- Prioritizes: Successful deals + Same category > Successful deals > Same category > Others\n")
            f.write("- Maintains semantic similarity within priority levels\n\n")
            
            f.write("### 4. Non_RAG\n")
            f.write("- Baseline without RAG\n")
            f.write("- No retrieval, pure generation\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Find best strategy by quality
            best_quality = max(summary.items(), key=lambda x: x[1]['avg_quality_score'])
            f.write(f"- **Best Quality Score**: {best_quality[0]} ({best_quality[1]['avg_quality_score']:.3f})\n")
            
            # Find fastest strategy
            fastest = min(summary.items(), key=lambda x: x[1]['avg_latency'])
            f.write(f"- **Fastest**: {fastest[0]} ({fastest[1]['avg_latency']:.2f}s)\n")
            
            # Compare RAG vs Non-RAG
            if 'Non_RAG' in summary:
                non_rag_quality = summary['Non_RAG']['avg_quality_score']
                rag_strategies = {k: v for k, v in summary.items() if k != 'Non_RAG'}
                best_rag = max(rag_strategies.items(), key=lambda x: x[1]['avg_quality_score'])
                if non_rag_quality > 0:
                    improvement = ((best_rag[1]['avg_quality_score'] - non_rag_quality) / non_rag_quality) * 100
                    f.write(f"- **RAG Improvement**: {best_rag[0]} improves over Non-RAG by {improvement:.1f}%\n")
                else:
                    f.write(f"- **RAG Improvement**: Cannot calculate (Non-RAG quality score is 0)\n")
        
        logger.info(f"Generated markdown report: {output_file}")


def main():
    """Main comparison function."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Setup LMs
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    generator_lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    evaluator_lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",  # Updated to 3.3 (3.1 was decommissioned)
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    # Run comparison
    comparator = RAGStrategyComparator(
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        test_size=10,  # Start with 10 for testing
        top_k=5
    )
    
    logger.info("Starting comparison...")
    results_df = comparator.run_comparison()
    
    logger.info("Generating report...")
    summary = comparator.generate_report(results_df)
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*80)
    logger.info("\nSummary:")
    for strategy, stats in summary.items():
        logger.info(f"\n{strategy}:")
        logger.info(f"  Quality Score: {stats['avg_quality_score']:.3f}")
        logger.info(f"  Success Rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"  Avg Latency: {stats['avg_latency']:.2f}s")


if __name__ == "__main__":
    main()

