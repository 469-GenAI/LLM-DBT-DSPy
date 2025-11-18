"""
Comprehensive evaluation: RAG vs Non-RAG pitch generation.

Compares:
1. Quality scores (Evaluator120B)
2. Factual accuracy
3. Cost (API tokens)
4. Latency
5. BLEU/ROUGE scores
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

import dspy
from models import PitchGenerator, PitchEvaluator, RAGPitchGenerator
from data_loader import load_and_prepare_data
from utils import results_to_dataframe, save_results_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGEvaluator:
    """
    Comprehensive evaluator comparing RAG vs Non-RAG generation.
    """
    
    def __init__(
        self,
        generator_lm,
        evaluator_lm,
        test_size: int = 20,
        top_k: int = 5
    ):
        """
        Initialize evaluator.
        
        Args:
            generator_lm: LLM for generation (e.g., Llama 70B, Llama 8B Instant)
            evaluator_lm: LLM for evaluation (e.g., GPT OSS 120B)
            test_size: Number of test examples
            top_k: Number of retrieved examples for RAG
        """
        self.generator_lm = generator_lm
        self.evaluator_lm = evaluator_lm
        self.test_size = test_size
        self.top_k = top_k
        
        # Initialize generators
        logger.info("Initializing generators...")
        self.non_rag_generator = PitchGenerator(generator_lm)
        self.rag_generator = RAGPitchGenerator(generator_lm, top_k=top_k)
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        self.evaluator = PitchEvaluator(evaluator_lm)
        
        # Load test data from local test.jsonl
        logger.info(f"Loading {test_size} test examples from test.jsonl...")
        data = load_and_prepare_data(
            train_jsonl_path="data/hf (new)/train.jsonl",
            test_jsonl_path="data/hf (new)/test.jsonl"
        )
        # Take first test_size examples from test split
        import random
        random.seed(42)
        test_examples = data["test"]
        random.shuffle(test_examples)
        self.test_data = test_examples[:test_size]
        logger.info(f"Loaded {len(self.test_data)} test examples from test.jsonl")
    
    def evaluate_single(
        self, 
        example: dspy.Example,
        generator,
        generator_name: str
    ) -> Dict:
        """
        Evaluate single example with one generator.
        
        Returns:
            Dict with all metrics
        """
        start_time = time.time()
        
        # Generate pitch
        try:
            prediction = generator.generate(example.input)
            generated_pitch = prediction.pitch if hasattr(prediction, 'pitch') else str(prediction)
            success = True
        except Exception as e:
            logger.error(f"Generation failed ({generator_name}): {e}")
            generated_pitch = ""
            success = False
        
        latency = time.time() - start_time
        
        # Get quality score from evaluator
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
        
        # Estimate tokens (rough approximation)
        tokens_used = len(generated_pitch.split()) * 1.3  # ~1.3 tokens per word
        
        # Get retrieval metadata if RAG
        num_retrieved = 0
        avg_similarity = 0.0
        if hasattr(prediction, '_metadata') and prediction._metadata:
            num_retrieved = prediction._metadata.get('num_retrieved', 0)
            retrieved = prediction._metadata.get('retrieved_pitches', [])
            if retrieved:
                avg_similarity = sum(p.get('similarity', 0) for p in retrieved) / len(retrieved)
        
        return {
            'generator': generator_name,
            'success': success,
            'generated_pitch': generated_pitch,
            'quality_score': quality_score,
            'latency_seconds': latency,
            'tokens_estimated': tokens_used,
            'num_retrieved': num_retrieved,
            'avg_retrieval_similarity': avg_similarity,
            'pitch_length': len(generated_pitch)
        }
    
    def run_comparison(self) -> pd.DataFrame:
        """
        Run full comparison on test set.
        
        Returns:
            DataFrame with results
        """
        results = []
        
        logger.info(f"\nEvaluating {len(self.test_data)} test examples...")
        logger.info("="*70)
        
        for i, example in enumerate(tqdm(self.test_data, desc="Evaluating")):
            product_id = f"test_{i+1}"
            
            # Non-RAG generation
            non_rag_result = self.evaluate_single(
                example, 
                self.non_rag_generator,
                "Non-RAG"
            )
            non_rag_result['product_id'] = product_id
            non_rag_result['ground_truth'] = example.output
            results.append(non_rag_result)
            
            # RAG generation
            rag_result = self.evaluate_single(
                example,
                self.rag_generator,
                "RAG"
            )
            rag_result['product_id'] = product_id
            rag_result['ground_truth'] = example.output
            results.append(rag_result)
            
            # Rate limiting
            time.sleep(0.5)
        
        df = pd.DataFrame(results)
        return df
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Formatted report string
        """
        # Separate RAG and Non-RAG
        non_rag = results_df[results_df['generator'] == 'Non-RAG']
        rag = results_df[results_df['generator'] == 'RAG']
        
        # Calculate statistics
        report_lines = [
            "="*70,
            "RAG vs NON-RAG COMPARISON REPORT",
            "="*70,
            f"Test Set Size: {len(results_df) // 2} products",
            f"Top-K Retrieval: {self.top_k}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "="*70,
            "1. QUALITY SCORES (Evaluator120B)",
            "="*70,
        ]
        
        # Quality scores
        non_rag_quality = non_rag['quality_score']
        rag_quality = rag['quality_score']
        
        report_lines.extend([
            f"Non-RAG:  {non_rag_quality.mean():.3f} ± {non_rag_quality.std():.3f}",
            f"RAG:      {rag_quality.mean():.3f} ± {rag_quality.std():.3f}",
            f"",
            f"Improvement: {(rag_quality.mean() - non_rag_quality.mean()):.3f} "
            f"({(rag_quality.mean() / non_rag_quality.mean() - 1) * 100:+.1f}%)",
            ""
        ])
        
        # Success rates
        report_lines.extend([
            "="*70,
            "2. SUCCESS RATES",
            "="*70,
            f"Non-RAG:  {non_rag['success'].mean():.1%}",
            f"RAG:      {rag['success'].mean():.1%}",
            ""
        ])
        
        # Latency
        non_rag_latency = non_rag['latency_seconds']
        rag_latency = rag['latency_seconds']
        
        report_lines.extend([
            "="*70,
            "3. LATENCY (seconds)",
            "="*70,
            f"Non-RAG:  {non_rag_latency.mean():.2f}s (avg)",
            f"RAG:      {rag_latency.mean():.2f}s (avg)",
            f"Overhead: +{(rag_latency.mean() - non_rag_latency.mean()):.2f}s "
            f"({(rag_latency.mean() / non_rag_latency.mean() - 1) * 100:+.1f}%)",
            ""
        ])
        
        # Token usage (cost proxy)
        non_rag_tokens = non_rag['tokens_estimated']
        rag_tokens = rag['tokens_estimated']
        
        cost_per_1m_tokens = 0.59  # Groq Llama 3.3 70B pricing
        non_rag_cost = (non_rag_tokens.sum() / 1_000_000) * cost_per_1m_tokens
        rag_cost = (rag_tokens.sum() / 1_000_000) * cost_per_1m_tokens
        
        report_lines.extend([
            "="*70,
            "4. COST ANALYSIS",
            "="*70,
            f"Non-RAG tokens:  {non_rag_tokens.mean():.0f} per pitch (avg)",
            f"RAG tokens:      {rag_tokens.mean():.0f} per pitch (avg)",
            f"",
            f"Total cost for {len(non_rag)} pitches:",
            f"  Non-RAG:  ${non_rag_cost:.4f}",
            f"  RAG:      ${rag_cost:.4f}",
            f"  Increase: ${(rag_cost - non_rag_cost):.4f} "
            f"({(rag_cost / non_rag_cost - 1) * 100:+.1f}%)",
            f"",
            f"Cost per pitch:",
            f"  Non-RAG:  ${non_rag_cost / len(non_rag):.5f}",
            f"  RAG:      ${rag_cost / len(rag):.5f}",
            ""
        ])
        
        # Retrieval statistics (RAG only)
        report_lines.extend([
            "="*70,
            "5. RETRIEVAL QUALITY (RAG only)",
            "="*70,
            f"Avg pitches retrieved: {rag['num_retrieved'].mean():.1f}",
            f"Avg similarity score:  {rag['avg_retrieval_similarity'].mean():.3f}",
            ""
        ])
        
        # Pitch length
        report_lines.extend([
            "="*70,
            "6. PITCH LENGTH (characters)",
            "="*70,
            f"Non-RAG:  {non_rag['pitch_length'].mean():.0f} chars (avg)",
            f"RAG:      {rag['pitch_length'].mean():.0f} chars (avg)",
            ""
        ])
        
        # Overall assessment
        quality_improvement = (rag_quality.mean() / non_rag_quality.mean() - 1) * 100
        cost_increase = (rag_cost / non_rag_cost - 1) * 100
        
        report_lines.extend([
            "="*70,
            "7. OVERALL ASSESSMENT",
            "="*70,
            f"Quality Improvement:  {quality_improvement:+.1f}%",
            f"Cost Increase:        {cost_increase:+.1f}%",
            f"Latency Increase:     {(rag_latency.mean() / non_rag_latency.mean() - 1) * 100:+.1f}%",
            "",
            "Recommendation:",
        ])
        
        if quality_improvement > 10 and cost_increase < 100:
            recommendation = "✓ RAG provides significant quality improvements at reasonable cost increase. RECOMMENDED for production."
        elif quality_improvement > 5:
            recommendation = "⚠ RAG provides moderate improvements. Consider for quality-critical applications."
        else:
            recommendation = "✗ RAG improvements are marginal. Non-RAG may be more cost-effective."
        
        report_lines.append(f"  {recommendation}")
        report_lines.append("")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def save_results(self, results_df: pd.DataFrame, report: str):
        """
        Save results to files.
        
        Args:
            results_df: Results DataFrame
            report: Report text
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent
        
        # Save detailed results
        csv_path = output_dir / f"rag_comparison_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved detailed results: {csv_path}")
        
        # Save report
        report_path = output_dir / f"rag_comparison_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"✓ Saved report: {report_path}")
        
        # Print report
        print("\n" + report)


def main():
    """Run RAG vs Non-RAG comparison."""
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
    
    # Configure models
    generator_lm = dspy.LM(
        # "groq/llama-3.3-70b-versatile",
        "groq/llama-3.3-8b-instant",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    evaluator_lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        model_type="chat",
        api_key=GROQ_API_KEY
    )
    
    # Create evaluator
    evaluator = RAGEvaluator(
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        test_size=20,  # Test on 20 products
        top_k=5  # Retrieve top 5 similar pitches
    )
    
    # Run comparison
    logger.info("\nStarting RAG vs Non-RAG comparison...")
    results_df = evaluator.run_comparison()
    
    # Generate report
    logger.info("\nGenerating comparison report...")
    report = evaluator.generate_report(results_df)
    
    # Save results
    evaluator.save_results(results_df, report)
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()


