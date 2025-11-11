# AI/ML Improvements - Detailed Implementation Guide

**Target Audience**: AI/ML Engineers, Data Scientists, Researchers  
**Skill Level**: Intermediate to Advanced  
**Time Investment**: 1 week to 3 months depending on chosen improvements

---

## 🎯 Quick Reference Table

| # | Improvement | Effort | Cost | Impact | Priority | ROI |
|---|-------------|--------|------|--------|----------|-----|
| 1 | Automated Model Comparison | 4h | $10 | High | ⭐⭐⭐ | Excellent |
| 2 | Ensemble System | 6h | $20 | Very High | ⭐⭐⭐ | Excellent |
| 3 | Advanced DSPy Optimization | 8h | $30 | Very High | ⭐⭐⭐ | Excellent |
| 4 | Few-Shot Learning System | 8h | $15 | High | ⭐⭐ | Good |
| 5 | Semantic Caching | 4h | $0 | Medium | ⭐⭐ | Excellent |
| 6 | Fine-Tune Specialist Models | 2wk | $200 | Very High | ⭐⭐⭐ | Good |
| 7 | RAFT Implementation | 1wk | $50 | High | ⭐⭐ | Good |
| 8 | RLHF Pipeline | 3wk | $500 | Very High | ⭐⭐⭐ | Medium |
| 9 | Multi-Task Learning | 3wk | $100 | Very High | ⭐⭐ | Medium |
| 10 | Constitutional AI | 1wk | $30 | Medium | ⭐⭐ | Good |

---

## 1. Automated Model Comparison Framework

### Goal
Systematically compare different models and approaches to find the optimal cost-quality tradeoff.

### Implementation

```python
# File: src/evaluation/model_comparison.py

import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm
import time

@dataclass
class ModelConfig:
    name: str
    provider: str  # 'openai', 'groq', 'anthropic'
    model_id: str
    cost_per_1k_tokens: float
    max_tokens: int = 4096

@dataclass
class ComparisonResult:
    model_name: str
    avg_quality_score: float
    avg_cost_per_pitch: float
    avg_latency_seconds: float
    success_rate: float
    pitches_generated: int
    
    def cost_quality_ratio(self):
        """Lower is better"""
        return self.avg_cost_per_pitch / self.avg_quality_score
    
    def quality_per_dollar(self):
        """Higher is better"""
        return self.avg_quality_score / self.avg_cost_per_pitch


class ModelComparison:
    """
    Compare multiple models/approaches on the same dataset.
    """
    
    def __init__(self, test_products: List[Dict], output_dir: str = "./comparison_results"):
        self.test_products = test_products
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define models to compare
        self.models = [
            ModelConfig("gpt-4o-mini", "openai", "gpt-4o-mini", 0.00015),
            ModelConfig("gpt-4o", "openai", "gpt-4o", 0.0050),
            ModelConfig("llama-3.3-70b", "groq", "llama-3.3-70b-versatile", 0.00079),
            ModelConfig("llama-3.1-8b", "groq", "llama-3.1-8b-instant", 0.00018),
            ModelConfig("claude-3.5-sonnet", "anthropic", "claude-3-5-sonnet-20241022", 0.0030),
        ]
    
    def run_comparison(self, pitch_generator, evaluator, num_samples: int = 20):
        """
        Run comparison across all models.
        
        Args:
            pitch_generator: Function that takes (model_config, product_data) -> pitch
            evaluator: Function that takes pitch -> quality_score
            num_samples: Number of products to test per model
        """
        results = []
        
        for model_config in self.models:
            print(f"\n{'='*70}")
            print(f"Testing: {model_config.name}")
            print(f"{'='*70}")
            
            model_results = self._test_model(
                model_config, 
                pitch_generator, 
                evaluator, 
                num_samples
            )
            results.append(model_results)
        
        # Generate comparison report
        self._generate_report(results)
        return results
    
    def _test_model(self, model_config, pitch_generator, evaluator, num_samples):
        """Test a single model on sample products."""
        quality_scores = []
        costs = []
        latencies = []
        successes = 0
        
        for product in tqdm(self.test_products[:num_samples], 
                           desc=f"Testing {model_config.name}"):
            try:
                start_time = time.time()
                
                # Generate pitch
                pitch, tokens_used = pitch_generator(model_config, product)
                
                latency = time.time() - start_time
                
                # Evaluate quality
                quality_score = evaluator(pitch, product)
                
                # Calculate cost
                cost = (tokens_used / 1000) * model_config.cost_per_1k_tokens
                
                quality_scores.append(quality_score)
                costs.append(cost)
                latencies.append(latency)
                successes += 1
                
            except Exception as e:
                print(f"Error with {model_config.name}: {e}")
                continue
        
        return ComparisonResult(
            model_name=model_config.name,
            avg_quality_score=sum(quality_scores) / len(quality_scores),
            avg_cost_per_pitch=sum(costs) / len(costs),
            avg_latency_seconds=sum(latencies) / len(latencies),
            success_rate=successes / num_samples,
            pitches_generated=successes
        )
    
    def _generate_report(self, results: List[ComparisonResult]):
        """Generate comparison report with visualizations."""
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'Model': r.model_name,
                'Quality Score': f"{r.avg_quality_score:.3f}",
                'Cost per Pitch': f"${r.avg_cost_per_pitch:.4f}",
                'Latency (s)': f"{r.avg_latency_seconds:.1f}",
                'Success Rate': f"{r.success_rate:.1%}",
                'Quality per $': f"{r.quality_per_dollar():.1f}",
                'Cost/Quality Ratio': f"{r.cost_quality_ratio():.4f}",
            }
            for r in results
        ])
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"model_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Print summary
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print(f"\n✓ Full results saved to: {csv_path}")
        
        # Find best models
        best_quality = max(results, key=lambda x: x.avg_quality_score)
        best_cost = min(results, key=lambda x: x.avg_cost_per_pitch)
        best_value = max(results, key=lambda x: x.quality_per_dollar())
        
        print("\n" + "="*70)
        print("WINNERS")
        print("="*70)
        print(f"🏆 Best Quality: {best_quality.model_name} ({best_quality.avg_quality_score:.3f})")
        print(f"💰 Cheapest: {best_cost.model_name} (${best_cost.avg_cost_per_pitch:.4f})")
        print(f"⭐ Best Value: {best_value.model_name} ({best_value.quality_per_dollar():.1f} quality/$)")


# Usage example
if __name__ == "__main__":
    from dataset_prep import load_dataset
    from metrics import get_composite_metric
    import dspy
    
    # Load test data
    _, _, test_set = load_dataset(test_size=20)
    
    # Define pitch generator
    def generate_pitch(model_config, product_data):
        """Generate pitch using DSPy."""
        lm = dspy.LM(f"{model_config.provider}/{model_config.model_id}")
        dspy.configure(lm=lm)
        
        # ... your pitch generation logic
        # Return: (pitch_text, tokens_used)
        pass
    
    # Define evaluator
    def evaluate_pitch(pitch, product_data):
        """Evaluate pitch quality."""
        metric = get_composite_metric()
        return metric(None, pitch, None)
    
    # Run comparison
    comparison = ModelComparison(test_set)
    results = comparison.run_comparison(generate_pitch, evaluate_pitch)
```

### Expected Output

```
======================================================================
MODEL COMPARISON RESULTS
======================================================================
Model                Quality Score  Cost per Pitch  Latency (s)  Success Rate  Quality per $  Cost/Quality Ratio
gpt-4o-mini          0.782          $0.0012         2.3          100.0%        651.7          0.0015
gpt-4o               0.891          $0.0089         3.1          100.0%        100.1          0.0100
llama-3.3-70b        0.801          $0.0021         1.8          100.0%        381.4          0.0026
llama-3.1-8b         0.712          $0.0004         1.2          95.0%         1780.0         0.0006
claude-3.5-sonnet    0.878          $0.0067         2.7          100.0%        131.0          0.0076

======================================================================
WINNERS
======================================================================
🏆 Best Quality: gpt-4o (0.891)
💰 Cheapest: llama-3.1-8b ($0.0004)
⭐ Best Value: llama-3.1-8b (1780.0 quality/$)
```

---

## 2. Ensemble System (Mixture of Experts)

### Goal
Combine outputs from multiple models/approaches to achieve better results than any single model.

### Implementation

```python
# File: src/ensemble/ensemble_pitcher.py

from typing import List, Dict, Tuple
from dataclasses import dataclass
import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

@dataclass
class EnsembleMember:
    name: str
    generator: callable
    weight: float = 1.0
    confidence_estimator: callable = None


class EnsemblePitcher:
    """
    Ensemble multiple pitch generation approaches.
    """
    
    def __init__(self, members: List[EnsembleMember], aggregation_method: str = "llm_judge"):
        self.members = members
        self.aggregation_method = aggregation_method
        
        # Configure LLM judge for aggregation
        if aggregation_method == "llm_judge":
            self.judge = dspy.LM("openai/gpt-4o")
    
    def generate_pitch(self, product_data: Dict) -> Dict:
        """
        Generate pitch using ensemble of methods.
        
        Returns:
            {
                'final_pitch': str,
                'confidence': float,
                'member_pitches': List[Dict],
                'aggregation_reasoning': str
            }
        """
        
        # Step 1: Generate pitches from all members in parallel
        member_pitches = self._generate_parallel(product_data)
        
        # Step 2: Aggregate pitches
        if self.aggregation_method == "llm_judge":
            final_pitch, reasoning = self._aggregate_with_llm_judge(
                product_data, member_pitches
            )
        elif self.aggregation_method == "weighted_vote":
            final_pitch, reasoning = self._aggregate_with_weighted_vote(
                product_data, member_pitches
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Step 3: Calculate confidence
        confidence = self._calculate_ensemble_confidence(member_pitches)
        
        return {
            'final_pitch': final_pitch,
            'confidence': confidence,
            'member_pitches': member_pitches,
            'aggregation_reasoning': reasoning
        }
    
    def _generate_parallel(self, product_data: Dict) -> List[Dict]:
        """Generate pitches from all members in parallel."""
        pitches = []
        
        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            future_to_member = {
                executor.submit(member.generator, product_data): member
                for member in self.members
            }
            
            for future in as_completed(future_to_member):
                member = future_to_member[future]
                try:
                    pitch = future.result()
                    
                    # Estimate confidence if estimator provided
                    confidence = 0.8  # default
                    if member.confidence_estimator:
                        confidence = member.confidence_estimator(pitch, product_data)
                    
                    pitches.append({
                        'member_name': member.name,
                        'pitch': pitch,
                        'weight': member.weight,
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"Error generating pitch with {member.name}: {e}")
        
        return pitches
    
    def _aggregate_with_llm_judge(self, product_data: Dict, 
                                   member_pitches: List[Dict]) -> Tuple[str, str]:
        """
        Use LLM as judge to select/synthesize best pitch.
        """
        
        # Prepare prompt
        pitches_text = "\n\n".join([
            f"### PITCH {i+1} (from {p['member_name']}, confidence={p['confidence']:.2f}):\n{p['pitch']}"
            for i, p in enumerate(member_pitches)
        ])
        
        product_summary = json.dumps(product_data, indent=2)
        
        prompt = f"""You are an expert Shark Tank pitch evaluator. You have been given {len(member_pitches)} different pitch versions for the same product, generated by different AI systems.

PRODUCT DATA:
{product_summary}

CANDIDATE PITCHES:
{pitches_text}

Your task:
1. Evaluate each pitch for quality, persuasiveness, structure, and accuracy
2. Either select the best pitch OR synthesize a new pitch that combines the best elements
3. Provide brief reasoning for your choice

Output format:
REASONING: [your reasoning]

FINAL_PITCH:
[the selected or synthesized pitch]
"""
        
        dspy.configure(lm=self.judge)
        response = self.judge(prompt)
        
        # Parse response
        parts = response.split("FINAL_PITCH:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        final_pitch = parts[1].strip() if len(parts) > 1 else member_pitches[0]['pitch']
        
        return final_pitch, reasoning
    
    def _aggregate_with_weighted_vote(self, product_data: Dict,
                                       member_pitches: List[Dict]) -> Tuple[str, str]:
        """
        Select pitch with highest weighted confidence score.
        """
        best_pitch = max(
            member_pitches,
            key=lambda x: x['weight'] * x['confidence']
        )
        
        reasoning = f"Selected pitch from {best_pitch['member_name']} " \
                   f"(weighted score: {best_pitch['weight'] * best_pitch['confidence']:.3f})"
        
        return best_pitch['pitch'], reasoning
    
    def _calculate_ensemble_confidence(self, member_pitches: List[Dict]) -> float:
        """
        Calculate overall ensemble confidence.
        Higher when members agree, lower when they diverge.
        """
        if len(member_pitches) < 2:
            return member_pitches[0]['confidence']
        
        # Simple average of confidences (can be made more sophisticated)
        avg_confidence = sum(p['confidence'] for p in member_pitches) / len(member_pitches)
        
        # Penalty for disagreement (measured by variance)
        confidences = [p['confidence'] for p in member_pitches]
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        agreement_penalty = 1.0 - min(variance, 0.3)  # Cap penalty at 30%
        
        return avg_confidence * agreement_penalty


# Usage Example
def create_ensemble():
    """Create an ensemble of different approaches."""
    
    # Member 1: DSPy Optimized
    def dspy_generator(product_data):
        from agents.pitchLLM import PitchProgram
        program = PitchProgram()
        # Load optimized version if available
        result = program(product_data)
        return result['response'].Pitch
    
    # Member 2: Multi-Agent RAG
    def multiagent_generator(product_data):
        # Your multi-agent RAG system
        # ...
        pass
    
    # Member 3: MoA (Mixture of Agents)
    def moa_generator(product_data):
        # Your MoA system
        # ...
        pass
    
    members = [
        EnsembleMember("DSPy Optimized", dspy_generator, weight=1.2),
        EnsembleMember("Multi-Agent RAG", multiagent_generator, weight=1.0),
        EnsembleMember("MoA", moa_generator, weight=0.9),
    ]
    
    return EnsemblePitcher(members, aggregation_method="llm_judge")


if __name__ == "__main__":
    # Load test product
    import json
    with open("data/all_processed_facts.json") as f:
        products = json.load(f)
    
    product_key = list(products.keys())[0]
    product_data = products[product_key]
    
    # Create and run ensemble
    ensemble = create_ensemble()
    result = ensemble.generate_pitch(product_data)
    
    print("="*70)
    print("ENSEMBLE RESULT")
    print("="*70)
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nAggregation Reasoning:\n{result['aggregation_reasoning']}")
    print(f"\nFinal Pitch:\n{result['final_pitch']}")
```

### Expected Improvement
- **Quality**: +15-25% over best single model
- **Robustness**: More consistent across different product types
- **Confidence**: Better calibration of uncertainty

---

## 3. Advanced DSPy Optimization with Custom Metrics

### Goal
Go beyond basic optimization by creating domain-specific metrics and using advanced DSPy features.

### Implementation

```python
# File: src/dspy_optimization/advanced_metrics.py

import dspy
from typing import Dict, List
import re
from dataclasses import dataclass

@dataclass
class PitchAnalysis:
    """Detailed pitch analysis results."""
    has_hook: bool
    has_problem_statement: bool
    has_solution: bool
    has_market_size: bool
    has_traction: bool
    has_ask: bool
    has_cta: bool
    
    hook_quality: float  # 0-1
    storytelling_score: float  # 0-1
    persuasiveness_score: float  # 0-1
    financial_accuracy: float  # 0-1
    
    word_count: int
    estimated_speaking_time: float  # minutes


class AdvancedPitchMetric:
    """
    Comprehensive pitch evaluation metric using LLM-as-judge
    with structured output.
    """
    
    def __init__(self):
        # Use GPT-4 for evaluation (more reliable)
        self.evaluator = dspy.LM("openai/gpt-4o")
    
    def __call__(self, example, pred, trace=None) -> float:
        """
        Evaluate pitch quality with detailed analysis.
        
        Returns score between 0 and 1.
        """
        
        product_data = example.product_data
        generated_pitch = pred.response.Pitch if hasattr(pred, 'response') else pred
        
        # Perform detailed analysis
        analysis = self._analyze_pitch(generated_pitch, product_data)
        
        # Calculate composite score
        structure_score = self._structure_score(analysis)
        content_score = self._content_score(analysis)
        delivery_score = self._delivery_score(analysis)
        accuracy_score = analysis.financial_accuracy
        
        # Weighted combination
        final_score = (
            0.25 * structure_score +
            0.35 * content_score +
            0.25 * delivery_score +
            0.15 * accuracy_score
        )
        
        # Store analysis for debugging
        if hasattr(pred, 'analysis'):
            pred.analysis = analysis
        
        return final_score
    
    def _analyze_pitch(self, pitch: str, product_data: Dict) -> PitchAnalysis:
        """Use LLM to perform detailed pitch analysis."""
        
        dspy.configure(lm=self.evaluator)
        
        analysis_prompt = f"""Analyze this Shark Tank pitch in detail.

PRODUCT DATA:
{product_data}

PITCH:
{pitch}

Evaluate the following (respond with JSON):
{{
    "has_hook": true/false,
    "has_problem_statement": true/false,
    "has_solution": true/false,
    "has_market_size": true/false,
    "has_traction": true/false,
    "has_ask": true/false,
    "has_cta": true/false,
    "hook_quality": 0.0-1.0,
    "storytelling_score": 0.0-1.0,
    "persuasiveness_score": 0.0-1.0,
    "financial_accuracy": 0.0-1.0
}}"""
        
        response = self.evaluator(analysis_prompt)
        
        # Parse JSON response
        import json
        try:
            analysis_dict = json.loads(response)
        except:
            # Fallback to basic analysis
            analysis_dict = self._basic_analysis(pitch)
        
        # Add word count and speaking time
        word_count = len(pitch.split())
        speaking_time = word_count / 150  # Average speaking pace
        
        return PitchAnalysis(
            **analysis_dict,
            word_count=word_count,
            estimated_speaking_time=speaking_time
        )
    
    def _basic_analysis(self, pitch: str) -> Dict:
        """Fallback: Rule-based analysis."""
        text_lower = pitch.lower()
        
        return {
            "has_hook": any(word in text_lower for word in ["imagine", "what if", "problem", "struggled"]),
            "has_problem_statement": "problem" in text_lower,
            "has_solution": any(word in text_lower for word in ["solution", "product", "solves"]),
            "has_market_size": any(word in text_lower for word in ["market", "billion", "million customers"]),
            "has_traction": any(word in text_lower for word in ["revenue", "sales", "customers", "sold"]),
            "has_ask": any(word in text_lower for word in ["asking", "seeking", "need", "investment"]),
            "has_cta": any(word in text_lower for word in ["join", "partner", "invest", "let's"]),
            "hook_quality": 0.7,
            "storytelling_score": 0.6,
            "persuasiveness_score": 0.6,
            "financial_accuracy": 0.8
        }
    
    def _structure_score(self, analysis: PitchAnalysis) -> float:
        """Score based on pitch structure."""
        required_elements = [
            analysis.has_hook,
            analysis.has_problem_statement,
            analysis.has_solution,
            analysis.has_ask,
            analysis.has_cta
        ]
        return sum(required_elements) / len(required_elements)
    
    def _content_score(self, analysis: PitchAnalysis) -> float:
        """Score based on content quality."""
        return (
            analysis.hook_quality +
            analysis.storytelling_score +
            analysis.persuasiveness_score
        ) / 3
    
    def _delivery_score(self, analysis: PitchAnalysis) -> float:
        """Score based on delivery aspects (length, pacing)."""
        # Ideal pitch: 2-3 minutes (300-450 words)
        ideal_min, ideal_max = 300, 450
        word_count = analysis.word_count
        
        if ideal_min <= word_count <= ideal_max:
            length_score = 1.0
        elif word_count < ideal_min:
            length_score = max(0.5, word_count / ideal_min)
        else:
            length_score = max(0.5, ideal_max / word_count)
        
        return length_score


# Enhanced optimization with curriculum learning
class CurriculumOptimizer:
    """
    Optimize DSPy program with curriculum learning.
    Start with easy examples, gradually increase difficulty.
    """
    
    def __init__(self, train_set, difficulty_fn):
        self.train_set = train_set
        self.difficulty_fn = difficulty_fn
        
        # Sort by difficulty
        self.sorted_train = sorted(
            train_set,
            key=lambda x: difficulty_fn(x)
        )
    
    def optimize(self, program, metric, num_stages=3):
        """
        Optimize in stages, gradually introducing harder examples.
        """
        n = len(self.sorted_train)
        stage_size = n // num_stages
        
        optimized_program = program
        
        for stage in range(num_stages):
            print(f"\n{'='*70}")
            print(f"CURRICULUM STAGE {stage + 1}/{num_stages}")
            print(f"{'='*70}")
            
            # Get examples for this stage
            start_idx = 0
            end_idx = stage_size * (stage + 1)
            stage_examples = self.sorted_train[start_idx:end_idx]
            
            print(f"Training on {len(stage_examples)} examples")
            print(f"Difficulty range: {self.difficulty_fn(stage_examples[0]):.2f} - {self.difficulty_fn(stage_examples[-1]):.2f}")
            
            # Optimize on this stage
            optimizer = dspy.MIPROv2(
                metric=metric,
                num_candidates=10,
                init_temperature=0.5
            )
            
            optimized_program = optimizer.compile(
                optimized_program,
                trainset=stage_examples,
                max_bootstrapped_demos=3,
                max_labeled_demos=2
            )
        
        return optimized_program


# Difficulty estimator for curriculum learning
def estimate_difficulty(example) -> float:
    """
    Estimate difficulty of a product (0=easy, 1=hard).
    
    Factors:
    - Missing information
    - Complex financials
    - Unusual product category
    """
    product_data = example.product_data
    
    difficulty = 0.0
    
    # Factor 1: Information completeness (more missing info = harder)
    expected_keys = ['company_name', 'founder', 'product_category', 
                     'sales_to_date', 'investment_ask', 'equity_offer']
    missing_count = sum(1 for key in expected_keys if key not in product_data)
    difficulty += (missing_count / len(expected_keys)) * 0.3
    
    # Factor 2: Financial complexity
    if 'financial_projections' in product_data:
        difficulty += 0.2
    if 'multiple_revenue_streams' in product_data:
        difficulty += 0.2
    
    # Factor 3: Unusual category
    common_categories = ['food', 'fashion', 'tech', 'health', 'fitness']
    if product_data.get('product_category', '').lower() not in common_categories:
        difficulty += 0.3
    
    return min(difficulty, 1.0)


# Usage
if __name__ == "__main__":
    from agents.pitchLLM import PitchProgram
    from dspy_optimization.dataset_prep import load_dataset
    
    # Load data
    train_set, val_set, test_set = load_dataset(train_size=60, val_size=15, test_size=20)
    
    # Create program
    program = PitchProgram()
    
    # Create advanced metric
    metric = AdvancedPitchMetric()
    
    # Option 1: Standard optimization with advanced metric
    optimizer = dspy.MIPROv2(metric=metric, num_candidates=15)
    optimized = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
        max_bootstrapped_demos=4
    )
    
    # Option 2: Curriculum learning
    curriculum = CurriculumOptimizer(train_set, estimate_difficulty)
    optimized = curriculum.optimize(program, metric, num_stages=3)
    
    # Save
    optimized.save("optimized_models/advanced_curriculum.json")
```

### Expected Improvement
- **Baseline → Advanced Metric**: +10-15%
- **Advanced Metric → Curriculum**: +5-10% additional
- **Total**: 40-70% improvement over baseline

---

## 4. Fine-Tuning Specialist Models

### Goal
Fine-tune smaller models for specific subtasks to reduce costs while maintaining quality.

### Why Fine-Tuning?

| Aspect | GPT-4o (No FT) | Fine-Tuned Llama 8B |
|--------|----------------|---------------------|
| Cost per 1M tokens | $5,000 | $180 |
| Quality | Excellent | Good-Very Good |
| Speed | Moderate | Fast |
| Control | Limited | Full |

**ROI**: After ~10,000 pitches, fine-tuned model pays for itself.

### Implementation

```python
# File: src/fine_tuning/financial_analyzer_ft.py

"""
Fine-tune a specialist model for financial analysis.
This is one component of the pitch generation pipeline.
"""

from datasets import Dataset
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

class FinancialAnalysisDatasetBuilder:
    """
    Build training dataset for fine-tuning financial analysis model.
    """
    
    def __init__(self, source_data_path: str):
        with open(source_data_path) as f:
            self.source_data = json.load(f)
    
    def build_dataset(self, output_path: str = "./ft_data/financial_analysis.jsonl"):
        """
        Create training examples in the format:
        
        Input: Product facts (JSON)
        Output: Financial analysis (structured)
        """
        examples = []
        
        for product_key, product_data in self.source_data.items():
            # Create training example
            example = self._create_example(product_data)
            if example:
                examples.append(example)
        
        # Save in JSONL format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"✓ Created {len(examples)} training examples")
        print(f"✓ Saved to: {output_path}")
        
        return examples
    
    def _create_example(self, product_data: Dict) -> Dict:
        """Create a single training example."""
        
        # Extract facts
        facts = product_data.get('facts', [])
        if not facts:
            return None
        
        # Create input (facts as JSON)
        input_text = f"""Analyze the financial situation of this business:

{json.dumps({'facts': facts}, indent=2)}

Provide a structured financial analysis."""
        
        # Create output (what we want the model to learn)
        # In production, you'd have ground truth labels
        # For now, we'll use heuristics or human-labeled data
        output_text = self._generate_ideal_analysis(product_data)
        
        return {
            "messages": [
                {"role": "system", "content": "You are a financial analyst specializing in startup valuations."},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
        }
    
    def _generate_ideal_analysis(self, product_data: Dict) -> str:
        """
        Generate ideal financial analysis.
        
        In production, this would come from:
        1. Human expert annotations
        2. High-quality GPT-4 outputs (distillation)
        3. Actual pitch outcomes
        """
        
        # For now, template-based (replace with real labels)
        return json.dumps({
            "revenue": self._extract_revenue(product_data),
            "profit": self._extract_profit(product_data),
            "is_profitable": self._is_profitable(product_data),
            "growth_rate": self._extract_growth_rate(product_data),
            "valuation_estimate": self._estimate_valuation(product_data),
            "confidence": "medium",
            "reasoning": "Based on stated sales figures and industry benchmarks..."
        }, indent=2)
    
    def _extract_revenue(self, product_data: Dict) -> float:
        """Extract revenue from facts."""
        facts = product_data.get('facts', [])
        for fact in facts:
            if 'sales' in fact.lower() or 'revenue' in fact.lower():
                # Extract number (simplified)
                import re
                numbers = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)', fact)
                if numbers:
                    return float(numbers[0].replace(',', ''))
        return 0.0
    
    # ... other extraction methods


# Fine-tuning with OpenAI API (example)
def fine_tune_openai_model(training_file_path: str):
    """
    Fine-tune a model using OpenAI's fine-tuning API.
    
    Steps:
    1. Upload training file
    2. Create fine-tuning job
    3. Monitor progress
    4. Use fine-tuned model
    """
    from openai import OpenAI
    import os
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Step 1: Upload training file
    print("Uploading training file...")
    with open(training_file_path, "rb") as f:
        training_file = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    
    print(f"✓ File uploaded: {training_file.id}")
    
    # Step 2: Create fine-tuning job
    print("Creating fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model="gpt-4o-mini-2024-07-18",  # Base model
        hyperparameters={
            "n_epochs": 3,
            "batch_size": 4,
            "learning_rate_multiplier": 1.0
        }
    )
    
    print(f"✓ Job created: {job.id}")
    print(f"Status: {job.status}")
    print(f"\nMonitor progress:")
    print(f"  client.fine_tuning.jobs.retrieve('{job.id}')")
    
    return job.id


# Alternative: Fine-tune Llama with Unsloth (much faster, cheaper)
def fine_tune_llama_unsloth(training_examples: List[Dict]):
    """
    Fine-tune Llama 3.1 8B using Unsloth (4x faster, uses 70% less memory).
    
    Cost: ~$5-10 for full fine-tune on Colab/Runpod
    """
    
    # This runs in a separate environment (Colab/Runpod)
    script = """
# Install Unsloth
!pip install unsloth

from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.1-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# Prepare dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

# Save
model.save_pretrained("financial_analyzer_lora")
"""
    
    print("Fine-tuning script for Llama (run in Colab):")
    print(script)
    
    return script


if __name__ == "__main__":
    # Build dataset
    builder = FinancialAnalysisDatasetBuilder("data/all_processed_facts.json")
    examples = builder.build_dataset()
    
    print("\n" + "="*70)
    print("FINE-TUNING OPTIONS")
    print("="*70)
    print("\n1. OpenAI Fine-Tuning (GPT-4o-mini)")
    print("   Cost: ~$50-100")
    print("   Time: 1-2 hours")
    print("   Quality: Excellent")
    
    print("\n2. Llama 3.1 8B with Unsloth")
    print("   Cost: ~$5-10")
    print("   Time: 30-60 minutes")
    print("   Quality: Very Good")
    print("   Note: Requires GPU (Colab/Runpod)")
    
    print("\n3. Use existing LLMs (no fine-tuning)")
    print("   Cost: $0 upfront, pay per use")
    print("   Time: Immediate")
    print("   Quality: Good")
```

### Expected Results

**Before Fine-Tuning** (GPT-4o-mini):
- Cost per 1M tokens: $150
- Accuracy: 85%
- Speed: 2.5s per request

**After Fine-Tuning** (Llama 3.1 8B):
- Cost per 1M tokens: $18 (90% cheaper)
- Accuracy: 82% (slightly lower but acceptable)
- Speed: 0.8s per request (3x faster)

**Break-even**: After generating ~1,000 financial analyses

---

## 5. Semantic Caching System

### Goal
Cache LLM responses for similar queries to reduce costs by 50-80%.

### Implementation

```python
# File: src/caching/semantic_cache.py

from typing import Optional, Dict, Any
import json
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta

class SemanticCache:
    """
    Cache LLM responses with semantic similarity matching.
    
    If a query is semantically similar to a cached query,
    return the cached response instead of calling the LLM.
    """
    
    def __init__(
        self, 
        cache_dir: str = "./cache",
        similarity_threshold: float = 0.95,
        ttl_days: int = 30,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.ttl_days = ttl_days
        
        # Load embedding model for semantic similarity
        self.embedder = SentenceTransformer(embedding_model)
        
        # Load or create cache index
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def get(self, query: str, context: Dict = None) -> Optional[Dict]:
        """
        Get cached response for semantically similar query.
        
        Returns None if no similar query found in cache.
        """
        # Create query key
        query_embedding = self.embedder.encode(query)
        
        # Search for similar queries
        best_match = None
        best_similarity = 0.0
        
        for cached_key, cached_entry in self.index.items():
            # Check if cache entry is still valid (TTL)
            if self._is_expired(cached_entry['timestamp']):
                continue
            
            # Check context match (exact match required for context)
            if context and cached_entry.get('context') != context:
                continue
            
            # Calculate semantic similarity
            cached_embedding = np.array(cached_entry['embedding'])
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cached_key
        
        if best_match:
            # Load cached response
            cache_file = self.cache_dir / f"{best_match}.json"
            with open(cache_file) as f:
                cached_data = json.load(f)
            
            print(f"✓ Cache hit! (similarity: {best_similarity:.3f})")
            return cached_data['response']
        
        print("✗ Cache miss")
        return None
    
    def set(self, query: str, response: Any, context: Dict = None):
        """
        Cache a query-response pair.
        """
        # Generate cache key
        cache_key = hashlib.md5(
            (query + str(context)).encode()
        ).hexdigest()
        
        # Create query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Store in index
        self.index[cache_key] = {
            'query': query,
            'embedding': query_embedding,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store response
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'query': query,
                'response': response,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Update index
        self._save_index()
    
    def clear_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        
        for key, entry in self.index.items():
            if self._is_expired(entry['timestamp']):
                expired_keys.append(key)
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    cache_file.unlink()
        
        for key in expired_keys:
            del self.index[key]
        
        self._save_index()
        print(f"✓ Cleared {len(expired_keys)} expired entries")
    
    def _is_expired(self, timestamp_str: str) -> bool:
        """Check if cache entry has expired."""
        timestamp = datetime.fromisoformat(timestamp_str)
        age = datetime.now() - timestamp
        return age > timedelta(days=self.ttl_days)
    
    def _load_index(self) -> Dict:
        """Load cache index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save cache index."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        valid_entries = sum(
            1 for entry in self.index.values()
            if not self._is_expired(entry['timestamp'])
        )
        
        return {
            'total_entries': len(self.index),
            'valid_entries': valid_entries,
            'expired_entries': len(self.index) - valid_entries,
            'cache_dir': str(self.cache_dir),
            'size_mb': sum(
                f.stat().st_size for f in self.cache_dir.glob("*.json")
            ) / (1024 * 1024)
        }


# Cached pitch generator
class CachedPitchGenerator:
    """
    Wrap pitch generator with caching.
    """
    
    def __init__(self, base_generator, cache: SemanticCache):
        self.base_generator = base_generator
        self.cache = cache
        self.hits = 0
        self.misses = 0
    
    def generate(self, product_data: Dict) -> Dict:
        """Generate pitch with caching."""
        
        # Create query from product data
        query = json.dumps(product_data, sort_keys=True)
        
        # Check cache
        cached_result = self.cache.get(query)
        
        if cached_result:
            self.hits += 1
            return cached_result
        
        # Cache miss - generate new pitch
        self.misses += 1
        result = self.base_generator(product_data)
        
        # Store in cache
        self.cache.set(query, result)
        
        return result
    
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Usage example
if __name__ == "__main__":
    # Create cache
    cache = SemanticCache(
        cache_dir="./semantic_cache",
        similarity_threshold=0.95,
        ttl_days=30
    )
    
    # Define base generator
    def base_generator(product_data):
        from agents.pitchLLM import PitchProgram
        program = PitchProgram()
        return program(product_data)
    
    # Wrap with caching
    cached_generator = CachedPitchGenerator(base_generator, cache)
    
    # Generate pitches
    import json
    with open("data/all_processed_facts.json") as f:
        products = json.load(f)
    
    for i, product_data in enumerate(list(products.values())[:10]):
        print(f"\nGenerating pitch {i+1}/10...")
        result = cached_generator.generate(product_data)
        print(f"Cache hit rate: {cached_generator.cache_hit_rate():.1%}")
    
    # Show stats
    stats = cache.stats()
    print("\n" + "="*70)
    print("CACHE STATISTICS")
    print("="*70)
    for key, value in stats.items():
        print(f"{key}: {value}")
```

### Expected Savings
- **First run**: 0% savings (cache empty)
- **Similar products**: 80-90% cost reduction
- **Varied products**: 30-50% cost reduction
- **Overall (mixed workload)**: 40-60% cost reduction

---

## Summary: Recommended Implementation Order

### Week 1: Foundation (16 hours)
1. ✅ Model Comparison Framework (4h) - Understand what works best
2. ✅ Semantic Caching (4h) - Immediate 40-60% cost savings
3. ✅ Advanced DSPy Metrics (8h) - Better optimization

### Week 2-3: Advanced Features (40 hours)
4. ✅ Ensemble System (6h) - +15-25% quality improvement
5. ✅ Advanced DSPy Optimization with Curriculum (16h) - +40-60% overall improvement
6. ✅ Few-Shot Learning System (8h) - Consistency boost
7. ✅ Automated Evaluation Pipeline (10h) - Continuous measurement

### Month 2: Specialization (80 hours)
8. ✅ Fine-Tune Specialist Models (40h) - 80-90% cost reduction long-term
9. ✅ RAFT Implementation (20h) - Better factual accuracy
10. ✅ Constitutional AI (20h) - Consistent quality

### Month 3+: Research (120+ hours)
11. ✅ RLHF Pipeline (80h) - Human preference alignment
12. ✅ Multi-Task Learning (40h) - Better generalization

---

## 📊 Expected Total Impact

If you implement the Week 1-3 recommendations:

**Quality Improvements**:
- Baseline: 0.52 composite score
- After improvements: 0.78-0.85 composite score
- **Total improvement: +50-63%**

**Cost Reductions**:
- Caching: -40-60%
- Model optimization: -20-30%
- **Total savings: 50-70% on API costs**

**Speed Improvements**:
- Caching hits: instant (vs 2-3 seconds)
- Optimized prompts: -10-20% token usage
- **Total speedup: 2-3x for repeated queries**

**ROI Timeline**:
- Week 1 improvements: Immediate positive ROI
- Week 2-3 improvements: Positive ROI after ~1,000 pitches
- Month 2 (fine-tuning): Positive ROI after ~10,000 pitches

---

This is a comprehensive roadmap. Start with Week 1 for immediate wins, then gradually add more advanced features as needed. Would you like me to implement any of these for you?


