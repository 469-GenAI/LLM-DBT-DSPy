# evaluators/pitch_evaluator.py
import re
import json
from typing import Dict, Any, Optional
import dspy

class Assess(dspy.Signature):
    """Assess the quality of a pitch along a specified dimension."""
    
    pitch_text: str = dspy.InputField(desc="The pitch text to evaluate")
    financial_info: str = dspy.InputField(desc="Financial terms and context")
    assessment_question: str = dspy.InputField(desc="Specific question about what to assess")
    assessment_score: float = dspy.OutputField(desc="Score from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the score")

class PitchAssessor(dspy.Module):
    """DSPy module for pitch assessment using Chain of Thought reasoning"""
    
    def __init__(self):
        super().__init__()
        self.assessor = dspy.ChainOfThought(Assess)
    
    def forward(self, pitch_text: str, financial_info: str, assessment_question: str):
        """Assess pitch quality along a specific dimension"""
        return self.assessor(
            pitch_text=pitch_text,
            financial_info=financial_info,
            assessment_question=assessment_question
        )

class PitchEvaluator:
    """Enhanced evaluator using DSPy Signature pattern"""
    
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self._assessor = None
    
    @property
    def assessor(self):
        """Lazy load assessor to avoid import issues"""
        if self._assessor is None:
            self._assessor = PitchAssessor()
        return self._assessor
    
    def _format_financial_info(self, offer) -> str:
        """Format financial information for assessment"""
        return f"""
        FINANCIAL TERMS:
        - Valuation: {offer.Valuation}
        - Equity Offered: {offer.Equity_Offered}
        - Funding Requested: {offer.Funding_Amount}
        - Key Terms: {offer.Key_Terms}
        """
    
    def assess_problem_solution_fit(self, pitch_text: str, offer) -> tuple[float, str]:
        """Assess problem-solution fit (25% weight)"""
        question = """
        Evaluate the problem-solution fit of this pitch:
        1. Is the problem clearly identified and compelling?
        2. Is the solution well-defined and addresses the problem?
        3. Is there evidence of product-market fit?
        
        Rate from 0.0 to 1.0 based on clarity, relevance, and evidence.
        """
        
        financial_info = self._format_financial_info(offer)
        result = self.assessor(pitch_text=pitch_text, financial_info=financial_info, assessment_question=question)
        
        score = float(result.assessment_score)
        reasoning = result.reasoning
        
        if self.verbose:
            print(f"  Problem-Solution Fit: {score:.2f} - {reasoning}")
        
        return score, reasoning
    
    def assess_market_opportunity(self, pitch_text: str, offer) -> tuple[float, str]:
        """Assess market opportunity (25% weight)"""
        question = """
        Evaluate the market opportunity of this pitch:
        1. Is the market size clearly defined and substantial?
        2. Is there evidence of market growth potential?
        3. Is the target market well-defined?
        4. Are there competitive advantages mentioned?
        
        Rate from 0.0 to 1.0 based on market size, growth potential, and competitive positioning.
        """
        
        financial_info = self._format_financial_info(offer)
        result = self.assessor(pitch_text=pitch_text, financial_info=financial_info, assessment_question=question)
        
        score = float(result.assessment_score)
        reasoning = result.reasoning
        
        if self.verbose:
            print(f"  Market Opportunity: {score:.2f} - {reasoning}")
        
        return score, reasoning
    
    def assess_financial_logic(self, pitch_text: str, offer) -> tuple[float, str]:
        """Assess financial logic (25% weight)"""
        question = """
        Evaluate the financial logic of this pitch:
        1. Is the valuation reasonable given the financial metrics?
        2. Is the equity ask proportional to the funding amount?
        3. Are the financial projections realistic?
        4. Is there alignment between valuation and market opportunity?
        
        Rate from 0.0 to 1.0 based on financial coherence and reasonableness.
        """
        
        financial_info = self._format_financial_info(offer)
        result = self.assessor(pitch_text=pitch_text, financial_info=financial_info, assessment_question=question)
        
        score = float(result.assessment_score)
        reasoning = result.reasoning
        
        if self.verbose:
            print(f"  Financial Logic: {score:.2f} - {reasoning}")
        
        return score, reasoning
    
    def assess_persuasiveness(self, pitch_text: str, offer) -> tuple[float, str]:
        """Assess persuasiveness (25% weight)"""
        question = """
        Evaluate the persuasiveness of this pitch:
        1. Is the narrative compelling and well-structured?
        2. Is there a clear call-to-action?
        3. Does it create urgency and excitement?
        4. Is the tone appropriate for investors?
        
        Rate from 0.0 to 1.0 based on narrative quality and persuasive power.
        """
        
        financial_info = self._format_financial_info(offer)
        result = self.assessor(pitch_text=pitch_text, financial_info=financial_info, assessment_question=question)
        
        score = float(result.assessment_score)
        reasoning = result.reasoning
        
        if self.verbose:
            print(f"  Persuasiveness: {score:.2f} - {reasoning}")
        
        return score, reasoning
    
    def evaluate_pitch(self, response, offer, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate pitch quality across all dimensions"""
        try:
            pitch_text = response.Pitch
            
            # Assess each dimension
            ps_score, ps_reasoning = self.assess_problem_solution_fit(pitch_text, offer)
            mo_score, mo_reasoning = self.assess_market_opportunity(pitch_text, offer)
            fl_score, fl_reasoning = self.assess_financial_logic(pitch_text, offer)
            p_score, p_reasoning = self.assess_persuasiveness(pitch_text, offer)
            
            # Calculate weighted composite score
            composite_score = (ps_score * 0.25 + mo_score * 0.25 + fl_score * 0.25 + p_score * 0.25)
            
            if self.verbose:
                print(f"\n📊 PITCH EVALUATION BREAKDOWN:")
                print(f"  Problem-Solution Fit: {ps_score:.2f}")
                print(f"  Market Opportunity: {mo_score:.2f}")
                print(f"  Financial Logic: {fl_score:.2f}")
                print(f"  Persuasiveness: {p_score:.2f}")
                print(f"  🎯 COMPOSITE SCORE: {composite_score:.2f}")
                print()
            
            return {
                "composite_score": composite_score,
                "dimensions": {
                    "problem_solution": {"score": ps_score, "reasoning": ps_reasoning},
                    "market_opportunity": {"score": mo_score, "reasoning": mo_reasoning},
                    "financial_logic": {"score": fl_score, "reasoning": fl_reasoning},
                    "persuasiveness": {"score": p_score, "reasoning": p_reasoning}
                }
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                "composite_score": 0.0,
                "dimensions": {
                    "problem_solution": {"score": 0.0, "reasoning": f"Error: {e}"},
                    "market_opportunity": {"score": 0.0, "reasoning": f"Error: {e}"},
                    "financial_logic": {"score": 0.0, "reasoning": f"Error: {e}"},
                    "persuasiveness": {"score": 0.0, "reasoning": f"Error: {e}"}
                }
            }

# Global evaluator instance
_evaluator = None

def get_pitch_evaluator(verbose: bool = False) -> PitchEvaluator:
    """Get singleton evaluator instance"""
    global _evaluator
    if _evaluator is None or _evaluator.verbose != verbose:
        _evaluator = PitchEvaluator(verbose=verbose)
    return _evaluator

def llm_pitch_metric(example, pred, trace=None, verbose: bool = False) -> float:
    """DSPy-compatible metric function using enhanced evaluator"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            if verbose:
                print("❌ Missing required fields for evaluation")
            return 0.0
        
        evaluator = get_pitch_evaluator(verbose=verbose)
        context = example.get('product_data', {})
        evaluation_result = evaluator.evaluate_pitch(resp, offer, context)
        
        score = evaluation_result["composite_score"]
        
        # Handle trace for optimization vs evaluation
        if trace is not None:
            # During optimization: return bool with strict threshold
            threshold = 0.7
            passes = score >= threshold
            if verbose:
                print(f"🔍 Optimization check: {score:.2f} {'✅ PASSES' if passes else '❌ FAILS'} (threshold: {threshold})")
            return passes
        else:
            # During evaluation: return float score
            if verbose:
                print(f"📈 Evaluation score: {score:.2f}")
            return score
        
    except Exception as e:
        print(f"Metric evaluation error: {e}")
        return 0.0

# Individual dimension metrics
def problem_solution_metric(example, pred, trace=None, verbose: bool = False) -> float:
    """Individual metric for problem-solution fit"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            return 0.0
        
        evaluator = get_pitch_evaluator(verbose=verbose)
        score, _ = evaluator.assess_problem_solution_fit(resp.Pitch, offer)
        
        if trace is not None:
            return score >= 0.7
        return score
        
    except Exception as e:
        print(f"Problem-solution metric error: {e}")
        return 0.0

def market_opportunity_metric(example, pred, trace=None, verbose: bool = False) -> float:
    """Individual metric for market opportunity"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            return 0.0
        
        evaluator = get_pitch_evaluator(verbose=verbose)
        score, _ = evaluator.assess_market_opportunity(resp.Pitch, offer)
        
        if trace is not None:
            return score >= 0.7
        return score
        
    except Exception as e:
        print(f"Market opportunity metric error: {e}")
        return 0.0

def financial_logic_metric(example, pred, trace=None, verbose: bool = False) -> float:
    """Individual metric for financial logic"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            return 0.0
        
        evaluator = get_pitch_evaluator(verbose=verbose)
        score, _ = evaluator.assess_financial_logic(resp.Pitch, offer)
        
        if trace is not None:
            return score >= 0.7
        return score
        
    except Exception as e:
        print(f"Financial logic metric error: {e}")
        return 0.0

def persuasiveness_metric(example, pred, trace=None, verbose: bool = False) -> float:
    """Individual metric for persuasiveness"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            return 0.0
        
        evaluator = get_pitch_evaluator(verbose=verbose)
        score, _ = evaluator.assess_persuasiveness(resp.Pitch, offer)
        
        if trace is not None:
            return score >= 0.7
        return score
        
    except Exception as e:
        print(f"Persuasiveness metric error: {e}")
        return 0.0
