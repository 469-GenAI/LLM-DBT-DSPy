# evaluators/pitch_evaluator.py
import re
import json
from typing import Dict, Any, Optional

class PitchEvaluator:
    """LLM-as-a-judge evaluator for pitch quality"""
    
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model_name
        self._lm = None
    
    @property
    def lm(self):
        """Lazy load LM to avoid import issues"""
        if self._lm is None:
            import dspy
            self._lm = dspy.LM(self.model_name)
        return self._lm
    
    def evaluate_pitch(self, response, offer, context: Optional[Dict] = None) -> float:
        """Evaluate pitch quality using LLM-as-a-judge"""
        try:
            prompt = self._build_evaluation_prompt(response, offer, context)
            evaluation_response = self.lm(prompt)
            return self._extract_score(evaluation_response)
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return self._fallback_scoring(response, offer)
    
    def _build_evaluation_prompt(self, response, offer, context=None) -> str:
        """Build evaluation prompt"""
        return f"""
            You are an expert venture capitalist evaluating a startup pitch. Rate from 0.0 to 1.0.

            PITCH: {response.Pitch}

            FINANCIAL TERMS:
            - Valuation: {offer.Valuation}
            - Equity Offered: {offer.Equity_Offered}
            - Funding Requested: {offer.Funding_Amount}
            - Key Terms: {offer.Key_Terms}

            EVALUATION CRITERIA:
            1. Problem-Solution Fit (25%): Clear problem identification and compelling solution
            2. Market Opportunity (25%): Market size and growth potential
            3. Financial Logic (25%): Reasonable valuation, equity, and funding alignment
            4. Persuasiveness (25%): Compelling narrative and call-to-action

            CONTEXT: {json.dumps(context or {}, indent=2)}

            Respond with ONLY a number between 0.0 and 1.0.
            """
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response"""
        score_match = re.search(r'(\d+\.?\d*)', str(response))
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)
        return 0.5
    
    def _fallback_scoring(self, response, offer) -> float:
        """Fallback keyword-based scoring"""
        score = 0.0
        pitch_text = response.Pitch.lower()
        
        # Basic quality indicators
        if len(response.Pitch) > 100:
            score += 0.2
        if any(word in pitch_text for word in ["problem", "solution", "market", "opportunity"]):
            score += 0.3
        if any(word in pitch_text for word in ["invest", "growth", "potential"]):
            score += 0.2
        
        # Financial logic check
        try:
            valuation = float(offer.Valuation.replace('$', '').replace(',', '').replace(' million', '000000'))
            funding = float(offer.Funding_Amount.replace('$', '').replace(',', ''))
            equity = float(offer.Equity_Offered.replace('%', ''))
            
            implied_valuation = (funding / equity) * 100 if equity > 0 else 0
            if abs(valuation - implied_valuation) / max(valuation, implied_valuation) < 0.3:
                score += 0.3
        except:
            pass
        
        return min(score, 1.0)

# Global evaluator instance
_evaluator = None

def get_pitch_evaluator() -> PitchEvaluator:
    """Get singleton evaluator instance"""
    global _evaluator
    if _evaluator is None:
        _evaluator = PitchEvaluator()
    return _evaluator

def llm_pitch_metric(example, pred, trace=None) -> float:
    """DSPy-compatible metric function using LLM-as-a-judge"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        
        if not (resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount):
            return 0.0
        
        evaluator = get_pitch_evaluator()
        context = example.get('product_data', {})
        return evaluator.evaluate_pitch(resp, offer, context)
        
    except Exception as e:
        print(f"Metric evaluation error: {e}")
        return 0.0
