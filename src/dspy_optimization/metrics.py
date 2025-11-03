"""
Comprehensive metrics for evaluating DSPy pitch generation quality
"""

import json
import re
from typing import Dict, Any, Optional
import dspy


class PitchQualityMetrics:
    """Comprehensive metrics for pitch evaluation"""
    
    @staticmethod
    def basic_structure_metric(example: dspy.Example, pred: Any, trace=None) -> bool:
        """
        Check if prediction has basic required structure (Pitch + Initial_Offer)
        This is a binary metric: True if valid structure, False otherwise.
        """
        try:
            # Handle both dict and object responses
            if isinstance(pred, dict):
                resp = pred.get("response")
            else:
                resp = pred.response if hasattr(pred, 'response') else pred
            
            # Check for Pitch
            has_pitch = hasattr(resp, 'Pitch') and resp.Pitch and len(str(resp.Pitch)) > 50
            
            # Check for Initial_Offer
            if hasattr(resp, 'Initial_Offer'):
                offer = resp.Initial_Offer
                has_valuation = hasattr(offer, 'Valuation') and offer.Valuation
                has_equity = hasattr(offer, 'Equity_Offered') and offer.Equity_Offered
                has_funding = hasattr(offer, 'Funding_Amount') and offer.Funding_Amount
                has_offer = has_valuation and has_equity and has_funding
            else:
                has_offer = False
            
            return has_pitch and has_offer
            
        except Exception as e:
            print(f"Error in basic_structure_metric: {e}")
            return False
    
    @staticmethod
    def completeness_score(example: dspy.Example, pred: Any, trace=None) -> float:
        """
        Score from 0.0 to 1.0 based on completeness of the response
        Checks for: Pitch, Valuation, Equity, Funding Amount, Key Terms
        """
        score = 0.0
        
        try:
            if isinstance(pred, dict):
                resp = pred.get("response")
            else:
                resp = pred.response if hasattr(pred, 'response') else pred
            
            # Pitch quality (0.4 points)
            if hasattr(resp, 'Pitch') and resp.Pitch:
                pitch_len = len(str(resp.Pitch))
                if pitch_len > 200:
                    score += 0.4
                elif pitch_len > 100:
                    score += 0.3
                elif pitch_len > 50:
                    score += 0.2
                else:
                    score += 0.1
            
            # Initial Offer components (0.6 points total)
            if hasattr(resp, 'Initial_Offer'):
                offer = resp.Initial_Offer
                
                # Valuation (0.15)
                if hasattr(offer, 'Valuation') and offer.Valuation:
                    if _has_dollar_amount(offer.Valuation):
                        score += 0.15
                
                # Equity (0.15)
                if hasattr(offer, 'Equity_Offered') and offer.Equity_Offered:
                    if '%' in str(offer.Equity_Offered) or 'percent' in str(offer.Equity_Offered).lower():
                        score += 0.15
                
                # Funding Amount (0.15)
                if hasattr(offer, 'Funding_Amount') and offer.Funding_Amount:
                    if _has_dollar_amount(offer.Funding_Amount):
                        score += 0.15
                
                # Key Terms (0.15)
                if hasattr(offer, 'Key_Terms') and offer.Key_Terms and len(str(offer.Key_Terms)) > 20:
                    score += 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in completeness_score: {e}")
            return 0.0
    
    @staticmethod
    def pitch_quality_score(example: dspy.Example, pred: Any, trace=None) -> float:
        """
        Score from 0.0 to 1.0 based on pitch content quality
        Checks for key elements: value proposition, market opportunity, traction, call to action
        """
        score = 0.0
        
        try:
            if isinstance(pred, dict):
                resp = pred.get("response")
            else:
                resp = pred.response if hasattr(pred, 'response') else pred
            
            if not hasattr(resp, 'Pitch') or not resp.Pitch:
                return 0.0
            
            pitch_text = str(resp.Pitch).lower()
            
            # Check for value proposition keywords (0.25)
            value_keywords = ['unique', 'innovative', 'solution', 'problem', 'solves', 'addresses', 'revolutionary']
            if any(keyword in pitch_text for keyword in value_keywords):
                score += 0.25
            
            # Check for market/opportunity keywords (0.25)
            market_keywords = ['market', 'opportunity', 'industry', 'customers', 'demand', 'growing']
            if any(keyword in pitch_text for keyword in market_keywords):
                score += 0.25
            
            # Check for traction/proof keywords (0.25)
            traction_keywords = ['revenue', 'sales', 'customers', 'users', 'growth', 'profit', 'traction']
            if any(keyword in pitch_text for keyword in traction_keywords):
                score += 0.25
            
            # Check for call to action (0.25)
            cta_keywords = ['invest', 'join', 'partner', 'opportunity', 'together', 'funding']
            if any(keyword in pitch_text for keyword in cta_keywords):
                score += 0.25
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in pitch_quality_score: {e}")
            return 0.0
    
    @staticmethod
    def financial_consistency_score(example: dspy.Example, pred: Any, trace=None) -> float:
        """
        Score from 0.0 to 1.0 based on financial consistency
        Checks if valuation and funding amount are reasonable and consistent
        """
        score = 0.0
        
        try:
            if isinstance(pred, dict):
                resp = pred.get("response")
                valuation = pred.get("valuation", 0)
                funding_amount = pred.get("funding_amount", 0)
            else:
                resp = pred.response if hasattr(pred, 'response') else pred
                valuation = pred.valuation if hasattr(pred, 'valuation') else 0
                funding_amount = pred.funding_amount if hasattr(pred, 'funding_amount') else 0
            
            if not hasattr(resp, 'Initial_Offer'):
                return 0.0
            
            offer = resp.Initial_Offer
            
            # Extract dollar amounts from strings
            offered_valuation = _extract_dollar_amount(offer.Valuation) if hasattr(offer, 'Valuation') else 0
            offered_funding = _extract_dollar_amount(offer.Funding_Amount) if hasattr(offer, 'Funding_Amount') else 0
            
            # Check if both amounts are present (0.3)
            if offered_valuation > 0 and offered_funding > 0:
                score += 0.3
            
            # Check if funding is reasonable percentage of valuation (0.4)
            if offered_valuation > 0 and offered_funding > 0:
                funding_ratio = offered_funding / offered_valuation
                if 0.05 <= funding_ratio <= 0.25:  # 5% to 25% is reasonable
                    score += 0.4
                elif 0.01 <= funding_ratio <= 0.40:  # 1% to 40% is acceptable
                    score += 0.2
            
            # Check consistency with calculated values if available (0.3)
            if valuation > 0 and funding_amount > 0:
                val_diff = abs(offered_valuation - valuation) / valuation if valuation > 0 else 1.0
                fund_diff = abs(offered_funding - funding_amount) / funding_amount if funding_amount > 0 else 1.0
                
                if val_diff < 0.1 and fund_diff < 0.1:  # Within 10%
                    score += 0.3
                elif val_diff < 0.3 and fund_diff < 0.3:  # Within 30%
                    score += 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Error in financial_consistency_score: {e}")
            return 0.0
    
    @staticmethod
    def composite_metric(example: dspy.Example, pred: Any, trace=None) -> float:
        """
        Composite metric combining all quality scores
        Returns weighted average of all metrics (0.0 to 1.0)
        """
        weights = {
            'structure': 0.3,
            'completeness': 0.25,
            'pitch_quality': 0.25,
            'financial_consistency': 0.2
        }
        
        structure_score = 1.0 if PitchQualityMetrics.basic_structure_metric(example, pred, trace) else 0.0
        completeness = PitchQualityMetrics.completeness_score(example, pred, trace)
        pitch_quality = PitchQualityMetrics.pitch_quality_score(example, pred, trace)
        financial = PitchQualityMetrics.financial_consistency_score(example, pred, trace)
        
        composite = (
            weights['structure'] * structure_score +
            weights['completeness'] * completeness +
            weights['pitch_quality'] * pitch_quality +
            weights['financial_consistency'] * financial
        )
        
        return composite


def _has_dollar_amount(text: str) -> bool:
    """Check if text contains a dollar amount"""
    if not text:
        return False
    return '$' in str(text) or 'dollar' in str(text).lower()


def _extract_dollar_amount(text: str) -> float:
    """Extract numeric dollar amount from text like '$1,234,567' or '$1.5 million'"""
    if not text:
        return 0.0
    
    text = str(text).replace(',', '')
    
    # Try direct number extraction
    match = re.search(r'\$?\s*([0-9]+\.?[0-9]*)', text)
    if match:
        amount = float(match.group(1))
        
        # Check for multipliers
        text_lower = text.lower()
        if 'million' in text_lower or 'm' in text_lower:
            amount *= 1_000_000
        elif 'billion' in text_lower or 'b' in text_lower:
            amount *= 1_000_000_000
        elif 'thousand' in text_lower or 'k' in text_lower:
            amount *= 1_000
        
        return amount
    
    return 0.0


# Convenience functions for common use cases
def get_structure_metric():
    """Get basic structure validation metric (binary: True/False)"""
    return PitchQualityMetrics.basic_structure_metric


def get_composite_metric():
    """Get composite quality metric (0.0 to 1.0)"""
    return PitchQualityMetrics.composite_metric


def get_all_metrics() -> Dict[str, callable]:
    """Get dictionary of all available metrics"""
    return {
        'structure': PitchQualityMetrics.basic_structure_metric,
        'completeness': PitchQualityMetrics.completeness_score,
        'pitch_quality': PitchQualityMetrics.pitch_quality_score,
        'financial_consistency': PitchQualityMetrics.financial_consistency_score,
        'composite': PitchQualityMetrics.composite_metric
    }


if __name__ == "__main__":
    print("Available metrics:")
    for name, metric in get_all_metrics().items():
        print(f"  - {name}: {metric.__doc__.split(chr(10))[0] if metric.__doc__ else 'No description'}")




