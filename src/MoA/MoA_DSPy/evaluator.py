"""
Evaluator module for pitch quality assessment.
"""
import dspy
import json
from AssessPitch import AssessPitchQuality


class PitchEvaluatorModule(dspy.Module):
    """DSPy module for evaluating pitch quality using AssessPitchQuality signature."""
    
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(AssessPitchQuality)
    
    def forward(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str):
        """
        Assess the quality of a generated pitch.
        
        Args:
            pitch_facts: The structured facts the pitch was based on
            ground_truth_pitch: The gold-standard human-written pitch
            generated_pitch: The AI-generated pitch
            
        Returns:
            Assessment with Pydantic PitchAssessment model
        """
        assessment = self.assess(
            pitch_facts=pitch_facts,
            ground_truth_pitch=ground_truth_pitch,
            generated_pitch=generated_pitch
        )
        return assessment


class PitchEvaluator:
    """
    Dedicated pitch evaluator.
    
    This class ensures all pitch evaluation uses the assigned model exclusively,
    preventing any model confusion with the generator.
    """
    
    def __init__(self, lm):
        """
        Initialize evaluator with specified model.
        
        Args:
            lm: The dspy.LM instance for pitch evaluation
        """
        self.lm = lm
        self.evaluator = PitchEvaluatorModule()
        print(f"✓ PitchEvaluator initialized with model: {self.lm.model}")
    
    def evaluate(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str):
        """
        Evaluate pitch quality using the assigned evaluator model.
        
        Args:
            pitch_facts: Structured facts as string
            ground_truth_pitch: Gold-standard pitch
            generated_pitch: AI-generated pitch
            
        Returns:
            Assessment prediction with PitchAssessment model
        """
        with dspy.context(lm=self.lm):
            return self.evaluator(
                pitch_facts=pitch_facts,
                ground_truth_pitch=ground_truth_pitch,
                generated_pitch=generated_pitch
            )
    
    def get_score(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str) -> float:
        """
        Convenience method to get just the final score.
        
        Returns:
            Float score between 0.0 and 1.0
        """
        try:
            assessment = self.evaluate(pitch_facts, ground_truth_pitch, generated_pitch)
            return float(assessment.assessment.final_score)
        except (AttributeError, ValueError) as e:
            print(f"Warning: Could not extract score: {e}")
            return 0.0
    
    def get_full_assessment(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str) -> dict:
        """
        Convenience method to get full assessment as dictionary.
        
        Returns:
            Dictionary with all scores and reasoning
        """
        try:
            assessment = self.evaluate(pitch_facts, ground_truth_pitch, generated_pitch)
            pitch_assessment = assessment.assessment
            return {
                "factual_score": float(pitch_assessment.factual_score),
                "narrative_score": float(pitch_assessment.narrative_score),
                "style_score": float(pitch_assessment.style_score),
                "reasoning": str(pitch_assessment.reasoning),
                "final_score": float(pitch_assessment.final_score)
            }
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Failed to extract assessment: {e}")
            return {
                "factual_score": 0.0,
                "narrative_score": 0.0,
                "style_score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "final_score": 0.0
            }
    
    def __call__(self, pitch_facts: str, ground_truth_pitch: str, generated_pitch: str):
        """Allow instance to be called directly."""
        return self.evaluate(pitch_facts, ground_truth_pitch, generated_pitch)

