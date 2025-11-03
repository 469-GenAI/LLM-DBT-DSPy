import dspy
from pydantic import BaseModel, Field


class PitchAssessment(BaseModel):
    """Structured pitch assessment output."""
    factual_score: float = Field(..., description="0.0-1.0 score for factual accuracy")
    narrative_score: float = Field(..., description="0.0-1.0 score for narrative quality")
    style_score: float = Field(..., description="0.0-1.0 score for style and tone")
    reasoning: str = Field(..., description="Brief explanation for scores")
    final_score: float = Field(..., description="Weighted average of scores")


class AssessPitchQuality(dspy.Signature):
    """
    Assess the generated pitch against a gold-standard pitch.
    Evaluate factual inclusion, narrative structure, and persuasive tone.
    
    Return a structured assessment with:
    - factual_score (0.0-1.0): Whether all pitch_facts were included
    - narrative_score (0.0-1.0): Clear problem, solution, and story
    - style_score (0.0-1.0): Confident, persuasive, engaging tone
    - reasoning: Brief explanation for the scores
    - final_score: Weighted average (factual*0.4 + narrative*0.4 + style*0.2)
    """
    
    pitch_facts: str = dspy.InputField(desc="The specific facts the pitch was based on.")
    ground_truth_pitch: str = dspy.InputField(desc="The gold-standard pitch to compare against.")
    generated_pitch: str = dspy.InputField(desc="The AI-generated pitch.")
    
    assessment: PitchAssessment = dspy.OutputField(
        desc="Structured assessment with factual_score, narrative_score, style_score (each 0.0-1.0), reasoning, and final_score"
    )