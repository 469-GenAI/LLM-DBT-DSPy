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
    Critically assess the generated pitch against a gold-standard pitch.
    Be strict and penalize missing facts, weak narratives, and poor style.
    
    SCORING RUBRIC (be harsh):
    
    Factual Score (0.0-1.0):
    - 1.0: ALL facts present, correctly represented, no hallucinations
    - 0.7: Most facts present but 1-2 key details missing
    - 0.5: Several important facts missing or incorrect
    - 0.3: Many facts missing, significant errors
    - 0.0: Almost no factual accuracy
    
    Narrative Score (0.0-1.0):
    - 1.0: Crystal clear problem→solution→story, compelling flow
    - 0.7: Clear structure but lacks compelling emotional arc
    - 0.5: Basic structure present but disjointed or weak
    - 0.3: Poor structure, hard to follow
    - 0.0: No coherent narrative
    
    Style Score (0.0-1.0):
    - 1.0: Highly persuasive, confident, Shark Tank-ready
    - 0.7: Professional but lacks punch
    - 0.5: Generic business pitch, not engaging
    - 0.3: Weak tone, unconvincing
    - 0.0: Unprofessional or inappropriate
    
    BE CRITICAL: A baseline model should score 0.4-0.6. Only exceptional pitches deserve 0.9+.
    Final score is weighted average: factual*0.4 + narrative*0.4 + style*0.2
    """
    
    pitch_facts: str = dspy.InputField(desc="The specific facts the pitch was based on.")
    ground_truth_pitch: str = dspy.InputField(desc="The gold-standard pitch to compare against.")
    generated_pitch: str = dspy.InputField(desc="The AI-generated pitch to critically assess.")
    
    assessment: PitchAssessment = dspy.OutputField(
        desc="Harsh assessment with factual_score, narrative_score, style_score (0.0-1.0), reasoning explaining penalties, and final_score"
    )