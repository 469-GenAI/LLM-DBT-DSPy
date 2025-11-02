# utils.py - Pydantic models for structured pitch input
"""
Utility models for the structured pitch generation system.
These models match the input format from the HuggingFace sharktank_pitches dataset.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ProblemStory(BaseModel):
    """Model representing the problem narrative in a pitch."""
    persona: str = Field(..., description="The target customer persona")
    routine: List[str] = Field(..., description="List of routine behaviors or actions")
    core_problem: str = Field(..., description="The central problem being addressed")
    hygiene_gap: str = Field(..., description="The gap in current solutions")
    problem_keywords: List[str] = Field(..., description="Key problem descriptors")


class ProductSolution(BaseModel):
    """Model representing the product solution in a pitch."""
    name: str = Field(..., description="Product or company name")
    product_category: str = Field(..., description="Category of the product")
    key_differentiator: str = Field(..., description="What makes this product unique")
    application: str = Field(..., description="How the product is used")
    features_keywords: List[str] = Field(..., description="Key features of the product")
    benefits_keywords: List[str] = Field(..., description="Key benefits to customers")


class ClosingTheme(BaseModel):
    """Model representing the closing theme of a pitch."""
    call_to_action: str = Field(..., description="The call to action for investors")
    mission: str = Field(..., description="The company's mission statement")
    target_audience: str = Field(..., description="Target audience description")


class InitialOfferInput(BaseModel):
    """Model representing the investment offer details."""
    amount: str = Field(..., description="Funding amount requested (e.g., '$400k')")
    equity: str = Field(..., description="Equity percentage offered (e.g., '5%')")


class PitchInput(BaseModel):
    """Complete structured input for pitch generation."""
    founders: List[str] = Field(..., description="List of founder names")
    company_name: str = Field(..., description="Name of the company")
    initial_offer: InitialOfferInput = Field(..., description="Investment offer details")
    problem_story: ProblemStory = Field(..., description="The problem narrative")
    product_solution: ProductSolution = Field(..., description="The product solution")
    closing_theme: ClosingTheme = Field(..., description="Closing theme and call to action")


def format_pitch_input(pitch_input: PitchInput) -> str:
    """
    Format a PitchInput object into a structured string for DSPy processing.
    
    Args:
        pitch_input: Structured pitch input data
        
    Returns:
        Formatted string representation of the pitch input
    """
    return f"""
Company: {pitch_input.company_name}
Founders: {', '.join(pitch_input.founders)}
Investment Ask: {pitch_input.initial_offer.amount} for {pitch_input.initial_offer.equity} equity

PROBLEM:
Persona: {pitch_input.problem_story.persona}
Core Problem: {pitch_input.problem_story.core_problem}
Gap: {pitch_input.problem_story.hygiene_gap}

SOLUTION:
Product: {pitch_input.product_solution.name}
Category: {pitch_input.product_solution.product_category}
Differentiator: {pitch_input.product_solution.key_differentiator}
Application: {pitch_input.product_solution.application}
Key Features: {', '.join(pitch_input.product_solution.features_keywords)}
Key Benefits: {', '.join(pitch_input.product_solution.benefits_keywords)}

CLOSING:
Mission: {pitch_input.closing_theme.mission}
Call to Action: {pitch_input.closing_theme.call_to_action}
""".strip()

