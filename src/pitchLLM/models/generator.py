"""
Generator module for pitch generation.
"""
import json
import dspy
from utils import PitchInput, format_pitch_input


class PitchGenerationSig(dspy.Signature):
    """
    Generate a compelling Shark Tank pitch from structured input.
    
    The pitch should:
    - Start with an engaging introduction of the founders and company
    - Present the investment ask clearly
    - Tell a story about the problem from the customer's perspective
    - Introduce the solution with compelling details
    - End with a strong call to action for the Sharks
    """
    
    pitch_data: str = dspy.InputField(
        desc="Structured pitch data including company, founders, problem story, solution, and investment ask"
    )
    
    pitch: str = dspy.OutputField(
        desc="A compelling, narrative pitch in the style of Shark Tank presentations. "
             "Should be conversational, engaging, and tell a complete story from problem to solution."
    )


class StructuredPitchProgram(dspy.Module):
    """DSPy module for generating structured pitches."""
    
    def __init__(self):
        super().__init__()
        self.generate_pitch = dspy.ChainOfThought(PitchGenerationSig)
    
    def forward(self, input: dict, **kwargs):
        """
        Generate a pitch from structured input.
        
        Args:
            input: Dictionary containing structured pitch data
            **kwargs: Additional arguments passed to underlying predictors
                     (e.g., config={"rollout_id": "...", "temperature": 1.0})
            
        Returns:
            dspy.Prediction with pitch field
        """
        try:
            pitch_input = PitchInput(**input)
            formatted_input = format_pitch_input(pitch_input)
        except Exception as e:
            print(f"Warning: Could not parse input as PitchInput: {e}")
            formatted_input = json.dumps(input, indent=2)
        
        # Pass kwargs (including config) to the underlying predictor
        prediction = self.generate_pitch(pitch_data=formatted_input, **kwargs)
        return prediction


class PitchGenerator:
    """
    Dedicated pitch generator.
    
    This class ensures all pitch generation uses the assigned model exclusively,
    preventing any model confusion with the evaluator.
    """
    
    def __init__(self, lm):
        """
        Initialize generator with specified model.
        
        Args:
            lm: The dspy.LM instance for pitch generation
        """
        self.lm = lm
        self.program = StructuredPitchProgram()
        print(f"✓ PitchGenerator initialized with model: {self.lm.model}")
    
    def generate(self, input_data: dict, config: dict = None):
        """
        Generate a pitch using the assigned generator model.
        
        Args:
            input_data: Dictionary with structured pitch data
            config: Optional config dict with rollout_id and temperature for cache control
            
        Returns:
            dspy.Prediction with pitch field
        """
        # Build context parameters
        context_params = {"lm": self.lm}
        if config:
            context_params.update(config)
        
        with dspy.context(**context_params):
            return self.program(input=input_data)
    
    def __call__(self, input_data: dict):
        """Allow instance to be called directly."""
        return self.generate(input_data)

