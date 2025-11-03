"""
Models package for pitch generation and evaluation.

Provides dedicated classes with fixed LMs to prevent model confusion.
"""
from .generator import PitchGenerator
from .evaluator import PitchEvaluator

__all__ = ["PitchGenerator", "PitchEvaluator"]

