"""
DSPy Optimization Module for Pitch Generation

This module provides tools for optimizing DSPy pitch generation programs using
various optimizers like MIPROv2, BootstrapFewShot, and BootstrapRS.

Main components:
- dataset_prep: Load and prepare training/validation/test datasets
- metrics: Comprehensive evaluation metrics for pitch quality
- optimize_pitch: Main optimization script with multiple optimizers
"""

from .dataset_prep import load_dataset, SharkTankDataset
from .metrics import (
    get_structure_metric,
    get_composite_metric,
    get_all_metrics,
    PitchQualityMetrics
)

__all__ = [
    'load_dataset',
    'SharkTankDataset',
    'get_structure_metric',
    'get_composite_metric',
    'get_all_metrics',
    'PitchQualityMetrics'
]




