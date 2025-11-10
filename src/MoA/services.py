"""
Utility factories and compilation helpers for the MoA DSPy program.
"""

from typing import Callable, List, Optional

import dspy

from MoA.models.program import MoAGenerator, MoAMixtureProgram
from pitchLLM.models.evaluator import PitchEvaluator


def create_moa_generator(lm: dspy.LM, num_agents: int = 3) -> MoAMixtureProgram:
    """
    Instantiate the MoA generator with the provided language model.

    Args:
        lm: Language model used for planning, writing, and aggregation.
        num_agents: Number of agent sections to generate per pitch.

    Returns:
        MoAMixtureProgram ready for compilation or direct execution.
    """
    generator = MoAGenerator(lm=lm, num_agents=num_agents)
    return generator.program


def create_moa_evaluator(lm: dspy.LM) -> PitchEvaluator:
    """
    Instantiate the shared pitch evaluator used across structured pipelines.

    Args:
        lm: Dedicated evaluation language model.

    Returns:
        PitchEvaluator configured with the provided model.
    """
    if lm is None:
        raise ValueError("Evaluator language model is required.")

    return PitchEvaluator(lm)


def compile_moa_program(
    program: MoAMixtureProgram,
    trainset: List[dspy.Example],
    optimization_method: str = "none",
    metric: Optional[Callable] = None,
) -> MoAMixtureProgram:
    """
    Compile the MoA program with optional DSPy optimisation on the final module.

    Args:
        program: MoAMixtureProgram to compile.
        trainset: Training examples for optimisation.
        optimization_method: One of \"none\", \"bootstrap\", \"bootstrap_random\",
            \"knn\", or \"mipro\".
        metric: Optional evaluation metric function.

    Returns:
        Compiled MoAMixtureProgram (may be identical when optimisation is \"none\").
    """
    if optimization_method == "none":
        return program

    if metric is None:
        raise ValueError("Metric function is required for optimisation.")

    if optimization_method == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        return optimizer.compile(program, trainset=trainset)

    if optimization_method == "bootstrap_random":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=8,
        )
        return optimizer.compile(program, trainset=trainset)

    if optimization_method == "knn":
        from pitchLLM.utils import create_pitch_vectorizer

        vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
        optimizer = dspy.KNNFewShot(
            k=3,
            trainset=trainset,
            vectorizer=vectorizer,
        )
        return optimizer.compile(program)

    if optimization_method == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            init_temperature=1.0,
        )
        return optimizer.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )

    raise ValueError(f"Unknown optimisation method: {optimization_method}")

