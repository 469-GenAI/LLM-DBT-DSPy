"""
Compare RAG vs KNNFewShot: Extract and compare which examples each selects.

This script tests the hypothesis that RAG and KNNFewShot are "mining the same gold mine"
by comparing which training examples each mechanism selects for the same test case.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dotenv import load_dotenv
import dspy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from data_loader import load_and_prepare_data
from models import PitchGenerator
from models.rag_generator import RAGPitchGenerator
from rag.retriever import RetrievalStrategy
from utils import create_pitch_vectorizer
from models.generator import StructuredPitchProgram

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class KNNFewShotLogger:
    """
    Wrapper to intercept and log which examples KNNFewShot selects.
    
    This works by monkey-patching DSPy's internal retrieval mechanism.
    """
    def __init__(self, trainset: List[dspy.Example]):
        self.trainset = trainset
        self.selected_examples = []
        self.selected_ids = []
        
    def log_selection(self, example_ids: List[str]):
        """Log which examples were selected."""
        self.selected_ids = example_ids
        self.selected_examples = [
            ex for ex in self.trainset if ex.id in example_ids
        ]


def extract_knn_selections(
    program: StructuredPitchProgram,
    test_example: dspy.Example,
    trainset: List[dspy.Example],
    k: int = 3
) -> Tuple[List[str], List[Dict]]:
    """
    Extract which examples KNNFewShot selects for a given test example.
    
    This works by:
    1. Creating a vectorizer that matches KNNFewShot's
    2. Manually computing KNN to see which examples would be selected
    3. This replicates what DSPy's KNNFewShot does internally
    
    Returns:
        Tuple of (selected_ids, selected_examples_info)
    """
    import numpy as np
    
    # Create the same vectorizer KNNFewShot uses
    vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
    
    # Embed the test example
    try:
        test_embedding = np.array(vectorizer(test_example))
    except Exception as e:
        logger.error(f"Failed to embed test example: {e}")
        return [], []
    
    # Embed all training examples
    train_embeddings = []
    train_ids = []
    train_examples_map = {}
    
    for train_ex in trainset:
        try:
            emb = np.array(vectorizer(train_ex))
            train_embeddings.append(emb)
            train_ids.append(train_ex.id)
            train_examples_map[train_ex.id] = train_ex
        except Exception as e:
            logger.warning(f"Failed to embed train example {train_ex.id}: {e}")
            continue
    
    if not train_embeddings:
        logger.error("No training examples could be embedded")
        return [], []
    
    # Convert to numpy array
    train_embeddings = np.array(train_embeddings)
    
    # Normalize embeddings for cosine similarity
    test_emb_norm = test_embedding / (np.linalg.norm(test_embedding) + 1e-10)
    train_embs_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarities (dot product of normalized vectors)
    similarities = np.dot(train_embs_norm, test_emb_norm)
    
    # Get top-k indices (highest similarity first)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Extract selected IDs and info
    selected_ids = [train_ids[i] for i in top_k_indices]
    selected_info = []
    
    for idx in top_k_indices:
        train_ex = train_examples_map[train_ids[idx]]
        selected_info.append({
            'id': train_ids[idx],
            'similarity': float(similarities[idx]),
            'company': train_ex.input.get('company', 'N/A'),
            'problem_summary': str(train_ex.input.get('problem_summary', 'N/A'))[:100]
        })
    
    return selected_ids, selected_info


def extract_rag_selections(
    retriever,
    test_example: dspy.Example,
    top_k: int = 5,
    strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC,
    filter_successful_deals: bool = False,
    prioritize_successful: bool = True,
    prioritize_category: bool = True
) -> Tuple[List[str], List[Dict]]:
    """
    Extract which examples RAG retrieves for a given test example.
    
    Returns:
        Tuple of (selected_ids, selected_examples_info)
        Note: IDs are from vector store (product_key or id field from JSONL)
    """
    # Retrieve similar pitches
    retrieve_kwargs = {
        'product_data': test_example.input,
        'top_k': top_k,
        'include_scores': True,
        'strategy': strategy
    }
    
    # Add strategy-specific parameters
    if strategy == RetrievalStrategy.HYBRID_FILTER:
        retrieve_kwargs['filter_successful_deals'] = filter_successful_deals
    elif strategy == RetrievalStrategy.HYBRID_PRIORITIZE:
        retrieve_kwargs['prioritize_successful'] = prioritize_successful
        retrieve_kwargs['prioritize_category'] = prioritize_category
    
    similar_pitches = retriever.retrieve_similar_pitches(**retrieve_kwargs)
    
    # Extract IDs and info
    selected_ids = []
    selected_info = []
    
    for pitch in similar_pitches:
        # RAG stores IDs in metadata as 'product_key' which matches JSONL 'id' field
        # The ChromaDB document ID is also set to product_key
        metadata = pitch.get('metadata', {})
        
        # The product_key should match the JSONL 'id' field used by KNNFewShot
        # This is what we use for comparison
        pitch_id = metadata.get('product_key', 'unknown')
        
        # If product_key not found, try other fields
        if pitch_id == 'unknown':
            # Check if there's an 'id' field in metadata
            pitch_id = metadata.get('id', 'unknown')
            # Last resort: use product_name (but this won't match KNNFewShot IDs)
            if pitch_id == 'unknown':
                pitch_id = pitch.get('product_name', 'unknown')
        
        selected_ids.append(pitch_id)
        
        selected_info.append({
            'id': pitch_id,
            'product_name': pitch.get('product_name', 'N/A'),
            'similarity': pitch.get('similarity_score', 0.0),
            'has_deal': metadata.get('has_deal', False),
            'category': metadata.get('category', 'N/A')
        })
    
    return selected_ids, selected_info


def compute_overlap(rag_ids: List[str], knn_ids: List[str]) -> Dict:
    """
    Compute overlap metrics between RAG and KNNFewShot selections.
    
    Returns:
        Dict with overlap statistics
    """
    rag_set = set(rag_ids)
    knn_set = set(knn_ids)
    
    intersection = rag_set & knn_set
    union = rag_set | knn_set
    
    jaccard = len(intersection) / len(union) if union else 0.0
    overlap_ratio = len(intersection) / len(knn_set) if knn_set else 0.0
    
    return {
        'rag_selected': len(rag_set),
        'knn_selected': len(knn_set),
        'overlap_count': len(intersection),
        'overlap_ids': list(intersection),
        'jaccard_similarity': jaccard,
        'overlap_ratio': overlap_ratio,  # What % of KNN selections also appear in RAG
        'unique_to_rag': list(rag_set - knn_set),
        'unique_to_knn': list(knn_set - rag_set)
    }


def generate_and_evaluate_rag(
    test_example: dspy.Example,
    generator_lm,
    evaluator_lm,
    retriever,
    rag_top_k: int = 5,
    rag_strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC,
    filter_successful_deals: bool = False,
    prioritize_successful: bool = True,
    prioritize_category: bool = True
) -> Dict:
    """
    Generate pitch using RAG and evaluate it.
    
    Returns:
        Dict with generation results, prompts, outputs, and scores
    """
    from models.rag_generator import RAGPitchGenerator
    from models import PitchEvaluator
    
    logger.info("\n[Generating with RAG...]")
    
    # Create RAG generator with strategy-specific parameters
    rag_generator_kwargs = {
        'lm': generator_lm,
        'retriever': retriever,
        'top_k': rag_top_k,
        'retrieval_strategy': rag_strategy
    }
    
    # Add strategy-specific parameters
    if rag_strategy == RetrievalStrategy.HYBRID_FILTER:
        rag_generator_kwargs['filter_successful_deals'] = filter_successful_deals
    elif rag_strategy == RetrievalStrategy.HYBRID_PRIORITIZE:
        rag_generator_kwargs['prioritize_successful'] = prioritize_successful
        rag_generator_kwargs['prioritize_category'] = prioritize_category
    
    rag_generator = RAGPitchGenerator(**rag_generator_kwargs)
    
    # Get the context that will be used (before generation)
    retrieve_kwargs = {
        'product_data': test_example.input,
        'top_k': rag_top_k,
        'include_scores': True,
        'strategy': rag_strategy
    }
    
    # Add strategy-specific parameters
    if rag_strategy == RetrievalStrategy.HYBRID_FILTER:
        retrieve_kwargs['filter_successful_deals'] = filter_successful_deals
    elif rag_strategy == RetrievalStrategy.HYBRID_PRIORITIZE:
        retrieve_kwargs['prioritize_successful'] = prioritize_successful
        retrieve_kwargs['prioritize_category'] = prioritize_category
    
    similar_pitches = retriever.retrieve_similar_pitches(**retrieve_kwargs)
    
    context = retriever.format_context_for_prompt(similar_pitches, max_examples=rag_top_k)
    
    # Generate pitch with cache-busting config
    # Use strategy name in rollout_id to ensure different strategies don't cache each other
    import uuid
    cache_bust_config = {
        "rollout_id": f"{rag_strategy.value}_{uuid.uuid4().hex[:8]}",
        "temperature": 1.0  # Ensure non-deterministic generation
    }
    
    start_time = time.time()
    prediction = rag_generator.generate(test_example.input, config=cache_bust_config)
    generation_time = time.time() - start_time
    
    generated_pitch = prediction.pitch if hasattr(prediction, 'pitch') else str(prediction)
    
    # Evaluate
    evaluator = PitchEvaluator(evaluator_lm)
    pitch_facts = json.dumps(test_example.input, indent=2)
    
    eval_start = time.time()
    quality_score = evaluator.get_score(
        pitch_facts=pitch_facts,
        ground_truth_pitch=test_example.output,
        generated_pitch=generated_pitch
    )
    eval_time = time.time() - eval_start
    
    # Get full assessment if available
    try:
        full_assessment = evaluator.get_full_assessment(
            pitch_facts=pitch_facts,
            ground_truth_pitch=test_example.output,
            generated_pitch=generated_pitch
        )
    except:
        full_assessment = None
    
    return {
        'generated_pitch': generated_pitch,
        'context_used': context,
        'retrieved_examples': [
            {
                'id': p.get('metadata', {}).get('product_key', 'unknown'),
                'product_name': p.get('product_name', 'N/A'),
                'similarity': p.get('similarity_score', 0.0),
                'preview': p.get('pitch_text', '')[:200] + '...' if len(p.get('pitch_text', '')) > 200 else p.get('pitch_text', '')
            }
            for p in similar_pitches
        ],
        'quality_score': quality_score,
        'full_assessment': full_assessment,
        'generation_time': generation_time,
        'evaluation_time': eval_time,
        'prompt_length': len(context) + len(json.dumps(test_example.input, indent=2)),
        'output_length': len(generated_pitch)
    }


def generate_and_evaluate_knn(
    test_example: dspy.Example,
    trainset: List[dspy.Example],
    generator_lm,
    evaluator_lm,
    knn_k: int = 3
) -> Dict:
    """
    Generate pitch using KNNFewShot and evaluate it.
    
    Returns:
        Dict with generation results, prompts, outputs, and scores
    """
    from models import PitchGenerator, PitchEvaluator
    from models.generator import StructuredPitchProgram
    
    logger.info("\n[Generating with KNNFewShot...]")
    
    # Create base program
    base_program = StructuredPitchProgram()
    
    # Compile with KNNFewShot
    vectorizer = create_pitch_vectorizer(model_name="all-MiniLM-L6-v2")
    optimizer = dspy.KNNFewShot(
        k=knn_k,
        trainset=trainset,
        vectorizer=vectorizer
    )
    compiled_program = optimizer.compile(base_program)
    
    # Extract which examples will be used (by manually computing KNN)
    knn_ids, knn_info = extract_knn_selections(
        program=compiled_program,
        test_example=test_example,
        trainset=trainset,
        k=knn_k
    )
    
    # Get the actual examples that will be used
    knn_examples = [ex for ex in trainset if ex.id in knn_ids]
    
    # Format context that would be used (simulate what DSPy does)
    context_parts = []
    for i, ex in enumerate(knn_examples, 1):
        context_parts.append(f"Example {i}:")
        context_parts.append(f"Input: {json.dumps(ex.input, indent=2)}")
        context_parts.append(f"Output: {ex.output[:500]}...")
        context_parts.append("")
    knn_context = "\n".join(context_parts)
    
    # Generate pitch
    generator = PitchGenerator(generator_lm)
    generator.program = compiled_program  # Use compiled program
    
    start_time = time.time()
    prediction = generator.generate(test_example.input)
    generation_time = time.time() - start_time
    
    generated_pitch = prediction.pitch if hasattr(prediction, 'pitch') else str(prediction)
    
    # Evaluate
    evaluator = PitchEvaluator(evaluator_lm)
    pitch_facts = json.dumps(test_example.input, indent=2)
    
    eval_start = time.time()
    quality_score = evaluator.get_score(
        pitch_facts=pitch_facts,
        ground_truth_pitch=test_example.output,
        generated_pitch=generated_pitch
    )
    eval_time = time.time() - eval_start
    
    # Get full assessment if available
    try:
        full_assessment = evaluator.get_full_assessment(
            pitch_facts=pitch_facts,
            ground_truth_pitch=test_example.output,
            generated_pitch=generated_pitch
        )
    except:
        full_assessment = None
    
    return {
        'generated_pitch': generated_pitch,
        'context_used': knn_context,
        'selected_examples': [
            {
                'id': ex.id,
                'company': ex.input.get('company', 'N/A'),
                'problem_summary': ex.input.get('problem_summary', 'N/A')[:100],
                'output_preview': ex.output[:200] + '...' if len(ex.output) > 200 else ex.output
            }
            for ex in knn_examples
        ],
        'quality_score': quality_score,
        'full_assessment': full_assessment,
        'generation_time': generation_time,
        'evaluation_time': eval_time,
        'prompt_length': len(knn_context) + len(json.dumps(test_example.input, indent=2)),
        'output_length': len(generated_pitch)
    }


def compare_rag_vs_knn(
    test_example: dspy.Example,
    trainset: List[dspy.Example],
    generator_lm,
    evaluator_lm,
    rag_top_k: int = 5,
    knn_k: int = 3,
    rag_strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC
) -> Dict:
    """
    Compare RAG vs KNNFewShot for a single test example.
    
    Returns:
        Dict with comparison results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing RAG vs KNNFewShot for test example: {test_example.id}")
    logger.info(f"{'='*80}")
    
    # Extract RAG selections
    logger.info("\n[1] Extracting RAG selections...")
    from rag.retriever import PitchRetriever
    retriever = PitchRetriever(auto_index=True)
    
    rag_ids, rag_info = extract_rag_selections(
        retriever,
        test_example,
        top_k=rag_top_k,
        strategy=rag_strategy
    )
    
    logger.info(f"RAG selected {len(rag_ids)} examples:")
    for i, info in enumerate(rag_info, 1):
        logger.info(f"  {i}. {info['id']} (similarity: {info['similarity']:.4f})")
    
    # Extract KNNFewShot selections
    logger.info("\n[2] Extracting KNNFewShot selections...")
    knn_ids, knn_info = extract_knn_selections(
        program=None,  # Not needed for manual extraction
        test_example=test_example,
        trainset=trainset,
        k=knn_k
    )
    
    logger.info(f"KNNFewShot selected {len(knn_ids)} examples:")
    for i, info in enumerate(knn_info, 1):
        logger.info(f"  {i}. {info['id']} (similarity: {info['similarity']:.4f})")
    
    # Compute overlap
    logger.info("\n[3] Computing overlap...")
    overlap_stats = compute_overlap(rag_ids, knn_ids)
    
    logger.info(f"\nOverlap Statistics:")
    logger.info(f"  RAG selected: {overlap_stats['rag_selected']}")
    logger.info(f"  KNNFewShot selected: {overlap_stats['knn_selected']}")
    logger.info(f"  Overlap count: {overlap_stats['overlap_count']}")
    logger.info(f"  Jaccard similarity: {overlap_stats['jaccard_similarity']:.3f}")
    logger.info(f"  Overlap ratio: {overlap_stats['overlap_ratio']:.3f}")
    
    if overlap_stats['overlap_ids']:
        logger.info(f"  Overlapping IDs: {overlap_stats['overlap_ids']}")
    if overlap_stats['unique_to_rag']:
        logger.info(f"  Unique to RAG: {overlap_stats['unique_to_rag']}")
    if overlap_stats['unique_to_knn']:
        logger.info(f"  Unique to KNNFewShot: {overlap_stats['unique_to_knn']}")
    
    # Generate and evaluate with RAG
    logger.info("\n[4] Generating and evaluating with RAG...")
    rag_results = generate_and_evaluate_rag(
        test_example=test_example,
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        retriever=retriever,
        rag_top_k=rag_top_k,
        rag_strategy=rag_strategy
    )
    
    # Generate and evaluate with KNNFewShot
    logger.info("\n[5] Generating and evaluating with KNNFewShot...")
    knn_results = generate_and_evaluate_knn(
        test_example=test_example,
        trainset=trainset,
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        knn_k=knn_k
    )
    
    # Compile results
    results = {
        'test_example_id': test_example.id,
        'test_example_company': test_example.input.get('company', 'N/A'),
        'test_example_input': test_example.input,
        'ground_truth_pitch': test_example.output,
        'rag_selections': rag_info,
        'knn_selections': knn_info,
        'overlap_stats': overlap_stats,
        'rag_top_k': rag_top_k,
        'knn_k': knn_k,
        'rag_strategy': rag_strategy.value,
        'rag_generation': rag_results,
        'knn_generation': knn_results,
        'comparison': {
            'quality_score_diff': rag_results['quality_score'] - knn_results['quality_score'],
            'rag_score': rag_results['quality_score'],
            'knn_score': knn_results['quality_score'],
            'generation_time_diff': rag_results['generation_time'] - knn_results['generation_time'],
            'rag_generation_time': rag_results['generation_time'],
            'knn_generation_time': knn_results['generation_time'],
            'prompt_length_diff': rag_results['prompt_length'] - knn_results['prompt_length'],
            'rag_prompt_length': rag_results['prompt_length'],
            'knn_prompt_length': knn_results['prompt_length'],
            'output_length_diff': rag_results['output_length'] - knn_results['output_length'],
            'rag_output_length': rag_results['output_length'],
            'knn_output_length': knn_results['output_length']
        }
    }
    
    return results


def compare_all_rag_strategies_vs_knn_single(
    test_example: dspy.Example,
    trainset: List[dspy.Example],
    generator_lm,
    evaluator_lm,
    rag_top_k: int = 5,
    knn_k: int = 3
) -> Dict:
    """
    Compare all three RAG strategies against KNNFewShot for a single test example.
    
    Returns:
        Dict with results for each strategy
    """
    from rag.retriever import PitchRetriever
    retriever = PitchRetriever(auto_index=True)
    
    all_results = {
        'test_example_id': test_example.id,
        'test_example_company': test_example.input.get('company', 'N/A'),
        'test_example_input': test_example.input,
        'ground_truth_pitch': test_example.output,
        'strategies': {}
    }
    
    strategies_to_test = [
        ('pure_semantic', RetrievalStrategy.PURE_SEMANTIC, {'filter_successful_deals': False, 'prioritize_successful': False, 'prioritize_category': False}),
        ('hybrid_filter_deals', RetrievalStrategy.HYBRID_FILTER, {'filter_successful_deals': True, 'prioritize_successful': False, 'prioritize_category': False}),
        ('hybrid_prioritize', RetrievalStrategy.HYBRID_PRIORITIZE, {'filter_successful_deals': False, 'prioritize_successful': True, 'prioritize_category': True})
    ]
    
    for strategy_name, strategy_enum, extra_params in strategies_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing RAG Strategy: {strategy_name.upper()}")
        logger.info(f"{'='*80}")
        
        # Extract selections
        rag_ids, rag_info = extract_rag_selections(
            retriever,
            test_example,
            top_k=rag_top_k,
            strategy=strategy_enum,
            **extra_params
        )
        
        # Generate and evaluate with RAG
        rag_results = generate_and_evaluate_rag(
            test_example=test_example,
            generator_lm=generator_lm,
            evaluator_lm=evaluator_lm,
            retriever=retriever,
            rag_top_k=rag_top_k,
            rag_strategy=strategy_enum,
            **extra_params
        )
        
        all_results['strategies'][strategy_name] = {
            'rag_selections': rag_info,
            'rag_generation': rag_results
        }
    
    # Generate and evaluate with KNNFewShot (once, shared across all comparisons)
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing KNNFewShot")
    logger.info(f"{'='*80}")
    
    knn_ids, knn_info = extract_knn_selections(
        program=None,
        test_example=test_example,
        trainset=trainset,
        k=knn_k
    )
    
    knn_results = generate_and_evaluate_knn(
        test_example=test_example,
        trainset=trainset,
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        knn_k=knn_k
    )
    
    all_results['knn'] = {
        'knn_selections': knn_info,
        'knn_generation': knn_results
    }
    
    # Compute overlap for each strategy
    for strategy_name in all_results['strategies'].keys():
        rag_ids = [sel['id'] for sel in all_results['strategies'][strategy_name]['rag_selections']]
        overlap_stats = compute_overlap(rag_ids, knn_ids)
        all_results['strategies'][strategy_name]['overlap_stats'] = overlap_stats
        
        # Add comparison metrics
        rag_score = all_results['strategies'][strategy_name]['rag_generation']['quality_score']
        knn_score = knn_results['quality_score']
        
        all_results['strategies'][strategy_name]['comparison'] = {
            'quality_score_diff': rag_score - knn_score,
            'rag_score': rag_score,
            'knn_score': knn_score,
            'generation_time_diff': all_results['strategies'][strategy_name]['rag_generation']['generation_time'] - knn_results['generation_time'],
            'rag_generation_time': all_results['strategies'][strategy_name]['rag_generation']['generation_time'],
            'knn_generation_time': knn_results['generation_time']
        }
    
    return all_results


def compare_all_rag_strategies_vs_knn(
    test_examples: List[dspy.Example],
    trainset: List[dspy.Example],
    generator_lm,
    evaluator_lm,
    rag_top_k: int = 5,
    knn_k: int = 3
) -> Dict:
    """
    Compare all three RAG strategies against KNNFewShot across multiple test examples.
    Averages results for statistical reliability.
    
    Args:
        test_examples: List of test examples to evaluate
        trainset: Training examples for KNNFewShot
        generator_lm: Generator model
        evaluator_lm: Evaluator model
        rag_top_k: Number of RAG examples to retrieve
        knn_k: Number of KNN examples to retrieve
        
    Returns:
        Dict with averaged results and per-example breakdowns
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Comparing RAG vs KNNFewShot across {len(test_examples)} test examples")
    logger.info(f"{'='*80}")
    
    # Run comparison for each test example
    all_example_results = []
    for i, test_example in enumerate(test_examples, 1):
        logger.info(f"\n[Example {i}/{len(test_examples)}] {test_example.input.get('company', 'N/A')}")
        example_result = compare_all_rag_strategies_vs_knn_single(
            test_example=test_example,
            trainset=trainset,
            generator_lm=generator_lm,
            evaluator_lm=evaluator_lm,
            rag_top_k=rag_top_k,
            knn_k=knn_k
        )
        all_example_results.append(example_result)
    
    # Aggregate results across examples
    aggregated = {
        'num_examples': len(test_examples),
        'test_examples': [
            {
                'id': r['test_example_id'],
                'company': r['test_example_company']
            }
            for r in all_example_results
        ],
        'strategies': {},
        'knn': {}
    }
    
    # Average metrics for each strategy
    strategy_names = ['pure_semantic', 'hybrid_filter_deals', 'hybrid_prioritize']
    
    for strategy_name in strategy_names:
        # Collect scores and overlap stats
        rag_scores = []
        knn_scores = []
        jaccard_similarities = []
        overlap_counts = []
        rag_generation_times = []
        knn_generation_times = []
        
        for result in all_example_results:
            if strategy_name in result['strategies']:
                strategy_data = result['strategies'][strategy_name]
                rag_scores.append(strategy_data['rag_generation']['quality_score'])
                knn_scores.append(strategy_data['comparison']['knn_score'])
                jaccard_similarities.append(strategy_data['overlap_stats']['jaccard_similarity'])
                overlap_counts.append(strategy_data['overlap_stats']['overlap_count'])
                rag_generation_times.append(strategy_data['comparison']['rag_generation_time'])
                knn_generation_times.append(strategy_data['comparison']['knn_generation_time'])
        
        # Calculate averages
        aggregated['strategies'][strategy_name] = {
            'avg_rag_score': sum(rag_scores) / len(rag_scores) if rag_scores else 0.0,
            'avg_knn_score': sum(knn_scores) / len(knn_scores) if knn_scores else 0.0,
            'avg_score_diff': (sum(rag_scores) / len(rag_scores) - sum(knn_scores) / len(knn_scores)) if rag_scores and knn_scores else 0.0,
            'avg_jaccard_similarity': sum(jaccard_similarities) / len(jaccard_similarities) if jaccard_similarities else 0.0,
            'avg_overlap_count': sum(overlap_counts) / len(overlap_counts) if overlap_counts else 0.0,
            'avg_rag_generation_time': sum(rag_generation_times) / len(rag_generation_times) if rag_generation_times else 0.0,
            'avg_knn_generation_time': sum(knn_generation_times) / len(knn_generation_times) if knn_generation_times else 0.0,
            'per_example_results': [
                {
                    'example_id': r['test_example_id'],
                    'example_company': r['test_example_company'],
                    'rag_score': r['strategies'][strategy_name]['rag_generation']['quality_score'],
                    'knn_score': r['strategies'][strategy_name]['comparison']['knn_score'],
                    'jaccard_similarity': r['strategies'][strategy_name]['overlap_stats']['jaccard_similarity'],
                    'overlap_count': r['strategies'][strategy_name]['overlap_stats']['overlap_count']
                }
                for r in all_example_results
                if strategy_name in r['strategies']
            ]
        }
    
    # Average KNN metrics
    knn_scores = [r['knn']['knn_generation']['quality_score'] for r in all_example_results]
    knn_times = [r['knn']['knn_generation']['generation_time'] for r in all_example_results]
    
    aggregated['knn'] = {
        'avg_score': sum(knn_scores) / len(knn_scores) if knn_scores else 0.0,
        'avg_generation_time': sum(knn_times) / len(knn_times) if knn_times else 0.0
    }
    
    # Store detailed per-example results for reporting
    aggregated['detailed_results'] = all_example_results
    
    return aggregated


def main():
    """Main comparison script."""
    print("="*80)
    print("RAG vs KNNFewShot Comparison - All Strategies")
    print("Testing hypothesis: Are they mining the same gold mine?")
    print("="*80)
    
    # Setup models
    generator_lm = dspy.LM(
        "groq/llama-3.3-70b-versatile",  # Use 70B for better quality
        # "groq/llama-3.1-8b-instant",  # Fixed: llama-3.3-8b-instant doesn't exist, use 3.1 instead
        model_type="chat",
        api_key=GROQ_API_KEY,
        temperature=1.0
    )
    
    evaluator_lm = dspy.LM(
        "groq/openai/gpt-oss-120b",
        model_type="chat",
        api_key=GROQ_API_KEY,
        temperature=0.1
    )
    
    dspy.configure(lm=generator_lm)
    
    # Load data
    print("\n[1] Loading datasets...")
    data = load_and_prepare_data(
        train_jsonl_path="data/hf (new)/train.jsonl",
        test_jsonl_path="data/hf (new)/test.jsonl"
    )
    
    trainset = data["train"]
    testset = data["test"]
    
    print(f"  Train: {len(trainset)} examples")
    print(f"  Test: {len(testset)} examples")
    
    # Select test examples (same method as compare_rag_strategies.py)
    import random
    random.seed(42)  # Same seed for reproducibility
    test_examples = list(testset)
    random.shuffle(test_examples)
    test_size = 10  # Match compare_rag_strategies.py default
    selected_test_examples = test_examples[:test_size]
    
    print(f"\n[2] Selected {len(selected_test_examples)} test examples (seed=42):")
    for i, ex in enumerate(selected_test_examples[:5], 1):  # Show first 5
        print(f"  {i}. {ex.input.get('company', 'N/A')} (ID: {ex.id})")
    if len(selected_test_examples) > 5:
        print(f"  ... and {len(selected_test_examples) - 5} more")
    
    # Run comparison for all strategies across multiple examples
    print(f"\n[3] Running comprehensive comparison (all RAG strategies vs KNNFewShot)...")
    print(f"     Evaluating {len(selected_test_examples)} examples and averaging results...")
    results = compare_all_rag_strategies_vs_knn(
        test_examples=selected_test_examples,
        trainset=trainset,
        generator_lm=generator_lm,
        evaluator_lm=evaluator_lm,
        rag_top_k=5,
        knn_k=3
    )
    
    # Save results
    output_dir = Path("rag_comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"rag_all_strategies_vs_knn_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[4] Results saved to: {output_file}")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY - All RAG Strategies vs KNNFewShot")
    print("="*80)
    print(f"Test Examples: {results['num_examples']} (averaged)")
    print(f"KNNFewShot K: 3")
    print(f"RAG Top-K: 5")
    
    # Print comparison table (averaged results)
    print("\n" + "="*80)
    print("QUALITY SCORES COMPARISON (AVERAGED)")
    print("="*80)
    print(f"{'Strategy':<25} {'RAG Score':<12} {'KNN Score':<12} {'Difference':<12} {'Overlap':<10}")
    print("-" * 80)
    
    knn_score = results['knn']['avg_score']
    
    for strategy_name, strategy_data in results['strategies'].items():
        rag_score = strategy_data['avg_rag_score']
        diff = strategy_data['avg_score_diff']
        overlap = strategy_data['avg_jaccard_similarity']
        
        strategy_display = strategy_name.replace('_', ' ').title()
        print(f"{strategy_display:<25} {rag_score:<12.4f} {knn_score:<12.4f} {diff:+.4f} ({diff*100:+.1f}%){'':<4} {overlap:.3f}")
    
    # Print overlap statistics (averaged)
    print("\n" + "="*80)
    print("OVERLAP STATISTICS (Jaccard Similarity - AVERAGED)")
    print("="*80)
    for strategy_name, strategy_data in results['strategies'].items():
        strategy_display = strategy_name.replace('_', ' ').title()
        print(f"\n{strategy_display}:")
        print(f"  Avg Jaccard Similarity: {strategy_data['avg_jaccard_similarity']:.3f}")
        print(f"  Avg Overlap Count: {strategy_data['avg_overlap_count']:.2f}/3")
        print(f"  RAG Selected: 5 examples (per query)")
    
    # Print generation times (averaged)
    print("\n" + "="*80)
    print("GENERATION TIMES (AVERAGED)")
    print("="*80)
    print(f"{'Strategy':<25} {'RAG Time':<15} {'KNN Time':<15} {'Difference':<15}")
    print("-" * 80)
    knn_time = results['knn']['avg_generation_time']
    for strategy_name, strategy_data in results['strategies'].items():
        rag_time = strategy_data['avg_rag_generation_time']
        diff = rag_time - knn_time
        strategy_display = strategy_name.replace('_', ' ').title()
        print(f"{strategy_display:<25} {rag_time:<15.2f} {knn_time:<15.2f} {diff:+.2f}s")
    
    # Print per-example breakdown (first 3 examples)
    print("\n" + "="*80)
    print("PER-EXAMPLE BREAKDOWN (First 3 Examples)")
    print("="*80)
    
    for i, example_result in enumerate(results['detailed_results'][:3], 1):
        print(f"\nExample {i}: {example_result['test_example_company']}")
        print(f"  ID: {example_result['test_example_id']}")
        for strategy_name, strategy_data in example_result['strategies'].items():
            strategy_display = strategy_name.replace('_', ' ').title()
            rag_score = strategy_data['rag_generation']['quality_score']
            knn_score = strategy_data['comparison']['knn_score']
            overlap = strategy_data['overlap_stats']['jaccard_similarity']
            print(f"    {strategy_display}: RAG={rag_score:.3f}, KNN={knn_score:.3f}, Overlap={overlap:.3f}")
    
    # Save detailed markdown report
    markdown_file = output_dir / f"rag_all_strategies_vs_knn_detailed_{timestamp}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(f"# RAG All Strategies vs KNNFewShot Comprehensive Comparison\n\n")
        f.write(f"**Test Examples**: {results['num_examples']} examples (averaged)\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Test Examples\n\n")
        for i, ex_info in enumerate(results['test_examples'], 1):
            f.write(f"{i}. {ex_info['company']} (ID: {ex_info['id']})\n")
        f.write("\n")
        
        f.write("## Summary Comparison Table (Averaged)\n\n")
        f.write(f"| Strategy | RAG Score | KNN Score | Difference | Overlap (Jaccard) |\n")
        f.write(f"|----------|-----------|-----------|------------|-------------------|\n")
        
        knn_score = results['knn']['avg_score']
        for strategy_name, strategy_data in results['strategies'].items():
            rag_score = strategy_data['avg_rag_score']
            diff = strategy_data['avg_score_diff']
            overlap = strategy_data['avg_jaccard_similarity']
            strategy_display = strategy_name.replace('_', ' ').title()
            f.write(f"| {strategy_display} | {rag_score:.4f} | {knn_score:.4f} | {diff:+.4f} ({diff*100:+.1f}%) | {overlap:.3f} |\n")
        
        f.write("\n## Averaged Results by Strategy\n\n")
        
        for strategy_name, strategy_data in results['strategies'].items():
            strategy_display = strategy_name.replace('_', ' ').title()
            f.write(f"### {strategy_display}\n\n")
            
            f.write(f"#### Averaged Metrics\n\n")
            f.write(f"- **Avg RAG Score**: {strategy_data['avg_rag_score']:.4f}\n")
            f.write(f"- **Avg KNN Score**: {strategy_data['avg_knn_score']:.4f}\n")
            f.write(f"- **Avg Score Difference**: {strategy_data['avg_score_diff']:+.4f}\n")
            f.write(f"- **Avg Jaccard Similarity**: {strategy_data['avg_jaccard_similarity']:.3f}\n")
            f.write(f"- **Avg Overlap Count**: {strategy_data['avg_overlap_count']:.2f}/3\n")
            f.write(f"- **Avg RAG Generation Time**: {strategy_data['avg_rag_generation_time']:.2f}s\n")
            f.write(f"- **Avg KNN Generation Time**: {strategy_data['avg_knn_generation_time']:.2f}s\n\n")
            
            f.write(f"#### Per-Example Results\n\n")
            f.write(f"| Example | Company | RAG Score | KNN Score | Overlap |\n")
            f.write(f"|---------|---------|-----------|-----------|--------|\n")
            for ex_result in strategy_data['per_example_results']:
                f.write(f"| {ex_result['example_id'][:8]}... | {ex_result['example_company']} | {ex_result['rag_score']:.3f} | {ex_result['knn_score']:.3f} | {ex_result['jaccard_similarity']:.3f} |\n")
            f.write("\n")
        
        f.write("## KNNFewShot Averaged Results\n\n")
        f.write(f"- **Avg Score**: {results['knn']['avg_score']:.4f}\n")
        f.write(f"- **Avg Generation Time**: {results['knn']['avg_generation_time']:.2f}s\n\n")
        
        # Show detailed results for first example
        if results['detailed_results']:
            first_example = results['detailed_results'][0]
            f.write("## Detailed Analysis: First Example\n\n")
            f.write(f"**Example**: {first_example['test_example_company']} (ID: {first_example['test_example_id']})\n\n")
            f.write("### Test Example Input\n\n")
            f.write("```json\n")
            f.write(json.dumps(first_example['test_example_input'], indent=2))
            f.write("\n```\n\n")
            
            f.write("### RAG Selections (Pure Semantic)\n\n")
            if 'pure_semantic' in first_example['strategies']:
                for i, sel in enumerate(first_example['strategies']['pure_semantic']['rag_selections'], 1):
                    f.write(f"{i}. **{sel.get('product_name', 'N/A')}** (ID: {sel['id']})\n")
                    f.write(f"   - Similarity: {sel['similarity']:.4f}\n")
                    f.write(f"   - Category: {sel.get('category', 'N/A')}\n\n")
            
            f.write("### KNNFewShot Selections\n\n")
            for i, sel in enumerate(first_example['knn']['knn_selections'], 1):
                f.write(f"{i}. **{sel.get('company', 'N/A')}** (ID: {sel['id']})\n")
                f.write(f"   - Similarity: {sel['similarity']:.4f}\n\n")
            
            f.write("### Ground Truth Pitch\n\n")
            f.write(f"```\n{first_example['ground_truth_pitch']}\n```\n\n")
    
    print(f"\n[5] Detailed markdown report saved to: {markdown_file}")
    
    return results


if __name__ == "__main__":
    results = main()

