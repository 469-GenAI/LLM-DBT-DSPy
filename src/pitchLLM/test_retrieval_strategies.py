"""
Quick test script to verify all three RAG strategies work correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pitchLLM.rag.retriever import PitchRetriever, RetrievalStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_retrieval_strategies():
    """Test all three retrieval strategies."""
    
    print("="*80)
    print("Testing RAG Retrieval Strategies")
    print("="*80)
    
    # Initialize retriever
    retriever = PitchRetriever(auto_index=True)
    
    # Test product
    test_product = {
        'product_description': {
            'name': 'Smart Fitness Tracker',
            'description': 'Wearable device that tracks health metrics and provides AI insights',
            'category': 'Health & Fitness'
        },
        'facts': [
            'Uses AI to provide personalized health insights',
            '$500,000 in sales over 12 months',
            'Featured in TechCrunch and Wired'
        ]
    }
    
    strategies = [
        ("Pure Semantic", RetrievalStrategy.PURE_SEMANTIC),
        ("Hybrid Filter (Deals)", RetrievalStrategy.HYBRID_FILTER, {"filter_successful_deals": True}),
        ("Hybrid Prioritize", RetrievalStrategy.HYBRID_PRIORITIZE)
    ]
    
    for strategy_info in strategies:
        strategy_name = strategy_info[0]
        strategy = strategy_info[1]
        kwargs = strategy_info[2] if len(strategy_info) > 2 else {}
        
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*80}")
        
        results = retriever.retrieve_similar_pitches(
            product_data=test_product,
            top_k=5,
            include_scores=True,
            strategy=strategy,
            **kwargs
        )
        
        print(f"\nRetrieved {len(results)} pitches:\n")
        for i, pitch in enumerate(results, 1):
            print(f"{i}. {pitch['product_name']}")
            print(f"   Similarity: {pitch.get('similarity_score', 'N/A'):.4f}")
            if 'priority' in pitch:
                print(f"   Priority: {pitch['priority']}")
            if 'metadata' in pitch:
                metadata = pitch['metadata']
                print(f"   Has Deal: {metadata.get('has_deal', False)}")
                print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   Preview: {pitch['pitch_text'][:150]}...")
            print()
    
    print("\n" + "="*80)
    print("✓ All strategies tested successfully!")
    print("="*80)


if __name__ == "__main__":
    test_retrieval_strategies()

