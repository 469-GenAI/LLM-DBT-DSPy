"""
RAG-enhanced pitch generator using retrieved examples as context.
"""

import json
import dspy
from typing import Dict, Optional
import logging
from .generator import PitchGenerator, StructuredPitchProgram
from utils import PitchInput, format_pitch_input

# Import RAG components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.retriever import PitchRetriever, RetrievalStrategy

logger = logging.getLogger(__name__)


class RAGPitchGenerationSig(dspy.Signature):
    """
    Generate a compelling Shark Tank pitch using retrieved examples as context.
    
    The pitch should:
    - Learn from the style and structure of the example pitches
    - Start with an engaging introduction of the founders and company
    - Present the investment ask clearly
    - Tell a story about the problem from the customer's perspective
    - Introduce the solution with compelling details
    - End with a strong call to action for the Sharks
    """
    
    context: str = dspy.InputField(
        desc="Examples of successful Shark Tank pitches for similar products. "
             "Use these as inspiration for style, structure, and narrative flow."
    )
    
    pitch_data: str = dspy.InputField(
        desc="Structured pitch data including company, founders, problem story, solution, and investment ask"
    )
    
    pitch: str = dspy.OutputField(
        desc="A compelling, narrative pitch in the style of Shark Tank presentations. "
             "Should be conversational, engaging, and tell a complete story from problem to solution. "
             "Incorporate the narrative style and energy from the example pitches while staying true to this product's unique value."
    )


class RAGStructuredPitchProgram(dspy.Module):
    """
    DSPy module for generating pitches with RAG context.
    """
    
    def __init__(
        self, 
        retriever: Optional[PitchRetriever] = None, 
        top_k: int = 5,
        filter_successful_deals: bool = False,
        filter_category: Optional[str] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC,
        prioritize_successful: bool = True,
        prioritize_category: bool = True
    ):
        """
        Initialize RAG-enhanced pitch program.
        
        Args:
            retriever: PitchRetriever instance. If None, creates new one.
            top_k: Number of similar pitches to retrieve as context
            filter_successful_deals: If True, only retrieve pitches that closed deals (HYBRID_FILTER only)
            filter_category: Optional category to filter by (HYBRID_FILTER only)
            retrieval_strategy: Retrieval strategy (PURE_SEMANTIC, HYBRID_FILTER, HYBRID_PRIORITIZE)
            prioritize_successful: If True, prioritize successful deals (HYBRID_PRIORITIZE only)
            prioritize_category: If True, prioritize same category (HYBRID_PRIORITIZE only)
        """
        super().__init__()
        self.generate_pitch = dspy.ChainOfThought(RAGPitchGenerationSig)
        self.retriever = retriever or PitchRetriever(auto_index=True)
        self.top_k = top_k
        self.filter_successful_deals = filter_successful_deals
        self.filter_category = filter_category
        self.retrieval_strategy = retrieval_strategy
        self.prioritize_successful = prioritize_successful
        self.prioritize_category = prioritize_category
        
        logger.info(f"RAGStructuredPitchProgram initialized")
        logger.info(f"  Strategy: {retrieval_strategy.value}")
        logger.info(f"  Top-k: {top_k}")
    
    def forward(self, input: dict):
        """
        Generate a pitch with RAG context.
        
        Args:
            input: Dictionary containing structured pitch data
            
        Returns:
            dspy.Prediction with pitch field
        """
        # Format input data
        try:
            pitch_input = PitchInput(**input)
            formatted_input = format_pitch_input(pitch_input)
        except Exception as e:
            logger.warning(f"Could not parse input as PitchInput: {e}")
            formatted_input = json.dumps(input, indent=2)
        
        # Retrieve similar pitches using selected strategy
        similar_pitches = self.retriever.retrieve_similar_pitches(
            product_data=input,
            top_k=self.top_k,
            include_scores=True,
            filter_successful_deals=self.filter_successful_deals,
            filter_category=self.filter_category,
            strategy=self.retrieval_strategy,
            prioritize_successful=self.prioritize_successful,
            prioritize_category=self.prioritize_category
        )
        
        # Format as context
        context = self.retriever.format_context_for_prompt(
            similar_pitches,
            max_examples=self.top_k
        )
        
        # Generate pitch with context
        prediction = self.generate_pitch(
            context=context,
            pitch_data=formatted_input
        )
        
        # Add metadata about retrieval
        if hasattr(prediction, '_metadata'):
            prediction._metadata = {}
        else:
            prediction._metadata = {}
        
        prediction._metadata['retrieved_pitches'] = [
            {
                'product': p['product_name'],
                'similarity': p.get('similarity_score', 0)
            }
            for p in similar_pitches
        ]
        prediction._metadata['num_retrieved'] = len(similar_pitches)
        
        return prediction


class RAGPitchGenerator(PitchGenerator):
    """
    RAG-enhanced pitch generator.
    
    Extends PitchGenerator to use retrieved pitch examples as context.
    Maintains model separation (70B for generation, 120B for evaluation).
    """
    
    def __init__(
        self, 
        lm,
        retriever: Optional[PitchRetriever] = None,
        top_k: int = 5,
        filter_successful_deals: bool = False,
        filter_category: Optional[str] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC,
        prioritize_successful: bool = True,
        prioritize_category: bool = True
    ):
        """
        Initialize RAG-enhanced generator.
        
        Args:
            lm: The dspy.LM instance for pitch generation
            retriever: PitchRetriever instance. If None, creates new one.
            top_k: Number of similar pitches to retrieve
            filter_successful_deals: If True, only retrieve pitches that closed deals (HYBRID_FILTER only)
            filter_category: Optional category to filter by (HYBRID_FILTER only)
            retrieval_strategy: Retrieval strategy (PURE_SEMANTIC, HYBRID_FILTER, HYBRID_PRIORITIZE)
            prioritize_successful: If True, prioritize successful deals (HYBRID_PRIORITIZE only)
            prioritize_category: If True, prioritize same category (HYBRID_PRIORITIZE only)
        """
        # Don't call super().__init__ directly, set attributes manually
        self.lm = lm
        self.filter_successful_deals = filter_successful_deals
        self.filter_category = filter_category
        self.retrieval_strategy = retrieval_strategy
        self.program = RAGStructuredPitchProgram(
            retriever=retriever, 
            top_k=top_k,
            filter_successful_deals=filter_successful_deals,
            filter_category=filter_category,
            retrieval_strategy=retrieval_strategy,
            prioritize_successful=prioritize_successful,
            prioritize_category=prioritize_category
        )
        self.top_k = top_k
        
        print(f"✓ RAGPitchGenerator initialized")
        print(f"  Model: {self.lm.model}")
        print(f"  Strategy: {retrieval_strategy.value}")
        print(f"  Top-k retrieval: {top_k}")
        
        # Check retriever status
        stats = self.program.retriever.get_retriever_stats()
        print(f"  Vector store: {stats['total_documents']} pitches indexed")
    
    def generate(self, input_data: dict):
        """
        Generate a pitch with RAG context using the assigned generator model.
        
        Args:
            input_data: Dictionary containing structured pitch data
            
        Returns:
            dspy.Prediction with pitch field and retrieval metadata
        """
        with dspy.context(lm=self.lm):
            prediction = self.program(input=input_data)
        
        return prediction
    
    def __call__(self, input_data: dict):
        """Allow direct calling: generator(data)."""
        return self.generate(input_data)
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval system."""
        return self.program.retriever.get_retriever_stats()


if __name__ == "__main__":
    # Test RAG generator
    logging.basicConfig(level=logging.INFO)
    
    print("Testing RAGPitchGenerator...")
    print("="*70)
    
    # Create test LM (you'd use your actual model here)
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not found. Set it to test generation.")
        print("   Testing initialization only...")
        
        # Test without actual LM
        class MockLM:
            model = "mock-model-for-testing"
        
        generator = RAGPitchGenerator(MockLM(), top_k=3)
        print("\n✓ RAGPitchGenerator initialized successfully")
        
        stats = generator.get_retrieval_stats()
        print("\nRetrieval stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        # Full test with actual model
        generator_lm = dspy.LM(
            "groq/llama-3.3-70b-versatile", 
            model_type="chat", 
            api_key=GROQ_API_KEY
        )
        
        generator = RAGPitchGenerator(generator_lm, top_k=3)
        
        # Test with sample product
        test_product = {
            'product_description': {
                'name': 'EcoBottle',
                'description': 'Smart reusable water bottle that tracks hydration',
                'category': 'Health & Wellness'
            },
            'facts': [
                'Sold 50,000 units in first year',
                '$2.5M in revenue',
                'Featured in Men\'s Health magazine',
                'Partnered with major gyms nationwide'
            ],
            'company_name': 'EcoBottle Inc',
            'founder': 'Jane Smith',
            'ask': '$500,000 for 10% equity'
        }
        
        print("\nGenerating pitch...")
        prediction = generator.generate(test_product)
        
        print("\n" + "="*70)
        print("GENERATED PITCH:")
        print("="*70)
        print(prediction.pitch)
        
        print("\n" + "="*70)
        print("RETRIEVAL METADATA:")
        print("="*70)
        if hasattr(prediction, '_metadata'):
            print(f"Retrieved {prediction._metadata['num_retrieved']} similar pitches:")
            for p in prediction._metadata['retrieved_pitches']:
                print(f"  - {p['product']} (similarity: {p['similarity']:.4f})")

