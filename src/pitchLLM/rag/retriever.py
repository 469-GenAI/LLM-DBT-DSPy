"""
Pitch retriever for finding similar pitches to use as context.

Supports three retrieval strategies:
1. Pure Semantic: Pure similarity-based retrieval (no filters/prioritization)
2. Hybrid Filtering: Semantic search + metadata filtering
3. Hybrid Prioritization: Semantic search + rule-based prioritization
"""

import json
from typing import List, Dict, Optional
from enum import Enum
import logging
import sys
from pathlib import Path

# Handle imports when running directly vs as module
try:
    from .vector_store import PitchVectorStore
    from .data_indexer import load_and_prepare_all
except ImportError:
    # Running directly - add paths and use absolute imports
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from src.pitchLLM.rag.vector_store import PitchVectorStore
    from src.pitchLLM.rag.data_indexer import load_and_prepare_all

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """
    Different retrieval strategies for RAG.
    
    - PURE_SEMANTIC: Pure similarity-based retrieval (baseline)
    - HYBRID_FILTER: Semantic + metadata filtering (current approach)
    - HYBRID_PRIORITIZE: Semantic + rule-based prioritization (recommended)
    """
    PURE_SEMANTIC = "pure_semantic"
    HYBRID_FILTER = "hybrid_filter"
    HYBRID_PRIORITIZE = "hybrid_prioritize"


class PitchRetriever:
    """
    Retrieve similar pitch examples for RAG-based generation.
    """
    
    def __init__(
        self, 
        vector_store: Optional[PitchVectorStore] = None,
        auto_index: bool = True
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Existing PitchVectorStore instance. 
                         If None, creates new one.
            auto_index: If True and vector store is empty, 
                       automatically load and index transcripts
        """
        self.vector_store = vector_store or PitchVectorStore()
        
        # Check if we need to index
        stats = self.vector_store.get_stats()
        if auto_index and stats['total_documents'] == 0:
            logger.info("Vector store is empty. Auto-indexing transcripts...")
            # Try to find category mapping at default location
            from pathlib import Path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            default_mapping = project_root / "data" / "hf (new)" / "category_mapping.json"
            category_path = str(default_mapping) if default_mapping.exists() else None
            if category_path:
                logger.info(f"Found category mapping at default location: {category_path}")
            self.index_all_transcripts(category_mapping_path=category_path)
    
    def index_all_transcripts(self, force_reload: bool = False, use_jsonl: bool = True, jsonl_path: str = None, category_mapping_path: str = None):
        """
        Load and index transcripts from train.jsonl.
        
        Args:
            force_reload: If True, delete existing collection and reload
            use_jsonl: If True, load from train.jsonl (default: True)
            jsonl_path: Path to train.jsonl file (if None, uses default location)
            category_mapping_path: Path to category_mapping.json (if None, uses default location)
        """
        logger.info("Loading transcripts from train.jsonl...")
        documents = load_and_prepare_all(
            use_jsonl=use_jsonl, 
            jsonl_path=jsonl_path,
            category_mapping_path=category_mapping_path
        )
        
        if force_reload:
            logger.info("Force reload: resetting collection...")
            self.vector_store.reset_and_reload(documents)
        else:
            logger.info(f"Adding {len(documents)} documents...")
            self.vector_store.add_documents(documents)
        
        stats = self.vector_store.get_stats()
        logger.info(f"Indexing complete. Total documents: {stats['total_documents']}")
    
    def retrieve_similar_pitches(
        self, 
        product_data: Dict,
        top_k: int = 5,
        include_scores: bool = False,
        filter_successful_deals: bool = False,
        filter_category: Optional[str] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.PURE_SEMANTIC,
        prioritize_successful: bool = True,
        prioritize_category: bool = True
    ) -> List[Dict]:
        """
        Retrieve similar pitches for a given product.
        
        Supports three retrieval strategies:
        1. PURE_SEMANTIC: Pure similarity-based (no filters/prioritization)
        2. HYBRID_FILTER: Semantic + metadata filtering
        3. HYBRID_PRIORITIZE: Semantic + rule-based prioritization
        
        Args:
            product_data: Dict containing product info (facts, description, etc.)
            top_k: Number of similar pitches to retrieve
            include_scores: If True, include similarity scores
            filter_successful_deals: If True, only retrieve pitches that closed deals (HYBRID_FILTER only)
            filter_category: Optional category to filter by (HYBRID_FILTER only)
            strategy: Retrieval strategy to use
            prioritize_successful: If True, prioritize successful deals (HYBRID_PRIORITIZE only)
            prioritize_category: If True, prioritize same category (HYBRID_PRIORITIZE only)
            
        Returns:
            List of dicts with:
            - pitch_text: The retrieved pitch
            - product_name: Product name
            - metadata: Additional info
            - score: Similarity score (if include_scores=True)
            - priority: Priority level (if HYBRID_PRIORITIZE)
        """
        # Extract product category for prioritization
        product_category = None
        if 'product_description' in product_data:
            desc = product_data['product_description']
            if isinstance(desc, dict):
                product_category = desc.get('category')
        elif 'category' in product_data:
            product_category = product_data['category']
        
        # Route to appropriate retrieval method
        # Handle enum comparison robustly (works even with different import paths)
        # Primary method: use the enum's value attribute
        strategy_value_str = None
        strategy_name = None
        
        try:
            # Try to get the value directly
            if hasattr(strategy, 'value'):
                val = strategy.value
                # Handle case where value might be enum object itself
                if isinstance(val, RetrievalStrategy):
                    strategy_value_str = str(val.value).lower()
                else:
                    strategy_value_str = str(val).lower()
            
            # Also get the name
            if hasattr(strategy, 'name'):
                strategy_name = strategy.name
        except Exception as e:
            logger.debug(f"Error extracting enum attributes: {e}")
            strategy_value_str = str(strategy).lower()
        
        # Fallback: use string representation
        if strategy_value_str is None:
            strategy_str = str(strategy).lower()
            # Extract value from string representation if possible
            if "pure_semantic" in strategy_str:
                strategy_value_str = "pure_semantic"
            elif "hybrid_filter" in strategy_str:
                strategy_value_str = "hybrid_filter"
            elif "hybrid_prioritize" in strategy_str:
                strategy_value_str = "hybrid_prioritize"
            else:
                strategy_value_str = strategy_str
        
        # Compare using string value (most reliable across import paths)
        if strategy_value_str == "pure_semantic" or strategy_name == "PURE_SEMANTIC":
            return self._retrieve_pure_semantic(
                product_data, top_k, include_scores
            )
        elif strategy_value_str == "hybrid_filter" or strategy_name == "HYBRID_FILTER":
            return self._retrieve_hybrid_filter(
                product_data, top_k, include_scores,
                filter_successful_deals, filter_category
            )
        elif strategy_value_str == "hybrid_prioritize" or strategy_name == "HYBRID_PRIORITIZE":
            return self._retrieve_hybrid_prioritize(
                product_data, top_k, include_scores,
                product_category, prioritize_successful, prioritize_category
            )
        else:
            # Last resort: try direct enum comparison
            try:
                if strategy == RetrievalStrategy.PURE_SEMANTIC:
                    return self._retrieve_pure_semantic(product_data, top_k, include_scores)
                elif strategy == RetrievalStrategy.HYBRID_FILTER:
                    return self._retrieve_hybrid_filter(
                        product_data, top_k, include_scores,
                        filter_successful_deals, filter_category
                    )
                elif strategy == RetrievalStrategy.HYBRID_PRIORITIZE:
                    return self._retrieve_hybrid_prioritize(
                        product_data, top_k, include_scores,
                        product_category, prioritize_successful, prioritize_category
                    )
            except:
                pass
            
            raise ValueError(
                f"Unknown strategy: {strategy} "
                f"(value_str: {strategy_value_str}, name: {strategy_name}, "
                f"type: {type(strategy)}, str: {str(strategy)})"
            )
    
    def _retrieve_pure_semantic(
        self,
        product_data: Dict,
        top_k: int,
        include_scores: bool
    ) -> List[Dict]:
        """
        Pure semantic similarity retrieval (baseline approach).
        
        No filters, no prioritization - pure similarity ranking.
        """
        query_text = self._create_query_from_product(product_data)
        
        # Pure semantic search - no filters
        results = self.vector_store.query(query_text, n_results=top_k)
        
        similar_pitches = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            
            pitch_dict = {
                'pitch_text': results['documents'][i],
                'product_name': metadata.get('product_name', results['ids'][i]),
                'metadata': metadata,
                'strategy': 'pure_semantic'
            }
            
            if include_scores:
                distance = results['distances'][i]
                similarity = 1.0 / (1.0 + distance)
                pitch_dict['similarity_score'] = similarity
            
            similar_pitches.append(pitch_dict)
        
        logger.info(f"Pure semantic retrieval: {len(similar_pitches)} pitches")
        return similar_pitches
    
    def _retrieve_hybrid_filter(
        self,
        product_data: Dict,
        top_k: int,
        include_scores: bool,
        filter_successful_deals: bool,
        filter_category: Optional[str]
    ) -> List[Dict]:
        """
        Hybrid retrieval with metadata filtering (current approach).
        
        Semantic search + metadata filters (deal success, category).
        """
        query_text = self._create_query_from_product(product_data)
        
        # Build metadata filter
        where_filter = None
        if filter_successful_deals:
            where_filter = {"has_deal": True}
        elif filter_category:
            where_filter = {"category": filter_category}
        
        # Query vector store (retrieve more if filtering to ensure we get top_k after filtering)
        query_n = top_k * 3 if (filter_successful_deals or filter_category) else top_k
        results = self.vector_store.query(query_text, n_results=query_n, where=where_filter)
        
        # Format results and apply post-filtering if needed
        similar_pitches = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            
            # Apply category filter if specified (in case metadata filter didn't work)
            if filter_category and metadata.get('category') != filter_category:
                continue
            
            # Apply deal filter if specified
            if filter_successful_deals and not metadata.get('has_deal', False):
                continue
            
            pitch_dict = {
                'pitch_text': results['documents'][i],
                'product_name': metadata.get('product_name', results['ids'][i]),
                'metadata': metadata,
                'strategy': 'hybrid_filter'
            }
            
            if include_scores:
                distance = results['distances'][i]
                similarity = 1.0 / (1.0 + distance)
                pitch_dict['similarity_score'] = similarity
            
            similar_pitches.append(pitch_dict)
            
            # Stop if we have enough
            if len(similar_pitches) >= top_k:
                break
        
        logger.info(f"Hybrid filter retrieval: {len(similar_pitches)} pitches")
        if filter_successful_deals:
            logger.info(f"  (filtered to successful deals only)")
        if filter_category:
            logger.info(f"  (filtered to category: {filter_category})")
        return similar_pitches
    
    def _retrieve_hybrid_prioritize(
        self,
        product_data: Dict,
        top_k: int,
        include_scores: bool,
        product_category: Optional[str],
        prioritize_successful: bool,
        prioritize_category: bool
    ) -> List[Dict]:
        """
        Hybrid retrieval with rule-based prioritization (recommended approach).
        
        Semantic search finds candidates, then prioritizes by:
        Priority 1: Successful deals + Same category (highest)
        Priority 2: Successful deals + Different category
        Priority 3: Unsuccessful deals + Same category
        Priority 4: Unsuccessful deals + Different category (lowest)
        
        Within each priority level, maintains semantic similarity order.
        """
        query_text = self._create_query_from_product(product_data)
        
        # Retrieve more candidates for prioritization
        candidate_multiplier = 4  # Get 4x candidates for better prioritization
        results = self.vector_store.query(query_text, n_results=top_k * candidate_multiplier)
        
        # Prioritize candidates
        prioritized = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            has_deal = metadata.get('has_deal', False)
            candidate_category = metadata.get('category', '')
            
            # Determine category match
            category_match = False
            if prioritize_category and product_category and candidate_category:
                category_match = (candidate_category.lower() == product_category.lower())
            
            # Assign priority score (lower = higher priority)
            if prioritize_successful and prioritize_category:
                if has_deal and category_match:
                    priority = 1  # Highest priority
                elif has_deal:
                    priority = 2
                elif category_match:
                    priority = 3
                else:
                    priority = 4  # Lowest priority
            elif prioritize_successful:
                # Only prioritize by deal success
                priority = 1 if has_deal else 2
            elif prioritize_category:
                # Only prioritize by category
                priority = 1 if category_match else 2
            else:
                # No prioritization (fallback to pure semantic)
                priority = 1
            
            # Convert distance to similarity
            distance = results['distances'][i]
            similarity = 1.0 / (1.0 + distance)
            
            prioritized.append({
                'priority': priority,
                'similarity': similarity,
                'distance': distance,
                'pitch_text': results['documents'][i],
                'product_name': metadata.get('product_name', results['ids'][i]),
                'metadata': metadata,
                'has_deal': has_deal,
                'category_match': category_match,
                'strategy': 'hybrid_prioritize'
            })
        
        # Sort by priority (ascending), then by similarity (descending) within same priority
        prioritized.sort(key=lambda x: (x['priority'], -x['similarity']))
        
        # Format results
        similar_pitches = []
        for item in prioritized[:top_k]:
            pitch_dict = {
                'pitch_text': item['pitch_text'],
                'product_name': item['product_name'],
                'metadata': item['metadata'],
                'priority': item['priority'],
                'strategy': 'hybrid_prioritize'
            }
            
            if include_scores:
                pitch_dict['similarity_score'] = item['similarity']
            
            similar_pitches.append(pitch_dict)
        
        # Log prioritization stats
        priority_counts = {}
        for item in similar_pitches:
            p = item['priority']
            priority_counts[p] = priority_counts.get(p, 0) + 1
        
        logger.info(f"Hybrid prioritize retrieval: {len(similar_pitches)} pitches")
        logger.info(f"  Priority distribution: {priority_counts}")
        
        return similar_pitches
    
    def _create_query_from_product(self, product_data: Dict) -> str:
        """
        Create search query from product data.
        
        Extracts key information to search for similar pitches.
        
        Args:
            product_data: Product information dict
            
        Returns:
            Query string for vector search
        """
        query_parts = []
        
        # Add product description/name
        if 'product_description' in product_data:
            desc = product_data['product_description']
            if isinstance(desc, dict):
                query_parts.append(desc.get('name', ''))
                query_parts.append(desc.get('description', ''))
                query_parts.append(desc.get('category', ''))
            else:
                query_parts.append(str(desc))
        
        # Add facts
        if 'facts' in product_data:
            facts = product_data['facts']
            if isinstance(facts, list):
                query_parts.extend(facts[:5])  # Use first 5 facts
            elif isinstance(facts, dict):
                query_parts.extend(list(facts.values())[:5])
            else:
                query_parts.append(str(facts))
        
        # Add company/founder info
        for key in ['company_name', 'founder', 'problem', 'solution']:
            if key in product_data:
                query_parts.append(str(product_data[key]))
        
        # Combine into query
        query = " ".join(str(p) for p in query_parts if p)
        
        # Fallback to JSON dump if nothing extracted
        if not query.strip():
            query = json.dumps(product_data, indent=2)
        
        return query
    
    def format_context_for_prompt(
        self, 
        similar_pitches: List[Dict],
        max_examples: int = 5
    ) -> str:
        """
        Format retrieved pitches as context for prompt.
        
        Args:
            similar_pitches: List of retrieved pitch dicts
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted string with example pitches
        """
        context_parts = [
            "Here are some examples of successful Shark Tank pitches for similar products:\n"
        ]
        
        for i, pitch in enumerate(similar_pitches[:max_examples], 1):
            product_name = pitch['product_name']
            pitch_text = pitch['pitch_text']
            
            # Truncate very long pitches
            if len(pitch_text) > 1500:
                pitch_text = pitch_text[:1500] + "..."
            
            context_parts.append(f"\n--- Example {i}: {product_name} ---")
            context_parts.append(pitch_text)
            context_parts.append("\n")
        
        return "\n".join(context_parts)
    
    def get_retriever_stats(self) -> Dict:
        """Get statistics about the retriever."""
        store_stats = self.vector_store.get_stats()
        return {
            **store_stats,
            'retriever_ready': store_stats['total_documents'] > 0
        }


if __name__ == "__main__":
    # Fix imports when running directly
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Test the retriever
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing PitchRetriever...")
    retriever = PitchRetriever(auto_index=True)
    
    print("\nRetriever stats:")
    stats = retriever.get_retriever_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if stats['retriever_ready']:
        print("\nTesting retrieval...")
        
        # Load actual test product from test.jsonl
        test_jsonl_path = Path(project_root) / "data" / "hf (new)" / "test.jsonl"
        if test_jsonl_path.exists():
            import json
            import random
            random.seed(42)  # For reproducibility
            
            with open(test_jsonl_path, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f if line.strip()]
            
            if test_data:
                test_item = random.choice(test_data)
                test_product = test_item.get('input', {})
                print(f"\nUsing test product from test.jsonl:")
                print(f"  ID: {test_item.get('id', 'N/A')}")
                print(f"  Company: {test_product.get('company', 'N/A')}")
            else:
                print("⚠️  test.jsonl is empty, go fix it.")
      

        
        # Retrieve similar pitches
        similar = retriever.retrieve_similar_pitches(
            test_product, 
            top_k=3,
            include_scores=True
        )
        
        print(f"\nFound {len(similar)} similar pitches:")
        for i, pitch in enumerate(similar, 1):
            print(f"\n{i}. {pitch['product_name']}")
            print(f"   Similarity: {pitch.get('similarity_score', 'N/A'):.4f}")
            print(f"   Preview: {pitch['pitch_text'][:200]}...")
        
        # Test context formatting
        print("\n" + "="*70)
        print("FORMATTED CONTEXT FOR PROMPT:")
        print("="*70)
        context = retriever.format_context_for_prompt(similar, max_examples=2)
        print(context[:1000] + "..." if len(context) > 1000 else context)

