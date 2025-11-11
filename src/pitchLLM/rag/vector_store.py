

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PitchVectorStore:
    """
    ChromaDB-based vector store for pitch transcripts.
    
    Handles:
    - Creating/loading collection
    - Adding pitch documents
    - Querying for similar pitches
    """
    
    def __init__(
        self, 
        collection_name: str = "shark_tank_pitches",
        persist_directory: str = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Where to store the database. 
                             If None, uses default chromadb_data/
        """
        if persist_directory is None:
            # Use existing chromadb_data directory
            persist_directory = str(
                Path(__file__).parent.parent.parent.parent / "chromadb_data"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with error handling for corruption
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            # Handle ChromaDB corruption - reset and rebuild
            error_str = str(e)
            error_type = type(e).__name__
            if ("range start index" in error_str or 
                "PanicException" in error_type or
                "pyo3_runtime.PanicException" in error_str):
                logger.warning(f"ChromaDB database corruption detected: {error_type}")
                logger.warning(f"Resetting database...")
                self._reset_database()
                # Retry initialization after reset
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            # Test if collection is accessible (catches corruption)
            try:
                count = self.collection.count()
                logger.info(f"Loaded existing collection: {collection_name}")
                logger.info(f"Collection contains {count} documents")
            except Exception as e:
                # Collection exists but is corrupted - delete and recreate
                error_str = str(e)
                error_type = type(e).__name__
                if ("range start index" in error_str or 
                    "PanicException" in error_type or
                    "pyo3_runtime.PanicException" in error_str):
                    logger.warning(f"Collection '{collection_name}' is corrupted: {error_type}")
                    logger.warning(f"Recreating collection...")
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"description": "Shark Tank pitch transcripts for RAG"}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                else:
                    raise
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Shark Tank pitch transcripts for RAG"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
            batch_size: Number of documents to add per batch
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc['id'])
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        # Add in batches
        total = len(documents)
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            
            self.collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {batch_end}/{total} documents")
        
        logger.info(f"Successfully added {total} documents to {self.collection_name}")
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query for similar pitches.
        
        Args:
            query_text: Text to search for (e.g., product description)
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"category": "tech"})
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def _reset_database(self):
        """
        Reset corrupted ChromaDB database by deleting and recreating it.
        """
        import shutil
        logger.warning(f"Resetting corrupted database at: {self.persist_directory}")
        
        # Try to delete the directory
        try:
            if Path(self.persist_directory).exists():
                shutil.rmtree(self.persist_directory)
                logger.info(f"Deleted corrupted database directory")
        except Exception as e:
            logger.warning(f"Could not delete database directory: {e}")
            # Try to delete just the collection
            try:
                temp_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )
                temp_client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted corrupted collection")
            except Exception as e2:
                logger.warning(f"Could not delete collection: {e2}")
        
        # Recreate directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Database reset complete. Will recreate on next initialization.")
    
    def delete_collection(self):
        """Delete the collection (use with caution!)."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def reset_and_reload(self, documents: List[Dict]):
        """
        Delete collection and reload with new documents.
        
        Args:
            documents: List of documents to add
        """
        try:
            self.delete_collection()
        except Exception:
            pass
        
        # Recreate collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Shark Tank pitch transcripts for RAG"}
        )
        
        # Add documents
        self.add_documents(documents)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Test the vector store
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing PitchVectorStore...")
    store = PitchVectorStore()
    
    print("\nVector store stats:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query if collection has data
    if stats['total_documents'] > 0:
        print("\nTesting query...")
        query = "innovative technology product for health"
        results = store.query(query, n_results=3)
        
        print(f"\nTop 3 results for: '{query}'")
        for i, (doc_id, text, metadata, distance) in enumerate(
            zip(results['ids'], results['documents'], 
                results['metadatas'], results['distances'])
        ):
            print(f"\n{i+1}. {doc_id}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Product: {metadata.get('product_name', 'N/A')}")
            print(f"   Text preview: {text[:150]}...")


"""
Vector store using ChromaDB for pitch retrieval.

CHUNKING STRATEGY & CONSIDERATIONS
==================================

CURRENT APPROACH: NO CHUNKING (Default)
---------------------------------------
Each pitch is stored as ONE complete document in the vector database.
- Pitch length: ~500-8,000 characters (average ~2,000 chars)
- Documents: 245 total (one per product)
- Retrieval: Returns entire pitch contexts

WHY NO CHUNKING BY DEFAULT?
---------------------------
1. Pitch Structure: Pitches are self-contained narratives
   - Problem → Solution → Ask → Story arc
   - Breaking them up loses narrative coherence
   
2. Semantic Completeness: Entire pitch provides full context
   - Product description, founder story, financials, ask
   - LLM can extract relevant parts from full context
   
3. Retrieval Quality: Similarity search works better on complete pitches
   - "Smart fitness tracker" matches entire pitch, not fragments
   - Vector embeddings capture full semantic meaning
   
4. Token Limits: Modern LLMs handle ~8K tokens easily
   - Average pitch: ~500-2000 tokens (well within limits)
   - Even longest pitches (~8K chars) fit comfortably

WHEN TO CONSIDER CHUNKING?
--------------------------
Consider chunking if you encounter:

1. **Very Long Pitches** (>10,000 chars)
   - Some transcripts include full Q&A sessions
   - Solution: Filter/extract pitch portion only
   
2. **Fine-Grained Retrieval** (sub-pitch sections)
   - Need to find specific parts: "financial ask", "problem statement"
   - Solution: Semantic chunking by section (requires structure)
   
3. **Token Budget Constraints**
   - Strict limits on context window
   - Solution: Chunk to fit budget (trade-off: lose coherence)
   
4. **Hybrid Search Strategy**
   - Combine vector search + keyword search
   - Chunks enable better keyword matching
   - Solution: Use metadata filters instead (simpler)

CHUNKING FACTORS TO CONSIDER
----------------------------
If you decide to chunk, consider:

1. **Chunk Size**
   - Too small (<500 chars): Loses context, fragmented meaning
   - Too large (>3000 chars): Defeats purpose, still long
   - Recommended: 1000-2000 chars (if chunking needed)
   
2. **Chunk Overlap**
   - Purpose: Preserve context across boundaries
   - Example: "I need $500K" split mid-sentence
   - Recommended: 100-200 chars overlap
   
3. **Chunking Method**
   - Character-based: Simple, but may split mid-sentence
   - Sentence-based: Better semantic boundaries
   - Section-based: Best (Problem/Solution/Ask) but requires structure
   
4. **Metadata Tracking**
   - Track: chunk_index, total_chunks, product_key
   - Enables: Reconstructing full pitch, filtering by product
   - Current: Already implemented in data_indexer.py
   
5. **Retrieval Strategy**
   - With chunks: May retrieve multiple chunks from same pitch
   - Solution: Deduplicate by product_key, merge chunks
   - Alternative: Rerank and select best chunk per product

RECOMMENDED APPROACH
--------------------
For Shark Tank pitches, stick with NO CHUNKING unless:
- You have specific requirements (fine-grained search, token limits)
- You're experimenting with retrieval strategies
- Your pitches exceed 10K characters consistently

If chunking is needed:
```python
from pitchLLM.rag.data_indexer import load_and_prepare_all

# Chunk at 1500 chars with 200 char overlap
documents = load_and_prepare_all(
    chunk_size=1500,
    chunk_overlap=200
)
```

PERFORMANCE COMPARISON
----------------------
No Chunking (Current):
- Documents: 245
- Avg length: ~2000 chars
- Retrieval: Fast, simple
- Quality: High (full context)

With Chunking (1500 chars, 200 overlap):
- Documents: ~400-500 (estimated)
- Avg length: ~1500 chars
- Retrieval: Slightly slower (more docs)
- Quality: May decrease (fragmented context)
- Reassembly: Requires logic to merge chunks

CONCLUSION
----------
Default (no chunking) is optimal for most use cases. Only chunk if you have
specific requirements that benefit from finer granularity.
"""