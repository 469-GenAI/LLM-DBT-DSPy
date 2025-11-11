"""
RAG (Retrieval-Augmented Generation) package for pitch generation.

Provides:
- data_indexer: Load and prepare train.jsonl for indexing (~196 pitches)
- vector_store: ChromaDB interface for storing/retrieving pitches
- retriever: Retrieve similar pitches for context

NOTE: Only train.jsonl is indexed (test.jsonl excluded to prevent data leakage).
"""

from .vector_store import PitchVectorStore
from .retriever import PitchRetriever
from .data_indexer import prepare_pitch_documents, load_and_prepare_all

__all__ = [
    "prepare_pitch_documents",
    "load_and_prepare_all",  # Main function - uses train.jsonl
    "PitchVectorStore",
    "PitchRetriever"
]

# Deprecated functions (kept for backward compatibility but not exported)
# Use load_and_prepare_all(use_jsonl=True) instead
try:
    from .data_indexer import load_all_transcripts
except ImportError:
    # If import fails, that's okay - function is deprecated anyway
    pass


