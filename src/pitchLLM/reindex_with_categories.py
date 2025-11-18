"""
Re-index the vector database with category mapping loaded.

This script:
1. Loads category_mapping.json
2. Re-indexes train.jsonl with categories included
3. Verifies categories are stored in metadata
"""

import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pitchLLM.rag.retriever import PitchRetriever

def main():
    """Re-index with category mapping."""
    print("="*80)
    print("RE-INDEXING VECTOR DATABASE WITH CATEGORIES")
    print("="*80)
    
    # Get paths
    base_path = Path("data")
    category_mapping_path = base_path / "hf (new)" / "category_mapping.json"
    train_jsonl_path = base_path / "hf (new)" / "train.jsonl"
    
    print(f"\nCategory mapping: {category_mapping_path}")
    print(f"Train JSONL: {train_jsonl_path}")
    
    if not category_mapping_path.exists():
        print(f"\n⚠️  Category mapping file not found: {category_mapping_path}")
        print("   Run category_classifier.py first to generate categories.")
        return
    
    if not train_jsonl_path.exists():
        print(f"\n⚠️  Train JSONL file not found: {train_jsonl_path}")
        return
    
    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = PitchRetriever(auto_index=False)  # Don't auto-index, we'll do it manually
    
    # Re-index with category mapping
    print("\nRe-indexing with categories...")
    print("(This will delete existing collection and reload)")
    retriever.index_all_transcripts(
        force_reload=True,  # Delete and reload
        use_jsonl=True,
        jsonl_path=str(train_jsonl_path),
        category_mapping_path=str(category_mapping_path)
    )
    
    # Verify categories
    print("\n" + "="*80)
    print("VERIFYING CATEGORIES")
    print("="*80)
    
    stats = retriever.vector_store.get_stats()
    print(f"Total documents: {stats['total_documents']}")
    
    # Sample a few documents to check categories
    try:
        sample_docs = retriever.vector_store.collection.get(limit=5)
        if sample_docs and 'metadatas' in sample_docs:
            print("\nSample documents with categories:")
            for i, (doc_id, metadata) in enumerate(zip(sample_docs['ids'], sample_docs['metadatas']), 1):
                product_name = metadata.get('product_name', 'Unknown')
                category = metadata.get('category', 'Unknown')
                print(f"  {i}. {product_name}: {category}")
    except Exception as e:
        print(f"Could not verify categories: {e}")
    
    print("\n✓ Re-indexing complete!")
    print("\nTo verify categories are loaded, run:")
    print("  python src/pitchLLM/visualize_vector_db.py --samples 10")

if __name__ == "__main__":
    main()

