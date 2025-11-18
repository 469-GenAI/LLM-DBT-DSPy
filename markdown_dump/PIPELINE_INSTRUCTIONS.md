# RAG Pipeline Instructions

## Overview

This document provides instructions for running the RAG-based pitch generation pipeline. The system uses retrieval-augmented generation to find similar successful pitches and use them as examples when generating new pitches.

## Pipeline Components

### 1. Data Indexing (`src/pitchLLM/rag/data_indexer.py`)

**Purpose**: Loads pitch transcripts and enriches them with deal success information.

**Key Functions**:
- `load_all_processed_facts_with_deals()` - Extracts deal success from `all_processed_facts.json` based on `final_offer` field
- `load_po_samples_with_deals()` - Extracts deal info from `PO_samples.json` (smaller subset)
- `load_and_prepare_all()` - Main function that loads transcripts and enriches with deal info

**Deal Success Detection**:
- Successful deals: `final_offer` contains `investors` or `shark` field (not empty)
- Unsuccessful deals: `final_offer` contains `outcome` field saying "No deal" or is empty

### 2. Vector Store (`src/pitchLLM/rag/vector_store.py`)

**Purpose**: Stores pitch embeddings in ChromaDB for similarity search.

**Key Functions**:
- `PitchVectorStore.add_documents()` - Index pitches into vector store
- `PitchVectorStore.query()` - Search for similar pitches

### 3. Retriever (`src/pitchLLM/rag/retriever.py`)

**Purpose**: Retrieves similar pitches based on product data.

**Key Features**:
- `filter_successful_deals=True` - Only retrieve pitches that closed deals
- `filter_category="Health & Fitness"` - Filter by product category

### 4. Generator (`src/pitchLLM/models/rag_generator.py`)

**Purpose**: Generates pitches using retrieved examples as context.

## Running the Pipeline

### Step 1: Index Pitches (One-time setup)

The vector store is automatically indexed when you first use `RAGPitchGenerator`, but you can also do it manually:

```python
from src.pitchLLM.rag.retriever import PitchRetriever
from src.pitchLLM.rag.data_indexer import load_and_prepare_all
from src.pitchLLM.rag.vector_store import PitchVectorStore

# Load and prepare documents with deal information
documents = load_and_prepare_all(
    enrich_with_deals=True,
    use_all_processed_facts=True  # Use comprehensive data source
)

# Create vector store and index
vector_store = PitchVectorStore()
vector_store.add_documents(documents)

# Or use retriever (auto-indexes if empty)
retriever = PitchRetriever(auto_index=True)
```

### Step 2: Generate Pitches with RAG

**Basic Usage**:

```python
import os
from dotenv import load_dotenv
import dspy
from src.pitchLLM.models import RAGPitchGenerator
from src.pitchLLM.utils import format_pitch_input

load_dotenv()

# Configure LLM
lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    model_type="chat",
    api_key=os.getenv("GROQ_API_KEY")
)

# Create RAG generator
generator = RAGPitchGenerator(
    lm,
    top_k=5,  # Retrieve 5 similar pitches
    filter_successful_deals=False  # Set to True to only use successful pitches
)

# Generate pitch
product_data = {
    "product_solution": {
        "name": "SmartFit Tracker",
        "product_category": "Health & Fitness Wearables"
    },
    # ... other product data
}

result = generator.generate(product_data)
print(result.pitch)
```

**Filter by Successful Deals Only**:

```python
generator = RAGPitchGenerator(
    lm,
    top_k=5,
    filter_successful_deals=True  # Only retrieve pitches that closed deals
)
```

**Filter by Category**:

```python
generator = RAGPitchGenerator(
    lm,
    top_k=5,
    filter_category="Health & Fitness"  # Only retrieve pitches from this category
)
```

### Step 3: Test and Compare

Run the comparison script:

```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
python src/pitchLLM/test_rag_comparison.py
```

This will:
1. Generate a pitch without RAG
2. Generate a pitch with RAG
3. Show retrieved similar pitches
4. Display side-by-side comparison

## File Structure

```
src/pitchLLM/
├── rag/
│   ├── data_indexer.py      # Loads and enriches pitch data
│   ├── vector_store.py      # ChromaDB interface
│   └── retriever.py         # Retrieves similar pitches
├── models/
│   ├── generator.py         # Base pitch generator
│   └── rag_generator.py      # RAG-enhanced generator
├── test_rag_comparison.py    # Test script
└── evaluate_rag.py          # Evaluation script
```

## Data Sources

### Primary: `data/all_processed_facts.json`
- **Size**: ~245 pitches
- **Deal Info**: Extracted from `final_offer` field
- **Coverage**: Both successful and unsuccessful pitches
- **Used by**: `load_all_processed_facts_with_deals()`

### Secondary: `data/PO_samples.json`
- **Size**: ~10 pitches
- **Deal Info**: Extracted from `final_offer` field
- **Coverage**: Only successful pitches
- **Used by**: `load_po_samples_with_deals()`

### Pitch Transcripts: `data/pitches/`
- `existing_refined_pitches(119).json` - Clean, validated pitches
- `refined_pitches_(126).json` - Additional clean pitches
- Total: ~245 pitches

## Deal Success Labeling

The system automatically labels pitches as successful based on the `final_offer` structure:

**Successful Deal Patterns**:
```json
{
  "final_offer": {
    "investors": "Mark Cuban and Lori Greiner",  // OR
    "shark": "Blake",
    "amount_invested": "$1 million",  // OR "amount"
    "equity_stake": "10%"  // OR "equity"
  }
}
```

**Unsuccessful Deal Patterns**:
```json
{
  "final_offer": {
    "outcome": "No deal was made..."
  }
}
```

## Environment Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables** (`.env` file):
```
GROQ_API_KEY=your_key_here
```

3. **ChromaDB** (auto-created):
- Location: `chromadb_data/`
- No manual setup needed

## Common Operations

### Re-index Vector Store

```python
from src.pitchLLM.rag.retriever import PitchRetriever

retriever = PitchRetriever()
retriever.index_all_transcripts(force_reload=True)
```

### Check Vector Store Stats

```python
from src.pitchLLM.rag.retriever import PitchRetriever

retriever = PitchRetriever()
stats = retriever.get_retriever_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Has deals: {stats.get('has_deal_count', 0)}")
```

### Manual Deal Info Extraction

```python
from src.pitchLLM.rag.data_indexer import load_all_processed_facts_with_deals

deals = load_all_processed_facts_with_deals()
successful = [k for k, v in deals.items() if v.get('has_deal', False)]
print(f"Successful pitches: {len(successful)}")
```

## Troubleshooting

### Issue: Vector store is empty
**Solution**: The retriever auto-indexes on first use. You can also manually index:
```python
retriever = PitchRetriever(auto_index=True)
```

### Issue: Deal information not found
**Solution**: Check that `all_processed_facts.json` exists and has `final_offer` fields. The system uses fuzzy matching to match product names.

### Issue: No similar pitches retrieved
**Solution**: 
1. Check vector store has documents: `retriever.get_retriever_stats()`
2. Try reducing `filter_successful_deals` or `filter_category` restrictions
3. Check product data is formatted correctly

## Next Steps

1. **Evaluation**: Run `src/pitchLLM/evaluate_rag.py` to compare RAG vs non-RAG performance
2. **Fine-tuning**: Adjust `top_k` parameter based on results
3. **Filtering**: Experiment with `filter_successful_deals` and `filter_category`
4. **Chunking**: If needed, enable chunking for very long pitches (default: no chunking)

