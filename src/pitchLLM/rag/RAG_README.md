# RAG Implementation for Shark Tank Pitch Generation

A comprehensive Retrieval-Augmented Generation (RAG) system that enhances LLM pitch generation by retrieving and using similar successful pitches as context. This implementation compares custom RAG strategies against DSPy's KNNFewShot optimizer to understand their relative effectiveness.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Key Findings](#key-findings)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Components](#components)
7. [Usage Guide](#usage-guide)
8. [Retrieval Strategies](#retrieval-strategies)
9. [Evaluation & Comparison](#evaluation--comparison)
10. [File Structure](#file-structure)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### What This RAG System Does

This RAG pipeline enhances LLM pitch generation by:

1. **Indexing** 196 training pitches from `train.jsonl` into a ChromaDB vector store
2. **Retrieving** similar pitches based on product information (company, problem, solution)
3. **Generating** new pitches using retrieved examples as style/context guidance
4. **Evaluating** quality using LLM-as-a-Judge methodology
5. **Comparing** different retrieval strategies and against DSPy optimizers

### Key Design Principles

- ✅ **No Data Leakage**: Only `train.jsonl` is indexed; `test.jsonl` is never used for retrieval
- ✅ **Semantic Similarity**: Uses embeddings to find semantically similar products
- ✅ **Multiple Strategies**: Supports pure semantic, filtering, and prioritization approaches
- ✅ **Fair Comparison**: Separates RAG from DSPy optimizers for unbiased evaluation

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (One-time)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  train.jsonl (196 pitches)                                 │
│       ↓                                                     │
│  [Extract product info + pitch text]                       │
│       ↓                                                     │
│  [Create embeddings]                                        │
│       ↓                                                     │
│  [Store in ChromaDB]                                        │
│       ↓                                                     │
│  Vector Store: 196 indexed documents                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              GENERATION PHASE (Per test example)            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  test.jsonl INPUT (product info)                           │
│       ↓                                                     │
│  [Create query from product info]                          │
│       ↓                                                     │
│  [Query vector store: find similar products]               │
│       ↓                                                     │
│  [Retrieve top-K similar pitches from train.jsonl]         │
│       ↓                                                     │
│  [Format as context]                                        │
│       ↓                                                     │
│  [Generate NEW pitch using INPUT + context]                │
│       ↓                                                     │
│  [Compare with test.jsonl OUTPUT (ground truth)]          │
│       ↓                                                     │
│  Quality Score                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Core RAG Components                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────┐            │
│  │  data_indexer.py │───▶│  vector_store.py  │            │
│  │                  │    │                  │            │
│  │  - Load JSONL    │    │  - ChromaDB      │            │
│  │  - Extract data  │    │  - Embeddings    │            │
│  │  - Prepare docs  │    │  - Storage       │            │
│  └──────────────────┘    └──────────────────┘            │
│         │                          │                       │
│         └──────────┬───────────────┘                       │
│                    ↓                                       │
│         ┌──────────────────┐                              │
│         │   retriever.py   │                              │
│         │                  │                              │
│         │  - Query vector  │                              │
│         │  - Apply filters  │                              │
│         │  - Prioritize    │                              │
│         └──────────────────┘                              │
│                    │                                       │
│                    ↓                                       │
│         ┌──────────────────┐                              │
│         │ rag_generator.py  │                              │
│         │                  │                              │
│         │  - Format context │                              │
│         │  - Generate pitch │                              │
│         └──────────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Findings

### Performance Comparison (10 test examples, averaged)

| Strategy | RAG Score | KNN Score | Difference | Overlap (Jaccard) |
|----------|-----------|-----------|------------|-------------------|
| **Pure Semantic** | 0.7380 | 0.7710 | -0.0330 (-3.3%) | 0.029 |
| **Hybrid Filter Deals** | 0.7120 | 0.7710 | -0.0590 (-5.9%) | 0.000 |
| **Hybrid Prioritize** | 0.7260 | 0.7710 | -0.0450 (-4.5%) | 0.014 |

**KNNFewShot Baseline**: 0.7710

### Key Insights

1. **KNNFewShot Outperforms RAG**: All RAG strategies score 3-6% lower than KNNFewShot
2. **Low Overlap**: RAG and KNNFewShot select different examples (Jaccard similarity: 0.000-0.029)
3. **Pure Semantic Works Best**: Among RAG strategies, pure semantic similarity performs best
4. **Domain Knowledge Doesn't Help**: Filtering/prioritizing by deals/categories reduces performance
5. **Different Selection Strategies**: KNNFewShot selects higher-similarity examples (0.437-0.442 vs 0.387-0.400)

### Conclusion

**RAG and KNNFewShot are NOT "mining the same gold mine"**:
- They select different examples (low overlap)
- KNNFewShot's optimization-based selection is more effective
- RAG's domain knowledge (deals/categories) doesn't improve performance
- KNNFewShot's internal optimization outperforms explicit RAG retrieval

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Groq API key

### Step 1: Activate Virtual Environment

```bash
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install dspy-ai chromadb pandas python-dotenv tqdm pydantic sentence-transformers numpy
```

### Step 3: Set Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Verify Setup

```bash
python -c "import dspy; import chromadb; print('✓ Dependencies installed')"
```

---

## Quick Start

### 1. First-Time Setup

```bash
# Classify categories (optional but recommended)
python src/pitchLLM/rag/category_classifier.py

# Index training data into vector store
python src/pitchLLM/reindex_with_categories.py
```

### 2. Run Comparisons

```bash
# Compare RAG strategies vs Non-RAG baseline
python src/pitchLLM/compare_rag_strategies.py

# Compare RAG strategies vs KNNFewShot
python src/pitchLLM/compare_rag_vs_knn.py

# Quick RAG vs Non-RAG test
python src/pitchLLM/evaluate_rag.py
```

### 3. View Results

Results are saved in `rag_comparison_results/`:
- `comparison_report_{timestamp}.md` - Summary report
- `detailed_results_{timestamp}.csv` - Per-example results
- `summary_{timestamp}.json` - Aggregated metrics

---

## Components

### Core RAG Modules

#### `rag/vector_store.py`
**Purpose**: ChromaDB wrapper for storing and querying pitch embeddings

**Key Functions**:
- `PitchVectorStore`: Manages ChromaDB collection
- Stores embeddings of pitch documents
- Handles similarity search queries

#### `rag/data_indexer.py`
**Purpose**: Load and prepare training data for indexing

**Key Functions**:
- `load_and_prepare_all()`: Main function to load `train.jsonl` and prepare documents
- `prepare_pitch_documents()`: Extract product info and pitch text
- Handles metadata (category, deal info) from mapping files

#### `rag/retriever.py`
**Purpose**: Retrieve similar pitches using different strategies

**Key Classes**:
- `PitchRetriever`: Main retriever class
- `RetrievalStrategy`: Enum for different strategies

**Key Methods**:
- `retrieve_similar_pitches()`: Main retrieval function
- `format_context_for_prompt()`: Format retrieved pitches as context

#### `models/rag_generator.py`
**Purpose**: RAG-enhanced pitch generator

**Key Classes**:
- `RAGStructuredPitchProgram`: DSPy module for RAG generation
- `RAGPitchGenerator`: High-level generator interface

**Key Features**:
- Integrates retrieval with generation
- Supports all three retrieval strategies
- Handles context formatting and prompt construction

### Evaluation & Comparison Scripts

#### `compare_rag_strategies.py`
**Purpose**: Compare all RAG strategies against Non-RAG baseline

**What it does**:
- Tests Pure_Semantic, Hybrid_Filter_Deals, Hybrid_Prioritize, and Non_RAG
- Evaluates on 10 test examples (configurable)
- Generates comprehensive comparison reports

**Usage**:
```bash
python src/pitchLLM/compare_rag_strategies.py
```

**Output**:
- CSV with per-example results
- JSON summary with averaged metrics
- Markdown comparison report

#### `compare_rag_vs_knn.py`
**Purpose**: Compare RAG strategies against DSPy's KNNFewShot

**What it does**:
- Tests all three RAG strategies
- Compares against KNNFewShot optimizer
- Computes overlap statistics (Jaccard similarity)
- Analyzes which examples each method selects

**Usage**:
```bash
python src/pitchLLM/compare_rag_vs_knn.py
```

**Output**:
- JSON with detailed comparison results
- Markdown report with overlap analysis
- Per-example breakdowns

#### `evaluate_rag.py`
**Purpose**: Simple RAG vs Non-RAG evaluation

**What it does**:
- Quick comparison of RAG (default: Pure_Semantic) vs Non-RAG
- Configurable test size and retrieval parameters

**Usage**:
```bash
python src/pitchLLM/evaluate_rag.py
```

### Utility Scripts

#### `reindex_with_categories.py`
**Purpose**: Re-index vector database with latest data

**When to use**:
- After updating `train.jsonl`
- After updating `category_mapping.json` or `deal_mapping.json`
- If ChromaDB becomes corrupted

**Usage**:
```bash
python src/pitchLLM/reindex_with_categories.py
```

#### `rag/category_classifier.py`
**Purpose**: Classify pitches into categories

**What it does**:
- Uses LLM to classify products into categories
- Creates/updates `category_mapping.json`
- Categories: Technology, Health & Fitness, Food & Beverage, etc.

**Usage**:
```bash
python src/pitchLLM/rag/category_classifier.py
```

---

## Usage Guide

### Basic RAG Generation

```python
from pitchLLM.models.rag_generator import RAGPitchGenerator
from pitchLLM.rag.retriever import RetrievalStrategy
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# Setup model
lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    model_type="chat",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.0
)

# Create RAG generator
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.PURE_SEMANTIC,
    top_k=5
)

# Generate pitch
product_data = {
    "company": "My Product",
    "problem_summary": "People struggle with...",
    "solution_summary": "Our solution is...",
    "offer": "100,000 for 10%"
}

prediction = generator.generate(product_data)
print(prediction.pitch)
```

### Using Different Retrieval Strategies

```python
# Pure Semantic (baseline)
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.PURE_SEMANTIC,
    top_k=5
)

# Hybrid Filter (only successful deals)
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.HYBRID_FILTER,
    filter_successful_deals=True,
    top_k=5
)

# Hybrid Prioritize (recommended)
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.HYBRID_PRIORITIZE,
    prioritize_successful=True,
    prioritize_category=True,
    top_k=5
)
```

### Manual Retrieval

```python
from pitchLLM.rag.retriever import PitchRetriever, RetrievalStrategy

retriever = PitchRetriever(auto_index=True)

# Retrieve similar pitches
similar = retriever.retrieve_similar_pitches(
    product_data=product_data,
    top_k=5,
    strategy=RetrievalStrategy.PURE_SEMANTIC,
    include_scores=True
)

# View results
for pitch in similar:
    print(f"{pitch['product_name']}: {pitch['similarity_score']:.3f}")
    print(f"  Category: {pitch['metadata'].get('category', 'N/A')}")
    print(f"  Has Deal: {pitch['metadata'].get('has_deal', False)}")
```

---

## Retrieval Strategies

### 1. Pure Semantic (`PURE_SEMANTIC`)

**Description**: Pure similarity-based retrieval without any filtering or prioritization

**How it works**:
- Creates query from product info (company, problem, solution, offer)
- Performs semantic similarity search
- Returns top-K most similar pitches

**Use when**:
- You want baseline performance
- No domain knowledge available
- Simple, fast retrieval needed

**Performance**: Best among RAG strategies (0.7380 avg score)

### 2. Hybrid Filter (`HYBRID_FILTER`)

**Description**: Semantic search + metadata filtering

**How it works**:
- Performs semantic search
- Filters results by metadata:
  - `filter_successful_deals=True`: Only pitches that closed deals
  - `filter_category="Technology"`: Only pitches in specific category

**Use when**:
- You want to learn only from successful deals
- Category-specific retrieval needed

**Performance**: Lowest among RAG strategies (0.7120 avg score)

### 3. Hybrid Prioritize (`HYBRID_PRIORITIZE`)

**Description**: Semantic search + rule-based prioritization

**How it works**:
- Performs semantic search
- Prioritizes results by:
  1. Successful deals + Same category (highest priority)
  2. Successful deals only
  3. Same category only
  4. Others (lowest priority)
- Maintains semantic similarity within priority levels

**Use when**:
- You want domain-informed retrieval
- Category and deal success matter

**Performance**: Middle among RAG strategies (0.7260 avg score)

---

## Evaluation & Comparison

### Evaluation Methodology

**Quality Score**: Composite metric (0.0-1.0)
- **Factual Score (40%)**: Accuracy of product information
- **Narrative Score (40%)**: Story structure and flow
- **Style Score (20%)**: Shark Tank presentation style

**Evaluator**: LLM-as-a-Judge using `groq/openai/gpt-oss-120b`

### Running Comparisons

#### Compare RAG Strategies

```bash
python src/pitchLLM/compare_rag_strategies.py
```

**Configuration** (in script):
- `test_size=10`: Number of test examples
- `top_k=5`: Number of retrieved examples
- Models: 70B generator, 120B evaluator

**Output**:
- Average quality scores for each strategy
- Success rates, latencies, token usage
- Per-example detailed results

#### Compare RAG vs KNNFewShot

```bash
python src/pitchLLM/compare_rag_vs_knn.py
```

**What it analyzes**:
- Which examples each method selects
- Overlap statistics (Jaccard similarity)
- Performance comparison
- Selection quality (similarity scores)

**Output**:
- Overlap analysis (which examples overlap)
- Performance comparison table
- Per-example breakdowns

### Understanding Results

**Quality Score Interpretation**:
- `0.4-0.6`: Baseline performance
- `0.6-0.8`: Good performance
- `0.8-0.9`: Excellent performance
- `0.9+`: Exceptional performance

**Overlap Metrics**:
- **Jaccard Similarity**: Measure of overlap between selected examples (0.0-1.0)
- **Overlap Count**: Number of examples that appear in both selections
- **Low overlap (<0.1)**: Methods select different examples
- **High overlap (>0.5)**: Methods select similar examples

---

## File Structure

```
src/pitchLLM/
├── rag/                          # Core RAG components
│   ├── __init__.py
│   ├── vector_store.py           # ChromaDB wrapper
│   ├── data_indexer.py           # Data loading & preparation
│   ├── retriever.py              # Retrieval logic & strategies
│   └── category_classifier.py    # Category classification
│
├── models/                        # Generator & evaluator models
│   ├── rag_generator.py          # RAG-enhanced generator
│   ├── generator.py               # Non-RAG generator
│   └── evaluator.py              # LLM-as-a-Judge evaluator
│
├── compare_rag_strategies.py     # Compare RAG strategies vs Non-RAG
├── compare_rag_vs_knn.py         # Compare RAG vs KNNFewShot
├── evaluate_rag.py               # Simple RAG evaluation
├── optimize_rag.py                # DSPy optimization of RAG
├── reindex_with_categories.py    # Re-index vector database
└── test_rag_comparison.py        # Quick test script

data/hf (new)/
├── train.jsonl                   # Training data (196 pitches) - INDEXED
├── test.jsonl                    # Test data (49 pitches) - NOT indexed
├── category_mapping.json         # Category classifications
└── deal_mapping.json             # Deal outcome information

rag_comparison_results/            # Comparison outputs
├── comparison_report_*.md        # Summary reports
├── detailed_results_*.csv        # Per-example results
└── summary_*.json                # Aggregated metrics

chromadb_data/                     # Vector database storage
└── shark_tank_pitches/           # ChromaDB collection
```

---

## Troubleshooting

### Vector Database Issues

**Problem**: ChromaDB corruption errors
```
pyo3_runtime.PanicException: range start index out of range
```

**Solution**: Re-index the database
```bash
python src/pitchLLM/reindex_with_categories.py
```

### Missing Categories

**Problem**: Categories not found in metadata
```
⚠️ Category mapping file not found
```

**Solution**: Run category classifier
```bash
python src/pitchLLM/rag/category_classifier.py
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'dspy'`

**Solution**: Activate virtual environment
```bash
source venv/bin/activate
python --version  # Should show Python 3.11+
```

### Rate Limiting

**Problem**: `RateLimitError: Rate limit reached`

**Solution**:
- Wait a few minutes and retry
- Reduce `test_size` in evaluation scripts
- Check API key limits

### Low Quality Scores

**Possible causes**:
1. **Insufficient training data**: Need at least 50+ examples
2. **Poor retrieval**: Retrieved examples not relevant
3. **Model issues**: Check API key and model availability

**Debugging**:
```python
from pitchLLM.rag.retriever import PitchRetriever

retriever = PitchRetriever()
similar = retriever.retrieve_similar_pitches(
    product_data=test_product,
    top_k=5,
    include_scores=True
)

# Check retrieval quality
for pitch in similar:
    print(f"{pitch['product_name']}: {pitch['similarity_score']:.3f}")
    # Low similarity (<0.3) suggests poor retrieval
```

### Model Not Found

**Problem**: `The model 'llama-3.3-8b-instant' does not exist`

**Solution**: Use correct model names:
- ✅ `groq/llama-3.1-8b-instant` (8B model)
- ✅ `groq/llama-3.3-70b-versatile` (70B model)
- ❌ `groq/llama-3.3-8b-instant` (doesn't exist)

---

## Configuration

### Model Configuration

**Generator Models**:
- `groq/llama-3.3-70b-versatile`: Best quality (recommended)
- `groq/llama-3.1-8b-instant`: Faster, cheaper (for testing)

**Evaluator Models**:
- `groq/openai/gpt-oss-120b`: Best evaluation quality (recommended)

**Temperature Settings**:
- Generator: `1.0` (for diverse outputs)
- Evaluator: `0.1` (for consistent scoring)

### Retrieval Configuration

**Top-K**: Number of examples to retrieve
- Default: `5`
- Range: `1-10` (recommended)
- Higher = more context but longer prompts

**Embedding Model**: Used for semantic similarity
- Default: `all-MiniLM-L6-v2` (via ChromaDB)
- 384-dimensional embeddings
- Fast and efficient

---

## Research Findings Summary

### Hypothesis Tested

> "RAG and KNNFewShot are mining the same gold mine" - i.e., they select similar examples from the same training corpus.

### Results

**Hypothesis REJECTED**:
- Low overlap (Jaccard similarity: 0.000-0.029)
- Different example selection strategies
- KNNFewShot selects higher-quality examples

### Performance Comparison

| Method | Avg Score | vs KNNFewShot |
|--------|-----------|---------------|
| KNNFewShot | 0.7710 | Baseline |
| Pure Semantic RAG | 0.7380 | -3.3% |
| Hybrid Prioritize RAG | 0.7260 | -4.5% |
| Hybrid Filter RAG | 0.7120 | -5.9% |

### Key Insights

1. **KNNFewShot's optimization is more effective** than explicit RAG retrieval
2. **Domain knowledge (deals/categories) doesn't help** RAG performance
3. **Pure semantic similarity works best** among RAG strategies
4. **Different selection mechanisms** lead to different example choices
5. **RAG adds complexity without clear benefit** over KNNFewShot

---

## Future Work

### Potential Improvements

1. **Better Embeddings**: Use larger embedding models (e.g., `all-mpnet-base-v2`)
2. **Hybrid Retrieval**: Combine semantic + keyword-based retrieval
3. **Reranking**: Use cross-encoder to rerank retrieved examples
4. **Prompt Optimization**: Optimize how retrieved examples are formatted
5. **Dynamic Top-K**: Adjust number of examples based on query complexity

### Research Questions

1. Why does KNNFewShot select better examples despite lower similarity scores?
2. Can RAG be improved to match KNNFewShot performance?
3. Would combining RAG + KNNFewShot improve results?
4. How does retrieval quality affect generation quality?

---

## Citation & References

If you use this RAG implementation in your research, please cite:

```bibtex
@software{rag_pitch_generation,
  title = {RAG Implementation for Shark Tank Pitch Generation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

### Related Work

- **DSPy**: Stanford's framework for programming LLMs
- **KNNFewShot**: DSPy's k-nearest neighbor few-shot optimizer
- **ChromaDB**: Vector database for embeddings
- **LLM-as-a-Judge**: Evaluation methodology using LLMs

---

## License

[Your License Here]

---

## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email]

---

## Quick Reference

For quick commands and code snippets, see `RAG_QUICK_REFERENCE.md`.

For detailed process explanation, see `RAG_RETRIEVAL_PROCESS_EXPLANATION.md`.

---

**Last Updated**: November 2024

