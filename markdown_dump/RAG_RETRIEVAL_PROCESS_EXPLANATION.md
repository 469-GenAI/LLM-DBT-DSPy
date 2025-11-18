# RAG Pipeline - Complete Guide

A comprehensive guide to understanding, running, and optimizing the RAG (Retrieval-Augmented Generation) pipeline for Shark Tank pitch generation.

---

## Table of Contents

1. [Overview](#overview)
2. [How RAG Works](#how-rag-works)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Running the RAG Pipeline](#running-the-rag-pipeline)
6. [Optimizing RAG with DSPy](#optimizing-rag-with-dspy)
7. [Comparing RAG Strategies](#comparing-rag-strategies)
8. [Understanding Results](#understanding-results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This RAG pipeline enhances LLM pitch generation by:
- **Retrieving** similar successful pitches from a vector database
- **Using** retrieved examples as context for style and structure
- **Generating** new pitches that learn from successful patterns
- **Evaluating** quality using LLM-as-a-Judge methodology

---

## How RAG Works

### Your Question

> "Why is the retrieval a query of the product? I thought RAG is taking the entire input of a pitch from test.jsonl."

### Answer: How RAG Actually Works

RAG does NOT take the entire pitch from test.jsonl. Instead, it:

1. **Takes INPUT** (product info) from test.jsonl
2. **Queries vector DB** using that INPUT to find similar products
3. **Retrieves similar pitches** from train.jsonl (already indexed)
4. **Generates NEW pitch** using INPUT + retrieved examples
5. **Compares** generated pitch vs OUTPUT (ground truth) from test.jsonl

---

## The Complete RAG Flow

### Step 1: Indexing Phase (One-time setup)

```
train.jsonl (196 products)
    ↓
[Load & Process]
    ↓
[Create Embeddings]
    ↓
[Store in Vector DB]
    ↓
ChromaDB Vector Store
  - 196 pitch documents indexed
  - Each has: pitch text + metadata (company, category, deal info)
```

**What's indexed:**
- Full pitch text from `train.jsonl` → `output` field
- Metadata from `train.jsonl` → `input` field (company, problem, solution, etc.)

---

### Step 2: Evaluation Phase (For each test product)

```
┌─────────────────────────────────────────────────────────────┐
│ FROM test.jsonl (Test Product #1)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ INPUT (Product Info):                                      │
│ {                                                           │
│   "company": "Chubby Buttons",                             │
│   "problem_summary": "Active people struggle to control...",│
│   "solution_summary": "Wearable remote for smartphones...",│
│   "offer": "250,000 for 8%"                                │
│ }                                                           │
│                                                             │
│ OUTPUT (Ground Truth Pitch):                                │
│ "Hi Sharks my name is Justin..."                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Extract Product Info from INPUT                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Query Text Created:                                         │
│ "Chubby Buttons Active people struggle to control          │
│  smartphones wearable remote..."                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Query Vector DB (Search train.jsonl pitches)      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Vector Similarity Search:                                   │
│ "Find pitches similar to 'Chubby Buttons...'"             │
│                                                             │
│ Returns Top-K (e.g., 5) similar pitches:                   │
│                                                             │
│ 1. "Breathometer" (similarity: 0.85)                       │
│    - Problem: Smartphone accessory for safety              │
│    - Pitch: "hello sharks my name is charles..."           │
│                                                             │
│ 2. "MuteMe" (similarity: 0.82)                             │
│    - Problem: Tech product for professionals               │
│    - Pitch: "Hi Sharks, my name is Parm..."                │
│                                                             │
│ 3. "Beulr" (similarity: 0.79)                              │
│    - Problem: Software solution                            │
│    - Pitch: "Hi Sharks, uh my name's Peter..."             │
│                                                             │
│ ... (2 more similar pitches)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Format Retrieved Pitches as Context                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Context String:                                             │
│ """                                                         │
│ Here are some examples of successful Shark Tank pitches:   │
│                                                             │
│ --- Example 1: Breathometer ---                             │
│ hello sharks my name is charles michael yim...             │
│                                                             │
│ --- Example 2: MuteMe ---                                   │
│ Hi Sharks, my name is Parm like parmesan...                │
│                                                             │
│ ... (3 more examples)                                      │
│ """                                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Generate NEW Pitch                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ LLM Prompt:                                                 │
│ """                                                         │
│ [Context: Retrieved similar pitches]                       │
│                                                             │
│ Generate a Shark Tank pitch for:                            │
│ Company: Chubby Buttons                                     │
│ Problem: Active people struggle to control smartphones...  │
│ Solution: Wearable remote for smartphones...               │
│ Ask: 250,000 for 8%                                         │
│ """                                                         │
│                                                             │
│ ↓ LLM Generates ↓                                           │
│                                                             │
│ Generated Pitch:                                            │
│ "Hi Sharks, my name is Justin and this is my business      │
│  partner Mike. We're seeking $250,000 for 8% of our        │
│  company that revolutionizes how active people interact..." │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Compare Generated vs Ground Truth                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Generated Pitch: "Hi Sharks, my name is Justin..."         │
│                                                             │
│ Ground Truth (from test.jsonl OUTPUT):                      │
│ "Hi Sharks my name is Justin this is my business partner..."│
│                                                             │
│ ↓ Evaluator (LLM-as-a-Judge) ↓                             │
│                                                             │
│ Quality Score: 0.85/1.0                                     │
│ - Factual: 0.90 (matches product info)                      │
│ - Narrative: 0.85 (good story flow)                        │
│ - Style: 0.80 (similar to Shark Tank style)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---


### ✅ What RAG Actually Does:

```
test.jsonl INPUT → Extract product info → Search train.jsonl → Retrieve similar → Generate
```

**Why this works:**
1. **No data leakage**: We only use INPUT (product description), not OUTPUT (pitch)
2. **Semantic similarity**: Find products with similar problems/solutions
3. **Learn from examples**: Use retrieved pitches as style/tone examples
4. **Generate new content**: Create pitch for new product using learned patterns

---

## Code Flow Example

### From `evaluate_rag.py`:

```python
# Load test example from test.jsonl
example = test_data[0]  # Chubby Buttons

# example.input = {
#     "company": "Chubby Buttons",
#     "problem_summary": "...",
#     "solution_summary": "...",
#     "offer": "250,000 for 8%"
# }
# example.output = "Hi Sharks my name is Justin..." (ground truth)

# Generate with RAG
prediction = rag_generator.generate(example.input)
# ↑ Uses INPUT only, not OUTPUT!

# Inside rag_generator.generate():
#   1. Extract product info from example.input
#   2. Query vector DB: retriever.retrieve_similar_pitches(product_data=example.input)
#   3. Get top-5 similar pitches from train.jsonl
#   4. Format as context
#   5. Generate new pitch using INPUT + context
#   6. Return generated pitch

# Compare
quality_score = evaluator.get_score(
    pitch_facts=example.input,      # Product info
    ground_truth_pitch=example.output,  # Ground truth
    generated_pitch=prediction.pitch    # Generated pitch
)
```

---

## What Gets Queried?

### The Query Creation Process (`retriever.py`):

```python
def _create_query_from_product(self, product_data: Dict) -> str:
    """Create search query from product data."""
    query_parts = []
    
    # Extract from INPUT:
    query_parts.append(product_data.get('company', ''))        # "Chubby Buttons"
    query_parts.append(product_data.get('problem_summary', '')) # "Active people..."
    query_parts.append(product_data.get('solution_summary', '')) # "Wearable remote..."
    query_parts.append(product_data.get('offer', ''))          # "250,000 for 8%"
    
    # Combine into query string
    query = " ".join(query_parts)
    # → "Chubby Buttons Active people struggle to control smartphones..."
    
    return query
```

### Vector Search:

```python
# Query vector DB with product info
results = vector_store.query(
    query_text="Chubby Buttons Active people struggle...",
    n_results=5  # Top-5 similar
)

# Returns pitches from train.jsonl that are semantically similar
# Based on: product description, problem, solution similarity
```

---

## Key Points

1. **INPUT is used for querying** (product info: company, problem, solution)
2. **OUTPUT is NOT used** (ground truth pitch is only for comparison)
3. **Vector DB contains train.jsonl pitches** (196 indexed documents)
4. **Retrieval finds similar products** (not similar pitches)
5. **Generated pitch is NEW** (created using INPUT + retrieved examples)

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│ test.jsonl                                              │
│ ┌─────────────────┐  ┌──────────────────────────────┐ │
│ │ INPUT (used)     │  │ OUTPUT (not used for query) │ │
│ │ - company        │  │ - Ground truth pitch        │ │
│ │ - problem        │  │ - Only for comparison       │ │
│ │ - solution       │  │                             │ │
│ └─────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         │ Extract product info
         ↓
┌─────────────────────────────────────────────────────────┐
│ Query Vector DB                                         │
│ "Find products similar to: Chubby Buttons..."          │
└─────────────────────────────────────────────────────────┘
         │
         │ Semantic search
         ↓
┌─────────────────────────────────────────────────────────┐
│ train.jsonl (Indexed in Vector DB)                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Retrieved Similar Pitches:                           │ │
│ │ 1. Breathometer (similarity: 0.85)                  │ │
│ │ 2. MuteMe (similarity: 0.82)                        │ │
│ │ 3. Beulr (similarity: 0.79)                         │ │
│ │ ...                                                  │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         │ Use as examples/context
         ↓
┌─────────────────────────────────────────────────────────┐
│ Generate NEW Pitch                                      │
│ INPUT + Retrieved Examples → LLM → Generated Pitch     │
└─────────────────────────────────────────────────────────┘
         │
         │ Compare
         ↓
┌─────────────────────────────────────────────────────────┐
│ Evaluation                                               │
│ Generated Pitch vs OUTPUT (ground truth)                │
│ → Quality Score                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Summary

- **RAG queries with product INPUT** (company, problem, solution) from test.jsonl
- **NOT the full pitch OUTPUT** (that's ground truth for comparison)
- **Finds similar products** from train.jsonl using semantic similarity
- **Uses retrieved pitches as examples** to learn style/tone
- **Generates NEW pitch** combining INPUT + learned patterns
- **Compares** generated vs ground truth OUTPUT for evaluation

This is the standard RAG approach: use structured input to retrieve relevant examples, then generate new content using those examples as context.

---

## Prerequisites

### Required Files

- `data/hf (new)/train.jsonl` - Training data (196 pitches)
- `data/hf (new)/test.jsonl` - Test data (49 pitches)
- `data/hf (new)/category_mapping.json` - Category classifications (optional but recommended)

### Environment Setup

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Set environment variables:**
   Create a `.env` file in the project root:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Install dependencies:**
   ```bash
   pip install dspy-ai chromadb pandas python-dotenv tqdm pydantic
   ```

### Verify Setup

```bash
python -c "import dspy; import chromadb; print('✓ Dependencies installed')"
```

---

## Quick Start

### 1. Classify Categories (First Time Only)

If you haven't classified categories yet:

```bash
python src/pitchLLM/rag/category_classifier.py
```

This creates `data/hf (new)/category_mapping.json` with category assignments for all pitches.

### 2. Index Training Data

Index the training data into the vector database:

```bash
python src/pitchLLM/reindex_with_categories.py
```

This will:
- Load `train.jsonl` (196 pitches)
- Load `category_mapping.json` (if available)
- Create embeddings and store in ChromaDB
- Verify indexing completed successfully

**Output:** Vector database at `chromadb_data/` directory

### 3. Run Quick Test

Test RAG generation on a single product:

```bash
python src/pitchLLM/test_rag_comparison.py
```

This generates pitches using RAG and Non-RAG approaches for comparison.

---

## Running the RAG Pipeline

### Option 1: Compare RAG vs Non-RAG

Run a comprehensive comparison:

```bash
python src/pitchLLM/evaluate_rag.py
```

**What it does:**
- Loads test data from `test.jsonl`
- Generates pitches with RAG (using retrieved examples)
- Generates pitches without RAG (baseline)
- Evaluates quality using LLM-as-a-Judge
- Generates comparison report

**Output:**
- `rag_outputs/rag_output_{timestamp}.json` - Detailed results
- Console output with quality scores and metrics

**Configuration:**
Edit `evaluate_rag.py` to adjust:
- `test_size`: Number of test examples (default: 20)
- `top_k`: Number of retrieved examples (default: 5)
- Retrieval strategy (default: HYBRID_PRIORITIZE)

### Option 2: Compare Multiple RAG Strategies

Compare all three retrieval strategies:

```bash
python src/pitchLLM/compare_rag_strategies.py
```

**Strategies compared:**
1. **Pure Semantic** - Pure similarity-based retrieval
2. **Hybrid Filter** - Semantic + filter by successful deals
3. **Hybrid Prioritize** - Semantic + prioritize successful deals + same category
4. **Non-RAG** - Baseline without retrieval

**Output:**
- `rag_comparison_results/detailed_results_{timestamp}.csv` - Per-example results
- `rag_comparison_results/summary_{timestamp}.json` - Summary metrics
- `rag_comparison_results/comparison_report_{timestamp}.md` - Markdown report

**Example output:**
```
Pure_Semantic:
  Quality Score: 0.629
  Success Rate: 100.0%
  Avg Latency: 0.14s

Hybrid_Filter_Deals:
  Quality Score: 0.630
  Success Rate: 100.0%
  Avg Latency: 0.12s

Hybrid_Prioritize:
  Quality Score: 0.604
  Success Rate: 100.0%
  Avg Latency: 0.13s

Non_RAG:
  Quality Score: 0.624
  Success Rate: 100.0%
  Avg Latency: 0.00s
```

---

## Optimizing RAG with DSPy

Optimize how the RAG program uses retrieved examples using DSPy's MIPROv2 optimizer.

### Run Optimization

```bash
python src/pitchLLM/optimize_rag.py
```

**What it does:**
1. Loads train/val/test sets (all available data)
2. Creates baseline RAG program
3. Evaluates baseline performance
4. Optimizes RAG program with MIPROv2
5. Evaluates optimized performance
6. Compares: Optimized RAG vs Baseline RAG vs Non-RAG

**Configuration:**
Edit `optimize_rag.py` to adjust:
- `train_size`: Training examples (default: None = all ~196)
- `val_size`: Validation examples (default: 16)
- `test_size`: Test examples (default: None = all ~49)
- `mipro_mode`: Optimization intensity - `"light"`, `"medium"`, or `"heavy"` (default: `"light"`)

**Time & Cost:**
- Light mode: ~15-30 minutes, ~$2-5
- Medium mode: ~30-60 minutes, ~$5-10
- Heavy mode: ~60-120 minutes, ~$10-20

**Output:**
- `rag_optimization_results/optimization_results_{timestamp}.json` - Full results

**Example output:**
```
Method                    Quality Score   Success Rate   
-------------------------------------------------------
Non-RAG                   0.609           100.0%         
Baseline RAG              0.643           100.0%         
Optimized RAG             0.643           100.0%         

RAG Improvement: +0.0% (Optimized vs Baseline)
vs Non-RAG: +5.5% improvement
```

### Understanding Optimization

**What MIPROv2 optimizes:**
- **Instructions**: How to use retrieved examples in prompts
- **Few-shot examples**: Which examples to include and how to format them
- **Prompt structure**: Best way to combine context with product data

**What it doesn't optimize:**
- Retrieval strategy (still uses HYBRID_PRIORITIZE)
- Number of retrieved examples (still uses top_k=5)
- Vector database contents (indexing is separate)

**When optimization helps:**
- Baseline RAG underperforms
- You want to improve prompt engineering automatically
- You have sufficient training data (100+ examples)

**When optimization may not help:**
- Baseline RAG already performs well
- Limited training data (<50 examples)
- You want faster iteration (optimization takes time)

---

## Comparing RAG Strategies

### Strategy Comparison Script

```bash
python src/pitchLLM/compare_rag_strategies.py
```

### Manual Strategy Testing

Test individual strategies:

```python
from pitchLLM.models.rag_generator import RAGPitchGenerator
from pitchLLM.rag.retriever import RetrievalStrategy
import dspy

# Setup
lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Pure Semantic
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.PURE_SEMANTIC,
    top_k=5
)

# Hybrid Filter (successful deals only)
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

---

## Understanding Results

### Quality Score

The quality score (0.0-1.0) is a weighted composite:

- **Factual Score (40%)**: Accuracy of product information
- **Narrative Score (40%)**: Story structure and flow
- **Style Score (20%)**: Shark Tank presentation style

**Interpretation:**
- `0.4-0.6`: Baseline performance
- `0.6-0.8`: Good performance
- `0.8-0.9`: Excellent performance
- `0.9+`: Exceptional performance

### Success Rate

Percentage of pitches generated without errors (should be 100% in normal operation).

### Latency

Time to generate one pitch (includes retrieval + generation).

### Retrieval Quality

- **Retrieved**: Number of examples found (should match `top_k`)
- **Similarity**: Average similarity score of retrieved examples (higher = more relevant)

---

## Troubleshooting

### Vector Database Issues

**Problem:** ChromaDB corruption errors
```bash
pyo3_runtime.PanicException: range start index out of range
```

**Solution:** Re-index the database
```bash
python src/pitchLLM/reindex_with_categories.py
```

This automatically resets corrupted databases.

### Missing Categories

**Problem:** Categories not found in metadata
```
⚠️ Category mapping file not found
```

**Solution:** Run category classifier first
```bash
python src/pitchLLM/rag/category_classifier.py
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'dspy'`

**Solution:** Ensure virtual environment is activated
```bash
source venv/bin/activate
python --version  # Should show Python 3.11+
```

### Rate Limiting

**Problem:** `RateLimitError: Rate limit reached`

**Solution:** 
- Wait a few minutes and retry
- Reduce `test_size` in evaluation scripts
- Use rate limiting delays (already implemented)

### Low Quality Scores

**Possible causes:**
1. **Insufficient training data**: Need at least 50+ examples
2. **Poor retrieval**: Retrieved examples not relevant
3. **Model issues**: Check API key and model availability

**Debugging:**
```python
# Check retrieval quality
from pitchLLM.rag.retriever import PitchRetriever

retriever = PitchRetriever()
similar = retriever.retrieve_similar_pitches(
    product_data=test_product,
    top_k=5,
    include_scores=True
)

# Print retrieved examples
for pitch in similar:
    print(f"{pitch['product_name']}: {pitch['similarity_score']:.3f}")
```

---

## Pipeline Execution Order

### First-Time Setup

```bash
# 1. Classify categories
python src/pitchLLM/rag/category_classifier.py

# 2. Index training data
python src/pitchLLM/reindex_with_categories.py

# 3. Verify setup
python src/pitchLLM/test_rag_comparison.py
```

### Regular Usage

```bash
# Option A: Quick comparison
python src/pitchLLM/evaluate_rag.py

# Option B: Strategy comparison
python src/pitchLLM/compare_rag_strategies.py

# Option C: Optimization (takes longer)
python src/pitchLLM/optimize_rag.py
```

### After Data Updates

If you update `train.jsonl` or `category_mapping.json`:

```bash
# Re-index with new data
python src/pitchLLM/reindex_with_categories.py
```

---

## File Structure

```
src/pitchLLM/
├── rag/
│   ├── category_classifier.py    # Classify pitches into categories
│   ├── data_indexer.py           # Index data into vector DB
│   ├── retriever.py              # RAG retrieval logic
│   └── vector_store.py           # ChromaDB wrapper
├── models/
│   ├── rag_generator.py          # RAG-enhanced generator
│   ├── generator.py               # Non-RAG generator
│   └── evaluator.py              # LLM-as-a-Judge evaluator
├── evaluate_rag.py               # RAG vs Non-RAG comparison
├── compare_rag_strategies.py     # Multi-strategy comparison
├── optimize_rag.py               # DSPy optimization
└── reindex_with_categories.py    # Re-index vector DB

data/hf (new)/
├── train.jsonl                   # Training data (196 pitches)
├── test.jsonl                    # Test data (49 pitches)
└── category_mapping.json         # Category assignments

rag_comparison_results/            # Strategy comparison outputs
rag_optimization_results/          # Optimization outputs
chromadb_data/                     # Vector database storage
```

---

## Next Steps

1. **Experiment with retrieval strategies**: Try different `top_k` values and strategies
2. **Optimize prompts**: Use MIPROv2 to automatically improve prompt engineering
3. **Analyze results**: Review detailed CSV outputs to understand what works best
4. **Fine-tune categories**: Adjust category taxonomy if needed
5. **Scale up**: Increase test size for more reliable metrics

---

## Additional Resources

- **Architecture Visualization**: See `ARCHITECTURE_VISUALIZATION.md`
- **Pipeline Instructions**: See `PIPELINE_INSTRUCTIONS.md`
- **RAG Optimization Guide**: See `RAG_OPTIMIZATION_GUIDE.md`

---

## Summary

This RAG pipeline:
- ✅ Uses product INPUT (not pitch OUTPUT) to query vector database
- ✅ Retrieves similar pitches from training data
- ✅ Generates new pitches using retrieved examples as context
- ✅ Evaluates quality using LLM-as-a-Judge
- ✅ Supports multiple retrieval strategies
- ✅ Can be optimized with DSPy MIPROv2

**Key Insight:** RAG queries with product information to find semantically similar products, then uses their pitches as style examples to generate new content. This avoids data leakage while learning from successful patterns.

