# RAG Quick Reference Guide

Quick commands and code snippets for common RAG operations.

---

## 🚀 Quick Commands

### Setup (First Time)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Classify categories
python src/pitchLLM/rag/category_classifier.py

# 3. Index training data
python src/pitchLLM/reindex_with_categories.py
```

### Run Comparisons

```bash
# Compare RAG strategies vs Non-RAG
python src/pitchLLM/compare_rag_strategies.py

# Compare RAG vs KNNFewShot
python src/pitchLLM/compare_rag_vs_knn.py

# Quick evaluation
python src/pitchLLM/evaluate_rag.py
```

### Troubleshooting

```bash
# Re-index if ChromaDB corrupted
python src/pitchLLM/reindex_with_categories.py

# Test retrieval
python src/pitchLLM/test_rag_comparison.py
```

---

## 📝 Code Snippets

### Basic RAG Generation

```python
from pitchLLM.models.rag_generator import RAGPitchGenerator
from pitchLLM.rag.retriever import RetrievalStrategy
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# Setup
lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",
    model_type="chat",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1.0
)

# Create generator
generator = RAGPitchGenerator(
    lm=lm,
    retrieval_strategy=RetrievalStrategy.PURE_SEMANTIC,
    top_k=5
)

# Generate
product_data = {
    "company": "My Product",
    "problem_summary": "...",
    "solution_summary": "...",
    "offer": "100,000 for 10%"
}

prediction = generator.generate(product_data)
print(prediction.pitch)
```

### Manual Retrieval

```python
from pitchLLM.rag.retriever import PitchRetriever, RetrievalStrategy

retriever = PitchRetriever(auto_index=True)

similar = retriever.retrieve_similar_pitches(
    product_data=product_data,
    top_k=5,
    strategy=RetrievalStrategy.PURE_SEMANTIC,
    include_scores=True
)

for pitch in similar:
    print(f"{pitch['product_name']}: {pitch['similarity_score']:.3f}")
```

### Evaluation

```python
from pitchLLM.models import PitchEvaluator
import dspy

evaluator_lm = dspy.LM(
    "groq/openai/gpt-oss-120b",
    model_type="chat",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

evaluator = PitchEvaluator(evaluator_lm)

score = evaluator.get_score(
    pitch_facts=json.dumps(product_data, indent=2),
    ground_truth_pitch=ground_truth,
    generated_pitch=generated_pitch
)
```

---

## 🔧 Configuration

### Model Settings

```python
# Generator (for pitch generation)
generator_lm = dspy.LM(
    "groq/llama-3.3-70b-versatile",  # or "groq/llama-3.1-8b-instant"
    model_type="chat",
    api_key=GROQ_API_KEY,
    temperature=1.0  # For diverse outputs
)

# Evaluator (for scoring)
evaluator_lm = dspy.LM(
    "groq/openai/gpt-oss-120b",
    model_type="chat",
    api_key=GROQ_API_KEY,
    temperature=0.1  # For consistent scoring
)
```

### Retrieval Strategies

```python
# Pure Semantic (best performance)
RetrievalStrategy.PURE_SEMANTIC

# Hybrid Filter (filter by deals)
RetrievalStrategy.HYBRID_FILTER
filter_successful_deals=True

# Hybrid Prioritize (prioritize deals + category)
RetrievalStrategy.HYBRID_PRIORITIZE
prioritize_successful=True
prioritize_category=True
```

---

## 📊 Understanding Results

### Quality Scores

- `0.4-0.6`: Baseline
- `0.6-0.8`: Good
- `0.8-0.9`: Excellent
- `0.9+`: Exceptional

### Overlap Metrics

- **Jaccard Similarity**: 0.0-1.0 (higher = more overlap)
- **Overlap Count**: Number of shared examples
- **Low (<0.1)**: Methods select different examples
- **High (>0.5)**: Methods select similar examples

---

## 🗂️ File Locations

```
Data:
  data/hf (new)/train.jsonl          # Training data (indexed)
  data/hf (new)/test.jsonl           # Test data (not indexed)
  data/hf (new)/category_mapping.json
  data/hf (new)/deal_mapping.json

Results:
  rag_comparison_results/            # Comparison outputs

Vector DB:
  chromadb_data/shark_tank_pitches/  # ChromaDB storage
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| ChromaDB corruption | `python src/pitchLLM/reindex_with_categories.py` |
| Missing categories | `python src/pitchLLM/rag/category_classifier.py` |
| Import errors | `source venv/bin/activate` |
| Model not found | Use `groq/llama-3.1-8b-instant` (not 3.3-8b) |
| Rate limiting | Wait and retry, reduce test_size |

---

## 📚 Full Documentation

See `RAG_README.md` for complete documentation.

