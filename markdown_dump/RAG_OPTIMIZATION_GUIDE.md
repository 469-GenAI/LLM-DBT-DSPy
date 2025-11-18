# RAG + DSPy Optimization Guide

## Question 1: Is RAG a Pre-Step Before DSPy?

**Answer: No, RAG is integrated INTO DSPy, not a pre-step.**

### How It Works:

```
RAGStructuredPitchProgram (DSPy Module)
    ↓
forward() method:
    1. Retrieves examples from vector DB (RAG retrieval)
    2. Formats retrieved examples as context
    3. Calls dspy.ChainOfThought (DSPy generation)
    4. Generates pitch using context + product data
```

**Key Point:** The RAG retrieval happens **inside** the DSPy module's `forward()` method. It's part of the DSPy program, not a separate preprocessing step.

### Comparison:

| Approach | When Retrieval Happens | Integration |
|----------|----------------------|------------|
| **Your RAG** | Inside DSPy Module (`forward()`) | ✅ Fully integrated |
| **Pre-step RAG** | Before DSPy (separate function) | ❌ Not your approach |
| **KNNFewShot** | Inside DSPy optimizer | ✅ DSPy's built-in RAG |

---

## Question 2: Optimizing RAG with DSPy

### What Gets Optimized?

When you optimize `RAGStructuredPitchProgram` with MIPROv2, DSPy optimizes:

1. **Instructions**: How to use retrieved examples
   - "Use these examples as inspiration for style"
   - "Learn from the narrative structure"
   - "Match the tone and energy"

2. **Example Formatting**: How to present retrieved examples
   - Order of examples
   - What information to include
   - How to structure the context

3. **Prompt Structure**: How to combine context + product data
   - Where to place examples
   - How to reference them
   - Balance between context and product info

### Expected Improvements:

- **Current RAG**: 0.629 quality score
- **Optimized RAG**: 0.75-0.80 quality score (+20-30%)
- **Better prompt instructions**
- **Better example selection/formatting**

---

## Running the Optimization

### Step 1: Run the Optimization Script

```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate
python src/pitchLLM/optimize_rag.py
```

### What It Does:

1. **Loads datasets** (train/val/test from `train.jsonl` and `test.jsonl`)
2. **Creates RAG program** with Hybrid Prioritize strategy
3. **Evaluates baseline RAG** (non-optimized) on test set
4. **Optimizes with MIPROv2** (takes 15-30 minutes, costs ~$2-5)
5. **Evaluates optimized RAG** on same test set
6. **Compares results**: Non-RAG vs Baseline RAG vs Optimized RAG

### Configuration:

You can modify these in `optimize_rag.py`:

```python
train_size = 50      # Training examples for optimization
val_size = 15        # Validation examples
test_size = 20       # Test examples for evaluation
mipro_mode = "light" # light/medium/heavy
```

### Expected Output:

```
================================================================================
FINAL COMPARISON
================================================================================

Method                   Quality Score   Success Rate   
-------------------------------------------------------
Non-RAG                  0.624           100.0%         
Baseline RAG             0.629           100.0%         
Optimized RAG            0.750           100.0%         

RAG Improvement: +19.2% (Optimized vs Baseline)
vs Non-RAG: +20.2% improvement
```

### Results Saved To:

- `rag_optimization_results/optimization_results_{timestamp}.json`

---

## Understanding the Optimization Process

### Before Optimization:

```python
# RAG program uses basic prompt
context = "Here are some example pitches: [retrieved examples]"
prompt = f"{context}\n\nGenerate a pitch for: {product_data}"
```

### After Optimization:

```python
# MIPROv2 optimizes the prompt structure
optimized_instructions = """
Study these successful pitches carefully:
1. Notice how they start with founder introduction
2. See how they present the problem from customer perspective
3. Observe the narrative flow: problem → solution → ask
4. Match the energy and style

Now generate a pitch following this structure...
"""

# Optimized example selection and formatting
optimized_context = format_examples_better(retrieved_examples)
optimized_prompt = combine_optimally(optimized_instructions, optimized_context, product_data)
```

---

## Next Steps After Optimization

1. **Compare Results**: Check if optimized RAG beats baseline
2. **Save Optimized Program**: The optimized program can be saved and reused
3. **Further Improvements**:
   - Try different retrieval strategies
   - Experiment with top-k values
   - Add reranking
   - Improve embeddings

---

## Troubleshooting

### If Optimization Fails:

- Check that vector store is indexed: `chromadb_data/` exists
- Verify train.jsonl and test.jsonl exist
- Ensure GROQ_API_KEY is set
- Try reducing `train_size` for faster iteration

### If Results Don't Improve:

- Try `mipro_mode="medium"` for more thorough optimization
- Increase `train_size` for more training examples
- Check retrieval quality (are retrieved examples relevant?)
- Consider improving embeddings or adding reranking

