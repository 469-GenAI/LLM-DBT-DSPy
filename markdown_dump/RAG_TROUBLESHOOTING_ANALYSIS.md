# RAG Troubleshooting Analysis

## ✅ DSPy Usage Confirmation

**All runs ARE going through DSPy!**

### Evidence:

1. **RAG Generators use DSPy Modules:**
   ```python
   # From rag_generator.py
   class RAGStructuredPitchProgram(dspy.Module):  # ← DSPy Module
       def __init__(self):
           super().__init__()
           self.generate_pitch = dspy.ChainOfThought(RAGPitchGenerationSig)  # ← DSPy ChainOfThought
       
       def forward(self, input: dict):
           # Retrieves examples
           similar_pitches = self.retriever.retrieve_similar_pitches(...)
           context = self.retriever.format_context_for_prompt(similar_pitches)
           
           # Uses DSPy to generate
           prediction = self.generate_pitch(  # ← DSPy ChainOfThought call
               context=context,
               pitch_data=formatted_input
           )
   ```

2. **Non-RAG Generator also uses DSPy:**
   ```python
   # From generator.py
   class StructuredPitchProgram(dspy.Module):  # ← DSPy Module
       def __init__(self):
           super().__init__()
           self.generate_pitch = dspy.ChainOfThought(PitchGenerationSig)  # ← DSPy ChainOfThought
   ```

3. **Both use `dspy.context()`:**
   ```python
   # RAG Generator
   with dspy.context(lm=self.lm):  # ← DSPy context
       prediction = self.program(input=input_data)
   
   # Non-RAG Generator
   with dspy.context(**context_params):  # ← DSPy context
       return self.program(input=input_data)
   ```

**Conclusion:** ✅ All generators are using DSPy properly. The issue is NOT with DSPy integration.

---

## 📊 Current Results Analysis

### Latest Run (2025-11-14):

| Strategy | Quality Score | vs Non-RAG | Improvement |
|----------|---------------|------------|-------------|
| **Non_RAG** | 0.624 | Baseline | - |
| **Pure_Semantic** | 0.629 | +0.8% | ✅ Small improvement |
| **Hybrid_Filter_Deals** | 0.630 | +1.0% | ✅ Best improvement |
| **Hybrid_Prioritize** | 0.604 | -3.2% | ❌ Worse! |

### Key Observations:

1. **Very small improvements** (+0.8% to +1.0%)
2. **Hybrid_Prioritize performs WORSE** (-3.2%)
3. **All strategies have 100% success rate** (no generation errors)
4. **RAG generates longer pitches** (1867 vs 1272 chars)
5. **RAG uses more tokens** (411 vs 277 tokens)

---

## 🔍 Why Improvements Are So Small

### 1. **Baseline (Non-RAG) is Already Strong**

**Non-RAG Quality: 0.624** - This is already quite good!

- The LLM (Llama 3.3 70B) is powerful enough to generate decent pitches without examples
- DSPy's ChainOfThought helps structure the generation
- The structured input format provides good context

**Implication:** There's less room for improvement because the baseline is already strong.

---

### 2. **Retrieved Examples May Not Be Highly Relevant**

**Average Similarity Scores:**
- Pure_Semantic: 0.396 (39.6% similarity)
- Hybrid_Filter: 0.368 (36.8% similarity)
- Hybrid_Prioritize: 0.390 (39.0% similarity)

**Analysis:**
- Similarity scores are relatively low (30-40%)
- This suggests retrieved examples may not be highly relevant
- Low relevance = less useful context = smaller improvement

**Example from results:**
```
Query: "Smart air filter" (Woosh)
Retrieved:
- Beulr (similarity: 0.400) - What is this?
- BeSomebody (similarity: 0.400) - Not related to air filters
- Black Sands Entertainment (similarity: 0.397) - Completely different
```

**Problem:** Retrieved examples may not be semantically similar enough to help.

---

### 3. **Context Formatting May Not Be Optimal**

**Current Context Format:**
```python
# From retriever.py lines 497-513
context = """
Here are some examples of successful Shark Tank pitches for similar products:

--- Example 1: Beulr ---
[Full pitch text...]

--- Example 2: BeSomebody ---
[Full pitch text...]
...
"""
```

**Issues:**
1. **No instruction on HOW to use examples** - LLM may not know what to learn
2. **Full pitch text** - May be too long, diluting signal
3. **No highlighting of key patterns** - LLM has to figure out what's important
4. **No comparison guidance** - LLM doesn't know what to extract

**What DSPy optimization would improve:**
- Better instructions: "Analyze the opening hooks in examples 1-2..."
- Pattern extraction: "Notice how successful pitches start with a question..."
- Structured examples: Extract key sections (hook, problem, solution, ask)

---

### 4. **Hybrid_Prioritize May Be Over-Prioritizing**

**Priority Distribution from Logs:**
```
Priority distribution: {2: 1, 4: 4}
```

**Analysis:**
- Priority 1 (✅ Deal + ✅ Same Category): 0 examples
- Priority 2 (✅ Deal + ❌ Different Category): 1 example
- Priority 4 (❌ No Deal + ❌ Different Category): 4 examples

**Problem:**
- No Priority 1 examples found (successful deals in same category)
- Mostly Priority 4 (lowest priority)
- Prioritization may be selecting LESS relevant examples

**Why this happens:**
- Limited successful deals in same category
- Category matching may be too strict
- Prioritization logic may be excluding good examples

---

### 5. **Small Test Set (10 examples)**

**Current test size: 10 examples**

**Issues:**
- Small sample = high variance
- One bad example can skew results significantly
- May not represent true performance

**Recommendation:** Test on full test set (49 examples) for more reliable results.

---

### 6. **LLM May Not Be Using Context Effectively**

**Evidence:**
- RAG generates longer pitches (1867 vs 1272 chars)
- But quality improvement is minimal (+0.8%)
- Suggests LLM is adding content but not improving quality

**Possible reasons:**
1. **Context is too generic** - "Here are some examples..." doesn't guide usage
2. **No explicit instructions** - LLM doesn't know what to learn from examples
3. **Examples may be distracting** - Low relevance examples may confuse LLM
4. **Temperature/randomness** - LLM may not consistently use context

---

## 🎯 Why Hybrid_Prioritize Performs Worse

### Priority Distribution Analysis:

From logs: `Priority distribution: {2: 1, 4: 4}`

**What this means:**
- **Priority 1** (✅ Deal + ✅ Same Category): **0 examples** ← Best, but none found!
- **Priority 2** (✅ Deal + ❌ Different Category): **1 example**
- **Priority 3** (❌ No Deal + ✅ Same Category): **0 examples**
- **Priority 4** (❌ No Deal + ❌ Different Category): **4 examples** ← Lowest quality

**Problem:**
1. **No high-priority examples** - System can't find successful deals in same category
2. **Mostly low-priority examples** - 4 out of 5 are Priority 4 (worst)
3. **Prioritization backfires** - By prioritizing, we're selecting LESS relevant examples

**Why this happens:**
- Limited successful deals in the dataset
- Category matching may be too strict
- Semantic similarity may find better examples, but prioritization overrides it

**Example:**
```
Semantic search finds:
1. FitBit (similarity: 0.88, has_deal: True, category: "Health & Fitness") ← Great!
2. Garmin (similarity: 0.82, has_deal: True, category: "Health & Fitness") ← Great!

But if query product category is "Technology":
- FitBit gets Priority 2 (✅ Deal + ❌ Different Category)
- Lower similarity but same category gets Priority 3
- Prioritization may select Priority 3 over Priority 2, even if Priority 2 is more similar
```

---

## 💡 Recommendations to Improve RAG Performance

### 1. **Improve Context Formatting**

**Current:**
```python
context = "Here are some examples of successful Shark Tank pitches..."
```

**Better:**
```python
context = """
Analyze these successful Shark Tank pitches and extract key patterns:

PATTERN 1 - Opening Hooks:
Example 1 (Beulr): "When was the last time you..."
Example 2 (FitBit): "Have you ever stopped to think..."

PATTERN 2 - Problem Storytelling:
Example 1: Focuses on customer pain point
Example 2: Uses emotional appeal

PATTERN 3 - Solution Presentation:
Example 1: Highlights key features
Example 2: Emphasizes benefits

Now generate a pitch that:
- Uses similar opening hook style
- Tells problem story like Example 1
- Presents solution like Example 2
"""
```

**Implementation:** Use DSPy optimization (MIPROv2) to learn optimal formatting.

---

### 2. **Improve Retrieval Quality**

**Current issues:**
- Low similarity scores (30-40%)
- Retrieved examples may not be relevant

**Solutions:**

**A. Better Query Construction:**
```python
# Current: Uses product description
query = product_data.get('product_description', '')

# Better: Use multiple fields with weights
query = f"""
Product: {product_data.get('company', '')}
Problem: {product_data.get('problem_summary', '')}
Solution: {product_data.get('solution_summary', '')}
Category: {product_data.get('category', '')}
"""
```

**B. Increase top_k:**
```python
# Current: top_k=5
# Try: top_k=10 or top_k=15
# Then rerank to select best 5
```

**C. Use Better Embedding Model:**
```python
# Current: all-MiniLM-L6-v2 (384 dims)
# Better: all-mpnet-base-v2 (768 dims, better quality)
# Or: BAAI/bge-small-en-v1.5 (384 dims, optimized for retrieval)
```

---

### 3. **Fix Hybrid_Prioritize Logic**

**Current problem:** Prioritization selects lower-quality examples

**Solutions:**

**A. Make prioritization less aggressive:**
```python
# Current: Strict priority ordering
# Better: Weighted combination
score = (similarity * 0.7) + (priority_bonus * 0.3)
# Where priority_bonus: Priority 1 = +0.2, Priority 2 = +0.1, etc.
```

**B. Require minimum similarity:**
```python
# Only prioritize if similarity > threshold (e.g., 0.35)
if similarity > 0.35:
    apply_prioritization()
else:
    use_pure_semantic()
```

**C. Fallback to semantic if no Priority 1:**
```python
# If no Priority 1 examples found, use pure semantic
if priority_1_count == 0:
    return pure_semantic_results()
```

---

### 4. **Optimize with DSPy MIPROv2**

**What MIPROv2 optimizes:**
- **Instructions:** How to use retrieved examples
- **Formatting:** How to present examples
- **Selection:** Which examples to use
- **Structure:** How to combine context + product data

**Expected improvement:** +10-20% quality score

**Run:**
```bash
python src/pitchLLM/optimize_rag.py
```

---

### 5. **Test on Larger Dataset**

**Current:** 10 examples (high variance)

**Better:** Use full test set (49 examples)

**Modify:**
```python
# In compare_rag_strategies.py
comparator = RAGStrategyComparator(
    test_size=49,  # Use all test examples
    top_k=5
)
```

---

### 6. **Improve Category Matching**

**Current issue:** Category matching may be too strict

**Solutions:**

**A. Fuzzy category matching:**
```python
# Current: Exact match
category_match = (candidate_category.lower() == product_category.lower())

# Better: Fuzzy matching
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, candidate_category.lower(), product_category.lower()).ratio()
category_match = similarity > 0.7  # 70% similarity threshold
```

**B. Category hierarchy:**
```python
# Define category relationships
CATEGORY_HIERARCHY = {
    "Health & Fitness": ["Technology & Software", "Home & Lifestyle"],
    "Food & Beverage": ["Home & Lifestyle"],
    ...
}
```

---

### 7. **Analyze Retrieved Examples**

**Add debugging:**
```python
# Log what examples are retrieved
logger.info(f"Retrieved examples for {product_name}:")
for i, pitch in enumerate(similar_pitches, 1):
    logger.info(f"  {i}. {pitch['product_name']} (similarity: {pitch['similarity_score']:.3f}, "
                f"has_deal: {pitch['metadata'].get('has_deal')}, "
                f"category: {pitch['metadata'].get('category')})")
```

**Check:**
- Are retrieved examples actually similar?
- Are they successful deals?
- Are they in the same category?

---

## 📈 Expected Improvements After Fixes

| Fix | Expected Improvement |
|-----|---------------------|
| Better context formatting | +2-5% |
| Better retrieval (higher similarity) | +3-7% |
| Fix Hybrid_Prioritize | +2-4% |
| DSPy MIPROv2 optimization | +10-20% |
| **Combined** | **+15-30%** |

**Target:** Quality score of 0.70-0.80 (vs current 0.63)

---

## 🔬 Diagnostic Steps

### Step 1: Check Retrieved Examples

```python
# Add to compare_rag_strategies.py
def analyze_retrieval(self, example, generator):
    """Analyze what examples are retrieved."""
    if hasattr(generator, 'program') and hasattr(generator.program, 'retriever'):
        similar_pitches = generator.program.retriever.retrieve_similar_pitches(
            product_data=example.input,
            top_k=5,
            include_scores=True
        )
        
        print(f"\nRetrieved examples:")
        for i, pitch in enumerate(similar_pitches, 1):
            print(f"  {i}. {pitch['product_name']}")
            print(f"     Similarity: {pitch.get('similarity_score', 0):.3f}")
            print(f"     Has Deal: {pitch['metadata'].get('has_deal', False)}")
            print(f"     Category: {pitch['metadata'].get('category', 'N/A')}")
```

### Step 2: Compare Context Quality

**Add logging to see what context is passed:**
```python
# In RAGStructuredPitchProgram.forward()
logger.debug(f"Context length: {len(context)} chars")
logger.debug(f"Context preview: {context[:500]}...")
```

### Step 3: Test Different top_k Values

```python
# Test with different top_k
for top_k in [3, 5, 7, 10]:
    generator = RAGPitchGenerator(lm, top_k=top_k)
    # Run comparison
```

---

## 📝 Summary

### ✅ Confirmed:
1. **DSPy IS being used** - All generators use `dspy.Module` and `dspy.ChainOfThought`
2. **RAG is working** - Examples are retrieved and passed to LLM
3. **Small improvements** - +0.8% to +1.0% improvement over baseline

### ❌ Issues Identified:
1. **Baseline is already strong** (0.624) - Less room for improvement
2. **Low retrieval similarity** (30-40%) - Examples may not be highly relevant
3. **Context formatting is generic** - No guidance on how to use examples
4. **Hybrid_Prioritize backfires** - Selects lower-quality examples
5. **Small test set** (10 examples) - High variance

### 🎯 Next Steps:
1. **Run DSPy optimization** (`optimize_rag.py`) - Expected +10-20% improvement
2. **Improve context formatting** - Add explicit instructions
3. **Fix Hybrid_Prioritize** - Make it less aggressive or add fallback
4. **Test on full dataset** (49 examples) - More reliable results
5. **Improve retrieval quality** - Better query construction, higher top_k, better embeddings

---

## 🚀 Quick Wins

1. **Increase test size to 49** (full test set)
2. **Add explicit context instructions** (tell LLM what to learn)
3. **Fix Hybrid_Prioritize** (add fallback to semantic if no Priority 1)
4. **Run MIPROv2 optimization** (biggest expected improvement)


