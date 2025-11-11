# RAG Retrieval Process - Complete Explanation

## Your Question

> "Why is the retrieval a query of the product? I thought RAG is taking the entire input of a pitch from test.jsonl."

## Answer: How RAG Actually Works

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

