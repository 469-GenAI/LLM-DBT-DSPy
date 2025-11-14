# DSPy Multi-Agent Intelligent LLM for Tackling Business-Pitches

Singapore Management University IS459 Generative AI with LLMs

## Setup Instructions

1. Create a new conda environment:

   ```sh
   conda create -n sharktank python=3.10
   ```

2. Activate the conda environment:

   ```sh
   conda activate sharktank
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements_clean.txt
   ```

4. Authenticate with HuggingFace to access the dataset:

   ```sh
   huggingface-cli login
   ```

   The project uses the `isaidchia/sharktank_pitches_modified` dataset, which requires authentication.

5. Create a `.env` file at the root directory with the following content:

   ```env
   # Required for Groq models (default provider)
   GROQ_API_KEY="your_groq_api_key"

   # Optional: For MLflow tracking and experiment management
   DATABRICKS_PATH="your_databricks_path"

   # Optional: For AWS Bedrock models
   BEDROCK_API_KEY="your_bedrock_api_key"

   # Optional: For other providers
   OPENAI_API_KEY="your_openai_api_key"
   ANTHROPIC_API_KEY="your_anthropic_api_key"
   DEEPSEEK_API_KEY="your_deepseek_api_key"
   ```

   **Note:** `GROQ_API_KEY` is required for the default Groq models. Other API keys are optional and only needed if you switch to different model providers.

## DSPy Pitch Generation Systems

This project implements two DSPy-based pitch generation systems for creating Shark Tank-style business pitches from structured input data.

### A. Structured Pitch Generation (`pitchLLM_structured.py`)

**Location:** `src/pitchLLM/pitchLLM_structured.py`

A simple, direct approach to pitch generation that takes structured input dictionaries and generates narrative pitches.

**Key Features:**

- Uses `PitchGenerator` with `StructuredPitchProgram` module
- Supports multiple optimization methods: `none`, `bootstrap`, `bootstrap_random`, `knn`, `mipro`
- Built-in rate limiting for Groq API (30 requests/minute, default 2.5s delay)
- Optional MLflow integration for experiment tracking
- Detailed evaluation with `AssessPitchQuality` signature

**Basic Usage:**

```sh
# Run without optimization (baseline)
python src/pitchLLM/pitchLLM_structured.py --optimization none --test-size 10

# Run with MIPRO optimization
python src/pitchLLM/pitchLLM_structured.py --optimization mipro --train-size 30 --test-size 10 --save-program

# Run with detailed evaluation
python src/pitchLLM/pitchLLM_structured.py --optimization mipro --train-size 30 --test-size 10 --evaluate

# Use different models
python src/pitchLLM/pitchLLM_structured.py \
  --optimization bootstrap \
  --generator-model "groq/llama-3.1-8b-instant" \
  --evaluator-model "groq/openai/gpt-oss-120b" \
  --test-size 5
```

**Available Arguments:**

- `--optimization`: Optimization method (`none`, `bootstrap`, `bootstrap_random`, `knn`, `mipro`)
- `--generator-model`: Model for pitch generation (default: `groq/llama-3.3-70b-versatile`)
- `--evaluator-model`: Model for evaluation (default: `groq/openai/gpt-oss-120b`)
- `--train-size`: Number of training examples (default: all)
- `--test-size`: Number of test examples (default: 10)
- `--evaluate`: Enable detailed AssessPitch evaluation
- `--save-program`: Save optimized program after compilation
- `--no-rate-limit`: Disable rate limiting (use with caution)

### B. Mixture-of-Agents (`MoA_DSPy.py`)

**Location:** `src/MoA/MoA_DSPy/MoA_DSPy.py`

A multi-agent pipeline that decomposes pitch generation into specialized agent roles, then synthesizes their outputs into a cohesive pitch.

**Architecture:**

1. **TaskPlanner**: Decomposes facts into agent roles and subtasks
2. **AgentEnsemble**: Multiple specialized agents write focused sections
3. **PitchSynthesizer**: Combines agent drafts into final pitch

**Key Features:**

- Multi-agent collaboration with configurable agent count (default: 3)
- Supports optimization: `none`, `mipro`, `simba`, `bootstrap`, `bootstrap_random`, `knn`
- Uses HuggingFace dataset: `isaidchia/sharktank_pitches_modified`
- Automatic task decomposition and role assignment
- Fallback to `PitchGenerator` if synthesis fails

**Basic Usage:**

```sh
# Run MoA pipeline with default settings
python src/MoA/MoA_DSPy/MoA_DSPy.py --optimization none --test-size 10

# Run with MIPRO optimization and custom agent count
python src/MoA/MoA_DSPy/MoA_DSPy.py \
  --optimization mipro \
  --train-size 30 \
  --test-size 10 \
  --num-agents 3 \
  --save-program

# Run with SIMBA optimizer
python src/MoA/MoA_DSPy/MoA_DSPy.py \
  --optimization simba \
  --train-size 20 \
  --test-size 10 \
  --num-agents 4

# Use different models
python src/MoA/MoA_DSPy/MoA_DSPy.py \
  --optimization bootstrap \
  --generator-model "groq/llama-3.3-70b-versatile" \
  --evaluator-model "groq/openai/gpt-oss-120b" \
  --temperature 1.0 \
  --max-tokens 2048
```

**Available Arguments:**

- `--optimization`: Optimization method (`none`, `mipro`, `simba`, `bootstrap`, `bootstrap_random`, `knn`)
- `--generator-model`: Model for pitch generation (default: `groq/llama-3.3-70b-versatile`)
- `--evaluator-model`: Model for evaluation (default: `groq/openai/gpt-oss-120b`)
- `--train-size`: Number of training examples (default: 20)
- `--test-size`: Number of test examples (default: 10)
- `--num-agents`: Number of agents in ensemble (default: 3)
- `--temperature`: Sampling temperature (default: 1.0)
- `--max-tokens`: Maximum tokens for generation (default: 2048)
- `--save-program`: Save optimized program after compilation
- `--save-dir`: Directory to save programs (default: `MoA/optimised_programs`)
- `--run-name`: Label for persisted artifacts (default: `moa_dspy_run`)
- `--seed`: Random seed for reproducibility (default: 42)

## Optimization Methods

The DSPy systems support several optimization methods to improve pitch quality:

- **`none`**: Baseline without optimization. Fastest option, useful for testing.

- **`bootstrap`**: BootstrapFewShot selects high-quality few-shot examples from the training set to improve performance.

- **`bootstrap_random`**: BootstrapFewShotWithRandomSearch extends bootstrap with random search over candidate programs for better exploration.

- **`knn`**: KNNFewShot uses K-nearest neighbors to dynamically select relevant training examples at inference time. Requires `sentence-transformers` for embeddings.

- **`mipro`**: MIPROv2 performs multi-stage instruction/prompt optimization to refine the generation process. Most effective but slower.

- **`simba`**: SIMBA optimizer (MoA only) uses feedback-based refinement. Provides detailed feedback for prompt improvement.

**Recommendation:** Start with `none` for baseline, then try `mipro` for best results, or `knn` for faster optimization with good performance.

## Output Locations

Results and optimized programs are saved in the following locations:

- **Results CSV**: `MoA/results/structured_pitch_results_{optimization_method}_{run_name}_{timestamp}_{mlflow_run_id}.csv`

  - Contains evaluation scores: `final_score`, `factual_score`, `narrative_score`, `style_score`
  - Includes generated pitches, ground truth, and assessment reasoning

- **Optimized Programs**: `MoA/optimised_programs/pitch_MoA_{optimization_method}_{timestamp}.json`
  - Saved when using `--save-program` flag
  - Contains full program state with metadata (models used, training info, MLflow run ID)

## Quick Start Examples

### Example 1: Basic Pitch Generation (No Optimization)

```sh
# Simple baseline run
python src/pitchLLM/pitchLLM_structured.py \
  --optimization none \
  --test-size 5
```

### Example 2: Optimized Pitch Generation with MIPRO

```sh
# Full optimization run with evaluation
python src/pitchLLM/pitchLLM_structured.py \
  --optimization mipro \
  --train-size 30 \
  --test-size 10 \
  --evaluate \
  --save-program
```

### Example 3: MoA Pipeline with Custom Configuration

```sh
# Multi-agent pipeline with 4 agents
python src/MoA/MoA_DSPy/MoA_DSPy.py \
  --optimization mipro \
  --train-size 30 \
  --test-size 10 \
  --num-agents 4 \
  --save-program
```

### Example 4: Fast Testing with Smaller Models

```sh
# Quick test with smaller model
python src/pitchLLM/pitchLLM_structured.py \
  --optimization bootstrap \
  --generator-model "groq/llama-3.1-8b-instant" \
  --train-size 10 \
  --test-size 3 \
  --no-rate-limit
```

**Note:** Rate limiting is enabled by default (2.5s delay) to respect Groq's 30 requests/minute limit. Disable with `--no-rate-limit` only for testing with small batches.

## Optional: RAG Pipeline (Legacy)

The RAG pipeline is available for document-based retrieval tasks but is not required for the DSPy pitch generation systems.

1. Run the 'indexer.py' to split the desired pdf into chunks and generate the embeddings into the specified embedding model and store into milvus db collection:

   ```sh
   python src/rag_pipeline/indexer.py
   ```

2. To get the top k results from the milvus client collection based on query, run the following:
   ```sh
   python src/rag_pipeline/retriever.py
   ```

## Optional: DeepEval Framework (Legacy)

DeepEval can be used for additional evaluation metrics, but the DSPy systems include built-in evaluation via `AssessPitchQuality`.

1. On your terminal, the UI will be opened after running the following command:
   ```sh
   deepeval login
   ```
2. Sign in accordingly as per recommendation to to use work email for login access.
3. Paste your API Key: XXXXX. You should see the following messages

```sh
🎉🥳 Congratulations! You've successfully logged in! 🙌
You're now using DeepEval with Confident AI. Follow our quickstart tutorial
here: https://docs.confident-ai.com/confident-ai/confident-ai-introduction
```

4. Run the sample test case via:

```sh
deepeval test run src/rag_pipeline/test_evaluation.py
```

Note: If you have not set the OpenAI key, then export it on your terminal via:

```sh
export OPENAI_API_KEY=XXXX
```

5. This is the expected via of the evaluation:

```sh

Evaluating 1 test case(s) in parallel: |█|100% (1/1) [Time Taken: 00:03,  3.78
.Running teardown with pytest sessionfinish...

============================ slowest 10 durations ============================
3.81s call     src/rag_pipeline/test_evaluation.py::test_answer_relevancy

(2 durations < 0.005s hidden.  Use -vv to show these durations.)
1 passed, 3 warnings in 3.82s
                                 Test Results
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃                ┃                ┃                ┃        ┃ Overall        ┃
┃ Test case      ┃ Metric         ┃ Score          ┃ Status ┃ Success Rate   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ test_answer_r… │                │                │        │ 100.0%         │
│                │ Answer         │ 1.0            │ PASSED │                │
│                │ Relevancy      │ (threshold=0.… │        │                │
│                │                │ evaluation     │        │                │
│                │                │ model=gpt-4o,  │        │                │
│                │                │ reason=The     │        │                │
│                │                │ score is 1.00  │        │                │
│                │                │ because the    │        │                │
│                │                │ actual output  │        │                │
│                │                │ flawlessly     │        │                │
│                │                │ addresses the  │        │                │
│                │                │ question       │        │                │
│                │                │ regarding shoe │        │                │
│                │                │ fit, with no   │        │                │
│                │                │ irrelevant     │        │                │
│                │                │ statements     │        │                │
│                │                │ distracting    │        │                │
│                │                │ from the       │        │                │
│                │                │ topic.         │        │                │
│                │                │ Excellent      │        │                │
│                │                │ job!,          │        │                │
│                │                │ error=None)    │        │                │
│ Note: Use      │                │                │        │                │
│ Confident AI   │                │                │        │                │
│ with DeepEval  │                │                │        │                │
│ to analyze     │                │                │        │                │
│ failed test    │                │                │        │                │
│ cases for more │                │                │        │                │
│ details        │                │                │        │                │
└────────────────┴────────────────┴────────────────┴────────┴────────────────┘
Total estimated evaluation tokens cost: 0.0037350000000000005 USD
✓ Tests finished 🎉! View results on
https://app.confident-ai.com/project/cm7k4hv8d5q5k7advs35m4pl5/evaluation/test-runs/cm7k5
119t00d6qyy0o7ywn80r/test-cases.

```
