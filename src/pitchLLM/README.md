# Structured Pitch Generation with DSPy

This directory contains a DSPy-based implementation for generating Shark Tank-style pitches from structured input data.

## Overview

The system takes structured pitch data (company info, problem story, product solution, etc.) and generates compelling narrative pitches similar to those presented on Shark Tank. It supports optimization via BootstrapFewShot and MIPROv2.

### Dual-Model Architecture

For objective evaluation, the system uses **two different models**:

- **Generator**: Llama 3.3 70B (`groq/llama-3.3-70b-versatile`) - Fast, efficient pitch generation
- **Evaluator**: GPT-OSS 120B (`groq/openai/gpt-oss-120b`) - More powerful, objective evaluation

This eliminates self-evaluation bias where a model might favor its own generation style.

## Files

- **`utils.py`**: Pydantic models for structured input validation
- **`data_loader.py`**: HuggingFace dataset loading and DSPy Example conversion
- **`pitchLLM_structured.py`**: Main implementation with DSPy signatures, modules, and evaluation
- **`eval/AssessPitch.py`**: Evaluation signature for assessing pitch quality

## Requirements

```bash
pip install dspy datasets pandas python-dotenv tqdm pydantic
```

## Environment Setup

Create a `.env` file in the project root with:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Authenticate with HuggingFace:

```bash
huggingface-cli login
```

## Usage

### 1. Baseline Generation (No Optimization)

Generate pitches without any optimization:

```bash
cd src/pitchLLM
python pitchLLM_structured.py --optimization none --test-size 5
```

### 2. BootstrapFewShot Optimization

Compile the program with few-shot learning:

```bash
python pitchLLM_structured.py --optimization bootstrap --train-size 20 --test-size 10
```

### 3. MIPROv2 Optimization

Compile with instruction optimization:

```bash
python pitchLLM_structured.py --optimization mipro --train-size 30 --test-size 10
```

### 4. With Detailed Evaluation

Add the `--evaluate` flag to use the AssessPitch evaluator:

```bash
python pitchLLM_structured.py --optimization bootstrap --test-size 5 --evaluate
```

## Command Line Arguments

- `--optimization`: Choose optimization method (`none`, `bootstrap`, `mipro`)
- `--train-size`: Number of training examples to use (default: all)
- `--test-size`: Number of test examples to use (default: 10)
- `--evaluate`: Enable detailed pitch quality evaluation with GPT-OSS-120B
- `--no-rate-limit`: Disable rate limiting (use with caution - may hit API limits)

### Rate Limiting

By default, the system includes **automatic rate limiting** to respect Groq's 30 requests/minute limit:

- **2.5 second delay** between API calls
- Safe margin for ~24 calls per minute
- With evaluation enabled: 2 calls per example (generation + evaluation)
- Estimated time is displayed before evaluation starts

To disable rate limiting (e.g., if you have higher limits), use `--no-rate-limit`.

## Output

Results are saved to CSV files with the naming pattern:

```
structured_pitch_results_{optimization_method}_{timestamp}.csv
```

The CSV includes:

- `id`: Unique example identifier
- `company_name`: Company name from input
- `ground_truth`: Original human-written pitch
- `generated_pitch`: AI-generated pitch
- `optimization_method`: Method used
- `model_name`: Language model used
- `timestamp`: Run timestamp
- `final_score`: Overall quality score (if evaluation enabled)
- `factual_score`: Factual accuracy score
- `narrative_score`: Narrative structure score
- `style_score`: Style and tone score
- `reasoning`: Detailed evaluation reasoning

## Data Format

The system expects data from the HuggingFace dataset `isaidchia/sharktank_pitches` with the following structure:

```json
{
  "id": "unique-id",
  "_meta_index": 123,
  "input": {
    "founders": ["Founder 1", "Founder 2"],
    "company_name": "Company Name",
    "initial_offer": {
      "amount": "$400k",
      "equity": "5%"
    },
    "problem_story": {
      "persona": "target customer",
      "routine": ["action 1", "action 2"],
      "core_problem": "main problem",
      "hygiene_gap": "gap in current solutions",
      "problem_keywords": ["keyword1", "keyword2"]
    },
    "product_solution": {
      "name": "Product Name",
      "product_category": "category",
      "key_differentiator": "what makes it unique",
      "application": "how it's used",
      "features_keywords": ["feature1", "feature2"],
      "benefits_keywords": ["benefit1", "benefit2"]
    },
    "closing_theme": {
      "call_to_action": "invest now",
      "mission": "company mission",
      "target_audience": "target audience description"
    }
  },
  "output": "The full narrative pitch text..."
}
```

## Optimization Methods

### Baseline (none)

Generates pitches using only the base prompt without any optimization or few-shot examples.

### BootstrapFewShot

Uses training examples to automatically generate few-shot demonstrations that improve pitch quality. The optimizer:

- Selects the most effective examples
- Bootstraps intermediate reasoning steps
- Creates optimized prompts with demonstrations

### MIPROv2

Advanced optimization that:

- Explores different instruction variations
- Optimizes both instructions and demonstrations
- Uses Bayesian optimization for efficiency
- Typically achieves the best results but takes longer

## Performance & Rate Limiting

### Execution Time (with rate limiting enabled)

With rate limiting enabled (default), processing times for 50 test examples:

- **Baseline**:

  - Without evaluation: ~2 minutes (50 calls)
  - With evaluation: ~10 minutes (100 calls: 50 generation + 50 evaluation)

- **Bootstrap**:

  - Training optimization: ~20-40 minutes
  - Testing with evaluation: ~10 minutes
  - **Total**: ~30-50 minutes

- **MIPRO**:
  - Training optimization (10 trials): ~1-2 hours
  - Testing with evaluation: ~10 minutes
  - **Total**: ~1.5-2.5 hours

Use `--no-rate-limit` to speed up by ~80%, but may hit API limits.

### API Usage

Each test example with evaluation makes:

1. **Generation call** (70B model): ~1-2 seconds
2. **Evaluation call** (120B model): ~2-3 seconds
3. **Rate limiting delay**: 2.5 seconds × 2 = 5 seconds
4. **Total per example**: ~8-10 seconds

For 50 examples with evaluation: ~400-500 seconds (~7-8 minutes base time + ~3 minutes overhead)

## Evaluation Metrics

When `--evaluate` is enabled, the system uses the `AssessPitchQuality` signature to score pitches on:

1. **Factual Score (40%)**: Whether all key facts are included
2. **Narrative Score (40%)**: Story structure and flow
3. **Style Score (20%)**: Tone, persuasiveness, and engagement

The final score is a weighted average of these three components.

### Why Two Different Models?

The evaluation uses **GPT-OSS-120B** (different from the 70B generator) for:

- **Objective assessment**: Different architecture eliminates self-evaluation bias
- **More powerful reasoning**: 120B parameters provide better judgment
- **Industry best practice**: Use stronger model as judge/evaluator

## Example Workflow

```bash
# 1. Quick test with baseline
python pitchLLM_structured.py --optimization none --test-size 3

# 2. Optimize with BootstrapFewShot
python pitchLLM_structured.py --optimization bootstrap --train-size 50 --test-size 10 --evaluate

# 3. Full evaluation with MIPRO
python pitchLLM_structured.py --optimization mipro --train-size 100 --test-size 50 --evaluate
```

## Troubleshooting

### Authentication Issues

- Ensure you have run `huggingface-cli login`
- Verify your HuggingFace token has access to the dataset

### Memory Issues

- Reduce `--train-size` and `--test-size`
- Use `--optimization none` for faster iteration

### API Rate Limits

- Add delays between requests if needed
- Consider using a different language model

## Development

To modify the pitch generation behavior:

1. Edit `PitchGenerationSig` in `pitchLLM_structured.py` to change instructions
2. Modify `format_pitch_input()` in `utils.py` to change input formatting
3. Adjust `AssessPitchQuality` in `eval/AssessPitch.py` to change evaluation criteria
