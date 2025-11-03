# Changes Summary - Dual-Model Architecture & Rate Limiting

## Overview

Updated the structured pitch generation system to use two different models and implement automatic rate limiting for Groq's API.

## Key Changes

### 1. Dual-Model Architecture

**Before**: Used single model (Llama 3.3 70B) for both generation and evaluation
**After**: Uses two different models:

```python
# Generator: Llama 3.3 70B - Fast, efficient pitch generation
generator_lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)

# Evaluator: GPT-OSS-120B - More powerful, objective evaluation
evaluator_lm = dspy.LM("groq/openai/gpt-oss-120b", model_type="chat", api_key=GROQ_API_KEY)
```

**Benefits**:
- Eliminates self-evaluation bias (different architecture: GPT vs Llama)
- More powerful evaluator (120B > 70B parameters)
- Objective assessment of pitch quality

### 2. Automatic Rate Limiting

**Added**: Built-in rate limiting to respect Groq's 30 requests/minute limit

```python
RATE_LIMIT_DELAY = 2.5  # Seconds between API calls (safe margin: 24 calls/min)
```

**Implementation**:
- 2.5 second delay between each API call
- Safe margin for ~24 calls per minute (vs 30 limit)
- Applies to both generation and evaluation calls
- Optional `--no-rate-limit` flag to disable

**Time Impact**:
- 50 examples with evaluation: ~10 minutes (vs ~2 minutes without rate limiting)
- Prevents hitting API rate limits
- Displays estimated time before starting

### 3. Updated Command-Line Interface

**New Argument**:
```bash
--no-rate-limit    # Disable rate limiting (use with caution)
```

**Updated Help Text**:
```bash
--evaluate         # Use detailed AssessPitch evaluation with GPT-OSS-120B
```

### 4. Enhanced Output & Reporting

**Console Output**:
- Shows which models are being used (generator vs evaluator)
- Displays rate limiting status and estimated time
- Reports mean and standard deviation for all scores

**Example Output**:
```
EVALUATION SUMMARY
================================================================================
Generator Model: groq/llama-3.3-70b-versatile
Evaluator Model: groq/openai/gpt-oss-120b
Optimization Method: bootstrap

Scores (0.0-1.0):
  Average Final Score: 0.823 ± 0.045
  Average Factual Score: 0.856 ± 0.062
  Average Narrative Score: 0.834 ± 0.051
  Average Style Score: 0.779 ± 0.073
```

### 5. Updated Documentation

**README.md**:
- Added "Dual-Model Architecture" section
- Added "Performance & Rate Limiting" section
- Added "Why Two Different Models?" explanation
- Updated all time estimates to reflect rate limiting
- Documented `--no-rate-limit` flag

## Files Modified

1. **pitchLLM_structured.py**:
   - Added `import time`
   - Created two separate LM instances (generator_lm, evaluator_lm)
   - Added `RATE_LIMIT_DELAY` constant
   - Updated `PitchEvaluator` to use evaluator_lm
   - Added rate limiting to `evaluate_program()` function
   - Added `--no-rate-limit` command-line argument
   - Enhanced summary statistics output

2. **README.md**:
   - Added dual-model architecture explanation
   - Added rate limiting documentation
   - Added performance & timing information
   - Updated command-line arguments section

3. **test_simple.py**:
   - Fixed `input_keys` access for DSPy 3.0.3 compatibility

## Usage Examples

### Standard Usage (with rate limiting)
```bash
# Baseline with evaluation (takes ~10 mins for 50 examples)
python pitchLLM_structured.py --optimization none --test-size 50 --evaluate

# Bootstrap with evaluation (takes ~30-50 mins for 50 examples)
python pitchLLM_structured.py --optimization bootstrap --train-size 100 --test-size 50 --evaluate

# MIPRO with evaluation (takes ~1.5-2.5 hours for 50 examples)
python pitchLLM_structured.py --optimization mipro --train-size 100 --test-size 50 --evaluate
```

### Fast Usage (no rate limiting - may hit limits)
```bash
# Same commands but with --no-rate-limit flag
python pitchLLM_structured.py --optimization none --test-size 50 --evaluate --no-rate-limit
```

## API Call Breakdown

With evaluation enabled, each test example makes:
1. Generation call (70B): ~1-2 seconds
2. Rate limit delay: 2.5 seconds
3. Evaluation call (120B): ~2-3 seconds
4. Rate limit delay: 2.5 seconds
5. **Total**: ~8-10 seconds per example

For 50 examples: ~400-500 seconds (~7-8 minutes)

## Verification

To verify the changes work:

```bash
cd src/pitchLLM

# Test with 3 examples
python pitchLLM_structured.py --optimization none --test-size 3 --evaluate

# Check output shows both models
# Should see: "Generator Model: groq/llama-3.3-70b-versatile"
#            "Evaluator Model: groq/openai/gpt-oss-120b"
```

## Benefits Summary

✅ **Objective Evaluation**: Different model architectures eliminate bias
✅ **Powerful Judge**: 120B evaluator provides better assessment
✅ **Rate Limit Safe**: Automatic delays prevent API limit errors
✅ **Cost Effective**: Only uses 120B for evaluation, not generation
✅ **Transparent**: Clear reporting of models used and timing
✅ **Flexible**: Can disable rate limiting if needed

