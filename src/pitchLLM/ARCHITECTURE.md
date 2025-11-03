# Pitch Generation Architecture - Dedicated Model Classes

## Overview

The system now uses **dedicated classes** for generation and evaluation to ensure complete separation between the 70B generator and 120B evaluator models. This eliminates model confusion and makes it crystal clear which model is being used for each operation.

## Architecture

```
┌─────────────────────────────────────────┐
│          pitchLLM_structured.py         │
│         (Main orchestration)            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Generator70B │  │ Evaluator120B │
│   (models/)  │  │   (models/)   │
└──────┬───────┘  └──────┬────────┘
       │                 │
       ▼                 ▼
┌─────────────┐  ┌──────────────┐
│ Llama 3.3   │  │ GPT-OSS      │
│ 70B Model   │  │ 120B Model   │
└─────────────┘  └──────────────┘
```

## File Structure

```
src/pitchLLM/
├── pitchLLM_structured.py      # Main orchestration
├── models/
│   ├── __init__.py             # Package exports
│   ├── generator.py            # Generator70B class
│   └── evaluator.py            # Evaluator120B class
├── utils.py                    # Pydantic models
├── data_loader.py              # HF dataset loading
└── eval/
    └── AssessPitch.py          # Evaluation signature
```

## Key Components

### 1. Generator70B (`models/generator.py`)

**Purpose**: Generate pitches using Llama 3.3 70B model exclusively.

**Features**:
- Encapsulates the StructuredPitchProgram DSPy module
- Guarantees all generation uses 70B model
- Uses context manager internally for explicit LM control
- Provides `.generate()` and `__call__()` methods

**Usage**:
```python
generator = Generator70B(generator_lm)
prediction = generator.generate(input_data)
pitch = prediction.pitch
```

### 2. Evaluator120B (`models/evaluator.py`)

**Purpose**: Evaluate pitch quality using GPT-OSS 120B model exclusively.

**Features**:
- Encapsulates the PitchEvaluatorModule DSPy module
- Guarantees all evaluation uses 120B model
- Uses context manager internally for explicit LM control
- Provides convenience methods:
  - `.evaluate()` - Full assessment prediction
  - `.get_score()` - Just the final score (float)
  - `.get_full_assessment()` - Dict with all scores
  - `__call__()` - Direct call interface

**Usage**:
```python
evaluator = Evaluator120B(evaluator_lm)

# Get just the score
score = evaluator.get_score(facts, ground_truth, generated)

# Get full assessment
assessment = evaluator.get_full_assessment(facts, ground_truth, generated)
# Returns: {"factual_score": 0.85, "narrative_score": 0.90, ...}
```

## Benefits of This Architecture

### 1. **Complete Model Separation**
- ✅ Generator ALWAYS uses 70B
- ✅ Evaluator ALWAYS uses 120B
- ✅ No possibility of model confusion
- ✅ No accidental model switching

### 2. **Explicit and Clear**
```python
# Old (fragile):
with dspy.context(lm=evaluator_lm):  # Hope it switches back!
    result = evaluator(...)

# New (bulletproof):
evaluator = Evaluator120B(evaluator_lm)  # ALWAYS 120B
result = evaluator.evaluate(...)
```

### 3. **Easy to Verify**
- Check initialization logs: "Generator initialized with model: groq/llama-3.3-70b-versatile"
- Check API logs: See exactly which model made each call
- No ambiguity in debugging

### 4. **Reusable Components**
- Generator and Evaluator can be used independently
- Easy to test each component in isolation
- Clear interfaces for future extensions

### 5. **Better Error Messages**
- Errors clearly indicate which model failed
- Stack traces show which class (Generator70B vs Evaluator120B)
- No confusion about which context the error occurred in

## Usage in Main Workflow

### Initialization

```python
# In main execution
generator = Generator70B(generator_lm)
evaluator = Evaluator120B(evaluator_lm)

# Prints:
# ✓ Generator initialized with model: groq/llama-3.3-70b-versatile
# ✓ Evaluator initialized with model: groq/openai/gpt-oss-120b
```

### Pitch Generation

```python
# Generate pitch
prediction = generator.generate(example.input)
pitch = prediction.pitch
```

### Pitch Evaluation

```python
# Evaluate pitch
assessment = evaluator.get_full_assessment(
    pitch_facts=json.dumps(example.input),
    ground_truth_pitch=example.output,
    generated_pitch=pitch
)

# assessment = {
#     "factual_score": 0.85,
#     "narrative_score": 0.90,
#     "style_score": 0.88,
#     "reasoning": "...",
#     "final_score": 0.88
# }
```

### Optimization Metrics

```python
def pitch_quality_metric(example, pred, trace=None):
    """Uses 120B evaluator for optimization."""
    evaluator = get_evaluator()  # Global Evaluator120B instance
    return evaluator.get_score(facts, ground_truth, generated)
```

## Context Manager Usage

Both classes use `with dspy.context(lm=...)` internally:

```python
class Generator70B:
    def generate(self, input_data):
        with dspy.context(lm=self.lm):  # Ensures 70B
            return self.program(input=input_data)

class Evaluator120B:
    def evaluate(self, ...):
        with dspy.context(lm=self.lm):  # Ensures 120B
            return self.evaluator(...)
```

This ensures that:
1. The correct model is used for each operation
2. The context is properly restored after each call
3. No global state pollution

## Migration Notes

### Old Code
```python
# Generation
program = StructuredPitchProgram()
prediction = program(input=data)

# Evaluation
evaluator = PitchEvaluator()
with dspy.context(lm=evaluator_lm):
    assessment = evaluator(...)
```

### New Code
```python
# Generation
generator = Generator70B(generator_lm)
prediction = generator.generate(data)

# Evaluation
evaluator = Evaluator120B(evaluator_lm)
assessment = evaluator.get_full_assessment(...)
```

## Testing

### Verify Model Separation

Check your API logs and confirm:
1. **Generation calls** show `llama-3.3-70b-versatile`
2. **Evaluation calls** show `openai/gpt-oss-120b`
3. **Equal distribution** of calls to both models (when evaluating)

### Expected Log Pattern

During evaluation with 3 examples:
```
✓ Generator initialized with model: groq/llama-3.3-70b-versatile
✓ Evaluator initialized with model: groq/openai/gpt-oss-120b

API Logs:
- llama-3.3-70b-versatile: 3 calls (generation)
- openai/gpt-oss-120b: 3 calls (evaluation)
```

## Troubleshooting

### Only seeing one model in logs?

**Check**:
1. Are both Generator70B and Evaluator120B being instantiated?
2. Is `use_evaluator=True` when calling evaluate_program?
3. Check initialization messages in console

### Model not switching?

**This architecture prevents this issue!** Each class has its own LM instance and uses context managers internally.

## Future Enhancements

This architecture makes it easy to:
- Add more models (e.g., different sizes for different tasks)
- A/B test different model combinations
- Implement model ensembles
- Add caching at the model level
- Track per-model metrics separately

## Summary

**Before**: Fragile context managers, easy to confuse models
**After**: Dedicated classes, impossible to confuse models

This architecture is:
- ✅ Explicit and clear
- ✅ Error-proof
- ✅ Easy to debug
- ✅ Easy to extend
- ✅ Production-ready

