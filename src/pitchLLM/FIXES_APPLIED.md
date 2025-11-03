# Fixes Applied - Evaluation Issues Resolved

## Problems Identified

1. **LM Not JSON Serializable**: Passing `lm=evaluator_lm` directly to `ChainOfThought` caused serialization errors
2. **JSON Parsing Failures**: Raw JSON output from model was unreliable and failed to parse
3. **Missing Score Fields**: Error handling only returned partial data (missing factual/narrative/style scores)
4. **Model Name Output**: Was printing entire LM object instead of model name string

## Solutions Implemented

### 1. Converted to Pydantic Output Model

**File**: `src/pitchLLM/eval/AssessPitch.py`

**Before**: Used raw JSON string with prefix template
```python
pitch_assessment = dspy.OutputField(
    desc="A JSON object with a final 0.0-1.0 score...",
    prefix="{ ... }"
)
```

**After**: Created Pydantic model for structured output
```python
class PitchAssessment(BaseModel):
    factual_score: float
    narrative_score: float
    style_score: float
    reasoning: str
    final_score: float

class AssessPitchQuality(dspy.Signature):
    assessment: PitchAssessment = dspy.OutputField(...)
```

**Benefits**:
- Type-safe output
- Automatic validation
- No JSON parsing errors
- Clear structure

### 2. Fixed LM Context Manager

**File**: `src/pitchLLM/pitchLLM_structured.py`

**Before**: Passed `lm` parameter directly (caused serialization error)
```python
self.assess = dspy.ChainOfThought(AssessPitchQuality, lm=evaluator_lm)
```

**After**: Used context manager to switch models
```python
def __init__(self):
    self.assess = dspy.ChainOfThought(AssessPitchQuality)

def forward(self, ...):
    with dspy.context(lm=evaluator_lm):
        assessment = self.assess(...)
```

**Benefits**:
- No serialization errors
- Clean model switching
- Compatible with DSPy 3.0.3

### 3. Updated Assessment Extraction

**Before**: JSON parsing with minimal error handling
```python
try:
    assessment_data = json.loads(assessment.pitch_assessment)
except (json.JSONDecodeError, AttributeError):
    assessment_data = {"final_score": 0.0, "reasoning": "Parse error"}
```

**After**: Direct Pydantic model access with complete error handling
```python
try:
    pitch_assessment = assessment.assessment  # Pydantic model
    assessment_data = {
        "factual_score": float(pitch_assessment.factual_score),
        "narrative_score": float(pitch_assessment.narrative_score),
        "style_score": float(pitch_assessment.style_score),
        "reasoning": str(pitch_assessment.reasoning),
        "final_score": float(pitch_assessment.final_score)
    }
except (AttributeError, ValueError, TypeError) as e:
    assessment_data = {
        "factual_score": 0.0,
        "narrative_score": 0.0,
        "style_score": 0.0,
        "reasoning": f"Parse error: {str(e)}",
        "final_score": 0.0
    }
```

**Benefits**:
- All scores always present
- Clear error messages
- Type conversion with validation

### 4. Fixed Error Handling

**Before**: Incomplete error data
```python
"assessment": {"final_score": 0.0, "reasoning": f"Error: {str(e)}"}
```

**After**: Complete error data structure
```python
"assessment": {
    "factual_score": 0.0,
    "narrative_score": 0.0,
    "style_score": 0.0,
    "reasoning": f"Error: {str(e)}",
    "final_score": 0.0
}
```

**Benefits**:
- Consistent data structure
- All fields present in CSV output
- No missing columns

### 5. Fixed Model Name Output

**Before**: Printed entire LM object
```python
"model_name": generator_lm,
"evaluator_model_name": evaluator_lm,
```

**After**: Extract model name string
```python
"model_name": generator_lm.model,
"evaluator_model_name": evaluator_lm.model,
```

**Benefits**:
- Readable CSV output
- Proper model identification

## Testing

Run the updated system:

```bash
cd src/pitchLLM

# Quick test with 3 examples
python pitchLLM_structured.py --optimization none --test-size 3 --evaluate
```

## Expected Output

CSV should now have:
- ✅ All score fields populated (factual, narrative, style, final)
- ✅ Reasoning text for all evaluated pitches
- ✅ No "Parse error" messages
- ✅ Readable model names in columns
- ✅ All rows complete with evaluations

## Files Modified

1. `src/pitchLLM/eval/AssessPitch.py` - Converted to Pydantic output
2. `src/pitchLLM/pitchLLM_structured.py` - Fixed context manager, extraction, error handling

## Verification

After running evaluation, check CSV for:
- Non-zero scores in factual_score, narrative_score, style_score columns
- Populated reasoning field for all rows
- Model names showing as strings (e.g., "groq/llama-3.3-70b-versatile")
- No empty assessment fields

## Technical Details

**DSPy 3.0.3 Compatibility**:
- Context manager (`with dspy.context(lm=...)`) is the correct way to switch models
- Pydantic models in OutputField are fully supported
- Direct `lm` parameter to ChainOfThought causes serialization issues

**Pydantic Benefits**:
- Automatic type validation and conversion
- Clear error messages when fields are missing
- No manual JSON parsing needed
- Better IDE support and type hints

