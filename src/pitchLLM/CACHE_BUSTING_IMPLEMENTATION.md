# Cache Busting Implementation

**Date**: November 4, 2025  
**Issue**: DSPy was caching LM calls, causing identical responses across runs  
**Solution**: Implemented cache busting with unique rollout IDs and non-zero temperature

## Problem

When running evaluations, the system was returning identical responses even with different optimization methods because:

1. **DSPy caches LM calls by default** - Same inputs produce cached outputs
2. **Zero/default temperature** - Made responses deterministic
3. **Fixed test set order** - Same examples in same order every run
4. **MLflow search hanging** - `mlflow.search_runs()` would hang when connecting to Databricks

## Solution Implemented

### 1. **Enable Usage Tracking and Non-Zero Temperature**

```python
# pitchLLM_structured.py (lines 66-82)
generator_lm = dspy.LM(
    "groq/llama-3.3-70b-versatile", 
    model_type="chat", 
    api_key=GROQ_API_KEY,
    temperature=1.0  # Non-zero to enable variation
)

evaluator_lm = dspy.LM(
    "groq/openai/gpt-oss-120b", 
    model_type="chat", 
    api_key=GROQ_API_KEY,
    temperature=0.7  # Lower for evaluation consistency
)

dspy.configure(lm=generator_lm, track_usage=True)
```

### 2. **Unique Rollout ID Per Example**

Added unique identifiers to each evaluation call to bypass DSPy cache:

```python
# pitchLLM_structured.py (lines 266-276)
# Generate unique run identifier for cache busting
run_timestamp = int(time.time())

for idx, example in enumerate(tqdm(testset, desc="Evaluating", unit="pitch")):
    prediction = generator.generate(
        example.input,
        config={
            "rollout_id": f"{run_timestamp}_{idx}",  # Unique per example
            "temperature": 1.0  # Bypass cache
        }
    )
```

### 3. **Updated Generator to Accept Config**

Modified `PitchGenerator.generate()` to accept and pass through config parameters:

```python
# models/generator.py (lines 78-95)
def generate(self, input_data: dict, config: dict = None):
    """
    Generate a pitch using the assigned generator model.
    
    Args:
        input_data: Dictionary with structured pitch data
        config: Optional config dict with rollout_id and temperature for cache control
        
    Returns:
        dspy.Prediction with pitch field
    """
    # Build context parameters
    context_params = {"lm": self.lm}
    if config:
        context_params.update(config)
    
    with dspy.context(**context_params):
        return self.program(input=input_data)
```

### 4. **Added Timeout to MLflow Search**

Fixed hanging issue when `mlflow.search_runs()` connects to Databricks:

```python
# utils.py (lines 122-153)
# Fall back to searching for the most recent run (needed for autolog)
# Use threading with timeout to prevent hanging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

def search_mlflow():
    """Search for the most recent run in this experiment."""
    experiment = mlflow.get_experiment_by_name(databricks_path + run_name)
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1,
            order_by=["start_time DESC"]
        )
        if not runs.empty:
            return runs.iloc[0]["run_id"]
    return None

# Execute search with 10 second timeout (Databricks can be slow)
print(f"   Searching for MLflow run...")
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(search_mlflow)
    try:
        mlflow_run_id = future.result(timeout=10.0)
        if mlflow_run_id:
            print(f"   ✓ MLflow Run ID: {mlflow_run_id}")
            return mlflow_run_id
        else:
            print(f"   ℹ No MLflow run found yet (may appear in next attempt)")
            return None
    except FutureTimeoutError:
        print(f"   ⚠ MLflow search timed out (network issue - continuing without run_id)")
        return None
```

## Files Modified

1. **`pitchLLM_structured.py`**
   - Added `temperature=1.0` to generator LM
   - Added `temperature=0.7` to evaluator LM
   - Added `track_usage=True` to dspy.configure
   - Added `run_timestamp` generation in evaluate_program
   - Pass `config` with rollout_id to generator.generate()

2. **`models/generator.py`**
   - Updated `generate()` method to accept `config` parameter
   - Pass config through to dspy.context

3. **`utils.py`**
   - Added timeout protection to `capture_mlflow_run_id()`
   - Uses ThreadPoolExecutor with 10-second timeout
   - Prevents hanging on network/Databricks issues

## Testing

✅ All files compile without errors  
✅ No linter errors  
✅ Syntax validated

## Expected Behavior

### Before Fix
- Identical pitches across runs (cached)
- Script hangs during MLflow search
- Cannot properly test optimization methods

### After Fix
- **Fresh responses every run** - Each evaluation bypasses cache
- **No hanging** - MLflow search times out gracefully after 10s
- **Proper testing** - Can compare optimization methods with fresh data
- **Usage tracking** - Can monitor API usage with `track_usage=True`

## Configuration Options

You can adjust cache-busting behavior by modifying:

- **Temperature**: Higher = more variation (0.7-1.0 recommended for testing)
- **Rollout ID**: Unique per call ensures fresh results
- **Timeout**: 10 seconds default, adjust based on network speed

## Usage

Just run your evaluations as normal:

```bash
# Fresh results every time
python pitchLLM_structured.py --optimization bootstrap --test-size 10 --evaluate

# Each run will generate new pitches (not cached)
python pitchLLM_structured.py --optimization none --test-size 10 --evaluate
```

## Important Notes

1. **Cache is still useful during optimization** - DSPy still caches during compile phase (good for efficiency)
2. **Fresh results for evaluation** - Only evaluation phase bypasses cache (good for testing)
3. **Controlled randomness** - Temperature adds variation while staying reasonable
4. **Fallback behavior** - If MLflow times out, run continues without run_id

## References

- DSPy Cache Documentation: https://dspy-docs.vercel.app/docs/building-blocks/language_models#forcing-fresh-lm-outputs
- MLflow Search API: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs

