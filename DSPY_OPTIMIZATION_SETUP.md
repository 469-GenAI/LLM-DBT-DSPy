# DSPy Prompt Optimization - Complete Setup Guide

## 🎉 Setup Complete!

Your DSPy optimization environment is fully configured and ready to use. All tests passed successfully!

## 📁 What Was Created

### 1. Core Optimization Module (`src/dspy_optimization/`)

```
src/dspy_optimization/
├── __init__.py              # Module exports
├── dataset_prep.py          # Dataset loading from all_processed_facts.json
├── metrics.py               # 5 comprehensive evaluation metrics  
├── optimize_pitch.py        # Main optimization script (MIPROv2, BootstrapFewShot, BootstrapRS)
├── test_setup.py           # Verification tests
└── README.md               # Detailed usage guide
```

### 2. Quick Start Scripts

- **`run_dspy_optimization.sh`** - Interactive script with 4 preset configurations
- **`test_setup.py`** - Validates all components before running

### 3. Key Features Implemented

✅ **Dataset Management** (119 SharkTank products from `all_processed_facts.json`)
- Automatic train/validation/test splitting (70/15/15)
- Configurable dataset sizes for cost control
- Proper DSPy Example formatting with input/output separation

✅ **5 Evaluation Metrics**
1. **Structure** - Binary check for required fields
2. **Completeness** - Score presence of all pitch components (0.0-1.0)
3. **Pitch Quality** - Evaluate content quality using NLP heuristics (0.0-1.0)
4. **Financial Consistency** - Verify calculations are reasonable (0.0-1.0)
5. **Composite** - Weighted combination of all metrics (0.0-1.0) ⭐ **Recommended**

✅ **3 Optimizers**
1. **MIPROv2** - Most powerful, proposes better instructions (light/medium/heavy modes)
2. **BootstrapFewShot** - Creates few-shot examples from successful runs
3. **BootstrapRS** - Random search over prompt variations

✅ **Cost Control**
- Light mode: ~$1-3 USD, 10-20 minutes
- Medium mode: ~$5-8 USD, 30-45 minutes
- Heavy mode: ~$15-25 USD, 1-2 hours

## 🚀 How to Use

### Option 1: Interactive Script (Recommended for First Time)

```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate
./run_dspy_optimization.sh
```

This will:
1. Run setup tests
2. Show you 4 preset configurations
3. Guide you through the optimization process

### Option 2: Direct Command (For Custom Runs)

```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate

# Quick test (recommended first run)
python src/dspy_optimization/optimize_pitch.py \
    --optimizer mipro \
    --model gpt-4o-mini \
    --train-size 20 \
    --test-size 5 \
    --mipro-mode light \
    --metric composite

# Standard optimization
python src/dspy_optimization/optimize_pitch.py \
    --optimizer mipro \
    --model gpt-4o-mini \
    --train-size 50 \
    --test-size 20 \
    --mipro-mode light

# Advanced optimization with validation set
python src/dspy_optimization/optimize_pitch.py \
    --optimizer mipro \
    --model gpt-4o-mini \
    --train-size 50 \
    --val-size 15 \
    --test-size 20 \
    --mipro-mode medium \
    --threads 12
```

### Option 3: Try Different Optimizers

```bash
# BootstrapFewShot (faster, cheaper)
python src/dspy_optimization/optimize_pitch.py \
    --optimizer bootstrap_fewshot \
    --train-size 30 \
    --test-size 10

# BootstrapRS (random search)
python src/dspy_optimization/optimize_pitch.py \
    --optimizer bootstrap_rs \
    --train-size 30 \
    --test-size 10
```

## 📊 Expected Output

When you run optimization, you'll see:

```
======================================================================
DSPy PITCH OPTIMIZATION
======================================================================
OptimizationConfig(
    optimizer: mipro
    model: openai/gpt-4o-mini
    train_size: 50
    val_size: 15
    test_size: 20
    metric: composite
    ...
)

======================================================================
LOADING DATASETS
======================================================================
✓ Train set: 50 examples
✓ Val set: 15 examples
✓ Test set: 20 examples

======================================================================
OPTIMIZING WITH MIPROv2 (LIGHT MODE)
======================================================================
Training on 50 examples...
Using 8 threads
Metric: composite
...

======================================================================
EVALUATING OPTIMIZED
======================================================================
Example 1/20: Score = 0.823
Example 2/20: Score = 0.791
...

Optimized Results:
  Average Score: 0.782
  Success Rate: 85.0%

======================================================================
COMPARISON
======================================================================
Baseline:  0.523 avg score, 45.0% success
Optimized: 0.782 avg score, 85.0% success

Improvement: +0.259 (+49.5%)

✓ Saved optimized program to: optimized_models/mipro_gpt-4o-mini_20251030_120000.json
✓ Saved results to: optimized_models/mipro_gpt-4o-mini_20251030_120000_results.json
```

## 📂 Output Files

All optimized models and results are saved in `optimized_models/`:

- **`mipro_gpt-4o-mini_YYYYMMDD_HHMMSS.json`** - The optimized DSPy program
- **`mipro_gpt-4o-mini_YYYYMMDD_HHMMSS_results.json`** - Detailed evaluation metrics

## 🔧 Using Optimized Models

To use an optimized model in your code:

```python
import dspy
from agno_agents.pitchLLM import PitchProgram

# Configure DSPy with your API key
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

# Load optimized program
program = PitchProgram()
program.load("optimized_models/mipro_gpt-4o-mini_20251030_120000.json")

# Use it
product_data = {
    "facts": {...},
    "product_description": {...}
}
result = program(product_data)
print(result['response'].Pitch)
```

## 🎯 Recommended Workflow

1. **First Run - Quick Test** (5-10 min, ~$1)
   ```bash
   python src/dspy_optimization/optimize_pitch.py --train-size 20 --test-size 5 --mipro-mode light
   ```
   This validates everything works and gives you a baseline improvement.

2. **Second Run - Standard** (15-20 min, ~$2-3)
   ```bash
   python src/dspy_optimization/optimize_pitch.py --train-size 50 --test-size 20 --mipro-mode light
   ```
   This gives you a solid optimized model for production use.

3. **Optional - Advanced** (30-45 min, ~$5-8)
   ```bash
   python src/dspy_optimization/optimize_pitch.py --train-size 50 --val-size 15 --test-size 20 --mipro-mode medium
   ```
   For maximum quality, use medium mode with validation set.

4. **Compare Optimizers**
   Try BootstrapFewShot and BootstrapRS to see which works best for your use case.

5. **Ensemble** (Advanced)
   Combine multiple optimized models for even better results.

## 📚 Key Files Modified

- **`src/agno_agents/pitchLLM.py`** - Made MLflow optional, fixed imports for module use
- **`src/agno_agents/__init__.py`** - Created for proper Python module structure
- **`requirements.txt`** - All dependencies already listed (dspy==3.0.3, datasets, etc.)

## ⚙️ Configuration

All optimization parameters can be customized via command-line arguments. See:
```bash
python src/dspy_optimization/optimize_pitch.py --help
```

## 🔍 Understanding the Metrics

- **Structure (0/1)**: Does it have all required fields?
- **Completeness (0-1)**: How complete is the response?
- **Pitch Quality (0-1)**: Does it contain good pitch elements (value prop, market, traction, CTA)?
- **Financial Consistency (0-1)**: Are the financial calculations reasonable?
- **Composite (0-1)**: Weighted average of all metrics ⭐ Use this!

## 💰 Cost Estimation

Based on DSPy documentation and your setup:

| Configuration | Train Size | Mode | Est. Cost | Est. Time |
|---------------|-----------|------|-----------|-----------|
| Quick Test | 20 | light | $1-2 | 5-10 min |
| Standard | 50 | light | $2-3 | 15-20 min |
| Advanced | 50 | medium | $5-8 | 30-45 min |
| Maximum | 80 | heavy | $15-25 | 1-2 hours |

Note: Costs vary based on:
- Model used (gpt-4o-mini is cheapest)
- Number of training examples
- Optimization mode
- Number of threads (more = faster but more API calls)

## 🐛 Troubleshooting

### Test Failures

```bash
python src/dspy_optimization/test_setup.py
```

### Import Errors

Ensure you're in the project root and virtual environment is activated:
```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate
```

## 📖 Additional Resources

- **Detailed README**: `src/dspy_optimization/README.md`
- **DSPy Docs**: https://dspy.ai/
- **MIPROv2 Tutorial**: https://dspy.ai/tutorials/optimization/

## ✅ Next Steps

1. **Run a quick test**:
   ```bash
   ./run_dspy_optimization.sh
   # Select option 1 (Quick test)
   ```

2. **Review the results** in `optimized_models/`

3. **Compare with baseline** to see improvement

4. **Integrate the optimized model** into your production code

5. **Experiment** with different optimizers and metrics

---

**Happy Optimizing! 🚀**

For questions or issues, refer to the detailed README at `src/dspy_optimization/README.md`.

