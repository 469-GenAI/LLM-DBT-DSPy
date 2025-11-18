# DSPy Pitch Optimization

This module provides comprehensive tools for optimizing DSPy pitch generation programs using state-of-the-art optimizers.

## 📋 Features

- **Multiple Optimizers**: MIPROv2, BootstrapFewShot, and BootstrapRS
- **Comprehensive Metrics**: Structure validation, completeness scoring, pitch quality, financial consistency
- **Automatic Dataset Splitting**: Train/validation/test split from SharkTank data
- **Parallel Processing**: Multi-threaded optimization for faster results
- **Model Saving**: Save and reload optimized programs

## 🚀 Quick Start

### 1. Basic Optimization (MIPROv2 Light Mode)

```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate
python src/dspy_optimization/optimize_pitch.py \
    --optimizer mipro \
    --model gpt-4o-mini \
    --train-size 30 \
    --test-size 10 \
    --mipro-mode light
```

**Expected Cost**: ~$2-3 USD, **Time**: ~15-20 minutes

### 2. Advanced Optimization (MIPROv2 Medium Mode)

```bash
python src/dspy_optimization/optimize_pitch.py \
    --optimizer mipro \
    --model gpt-4o-mini \
    --train-size 50 \
    --val-size 15 \
    --test-size 20 \
    --mipro-mode medium \
    --threads 12
```

**Expected Cost**: ~$5-8 USD, **Time**: ~30-45 minutes

### 3. Alternative: BootstrapFewShot

```bash
python src/dspy_optimization/optimize_pitch.py \
    --optimizer bootstrap_fewshot \
    --model gpt-4o-mini \
    --train-size 30 \
    --test-size 10
```

**Expected Cost**: ~$1-2 USD, **Time**: ~10-15 minutes

### 4. Alternative: BootstrapRS (Random Search)

```bash
python src/dspy_optimization/optimize_pitch.py \
    --optimizer bootstrap_rs \
    --model gpt-4o-mini \
    --train-size 30 \
    --test-size 10
```

**Expected Cost**: ~$1-2 USD, **Time**: ~10-15 minutes

## 📊 Available Metrics

You can choose which metric to optimize for using the `--metric` flag:

| Metric | Description | Use Case |
|--------|-------------|----------|
| `structure` | Binary check for required fields | Fast validation |
| `completeness` | Scores presence of all pitch components | Ensure comprehensive pitches |
| `pitch_quality` | Evaluates pitch content quality | Improve persuasiveness |
| `financial_consistency` | Checks financial calculations | Ensure accurate numbers |
| `composite` | Weighted combination of all metrics | **Recommended for best overall quality** |

Example:
```bash
python src/dspy_optimization/optimize_pitch.py \
    --metric composite \
    --optimizer mipro
```

## 🎯 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--optimizer` | `mipro` | Optimizer to use: `mipro`, `bootstrap_fewshot`, `bootstrap_rs` |
| `--model` | `gpt-4o-mini` | LLM model to use |
| `--train-size` | `50` | Number of training examples |
| `--val-size` | `15` | Number of validation examples (for MIPRO) |
| `--test-size` | `20` | Number of test examples |
| `--metric` | `composite` | Metric to optimize for |
| `--mipro-mode` | `light` | MIPRO mode: `light`, `medium`, `heavy` |
| `--threads` | `8` | Number of parallel threads |
| `--output-dir` | `./optimized_models` | Directory to save results |
| `--skip-baseline` | `False` | Skip baseline evaluation |

## 📈 Understanding the Output

After optimization, you'll see:

```
COMPARISON
======================================================================
Baseline:  0.523 avg score, 45.0% success
Optimized: 0.782 avg score, 85.0% success

Improvement: +0.259 (+49.5%)
```

The optimized program and results are saved to:
- `optimized_models/mipro_gpt-4o-mini_20251030_120000.json` - The optimized program
- `optimized_models/mipro_gpt-4o-mini_20251030_120000_results.json` - Detailed results

## 🔍 Understanding Each Optimizer

### MIPROv2 (Recommended)

**Best for**: Overall quality improvement  
**How it works**: 
1. Bootstraps few-shot examples from successful runs
2. Proposes and tests different instruction variations
3. Uses surrogate model to guide search for better prompts

**Modes**:
- `light`: Fast, cheap (~$2, 15-20 min) - Good for initial optimization
- `medium`: Balanced (~$5-8, 30-45 min) - Recommended for production
- `heavy`: Thorough (~$15-25, 1-2 hours) - Maximum quality

### BootstrapFewShot

**Best for**: Quick improvements with few-shot learning  
**How it works**: 
- Automatically creates few-shot examples from successful training runs
- Adds them to prompts to improve consistency

**Advantages**: Simple, fast, predictable cost

### BootstrapRS (Random Search)

**Best for**: Exploring different prompt variations  
**How it works**: 
- Randomly searches through different prompt configurations
- Evaluates each on a subset of training data

**Advantages**: Good for discovering unexpected improvements

## 📁 Module Structure

```
src/dspy_optimization/
├── __init__.py              # Module exports
├── README.md               # This file
├── dataset_prep.py         # Dataset loading and splitting
├── metrics.py              # Evaluation metrics
├── optimize_pitch.py       # Main optimization script
└── quick_start.sh          # Automated quick start script
```

## 💡 Tips for Best Results

1. **Start Small**: Begin with `--train-size 30 --mipro-mode light` to test
2. **Use Validation Set**: For MIPROv2, always include `--val-size` for better generalization
3. **Monitor Costs**: Use `light` mode first, upgrade to `medium` only if needed
4. **Try Different Metrics**: Test `structure` for speed, `composite` for quality
5. **Ensemble Models**: Run multiple optimizers and combine their results

## 🔄 Using Optimized Models

To use an optimized model in production:

```python
import dspy
from agents.pitchLLM import PitchProgram

# Load optimized program
optimized_program = PitchProgram()
optimized_program.load("optimized_models/mipro_gpt-4o-mini_20251030_120000.json")

# Use it
product_data = {...}  # Your product data
result = optimized_program(product_data)
```

## 🛠️ Troubleshooting

### Out of Memory

Reduce `--threads` to 4 or 2:
```bash
python src/dspy_optimization/optimize_pitch.py --threads 4
```

### Too Slow

Use fewer training examples or light mode:
```bash
python src/dspy_optimization/optimize_pitch.py --train-size 20 --mipro-mode light
```

### API Rate Limits

Reduce threads and add delays in the DSPy config

## 📚 References

- [DSPy Documentation](https://dspy.ai/)
- [MIPROv2 Paper](https://arxiv.org/abs/2406.11695)
- [DSPy Optimizers Guide](https://dspy.ai/tutorials/optimization)

## 🤝 Contributing

To add new metrics:
1. Add metric function to `metrics.py`
2. Update `get_all_metrics()` dictionary
3. Add to choices in `optimize_pitch.py` argument parser

