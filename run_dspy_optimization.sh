#!/bin/bash
# Quick start script for DSPy optimization

set -e  # Exit on error

echo "======================================================================"
echo "DSPy Pitch Optimization - Quick Start"
echo "======================================================================"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo "Loading environment from .env file..."
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "Warning: OPENAI_API_KEY not set and no .env file found"
        echo "Please create .env file with: OPENAI_API_KEY=your_key"
        exit 1
    fi
fi

# Run test first
echo ""
echo "Running setup tests..."
python src/dspy_optimization/test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup tests failed. Please fix the issues before proceeding."
    exit 1
fi

# Ask user for optimization mode
echo ""
echo "======================================================================"
echo "Select optimization mode:"
echo "======================================================================"
echo "1) Quick test (20 train, 5 test, light mode) - ~$1-2, 5-10 min"
echo "2) Standard (50 train, 20 test, light mode) - ~$2-3, 15-20 min"
echo "3) Advanced (50 train, 15 val, 20 test, medium mode) - ~$5-8, 30-45 min"
echo "4) Custom (specify your own parameters)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Running quick test optimization..."
        python src/dspy_optimization/optimize_pitch.py \
            --optimizer mipro \
            --model gpt-4o-mini \
            --train-size 20 \
            --test-size 5 \
            --mipro-mode light \
            --metric composite
        ;;
    2)
        echo "Running standard optimization..."
        python src/dspy_optimization/optimize_pitch.py \
            --optimizer mipro \
            --model gpt-4o-mini \
            --train-size 50 \
            --test-size 20 \
            --mipro-mode light \
            --metric composite
        ;;
    3)
        echo "Running advanced optimization..."
        python src/dspy_optimization/optimize_pitch.py \
            --optimizer mipro \
            --model gpt-4o-mini \
            --train-size 50 \
            --val-size 15 \
            --test-size 20 \
            --mipro-mode medium \
            --metric composite \
            --threads 12
        ;;
    4)
        echo ""
        read -p "Optimizer (mipro/bootstrap_fewshot/bootstrap_rs): " optimizer
        read -p "Train size: " train_size
        read -p "Test size: " test_size
        read -p "MIPRO mode (light/medium/heavy): " mipro_mode
        
        echo "Running custom optimization..."
        python src/dspy_optimization/optimize_pitch.py \
            --optimizer "$optimizer" \
            --model gpt-4o-mini \
            --train-size "$train_size" \
            --test-size "$test_size" \
            --mipro-mode "$mipro_mode" \
            --metric composite
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "Optimization complete! Check the optimized_models/ directory for results."
echo "======================================================================"

