#!/bin/bash
# Quick demo script for group presentation

echo "=========================================="
echo "RAG Implementation Demo"
echo "=========================================="
echo ""

echo "Step 1: Verifying setup..."
python src/pitchLLM/verify_and_reindex.py
echo ""

echo "Step 2: Running strategy comparison..."
python src/pitchLLM/compare_rag_strategies.py
echo ""

echo "Step 3: Running RAG vs KNNFewShot comparison..."
python src/pitchLLM/compare_rag_vs_knn.py
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "Results saved to: rag_comparison_results/"
echo "=========================================="
