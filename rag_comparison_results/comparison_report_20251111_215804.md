# RAG Retrieval Strategy Comparison Report

Generated: 2025-11-11 21:58:04

Test Examples: 10
Strategies Compared: 4

## Summary Statistics

| Strategy | Success Rate | Avg Quality | Avg Latency (s) | Avg Tokens | Avg Retrieved | Avg Similarity |
|----------|--------------|-------------|------------------|------------|---------------|----------------|
| Pure_Semantic | 100.0% | 0.629 | 0.14 | 411 | 5 | 0.396 |
| Hybrid_Filter_Deals | 100.0% | 0.630 | 0.12 | 354 | 5 | 0.368 |
| Hybrid_Prioritize | 100.0% | 0.604 | 0.13 | 352 | 5 | 0.390 |
| Non_RAG | 100.0% | 0.624 | 0.00 | 278 | nan | nan |

## Strategy Descriptions

### 1. Pure_Semantic
- Pure similarity-based retrieval
- No filters or prioritization
- Baseline approach

### 2. Hybrid_Filter_Deals
- Semantic search + metadata filtering
- Only retrieves successful deals
- Filters at query time

### 3. Hybrid_Prioritize
- Semantic search + rule-based prioritization
- Prioritizes: Successful deals + Same category > Successful deals > Same category > Others
- Maintains semantic similarity within priority levels

### 4. Non_RAG
- Baseline without RAG
- No retrieval, pure generation

## Key Findings

- **Best Quality Score**: Hybrid_Filter_Deals (0.630)
- **Fastest**: Non_RAG (0.00s)
- **RAG Improvement**: Hybrid_Filter_Deals improves over Non-RAG by 1.0%
