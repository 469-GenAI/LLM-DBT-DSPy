# Ganwang Pitch Generation Scripts

This folder contains three different multi-agent systems for generating SharkTank pitches:

1. **`async_specialist_team_single_rag_tool_tavily.py`** - Uses RAG (Retrieval-Augmented Generation) + Tavily search
2. **`async_specialist_team_single_rag_tool_tavily_norag.py`** - Uses Tavily search only (no RAG)
3. **`async_specialist_team_single_rag_tool_ddg.py`** - Uses RAG + DuckDuckGo search

## Prerequisites

### Required Environment Variables

You need to set the following environment variables (or create a `.env` file):

```bash
# Required for all scripts
export OPENAI_API_KEY="your-openai-api-key"

# Required for Tavily-based scripts (tavily.py and tavily_norag.py)
export TAVILY_API_KEY="your-tavily-api-key"
```

### Data File

The scripts automatically look for `all_processed_facts.json` in these locations:
- `./data/all_processed_facts.json`
- `./ganwang/all_processed_facts.json`
- Same directory as the script

The file is already present in the `ganwang/` folder.

## Running the Scripts

### Basic Usage

Run from the project root directory:

```bash
# With RAG + Tavily search
python ganwang/async_specialist_team_single_rag_tool_tavily.py

# With Tavily search only (no RAG)
python ganwang/async_specialist_team_single_rag_tool_tavily_norag.py

# With RAG + DuckDuckGo search
python ganwang/async_specialist_team_single_rag_tool_ddg.py
```

### Command-Line Options

All scripts support various options:

```bash
# Specify a different product to process
python ganwang/async_specialist_team_single_rag_tool_tavily.py --product-key "facts_shark_tank_transcript_0_GarmaGuard.txt"

# Change models (for tavily.py and tavily_norag.py)
python ganwang/async_specialist_team_single_rag_tool_tavily.py --specialist-model gpt-4o --pitch-model gpt-4o

# Change models (for ddg.py)
python ganwang/async_specialist_team_single_rag_tool_ddg.py --model gpt-4o

# Adjust generation parameters
python ganwang/async_specialist_team_single_rag_tool_tavily.py --max-tokens 4096 --temperature 0.7

# RAG-specific options (for scripts with RAG)
python ganwang/async_specialist_team_single_rag_tool_tavily.py --chunking fixed --force-reload

# Disable debug mode
python ganwang/async_specialist_team_single_rag_tool_tavily.py --no-debug
```

### Available Product Keys

To see available product keys, you can inspect the `all_processed_facts.json` file:
```bash
python -c "import json; data = json.load(open('ganwang/all_processed_facts.json')); print('\n'.join(list(data.keys())[:10]))"
```

## Output

The scripts create timestamped output directories in `./outputs/` containing:
- `final_pitch.json` - The generated pitch
- `metrics.json` - Execution metrics and statistics
- `input_data.json` - Input data used for the pitch
- `full_log_*.log` - Complete execution log

## Troubleshooting

1. **Missing API keys**: Make sure `OPENAI_API_KEY` and (if needed) `TAVILY_API_KEY` are set in your environment or `.env` file.

2. **Data file not found**: The scripts now automatically search for the data file in multiple locations. If it still can't find it, check that `all_processed_facts.json` exists in the `ganwang/` folder.

3. **Import errors**: Make sure you have installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

