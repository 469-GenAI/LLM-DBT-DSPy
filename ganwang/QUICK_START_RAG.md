# Quick Start - Running Scripts with RAG

## ✅ Setup Complete!

All scripts are now configured to:
1. ✅ Embed PDF documents (HBS pitch materials)
2. ✅ Embed all 119 products from `all_processed_facts.json`
3. ✅ Use ChromaDB automatically (no API keys needed)
4. ✅ Use RAG for Pitch Critic to find similar examples

## Setup Your .env File

Create or edit `.env` in the project root:

```bash
# Required for ALL scripts
OPENAI_API_KEY=sk-your-openai-key-here

# Required ONLY for tavily.py scripts
TAVILY_API_KEY=tvly-your-tavily-key-here
```

**Remember: NO quotes around the values!**

## Run the Scripts

### 1. Activate Virtual Environment
```bash
cd /Users/leozhengkai/Documents/GitHub/LLM-DBT-DSPy
source venv/bin/activate
```

### 2. Run Scripts

**Option A: DuckDuckGo (Simplest - only needs OpenAI)**
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py
```

**Option B: Tavily (Needs both API keys)**
```bash
python ganwang/async_specialist_team_single_rag_tool_tavily.py
```

**Option C: Tavily No-RAG (Needs both API keys, but no RAG)**
```bash
python ganwang/async_specialist_team_single_rag_tool_tavily_norag.py
```

## What Happens on First Run

1. **Creates ChromaDB** at `./chromadb_data/`
2. **Downloads PDFs** (if not found locally)
3. **Embeds PDFs** into ChromaDB (~2-3 minutes)
4. **Loads `all_processed_facts.json`**
5. **Embeds all 119 products** into ChromaDB (~5-7 minutes)
6. **Generates pitch** using multi-agent system

**Total first run**: ~10-15 minutes (mostly embedding)

**Subsequent runs**: Much faster (skips embedding)

## Verify RAG is Working

Look for these log messages:
```
Adding facts from all_processed_facts.json to knowledge base...
Converting 119 product entries into embeddable text...
Successfully added 119 product entries to knowledge base.
...
Starting Pitch Critic with RAG...
RAG used by Pitch Critic: True
```

## Force Re-embed Everything

If you want to refresh all embeddings:
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py --force-reload
```

## Custom Options

```bash
# Use a different product
python ganwang/async_specialist_team_single_rag_tool_ddg.py \
  --product-key "facts_shark_tank_transcript_0_Roadie.txt"

# Use different chunking strategy
python ganwang/async_specialist_team_single_rag_tool_ddg.py \
  --chunking semantic

# Disable debug mode
python ganwang/async_specialist_team_single_rag_tool_ddg.py \
  --no-debug
```

## Output Files

Results are saved to timestamped directories:
```
./outputs/async_specialist_team_rag_ddg_TIMESTAMP/
├── final_pitch.json          # Generated pitch
├── metrics.json              # Performance metrics
├── input_data.json           # Input data used
└── full_log_*.log            # Complete execution log
```

## Troubleshooting

**"OPENAI_API_KEY not found"**
→ Check your `.env` file exists and has the key (no quotes!)

**"ChromaDB error"**
→ Delete `chromadb_data/` and rerun (it will recreate)

**"Facts not embedding"**
→ Check `ganwang/all_processed_facts.json` exists
→ Run with `--force-reload`

**"Module not found"**
→ Make sure venv is activated
→ Run `pip install python-dotenv pydantic agno chromadb litellm ddgs`

## Summary

✅ Facts JSON is embedded automatically  
✅ RAG searches through 119 products for similar examples  
✅ Pitch Critic uses historical data for better feedback  
✅ ChromaDB runs locally (no API keys needed)  
✅ Everything is automatic - just run the scripts!

