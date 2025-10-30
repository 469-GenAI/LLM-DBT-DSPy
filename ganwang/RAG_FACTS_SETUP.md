# RAG with Facts JSON - Setup Complete ✅

## What Was Fixed

### ✅ All Three Scripts Now Embed `all_processed_facts.json`

**Before**: Only PDF documents (HBS pitch materials) were embedded into ChromaDB for RAG  
**After**: BOTH PDFs AND the `all_processed_facts.json` file (119 products) are embedded

### Scripts Updated:
1. ✅ `async_specialist_team_single_rag_tool_ddg.py` - Uses RAG with facts
2. ✅ `async_specialist_team_single_rag_tool_tavily.py` - Uses RAG with facts  
3. ❌ `async_specialist_team_single_rag_tool_tavily_norag.py` - NO RAG (by design)

## What Gets Embedded Now

### 1. PDF Documents (HBS Materials)
- HBS opportunities and pitch deck samples
- Provides best practices and examples

### 2. Product Facts (`all_processed_facts.json`) - **NEW!**
- **119 product entries** from your JSON file
- Each product includes:
  - Facts (sales, revenue, customer base, etc.)
  - Product descriptions
  - Pitch summaries (delivery, sentiment, story, key aspects)
  - Initial and final offers
- **Purpose**: Enables RAG to find:
  - Similar products for comparison
  - Successful pitch patterns
  - Comparable financial data
  - Similar market situations

## How It Works

### First Run:
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py
```

**What happens:**
1. Creates ChromaDB collection
2. Downloads/loads PDFs → embeds them
3. Loads `all_processed_facts.json` → converts each product to text → embeds all 119 entries
4. Total embedding time: 5-10 minutes (depends on your machine)

### Subsequent Runs:
- Checks if content exists in ChromaDB
- **Skips re-embedding** (saves time!)
- Only embeds if `--force-reload` is used

### Verification in Logs:
Look for these messages:
```
Adding facts from all_processed_facts.json to knowledge base...
Converting 119 product entries into embeddable text...
Successfully added 119 product entries to knowledge base.
```

## How RAG Uses the Facts

### Pitch Critic Agent:
When critiquing a draft pitch, the Pitch Critic agent:
1. Searches the knowledge base using the draft pitch as a query
2. Finds relevant examples from:
   - Similar products with comparable metrics
   - Successful pitches with similar characteristics
   - Products in the same industry/space
3. Uses these examples to provide better critiques:
   - "Similar products like X achieved Y valuation with Z sales"
   - "Products in this category typically offer X% equity"
   - "Successful pitches in this space emphasize A, B, C"

## ChromaDB Configuration

### No Setup Required!
ChromaDB is **automatically configured**. It:
- Creates storage at `./chromadb_data/` (local, no API keys needed)
- Uses default embedding models (no configuration needed)
- Handles all vector operations automatically

### Optional Customization:
```bash
# Custom storage location
export ASYNC_RAG_DB_PATH="./my_custom_path"

# Custom collection name
export ASYNC_RAG_COLLECTION_NAME="MyCustomCollection"

# Chunking strategy
export ASYNC_RAG_CHUNKING_STRATEGY="fixed"  # or "agentic" or "semantic"
```

## Testing the Setup

### Verify Facts Are Embedded:
```bash
# Run with debug output
python ganwang/async_specialist_team_single_rag_tool_ddg.py --product-key "facts_shark_tank_transcript_0_GarmaGuard.txt"

# Check logs for:
# - "Adding facts from all_processed_facts.json..."
# - "Successfully added 119 product entries..."
# - "RAG used by Pitch Critic: True"
```

### Force Re-embed Everything:
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py --force-reload
```

## API Keys Still Needed

### Required:
- **OPENAI_API_KEY** - For LLM models (required for all scripts)

### Optional:
- **TAVILY_API_KEY** - Only for `tavily.py` and `tavily_norag.py` scripts

### NOT Needed:
- ❌ ChromaDB API key (it's local, no API needed)
- ❌ Embedding API key (uses default models)
- ❌ Any other API keys

## Troubleshooting

### Facts not being embedded?
1. **Check if JSON file exists**:
   ```bash
   ls ganwang/all_processed_facts.json
   ```

2. **Force reload**:
   ```bash
   python ganwang/async_specialist_team_single_rag_tool_ddg.py --force-reload
   ```

3. **Check logs** for warning messages about missing files

### Want to disable facts embedding?
The scripts have `include_facts_json=True` by default. If you want to disable it, you'd need to modify the code (not recommended unless you have a reason).

### ChromaDB location issues?
- Default: `./chromadb_data/`
- Check you have write permissions
- Scripts will create the directory automatically

## Summary

✅ **All facts JSON data is now embedded into ChromaDB**  
✅ **RAG can now search through 119 products for similar examples**  
✅ **Pitch Critic will provide better feedback using historical data**  
✅ **No additional setup needed - it's automatic!**

Just run the scripts normally and the facts will be embedded automatically on first run!

