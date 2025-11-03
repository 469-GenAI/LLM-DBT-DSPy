# ChromaDB Setup Guide

## Overview

The scripts use **ChromaDB** as the vector database to store embeddings for RAG (Retrieval-Augmented Generation). ChromaDB automatically handles:
- Document embeddings (using default embedding models)
- Vector storage and retrieval
- Similarity search

## What Gets Embedded?

Currently, the scripts embed **TWO types of content** into ChromaDB:

### 1. PDF Documents (HBS Pitch Materials)
- **Source**: HBS opportunity and pitch deck sample PDFs
- **Purpose**: Provides best practices and examples for pitch creation
- **Location**: Downloaded from GitHub or loaded from `data/pdfs/` if available
- **Collection Name**: `HBS_{chunking_strategy}` (e.g., `HBS_fixed`)

### 2. Product Facts JSON (`all_processed_facts.json`)
- **Source**: `ganwang/all_processed_facts.json`
- **Content**: 119 product entries with facts, descriptions, pitch summaries, offers
- **Purpose**: Enables RAG to find similar products, successful pitches, and comparable data
- **Format**: Each product is converted to structured text and embedded

## ChromaDB Configuration

### Automatic Setup
ChromaDB is **automatically configured** by the scripts. You don't need to set it up manually!

### Storage Location
- **Default Path**: `./chromadb_data/`
- **Collection Names**: 
  - `HBS_fixed` (for fixed chunking strategy)
  - `HBS_agentic` (for agentic chunking)
  - `HBS_semantic` (for semantic chunking)

### Environment Variables (Optional)
You can customize ChromaDB behavior with environment variables:

```bash
# In your .env file or shell
export ASYNC_RAG_DB_PATH="./custom_chromadb_path"
export ASYNC_RAG_COLLECTION_NAME="MyCustomCollection"
export ASYNC_RAG_CHUNKING_STRATEGY="fixed"  # or "agentic" or "semantic"
export ASYNC_RAG_FORCE_RELOAD="true"  # Force re-embedding everything
export ASYNC_RAG_SKIP_EMBEDDING_CHECK="true"  # Skip checking for existing docs
```

## How RAG is Used

### In Scripts with RAG:
1. **`async_specialist_team_single_rag_tool_ddg.py`** ✅ Uses RAG
2. **`async_specialist_team_single_rag_tool_tavily.py`** ✅ Uses RAG
3. **`async_specialist_team_single_rag_tool_tavily_norag.py`** ❌ NO RAG (as name suggests)

### Where RAG is Used:
- **Pitch Critic Agent**: Uses RAG to find relevant examples and best practices when critiquing pitches
- The Critic searches the knowledge base to compare the draft pitch against:
  - Successful pitch examples from the JSON
  - Best practices from the HBS PDFs
  - Similar products and their outcomes

## First Run vs Subsequent Runs

### First Run:
1. Creates ChromaDB collection
2. Downloads/loads PDFs and embeds them
3. Loads `all_processed_facts.json` and embeds all 119 products
4. This may take several minutes

### Subsequent Runs:
1. Checks if documents already exist in ChromaDB
2. **Skips re-embedding** if found (saves time!)
3. Only adds new content if `--force-reload` is used

## Verifying ChromaDB is Working

### Check if ChromaDB data exists:
```bash
ls -la chromadb_data/
# Should see collection directories
```

### Force reload everything:
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py --force-reload
```

### Check logs:
The script logs what gets embedded:
- "Adding X PDF file(s)"
- "Adding X PDF URL(s)"
- "Adding facts from all_processed_facts.json to knowledge base..."
- "Successfully added 119 product entries to knowledge base"

## Troubleshooting

### Issue: ChromaDB seems empty
**Solution**: Run with `--force-reload` flag:
```bash
python ganwang/async_specialist_team_single_rag_tool_ddg.py --force-reload
```

### Issue: Want to start fresh
**Solution**: Delete the ChromaDB directory and rerun:
```bash
rm -rf chromadb_data/
python ganwang/async_specialist_team_single_rag_tool_ddg.py
```

### Issue: ChromaDB path not found
**Solution**: ChromaDB will create the directory automatically. If you want a custom path:
```bash
export ASYNC_RAG_DB_PATH="./my_custom_path"
```

## No Additional API Keys Needed!

**ChromaDB is local** - it doesn't require any API keys or external services. It runs entirely on your machine using:
- Default embedding models (handled by ChromaDB)
- Local file storage
- In-memory vector operations

## What You DO Need:

1. **OpenAI API Key** (for LLM models)
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

2. **Tavily API Key** (only for tavily.py scripts)
   ```bash
   TAVILY_API_KEY=tvly-your-key-here
   ```

That's it! ChromaDB is completely automatic.

