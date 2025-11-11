"""
Data indexer for loading and preparing pitch transcripts for RAG.

Data Sources:
1. JSONL files (REQUIRED for train/test split): Clean, validated data
   - train.jsonl: Training examples (~196 products) - INDEXED into vector DB
   - test.jsonl: Test examples (~49 products) - NOT indexed, used for evaluation only
   
NOTE: refined_pitches.json files are NO LONGER USED. Only train.jsonl is indexed.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import logging
import sys

# Import data path utilities with fallback
try:
    # Try absolute import first
    from src.utils.data_paths import get_all_processed_facts, get_po_samples, get_data_dir
except ImportError:
    try:
        # Try relative import from parent
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from utils.data_paths import get_all_processed_facts, get_po_samples, get_data_dir
    except ImportError:
        # Fallback: define functions locally
        def get_data_dir() -> Path:
            """Fallback: find data directory relative to this file."""
            current_file = Path(__file__).resolve()
            # Navigate: src/pitchLLM/rag/data_indexer.py -> src -> repo_root -> data
            repo_root = current_file.parent.parent.parent.parent
            data_dir = repo_root / "data"
            if not data_dir.exists():
                # Try current working directory
                cwd = Path.cwd()
                if (cwd / "data").exists():
                    return cwd / "data"
                raise FileNotFoundError(f"Could not find data/ directory")
            return data_dir
        
        def get_all_processed_facts() -> Path:
            """Fallback: get path to all_processed_facts.json"""
            return get_data_dir() / "all_processed_facts.json"
        
        def get_po_samples() -> Path:
            """Fallback: get path to PO_samples.json"""
            return get_data_dir() / "PO_samples.json"

# Import category mapping functions
try:
    from .category_classifier import load_category_mapping
except ImportError:
    # Fallback if category_classifier not available
    def load_category_mapping(mapping_path: Path) -> Dict:
        """Fallback: return empty dict if category_classifier not available."""
        return {}

logger = logging.getLogger(__name__)


def _load_json_safe(file_path: Path) -> Union[Dict, List, None]:
    """
    Safely load JSON file with error handling.
    
    Returns:
        Loaded data (dict or list) or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {file_path}: {e}")
        logger.error(f"Error at position {e.pos if hasattr(e, 'pos') else 'unknown'}")
        return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_refined_pitches(base_path: Path = None) -> Dict[str, Dict]:
    """
    Load refined pitches files (preferred source - clean and validated).
    
    Args:
        base_path: Base path to data directory (if None, uses centralized data paths)
        
    Returns:
        Dict mapping product_key -> pitch data
    """
    if base_path is None:
        base_path = get_data_dir()
    else:
        base_path = Path(base_path)
    
    all_transcripts = {}
    
    # File 1: existing_refined_pitches(119).json
    refined_path_1 = base_path / "pitches" / "existing_refined_pitches(119).json"
    if refined_path_1.exists():
        logger.info(f"Loading refined pitches from: {refined_path_1}")
        data_1 = _load_json_safe(refined_path_1)
        
        if data_1 is not None:
            if isinstance(data_1, list):
                # Array format: [{"Product": "...", "Full_Pitch": "..."}, ...]
                for item in data_1:
                    if isinstance(item, dict) and "Product" in item:
                        product_key = item["Product"]
                        all_transcripts[product_key] = item
                logger.info(f"Loaded {len(data_1)} refined pitches from file 1")
            elif isinstance(data_1, dict):
                # Dict format: {"ProductName": {...}, ...}
                all_transcripts.update(data_1)
                logger.info(f"Loaded {len(data_1)} refined pitches from file 1")
    else:
        logger.warning(f"File not found: {refined_path_1}")
    
    # File 2: refined_pitches_(126).json
    refined_path_2 = base_path / "pitches" / "refined_pitches_(126).json"
    if refined_path_2.exists():
        logger.info(f"Loading refined pitches from: {refined_path_2}")
        data_2 = _load_json_safe(refined_path_2)
        
        if data_2 is not None:
            if isinstance(data_2, list):
                # Array format
                for item in data_2:
                    if isinstance(item, dict) and "Product" in item:
                        product_key = item["Product"]
                        # Only add if not already present (avoid duplicates)
                        if product_key not in all_transcripts:
                            all_transcripts[product_key] = item
                logger.info(f"Loaded {len(data_2)} refined pitches from file 2")
            elif isinstance(data_2, dict):
                # Dict format
                for key, value in data_2.items():
                    if key not in all_transcripts:
                        all_transcripts[key] = value
                logger.info(f"Loaded {len(data_2)} refined pitches from file 2")
    else:
        logger.warning(f"File not found: {refined_path_2}")
    
    return all_transcripts


def load_raw_transcripts(base_path: Path = None) -> Dict[str, Any]:
    """
    Load raw transcript files (fallback if refined pitches unavailable).
    
    Args:
        base_path: Base path to data directory (if None, uses centralized data paths)
        
    Returns:
        Dict mapping product_key -> pitch data
    """
    if base_path is None:
        base_path = get_data_dir()
    else:
        base_path = Path(base_path)
    
    all_transcripts = {}
    
    # File 1: existing_transcripts_(119).json (known to work)
    raw_path_1 = base_path / "raw_transcripts" / "existing_transcripts_(119).json"
    if raw_path_1.exists():
        logger.info(f"Loading raw transcripts from: {raw_path_1}")
        data_1 = _load_json_safe(raw_path_1)
        
        if data_1 is not None and isinstance(data_1, dict):
            all_transcripts.update(data_1)
            logger.info(f"Loaded {len(data_1)} raw transcripts from file 1")
    else:
        logger.warning(f"File not found: {raw_path_1}")
    
    # File 2: transcripts_(126).json (may have parsing errors)
    raw_path_2 = base_path / "raw_transcripts" / "transcripts_(126).json"
    if raw_path_2.exists():
        logger.info(f"Attempting to load raw transcripts from: {raw_path_2}")
        data_2 = _load_json_safe(raw_path_2)
        
        if data_2 is not None and isinstance(data_2, dict):
            # Merge, preferring existing if overlap
            for key, value in data_2.items():
                if key not in all_transcripts:
                    all_transcripts[key] = value
            logger.info(f"Loaded {len(data_2)} raw transcripts from file 2")
        else:
            logger.warning(f"Failed to parse {raw_path_2} - skipping")
    else:
        logger.warning(f"File not found: {raw_path_2}")
    
    return all_transcripts


def load_jsonl_pitches(jsonl_path: Path) -> Dict[str, Dict]:
    """
    Load pitches from JSONL file (train.jsonl format).
    
    Format: Each line is a JSON object with:
    - id: unique identifier
    - input: dict with company, founder, offer, problem_summary, solution_summary
    - output: pitch text
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        Dict mapping id -> pitch data
    """
    transcripts = {}
    
    if not jsonl_path.exists():
        logger.warning(f"JSONL file not found: {jsonl_path}")
        return transcripts
    
    logger.info(f"Loading pitches from JSONL: {jsonl_path}")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    pitch_id = data.get('id', f'line_{line_num}')
                    
                    # Extract pitch text from 'output' field
                    pitch_text = data.get('output', '')
                    
                    # Extract input data for metadata
                    input_data = data.get('input', {})
                    
                    # Create pitch document structure
                    transcripts[pitch_id] = {
                        'Product': input_data.get('company', pitch_id),
                        'Full_Pitch': pitch_text,
                        'input': input_data,
                        'id': pitch_id,
                        'source': 'jsonl'
                    }
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num} in {jsonl_path}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(transcripts)} pitches from JSONL file")
        
    except Exception as e:
        logger.error(f"Error loading JSONL file {jsonl_path}: {e}")
    
    return transcripts


def load_all_transcripts(base_path: str = None, prefer_refined: bool = True) -> Dict[str, Dict]:
    """
    Load all pitch transcripts from available sources (~245 total).
    
    NOTE: This loads ALL transcripts. For train/test split, use load_and_prepare_all(use_jsonl=True) instead.
    
    Priority:
    1. refined_pitches files (if prefer_refined=True) - clean, validated, ~245 products
    2. raw_transcripts files (fallback) - may have parsing errors
    
    Args:
        base_path: Base path to data directory. If None, uses default location.
        prefer_refined: If True, try refined_pitches first (default: True)
        
    Returns:
        Dict mapping product_key -> pitch data (combined from all sources)
    """
    if base_path is None:
        # Use centralized data path
        base_path = get_data_dir()
    else:
        base_path = Path(base_path)
    
    all_transcripts = {}
    
    if prefer_refined:
        # Try refined pitches first (preferred)
        refined_data = load_refined_pitches(base_path)
        if refined_data:
            all_transcripts.update(refined_data)
            logger.info(f"Loaded {len(refined_data)} pitches from refined files")
    
    # If we don't have enough data, try raw transcripts
    if len(all_transcripts) < 200:  # Threshold for "enough" data
        logger.info("Loading additional data from raw transcripts...")
        raw_data = load_raw_transcripts(base_path)
        
        # Merge raw data, preferring refined if overlap
        for key, value in raw_data.items():
            if key not in all_transcripts:
                all_transcripts[key] = value
    
    total = len(all_transcripts)
    logger.info(f"Total transcripts loaded: {total}")
    
    if total == 0:
        raise ValueError(
            "No transcripts loaded. Check file paths.\n"
            "Expected files:\n"
            "  - data/pitches/existing_refined_pitches(119).json\n"
            "  - data/pitches/refined_pitches_(126).json\n"
            "  - data/raw_transcripts/existing_transcripts_(119).json"
        )
    
    return all_transcripts


def prepare_pitch_documents(
    transcripts: Dict[str, Dict], 
    chunk_size: int = None,
    chunk_overlap: int = 0,
    category_mapping: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Convert transcripts into documents suitable for vector store.
    
    Each document contains:
    - id: unique identifier
    - text: searchable pitch text (or chunk if chunking enabled)
    - metadata: product info, category, etc.
    
    Args:
        transcripts: Dict of product_key -> pitch data
        chunk_size: If provided, split long pitches into chunks of this size (chars).
                    If None, store entire pitch as one document (default).
        chunk_overlap: Number of characters to overlap between chunks (default: 0)
        
    Returns:
        List of document dicts ready for indexing
        
    Note:
        Chunking is OPTIONAL. By default, each pitch is stored as one document.
        See vector_store.py for chunking strategy documentation.
    """
    documents = []
    
    for product_key, pitch_data in transcripts.items():
        # Extract pitch text (handle different formats)
        pitch_text = ""
        
        if isinstance(pitch_data, dict):
            # Try common field names (prioritize Full_Pitch for refined data)
            pitch_text = (
                pitch_data.get('Full_Pitch') or 
                pitch_data.get('full_pitch') or 
                pitch_data.get('pitch') or 
                pitch_data.get('transcript') or
                str(pitch_data)
            )
        elif isinstance(pitch_data, str):
            pitch_text = pitch_data
        else:
            pitch_text = str(pitch_data)
        
        # Skip if no meaningful text
        if not pitch_text or len(pitch_text) < 50:
            logger.warning(f"Skipping {product_key}: insufficient pitch text")
            continue
        
        # Extract base metadata
        base_metadata = {
            'product_key': product_key,
            'source': pitch_data.get('source', 'shark_tank_transcripts')
        }
        
        # Add product name if available
        if isinstance(pitch_data, dict):
            product_name = pitch_data.get('Product', product_key)
            # For JSONL format, also check 'input' field
            if not product_name or product_name == product_key:
                input_data = pitch_data.get('input', {})
                product_name = input_data.get('company', product_key)
            
            base_metadata['product_name'] = product_name
            
            # Add split metadata (train/test) - default to 'train' for JSONL
            if pitch_data.get('source') == 'jsonl':
                base_metadata['split'] = 'train'  # train.jsonl is used for indexing
            else:
                base_metadata['split'] = 'unknown'
            
            # Add other available metadata
            for key in ['category', 'ask', 'equity', 'outcome', 'sales']:
                if key in pitch_data:
                    base_metadata[key] = pitch_data[key]
            
            # Extract category from input data if available (JSONL format)
            input_data = pitch_data.get('input', {})
            if input_data and not base_metadata.get('category'):
                # Try to infer category from problem/solution if available
                pass  # Category not in JSONL input format
            
            # Load category from mapping file (if available)
            # This allows categories to be stored separately without modifying original data
            if not base_metadata.get('category') and category_mapping:
                # Try to get category from mapping using product_key or id
                mapped_category = category_mapping.get(product_key)
                if not mapped_category and isinstance(pitch_data, dict):
                    # Try using id field if available
                    pitch_id = pitch_data.get('id')
                    if pitch_id:
                        mapped_category = category_mapping.get(pitch_id)
                
                if mapped_category:
                    base_metadata['category'] = mapped_category
                    logger.debug(f"Loaded category '{mapped_category}' for {product_key} from mapping")
            
            # Add deal information if available
            deal_info = pitch_data.get('deal_info', {})
            if deal_info:
                base_metadata['has_deal'] = deal_info.get('has_deal', False)
                base_metadata['investors'] = deal_info.get('investors', '')
                base_metadata['deal_category'] = deal_info.get('category', '')
                
                # Extract category from product_description if available
                if not base_metadata.get('category') and deal_info.get('product_description'):
                    category = deal_info['product_description'].get('category', '')
                    if category:
                        base_metadata['category'] = category
        
        # Apply chunking if requested
        if chunk_size and len(pitch_text) > chunk_size:
            # Split into chunks
            chunks = _chunk_text(pitch_text, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata['chunk_index'] = chunk_idx
                chunk_metadata['total_chunks'] = len(chunks)
                chunk_metadata['is_chunked'] = True
                
                document = {
                    'id': f"{product_key}_chunk_{chunk_idx}",
                    'text': chunk_text,
                    'metadata': chunk_metadata
                }
                documents.append(document)
            
            logger.debug(f"Split {product_key} into {len(chunks)} chunks")
        else:
            # Store entire pitch as one document (default behavior)
            document = {
                'id': product_key,
                'text': pitch_text,
                'metadata': base_metadata
            }
            documents.append(document)
    
    logger.info(f"Prepared {len(documents)} documents for indexing")
    if chunk_size:
        logger.info(f"Chunking enabled: {chunk_size} chars per chunk, {chunk_overlap} overlap")
    
    return documents


def _chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks with optional overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position forward, accounting for overlap
        start = end - overlap if overlap > 0 else end
        
        # Prevent infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start += 1
    
    return chunks


def extract_deal_from_final_offer(final_offer: Dict) -> Optional[Dict]:
    """
    Extract deal information from final_offer structure.
    
    Returns None if no deal, otherwise returns deal info dict.
    
    Patterns:
    - Successful: has 'investors' or 'shark' field (not empty)
    - Unsuccessful: has 'outcome' field saying "No deal" or empty final_offer
    """
    if not final_offer:
        return None
    
    # Check for "No deal" outcome
    outcome = final_offer.get('outcome', '').lower()
    if 'no deal' in outcome or outcome.startswith('no deal'):
        return None
    
    # Check for successful deal indicators
    investors = final_offer.get('investors', '') or final_offer.get('shark', '')
    if investors:
        deal_info = {
            'has_deal': True,
            'investors': investors,
            'amount_offered': final_offer.get('amount_invested') or final_offer.get('amount', ''),
            'equity_taken': final_offer.get('equity_stake') or final_offer.get('equity', ''),
            'additional_terms': final_offer.get('additional_terms', [])
        }
        return deal_info
    
    return None


def load_all_processed_facts_with_deals(base_path: Path = None) -> Dict[str, Dict]:
    """
    Load all_processed_facts.json and extract deal information from final_offer.
    
    Returns dict mapping product_name -> deal_info for all pitches (successful and unsuccessful).
    """
    if base_path is None:
        facts_path = get_all_processed_facts()
    else:
        facts_path = Path(base_path) / "all_processed_facts.json"
    
    if not facts_path.exists():
        logger.warning(f"all_processed_facts.json not found at {facts_path}")
        return {}
    
    try:
        with open(facts_path, 'r', encoding='utf-8') as f:
            facts_data = json.load(f)
        
        deals_info = {}
        for key, sample in facts_data.items():
            # Ensure sample is a dict
            if not isinstance(sample, dict):
                continue
            
            # Extract product name from various possible locations
            product_desc = sample.get('product_description', {})
            if not isinstance(product_desc, dict):
                product_desc = {}
            
            product_name = (
                product_desc.get('name') or
                product_desc.get('product_name') or
                key.replace('facts_shark_tank_transcript_', '').replace('.txt', '').split('_', 1)[-1] if '_' in key else key
            )
            
            if not product_name:
                continue
            
            # Extract deal info from final_offer
            final_offer = sample.get('final_offer', {})
            if not isinstance(final_offer, dict):
                final_offer = {}
            deal_info = extract_deal_from_final_offer(final_offer)
            
            # Store deal info (None means no deal, which is also useful information)
            if deal_info:
                # Successful deal
                deal_info['category'] = product_desc.get('category', '')
                deal_info['product_description'] = product_desc
                deals_info[product_name] = deal_info
            else:
                # Mark as no deal for filtering purposes
                deals_info[product_name] = {
                    'has_deal': False,
                    'category': product_desc.get('category', ''),
                    'product_description': product_desc
                }
        
        successful_count = sum(1 for d in deals_info.values() if d.get('has_deal', False))
        logger.info(f"Loaded deal information from all_processed_facts.json: {successful_count} successful, {len(deals_info) - successful_count} unsuccessful")
        return deals_info
    
    except Exception as e:
        logger.error(f"Error loading all_processed_facts.json: {e}")
        return {}


def load_po_samples_with_deals(base_path: Path = None) -> Dict[str, Dict]:
    """
    Load PO_samples.json and extract deal information.
    
    Returns dict mapping product_name -> deal_info for successful pitches.
    """
    if base_path is None:
        po_samples_path = get_po_samples()
    else:
        po_samples_path = Path(base_path) / "PO_samples.json"
    
    if not po_samples_path.exists():
        logger.warning(f"PO_samples.json not found at {po_samples_path}")
        return {}
    
    try:
        with open(po_samples_path, 'r', encoding='utf-8') as f:
            po_data = json.load(f)
        
        deals_info = {}
        for key, sample in po_data.items():
            # Extract product name
            product_name = sample.get('product_description', {}).get('name', '')
            if not product_name:
                continue
            
            # Check if deal was closed using final_offer
            final_offer = sample.get('final_offer', {})
            deal_info = extract_deal_from_final_offer(final_offer)
            
            if deal_info:
                deal_info['category'] = sample.get('product_description', {}).get('category', '')
                deal_info['product_description'] = sample.get('product_description', {})
                deals_info[product_name] = deal_info
        
        logger.info(f"Loaded deal information from PO_samples.json for {len(deals_info)} successful pitches")
        return deals_info
    
    except Exception as e:
        logger.error(f"Error loading PO_samples.json: {e}")
        return {}


def load_and_prepare_all(
    base_path: str = None,
    chunk_size: int = None,
    chunk_overlap: int = 0,
    prefer_refined: bool = True,
    enrich_with_deals: bool = True,
    use_all_processed_facts: bool = True,
    use_jsonl: bool = True,
    jsonl_path: str = None,
    category_mapping_path: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function: load transcripts and prepare documents in one step.
    
    Args:
        base_path: Base path to data directory (if None, uses centralized data paths)
        chunk_size: Optional chunk size for splitting long pitches
        chunk_overlap: Optional overlap between chunks
        prefer_refined: If True, prefer refined_pitches files (default: True)
        enrich_with_deals: If True, enrich metadata with deal information
        use_all_processed_facts: If True, use all_processed_facts.json for deal info (more comprehensive)
                                 If False, use PO_samples.json (smaller subset)
        use_jsonl: If True, load from train.jsonl instead of refined_pitches files
        jsonl_path: Path to train.jsonl file (if None, uses default location)
        category_mapping_path: Path to category mapping JSON file (optional)
        
    Returns:
        List of prepared documents ready for indexing
    """
    # Load category mapping if provided
    category_mapping = {}
    if category_mapping_path:
        category_mapping = load_category_mapping(Path(category_mapping_path))
        logger.info(f"Loaded {len(category_mapping)} categories from mapping file")
    elif use_jsonl:
        # Try default location
        if base_path is None:
            base_path = get_data_dir()
        default_mapping_path = Path(base_path) / "hf (new)" / "category_mapping.json"
        if default_mapping_path.exists():
            category_mapping = load_category_mapping(default_mapping_path)
            logger.info(f"Loaded {len(category_mapping)} categories from default mapping file")
        else:
            logger.warning(f"Category mapping file not found at {default_mapping_path}. Categories will not be included.")
    
    # Load from JSONL if requested (for train/test split)
    if use_jsonl:
        if jsonl_path is None:
            if base_path is None:
                base_path = get_data_dir()
            jsonl_path = Path(base_path) / "hf (new)" / "train.jsonl"
        else:
            jsonl_path = Path(jsonl_path)
        
        transcripts = load_jsonl_pitches(jsonl_path)
        
        if not transcripts:
            raise ValueError(
                f"No transcripts loaded from {jsonl_path}. "
                f"Please ensure train.jsonl exists at: {jsonl_path}"
            )
    else:
        transcripts = load_all_transcripts(base_path, prefer_refined=prefer_refined)
    
    # Load deal information if requested
    deals_info = {}
    if enrich_with_deals:
        base_path_obj = Path(base_path) if base_path else get_data_dir()
        
        if use_all_processed_facts:
            # Use all_processed_facts.json (more comprehensive, includes all pitches)
            deals_info = load_all_processed_facts_with_deals(base_path_obj)
            logger.info("Using all_processed_facts.json for deal information")
        else:
            # Use PO_samples.json (smaller subset, only successful deals)
            deals_info = load_po_samples_with_deals(base_path_obj)
            logger.info("Using PO_samples.json for deal information")
    
    # Enrich transcripts with deal info
    # Match by product name (try various formats)
    for product_key, pitch_data in transcripts.items():
        # Try to match deal info by product name
        matched_deal_info = None
        
        # Try exact match first
        if product_key in deals_info:
            matched_deal_info = deals_info[product_key]
        else:
            # Try fuzzy matching - check if any deal_info product name matches
            product_name_from_pitch = None
            if isinstance(pitch_data, dict):
                product_name_from_pitch = pitch_data.get('Product', product_key)
                # Also try 'company' field from JSONL input
                if not product_name_from_pitch or product_name_from_pitch == product_key:
                    input_data = pitch_data.get('input', {})
                    product_name_from_pitch = input_data.get('company', product_key)
            else:
                product_name_from_pitch = product_key
            
            # Try to find matching deal info
            for deal_product_name, deal_info in deals_info.items():
                if (deal_product_name.lower() in product_name_from_pitch.lower() or 
                    product_name_from_pitch.lower() in deal_product_name.lower()):
                    matched_deal_info = deal_info
                    break
        
        if matched_deal_info:
            if isinstance(pitch_data, dict):
                pitch_data['deal_info'] = matched_deal_info
            elif isinstance(pitch_data, str):
                # Convert to dict format
                transcripts[product_key] = {
                    'Product': product_key,
                    'Full_Pitch': pitch_data,
                    'deal_info': matched_deal_info
                }
    
    documents = prepare_pitch_documents(
        transcripts, 
        chunk_size, 
        chunk_overlap,
        category_mapping=category_mapping
    )
    return documents


if __name__ == "__main__":
    # Fix imports when running directly
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    
    # Test loading
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("STEP 1: LOADING train.jsonl (this is what gets INDEXED)")
    print("="*70)
    documents = load_and_prepare_all(use_jsonl=True)
    print(f"✓ Loaded {len(documents)} documents from train.jsonl")
    print(f"  → These {len(documents)} documents will be ADDED to vector DB")
    print(f"  → This is INDEXING/STORING (permanent storage)")
    
    print("\n" + "="*70)
    print("STEP 2: COMPARISON - Loading ALL transcripts (NOT indexed!)")
    print("="*70)
    print("⚠️  This is JUST for comparison - these are NEVER indexed!")
    transcripts = load_all_transcripts()
    print(f"  Total transcripts (all sources): {len(transcripts)}")
    print(f"  Difference: {len(transcripts) - len(documents)} documents excluded")
    print(f"\n✓ Only {len(documents)} documents from train.jsonl are indexed")
    print(f"  ✗ {len(transcripts) - len(documents)} documents (test set) are NOT indexed")
    
    if documents:
        print("\nSample document:")
        sample = documents[0]
        print(f"ID: {sample['id']}")
        print(f"Text length: {len(sample['text'])} chars")
        print(f"Metadata: {sample['metadata']}")
        print(f"Text preview: {sample['text'][:200]}...")

