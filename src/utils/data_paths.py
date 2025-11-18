"""
Centralized data path utilities.
Provides consistent access to data files from anywhere in the codebase.
"""
from pathlib import Path
from typing import Optional


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Assumes data/ is at the repository root level.
    Works from any subdirectory by finding the repo root.
    
    Returns:
        Path to data/ directory
    """
    # Start from current file location
    current_file = Path(__file__).resolve()
    
    # Navigate up to find repo root (containing data/ folder)
    # Current file is in: src/utils/data_paths.py
    # Need to go: src/utils -> src -> repo_root
    repo_root = current_file.parent.parent.parent
    
    data_dir = repo_root / "data"
    
    if not data_dir.exists():
        # Fallback: try current working directory
        cwd = Path.cwd()
        if (cwd / "data").exists():
            return cwd / "data"
        
        raise FileNotFoundError(
            f"Could not find data/ directory. "
            f"Looked in: {repo_root / 'data'}, {cwd / 'data'}"
        )
    
    return data_dir


def get_data_file(filename: str, subdirectory: Optional[str] = None) -> Path:
    """
    Get path to a data file.
    
    Args:
        filename: Name of the file (e.g., "all_processed_facts.json")
        subdirectory: Optional subdirectory within data/ (e.g., "pitches", "raw_transcripts")
        
    Returns:
        Path to the data file
        
    Examples:
        get_data_file("all_processed_facts.json")
        -> data/all_processed_facts.json
        
        get_data_file("existing_refined_pitches(119).json", "pitches")
        -> data/pitches/existing_refined_pitches(119).json
    """
    data_dir = get_data_dir()
    
    if subdirectory:
        file_path = data_dir / subdirectory / filename
    else:
        file_path = data_dir / filename
    
    return file_path


# Common data file paths (for convenience)
def get_all_processed_facts() -> Path:
    """Get path to all_processed_facts.json"""
    return get_data_file("all_processed_facts.json")


def get_po_samples() -> Path:
    """Get path to PO_samples.json"""
    return get_data_file("PO_samples.json")


def get_refined_pitches_119() -> Path:
    """Get path to existing_refined_pitches(119).json"""
    return get_data_file("existing_refined_pitches(119).json", "pitches")


def get_refined_pitches_126() -> Path:
    """Get path to refined_pitches_(126).json"""
    return get_data_file("refined_pitches_(126).json", "pitches")


def get_raw_transcripts_119() -> Path:
    """Get path to existing_transcripts_(119).json"""
    return get_data_file("existing_transcripts_(119).json", "raw_transcripts")


def get_raw_transcripts_126() -> Path:
    """Get path to transcripts_(126).json"""
    return get_data_file("transcripts_(126).json", "raw_transcripts")


if __name__ == "__main__":
    # Test the utility
    print("Testing data path utilities...")
    print(f"Data directory: {get_data_dir()}")
    print(f"all_processed_facts.json: {get_all_processed_facts()}")
    print(f"PO_samples.json: {get_po_samples()}")
    print(f"Refined pitches (119): {get_refined_pitches_119()}")
    print(f"Refined pitches (126): {get_refined_pitches_126()}")
    
    # Check if files exist
    print("\nFile existence check:")
    print(f"all_processed_facts.json exists: {get_all_processed_facts().exists()}")
    print(f"PO_samples.json exists: {get_po_samples().exists()}")
    print(f"Refined pitches (119) exists: {get_refined_pitches_119().exists()}")
    print(f"Refined pitches (126) exists: {get_refined_pitches_126().exists()}")


