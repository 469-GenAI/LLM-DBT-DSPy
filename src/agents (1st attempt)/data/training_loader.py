# training_loader.py - Load and transform PO_samples.json into DSPy training examples
import json
from pathlib import Path
from typing import List, Dict, Any
import dspy
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.data_paths import get_po_samples

# Import from agents.utils (need to check if it exists there)
try:
    from agents.utils import PitchResponse, InitialOffer
except ImportError:
    # Fallback: try relative import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import PitchResponse, InitialOffer


def load_po_samples(data_path: str = None) -> Dict[str, Any]:
    """Load and parse PO_samples.json file."""
    if data_path is None:
        data_path = str(get_po_samples())
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def flatten_facts(facts_dict: Dict[str, Any]) -> List[str]:
    """Convert facts dictionary to list of fact strings."""
    facts_list = []
    for key, value in facts_dict.items():
        if isinstance(value, (str, int, float)):
            facts_list.append(f"{key}: {value}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                facts_list.append(f"{key}_{sub_key}: {sub_value}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                facts_list.append(f"{key}_{i}: {item}")
    return facts_list


def calculate_implied_valuation(amount_requested: str, equity_offered: str) -> str:
    """Calculate implied valuation from amount requested and equity offered."""
    try:
        # Extract numeric values
        amount = float(amount_requested.replace('$', '').replace(',', ''))
        equity = float(equity_offered.replace('%', ''))
        
        if equity > 0:
            valuation = (amount / equity) * 100
            return f"${valuation:,.0f}"
        else:
            return "$0"
    except (ValueError, ZeroDivisionError):
        return "$0"


def synthesize_key_terms(sample_data: Dict[str, Any]) -> str:
    """Synthesize key terms from available context."""
    key_terms = []
    
    # Check for specific business metrics
    if "facts" in sample_data:
        facts = sample_data["facts"]
        if "profit_margin" in facts:
            key_terms.append(f"Profit margin: {facts['profit_margin']}")
        if "patent_status" in facts:
            key_terms.append(f"IP status: {facts['patent_status']}")
        if "retail_partners" in facts:
            key_terms.append(f"Market reach: {facts['retail_partners']}")
    
    # Check for growth indicators
    if "pitch_summary" in sample_data:
        pitch = sample_data["pitch_summary"]
        if "key_aspects" in pitch:
            if isinstance(pitch["key_aspects"], list):
                key_terms.extend(pitch["key_aspects"][:2])  # Take first 2 key aspects
            elif isinstance(pitch["key_aspects"], dict):
                key_terms.extend(list(pitch["key_aspects"].values())[:2])
    
    return "; ".join(key_terms) if key_terms else "Growth-focused investment opportunity"


def transform_to_dspy_example(sample_data: Dict[str, Any]) -> dspy.Example:
    """Transform a single sample into DSPy Example format."""
    try:
        # Prepare product data input
        product_data = {
            "description": sample_data.get("product_description", {}),
            "facts": flatten_facts(sample_data.get("facts", {}))
        }
        
        # Extract pitch from pitch_summary
        pitch_text = ""
        if "pitch_summary" in sample_data and "full_pitch" in sample_data["pitch_summary"]:
            pitch_text = sample_data["pitch_summary"]["full_pitch"]
        
        # Extract initial offer data
        initial_offer_data = sample_data.get("initial_offer", {})
        
        # Handle valuation - try to get from valuation_implied, calculate if missing
        valuation = initial_offer_data.get("valuation_implied", "")
        if not valuation and "amount_requested" in initial_offer_data and "equity_offered" in initial_offer_data:
            valuation = calculate_implied_valuation(
                initial_offer_data["amount_requested"],
                initial_offer_data["equity_offered"]
            )
        
        # Create InitialOffer object
        initial_offer = InitialOffer(
            Valuation=valuation or "$0",
            Equity_Offered=initial_offer_data.get("equity_offered", "0%"),
            Funding_Amount=initial_offer_data.get("amount_requested", "$0"),
            Key_Terms=synthesize_key_terms(sample_data)
        )
        
        # Create PitchResponse object
        response = PitchResponse(
            Pitch=pitch_text,
            Initial_Offer=initial_offer
        )
        
        # Create DSPy Example
        example = dspy.Example(
            product_data=product_data,
            response=response
        ).with_inputs("product_data")
        
        return example
        
    except Exception as e:
        print(f"Error transforming sample: {e}")
        return None


def create_training_set(data_path: str = None) -> List[dspy.Example]:
    """Generate complete list of training examples from PO_samples.json."""
    samples = load_po_samples(data_path)
    training_examples = []
    
    for sample_name, sample_data in samples.items():
        example = transform_to_dspy_example(sample_data)
        if example is not None:
            training_examples.append(example)
            print(f"✓ Created training example for: {sample_name}")
        else:
            print(f"✗ Failed to create training example for: {sample_name}")
    
    print(f"\nTotal training examples created: {len(training_examples)}")
    return training_examples


def validate_training_examples(examples: List[dspy.Example]) -> bool:
    """Validate that all training examples have correct structure."""
    for i, example in enumerate(examples):
        try:
            # Check that example has required fields
            assert hasattr(example, 'product_data'), f"Example {i} missing product_data"
            assert hasattr(example, 'response'), f"Example {i} missing response"
            
            # Check response structure
            response = example.response
            assert hasattr(response, 'Pitch'), f"Example {i} response missing Pitch"
            assert hasattr(response, 'Initial_Offer'), f"Example {i} response missing Initial_Offer"
            
            # Check Initial_Offer structure
            offer = response.Initial_Offer
            assert hasattr(offer, 'Valuation'), f"Example {i} Initial_Offer missing Valuation"
            assert hasattr(offer, 'Equity_Offered'), f"Example {i} Initial_Offer missing Equity_Offered"
            assert hasattr(offer, 'Funding_Amount'), f"Example {i} Initial_Offer missing Funding_Amount"
            assert hasattr(offer, 'Key_Terms'), f"Example {i} Initial_Offer missing Key_Terms"
            
        except AssertionError as e:
            print(f"Validation error: {e}")
            return False
    
    print("✓ All training examples validated successfully")
    return True


if __name__ == "__main__":
    # Test the training data loader
    examples = create_training_set()
    if validate_training_examples(examples):
        print(f"\nSuccessfully created {len(examples)} training examples")
        
        # Show first example structure
        if examples:
            print("\nFirst example structure:")
            print(f"Product data keys: {list(examples[0].product_data.keys())}")
            print(f"Response type: {type(examples[0].response)}")
            print(f"Initial offer type: {type(examples[0].response.Initial_Offer)}")
    else:
        print("Training examples validation failed")
