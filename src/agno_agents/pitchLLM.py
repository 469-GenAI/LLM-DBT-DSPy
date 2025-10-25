# pitchLLM.py - DSPy implementation for pitch generation
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import dspy
from pydantic import BaseModel, Field
from utils import ValuationTools, InitialOffer, PitchResponse, extract_metrics
from tqdm import tqdm
import pandas as pd
import mlflow

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABRICKS_PATH = os.getenv("DATABRICKS_PATH")

# Configure DSPy language model
lm = dspy.LM("groq/llama-3.3-70b-versatile", model_type="chat", api_key=GROQ_API_KEY)
dspy.configure(lm=lm)


# mlflow.set_experiment(DATABRICKS_PATH + "pitchLLM")
# mlflow.dspy.autolog()


# ---------- 1) DSPy SIGNATURES ----------
class FinancialsSig(dspy.Signature):
    """Infer simple financials from product facts."""
    facts: str = dspy.InputField(desc="JSON array of fact strings about the startup.")
    is_profitable: bool = dspy.OutputField(desc="True if profitable this year, else False.")
    revenue: float = dspy.OutputField(desc="Annual revenue in USD.")
    profit: float = dspy.OutputField(desc="Annual profit in USD (can be negative).")

class PitchSig(dspy.Signature):
    """Generate a persuasive Shark Tank pitch and investment offer."""
    financial_summary: str = dspy.InputField()
    product_json: str = dspy.InputField(desc="Full JSON with product description and facts.")
    response: PitchResponse = dspy.OutputField(
        desc="Structured pitch and initial offer; must be valid JSON per schema."
    )

# ---------- 2) DSPy MODULES ----------
class Financials(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(FinancialsSig)

    def forward(self, facts_json: str):
        pred = self.cot(facts=facts_json)
        return pred

class DraftPitch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(PitchSig)

    def forward(self, financial_summary: str, product_json: str):
        pred = self.cot(financial_summary=financial_summary, product_json=product_json)
        return pred

# ---------- 3) MAIN PROGRAM ----------
class PitchProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.financials = Financials()
        self.draft = DraftPitch()
        self.valuation_tools = ValuationTools()

    def forward(self, product_data: dict):
        try:
            # 1) Financial inference
            facts_json = json.dumps(product_data["facts"])
            fin = self.financials(facts_json)

            # 2) Extract financial metrics
            is_profitable = bool(fin.is_profitable)
            revenue = float(fin.revenue)
            profit = float(fin.profit)

            # 3) Calculate valuation using appropriate method
            if is_profitable and profit > 0:
                val1 = self.valuation_tools.profit_multiple_valuation(profit)
                val2 = self.valuation_tools.discounted_cash_flow_valuation(profit)
                valuation = (val1 + val2) / 2
            else:
                valuation = self.valuation_tools.revenue_multiple_valuation(revenue)

            funding_amount = 0.10 * valuation  # 10% ask

            financial_summary = (
                f"The startup has the following financial metrics:\n"
                f"- Revenue: ${revenue:,.0f}\n"
                f"- Profit: ${profit:,.0f}\n"
                f"- Is Profitable: {'Yes' if is_profitable else 'No'}\n\n"
                f"Estimated valuation: ${valuation:,.0f}.\n"
                f"Founders are asking for ~${funding_amount:,.0f}."
            )

            # 4) Generate pitch
            pred = self.draft(financial_summary=financial_summary,
                          product_json=json.dumps(product_data, indent=2))

                        # Capture the prompt from the LM's history immediately after the call
            prompt_str = "No prompt captured"
            try:
                if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                    # Access the last history entry
                    if hasattr(dspy.settings.lm, 'history') and dspy.settings.lm.history:
                        history_entry = dspy.settings.lm.history[-1]
                        if isinstance(history_entry, dict):
                            prompt_str = history_entry.get('messages', history_entry)
                        else:
                            prompt_str = str(history_entry)
            except Exception as e:
                print(f"Error capturing prompt: {e}")

            return {
                "is_profitable": is_profitable,
                "valuation": valuation,
                "funding_amount": funding_amount,
                "response": pred.response,
                "financial_summary": financial_summary,
                "prompt": prompt_str,
                "metrics": extract_metrics(pred) if hasattr(pred, 'metrics') else None
            }
        except Exception as e:
            print(f"Error in PitchProgram: {e}")
            return None

# ---------- 4) OPTIMIZATION ----------
def pitch_metric(example, pred, trace=None):
    """Metric for evaluating pitch quality"""
    try:
        resp = pred["response"] if isinstance(pred, dict) else pred.response
        offer = resp.Initial_Offer
        ok = bool(resp.Pitch and offer.Valuation and offer.Equity_Offered and offer.Funding_Amount)
        return ok
    except Exception:
        return False

def maybe_compile(program, train_examples=None, use_mipro=False):
    """Compile program with optimization if training examples provided"""
    if not train_examples:
        return program
    if use_mipro:
        opt = dspy.MIPROv2(metric=pitch_metric)
    else:
        opt = dspy.BootstrapFewShot(metric=pitch_metric)
    return opt.compile(program, trainset=train_examples)

# ---------- 5) MAIN EXECUTION ----------
if __name__ == "__main__":
    # Fix file path
    facts_path = Path("./src/agno_agents/data/outputs/facts_and_productdescriptions.json")
    
    if not facts_path.exists():
        print(f"Error: File not found at {facts_path}")
        exit(1)
    
    facts_dict = json.loads(facts_path.read_text(encoding="utf-8"))
    scenarios = list(facts_dict.keys())[:3]
    program = PitchProgram()

    # Optional compilation
    train_examples = None
    program = maybe_compile(program, train_examples, use_mipro=False)

    rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    framework = "dspy"
    layer = "N/A"
    model_name = lm.model

    for scenario in tqdm(scenarios, desc="Processing Pitches", unit="pitch"):
        product_data = facts_dict[scenario]
        out = program(product_data)
        
        if out is None:
            print(f"Failed to process scenario: {scenario}")
            continue
            
        resp = out["response"]

        rows.append({
            "scenario": scenario,
            "framework": framework,
            "layer": layer,
            "model_name": model_name,
            "model_identifier": f"{model_name}-{framework}_{layer}",
            "timestamp": timestamp,
            "valuation": f"${out['valuation']:,.0f}",
            "funding_amount": f"${out['funding_amount']:,.0f}",
            "is_profitable": out["is_profitable"],
            "response": resp.model_dump(),
            "prompt": out.get("prompt", "N/A"),
            "metrics": out.get("metrics", {})
        })

    df = pd.DataFrame(rows)
    out_csv = f"dspy_results_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")