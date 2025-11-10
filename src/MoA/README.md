MoA DSPy Pipeline
=================

Overview
--------
The DSPy-based Mixture-of-Agents pipeline refactors the legacy `MoA.py` script
into a modular program that:
- Plans specialised agent roles per pitch (`PlanMoATasks` signature)
- Generates agent contributions grounded in structured facts
- Optimises only the final aggregation stage to follow DSPy guidance
- Reuses the shared `AssessPitch` evaluator for consistent scoring

Key Files
---------
- `MoA_structured.py`: CLI entry point for running, compiling, and evaluating the program
- `data_interfaces.py`: Pydantic models and helpers for adapting Shark Tank facts
- `signatures.py`: DSPy signatures covering planning, drafting, and aggregation
- `models/program.py`: `MoAMixtureProgram` implementation and generator factory
- `services.py`: Helpers for generator/evaluator instantiation and optimisation

Running the Pipeline
--------------------
From the repository root:

```bash
python src/MoA/MoA_structured.py \
  --optimization bootstrap \
  --generator-model groq/llama-3.3-70b-versatile \
  --evaluator-model groq/openai/gpt-oss-120b \
  --train-size 20 \
  --test-size 5 \
  --evaluate
```

Flags of interest:
- `--num-agents`: override the default three-agent configuration
- `--no-rate-limit`: disable the Groq-safe 2.5s delay (use cautiously)
- `--save-program`: persist the compiled DSPy program with metadata

Validation & Troubleshooting
----------------------------
1. Ensure `.env` contains valid `GROQ_API_KEY` (and `BEDROCK_API_KEY` if required).
2. Authenticate with HuggingFace to access the `sharktank_pitches` dataset.
3. Enable `--evaluate` to retrieve numeric scores and qualitative feedback from `AssessPitch`.
4. Generated CSVs, metadata, and (optionally) compiled programs land in the working directory; review `agent_contributions` for interpretability.
5. If optimisation fails, rerun with `--optimization none` to confirm baseline behaviour before re-enabling advanced methods.

Testing Checklist
-----------------
- Baseline run: `python src/MoA/MoA_structured.py --test-size 2 --optimization none`
- Optimised run: `python src/MoA/MoA_structured.py --test-size 2 --train-size 10 --optimization bootstrap --evaluate`
- Inspect `agent_contributions` column in the output CSV to confirm multi-agent flow.

