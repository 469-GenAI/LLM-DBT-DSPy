# DSPy Mixture-of-Agents Pipeline

## Manual Verification

- Activate environment with `pip install -r requirements.txt` and ensure `dspy` plus Groq SDKs are available.
- Set `GROQ_API_KEY` in the shell before execution.
- Run `python -m src.MoA.run_moa_dspy --feedback-loops 2 --aggregator-training path/to/training.jsonl` to generate a pitch with self-improvement.
- Use `--scenario-key` to target a specific fact entry from `src/MoA/all_processed_facts.txt`.
- Pass `--save-output tmp/pitch.json` to persist results for future aggregator optimization runs.

## Rate Limiting Guidance

- Default delay is `2.5` seconds between runs; increase via `--rate-limit` when encountering Groq throttle warnings.
- Each loop issues approximately `num_agents + optional feedback` calls; multiplying by `feedback_loops` clarifies the total cost.
- Avoid running more than two instances of the script concurrently without increasing the delay.

## Follow-up Work

- **Unit tests**: Add fixtures under `tests/moa/` to validate planner parsing and aggregator serialization helpers.
- **MLflow integration**: Mirror the tracking pattern used in `pitchLLM_structured.py`, wiring `mlflow.dspy.autolog` behind a CLI flag.
- **Dataset curation**: Build and curate JSONL logs of agent outputs to strengthen aggregator compilation.
- **Evaluator variants**: Experiment with alternative feedback models exposed in `REFERENCE_MODELS` for robustness checks.
