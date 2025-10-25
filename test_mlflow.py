import mlflow
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

if experiment_id is None:
    print("No experiment ID found in .env")
else:
    experiment = mlflow.get_experiment(experiment_id)
    if experiment is None:
        print(f"Experiment with ID {experiment_id} not found.")
    else:
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Experiment Name: {experiment.name}")