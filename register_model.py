import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "Stock_Movement_Predictor"
REGISTERED_MODEL_NAME = "StockMovementPredictor"

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
)

best_run = runs[0]

print("Best Run ID:", best_run.info.run_id)
print("Best F1:", best_run.data.metrics["f1_score"])

model_uri = f"runs:/{best_run.info.run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print("\nModel Registered Successfully")
print("Name:", REGISTERED_MODEL_NAME)
print("Version:", registered_model.version)
