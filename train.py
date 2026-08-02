import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to CSV or Parquet dataset",
)

parser.add_argument(
    "--n_estimators",
    type=int,
    default=100,
)

parser.add_argument(
    "--max_depth",
    type=int,
    default=10,
)

args = parser.parse_args()


# Read dataset
if args.data.endswith(".parquet"):
    df = pd.read_parquet(args.data)
else:
    df = pd.read_csv(args.data)


# Remove rows where target cannot be computed
df = df.dropna(subset=["target"])

# Features
X = df[
    [
        "rolling_avg_10",
        "volume_sum_10",
    ]
]

# Target
y = df["target"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# MLflow
mlflow.set_experiment("Stock_Movement_Predictor")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("dataset", os.path.basename(args.data))
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Save model
    model_name = f"model_{os.path.splitext(os.path.basename(args.data))[0]}.joblib"

    joblib.dump(model, model_name)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
    )

    print("=" * 50)
    print("Training Complete")
    print("=" * 50)
    print("Dataset :", args.data)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Saved Model:", model_name)
