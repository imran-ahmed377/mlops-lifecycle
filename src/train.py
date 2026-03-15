from __future__ import annotations # for Python 3.7+ to allow postponed evaluation of type annotations

from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # for feature scaling

from src.common import ensure_parent_dir, load_params

TARGET_COLUMN = "target"

"""
This module provides functionality for training a machine learning model 
using the Iris dataset.
"""

def _load_datasets(train_csv: str, valid_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    return train_df, valid_df


def _build_model(C: float, max_iter: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )

# The run_train function orchestrates the training process by loading 
# the training and validation datasets, building a logistic regression model, 
# and logging the training parameters and metrics using MLflow. It also saves 
# the trained model to a specified file path and logs it as an artifact in MLflow.

def run_train() -> None:
    params = load_params()
    paths = params["paths"]
    training_cfg = params["training"]["logistic"]
    random_state = int(params["project"]["random_state"])

    train_df, valid_df = _load_datasets(paths["train_csv"], paths["valid_csv"])
    feature_columns: List[str] = [c for c in train_df.columns if c != TARGET_COLUMN] # Extract feature columns by excluding the target column

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN].astype(int)

    model = _build_model(
        C=float(training_cfg["C"]),
        max_iter=int(training_cfg["max_iter"]),
        random_state=random_state,
    )

    # Set the MLflow experiment name based on the configuration, 
    # defaulting to the project name if not specified
    experiment_name = params.get("mlflow", {}).get("experiment_name", params["project"]["name"])
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run with a descriptive name and log the training parameters,
    # including model type, random state, test size, regularization strength (C), and 
    # maximum iterations. After fitting the model, it logs the validation 
    # metrics to MLflow and saves the trained model artifact.
    with mlflow.start_run(run_name="train-logreg"):
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "random_state": random_state,
                "test_size": float(params["training"]["test_size"]),
                "C": float(training_cfg["C"]),
                "max_iter": int(training_cfg["max_iter"]),
            }
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)
        metrics: Dict[str, float] = {
            "valid_accuracy": float(accuracy_score(y_valid, predictions)),
            "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),
        }
        mlflow.log_metrics(metrics)

        ensure_parent_dir(paths["model_file"])
        joblib.dump(model, paths["model_file"])

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        mlflow.log_artifact(paths["model_file"], artifact_path="exported_model")

    print("Training complete.")
    print(f"Saved model to {paths['model_file']}")


if __name__ == "__main__":
    run_train()
