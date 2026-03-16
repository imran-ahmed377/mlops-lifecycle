from __future__ import annotations  # Allow postponed evaluation of type annotations.

from pathlib import Path  # Resolve repository-root-relative artifact paths.
from typing import Dict, List, Tuple  # Type hints for readability and editor support.

import joblib  # Serialize and save trained models.
import mlflow  # Track experiments, parameters, and metrics.
import mlflow.sklearn  # MLflow helpers for scikit-learn model logging.
import pandas as pd  # Load tabular datasets from CSV files.
from sklearn.linear_model import LogisticRegression  # Classifier used in this project.
from sklearn.metrics import accuracy_score, f1_score  # Validation metrics.
from sklearn.pipeline import Pipeline  # Chain preprocessing and model in one object.
from sklearn.preprocessing import StandardScaler  # Normalize feature scales before training.

from src.common import ensure_parent_dir, load_params, resolve_mlflow_tracking_uri  # Shared project utilities.

TARGET_COLUMN = "target"  # Name of the label column in train/validation CSV files.

"""Train and persist the Iris classifier, and log run metadata to MLflow."""


def _load_datasets(train_csv: str, valid_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:  # Load prepared train and validation splits.
    train_df = pd.read_csv(train_csv)  # Read training split from disk.
    valid_df = pd.read_csv(valid_csv)  # Read validation split from disk.
    return train_df, valid_df  # Return both datasets for downstream use.


def _build_model(C: float, max_iter: int, random_state: int) -> Pipeline:  # Construct the training pipeline from config values.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # Standardize features to improve optimizer stability.
            (
                "classifier",
                LogisticRegression(
                    C=C,  # Inverse regularization strength.
                    max_iter=max_iter,  # Maximum optimization iterations.
                    random_state=random_state,  # Reproducibility seed.
                ),
            ),
        ]
    )


def run_train() -> None:  # Execute end-to-end model training and artifact logging.
    params = load_params()  # Read project configuration from params.yaml.
    paths = params["paths"]  # File paths for input and output artifacts.
    training_cfg = params["training"]["logistic"]  # Logistic regression hyperparameters.
    random_state = int(params["project"]["random_state"])  # Global reproducibility seed.
    project_root = Path(__file__).resolve().parents[1]  # Resolve repository root regardless of current working directory.

    train_df, valid_df = _load_datasets(paths["train_csv"], paths["valid_csv"])  # Load prepared datasets.
    feature_columns: List[str] = [c for c in train_df.columns if c != TARGET_COLUMN]  # Keep all columns except label.

    X_train = train_df[feature_columns]  # Training features.
    y_train = train_df[TARGET_COLUMN].astype(int)  # Training labels as integers.

    X_valid = valid_df[feature_columns]  # Validation features.
    y_valid = valid_df[TARGET_COLUMN].astype(int)  # Validation labels as integers.

    model = _build_model(
        C=float(training_cfg["C"]),  # Regularization strength from config.
        max_iter=int(training_cfg["max_iter"]),  # Max iterations from config.
        random_state=random_state,  # Use project-wide random state.
    )

    tracking_uri = resolve_mlflow_tracking_uri(params, project_root=project_root)  # Lock runs to one deterministic tracking backend.
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = params.get("mlflow", {}).get("experiment_name", params["project"]["name"])  # Use configured MLflow experiment name when available.
    mlflow.set_experiment(experiment_name)  # Ensure run is logged under this experiment.

    with mlflow.start_run(run_name="train-logreg"):  # Open a tracked MLflow run for this training job.
        mlflow.log_params(
            {
                "model": "LogisticRegression",  # Model family identifier.
                "random_state": random_state,  # Seed used during training.
                "test_size": float(params["training"]["test_size"]),  # Validation split fraction.
                "C": float(training_cfg["C"]),  # Regularization parameter.
                "max_iter": int(training_cfg["max_iter"]),  # Iteration budget for solver.
            }
        )

        model.fit(X_train, y_train)  # Train pipeline on training split.

        predictions = model.predict(X_valid)  # Generate predictions on validation split.
        metrics: Dict[str, float] = {
            "valid_accuracy": float(accuracy_score(y_valid, predictions)),  # Fraction of correct predictions.
            "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),  # Class-balanced F1 score.
        }
        mlflow.log_metrics(metrics)  # Persist evaluation metrics in MLflow.

        ensure_parent_dir(paths["model_file"])  # Create model output directory if missing.
        joblib.dump(model, paths["model_file"])  # Save trained model for serving.

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")  # Store model artifact in MLflow run.
        mlflow.log_artifact(paths["model_file"], artifact_path="exported_model")  # Store exported model file in a separate artifact folder.

    print("Training complete.")  # Console signal that training succeeded.
    print(f"Saved model to {paths['model_file']}")  # Show where the model was written.


if __name__ == "__main__":  # Run training when file is executed directly.
    run_train()  # Script entry point.
