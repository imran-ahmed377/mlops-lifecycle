from __future__ import annotations  # Allow postponed evaluation of type annotations.

import json  # Save retraining reports as JSON artifacts.
from datetime import datetime, timezone  # Timestamp reports in UTC.
from pathlib import Path  # Handle filesystem paths safely.
from typing import Dict, List, Tuple  # Type hints for dataframes, metrics, and helper returns.

import joblib  # Load and save trained model artifacts.
import mlflow  # Track retraining runs, params, metrics, and artifacts.
import mlflow.sklearn  # MLflow helpers for logging sklearn models.
import pandas as pd  # Load and combine training/live CSV datasets.
from sklearn.linear_model import LogisticRegression  # Classifier used by this project.
from sklearn.metrics import accuracy_score, f1_score  # Validation metrics used by the promotion gate.
from sklearn.pipeline import Pipeline  # Wrap preprocessing and model into one reusable object.
from sklearn.preprocessing import StandardScaler  # Scale features before logistic regression training.

from src.common import ensure_parent_dir, load_params, resolve_mlflow_tracking_uri  # Shared config loading and directory helpers.

TARGET_COLUMN = "target"  # Ground-truth label column expected by the training pipeline.
PREDICTED_COLUMN = "predicted_class"  # Column logged by the inference API for pseudo-labeled live data.


def _build_model(C: float, max_iter: int, random_state: int) -> Pipeline:  # Create the retraining pipeline from config values.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # Normalize feature ranges before fitting the classifier.
            (
                "classifier",
                LogisticRegression(
                    C=C,  # Inverse regularization strength.
                    max_iter=max_iter,  # Maximum number of optimization iterations.
                    random_state=random_state,  # Seed for reproducible training behavior.
                ),
            ),
        ]
    )


def _score_model(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:  # Score a model on the validation split.
    predictions = model.predict(X_valid)  # Generate validation predictions.
    return {
        "valid_accuracy": float(accuracy_score(y_valid, predictions)),  # Fraction of correct predictions.
        "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),  # Class-balanced F1 score.
    }


def _load_base_datasets(train_csv: str, valid_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:  # Load the prepared train and validation splits.
    return pd.read_csv(train_csv), pd.read_csv(valid_csv)  # Return both datasets as pandas DataFrames.


def _load_labeled_live_rows(live_csv: str, feature_columns: List[str]) -> pd.DataFrame:  # Convert logged live predictions into retraining rows when possible.
    live_path = Path(live_csv)  # Wrap live CSV path for existence checks.
    if not live_path.exists():  # No live data has been collected yet.
        return pd.DataFrame(columns=feature_columns + [TARGET_COLUMN])  # Return an empty schema-compatible DataFrame.

    live_df = pd.read_csv(live_path)  # Load logged live requests.
    required_columns = set(feature_columns + [PREDICTED_COLUMN])  # Required columns for turning predictions into labels.
    if not required_columns.issubset(set(live_df.columns)):  # Bail out if the CSV does not contain the needed columns.
        return pd.DataFrame(columns=feature_columns + [TARGET_COLUMN])  # Return empty data instead of failing.

    labeled_live = live_df[feature_columns + [PREDICTED_COLUMN]].copy()  # Keep only features and predicted label column.
    labeled_live = labeled_live.rename(columns={PREDICTED_COLUMN: TARGET_COLUMN})  # Rename predicted label to expected training target name.
    labeled_live[TARGET_COLUMN] = labeled_live[TARGET_COLUMN].astype(int)  # Ensure labels are integers for sklearn.
    return labeled_live  # Return pseudo-labeled live rows ready for concatenation.


def _decide_promotion(
    current_model: Pipeline | None,
    current_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
) -> Tuple[bool, str]:  # Compare candidate metrics against the current model and decide promotion.
    if current_model is None:  # If no deployed model exists, accept the candidate by default.
        return True, "no_current_model"

    if (
        candidate_metrics["valid_accuracy"] >= current_metrics["valid_accuracy"]  # Candidate must not regress on accuracy.
        and candidate_metrics["valid_f1_macro"] >= current_metrics["valid_f1_macro"]  # Candidate must not regress on macro F1.
    ):
        return True, "candidate_meets_gate"  # Candidate satisfies both promotion checks.

    return False, "candidate_below_gate"  # Candidate is worse on at least one guarded metric.


def run_retrain() -> None:  # Execute retraining, gate evaluation, model promotion, and reporting.
    params = load_params()  # Load project configuration from params.yaml.
    paths = params["paths"]  # Resolve configured artifact locations.
    training_cfg = params["training"]["logistic"]  # Read logistic regression hyperparameters.
    random_state = int(params["project"]["random_state"])  # Reproducibility seed shared across the project.
    project_root = Path(__file__).resolve().parents[1]  # Resolve repository root regardless of current working directory.

    retrain_cfg = params.get("retrain", {})  # Optional retraining-specific settings.
    min_new_rows = int(retrain_cfg.get("min_new_rows", 5))  # Minimum live rows required before mixing live data into retraining.
    report_path = paths.get("retrain_report", "metrics/retrain_report.json")  # Destination for retraining decision report.

    train_df, valid_df = _load_base_datasets(paths["train_csv"], paths["valid_csv"])  # Load the base train/validation datasets.
    feature_columns = [c for c in train_df.columns if c != TARGET_COLUMN]  # Keep all non-label columns as features.

    labeled_live_df = _load_labeled_live_rows(paths["live_data"], feature_columns)  # Try to recover pseudo-labeled live rows.
    live_rows_count = int(len(labeled_live_df))  # Count usable live rows.

    if live_rows_count >= min_new_rows:  # Use live rows only when there is enough data to be meaningful.
        retrain_df = pd.concat([train_df, labeled_live_df], ignore_index=True)  # Blend original training data with live rows.
        data_source = "train_plus_live"  # Record that retraining used both sources.
    else:
        retrain_df = train_df.copy()  # Fall back to the original training data only.
        data_source = "train_only"  # Record that live data was not used.

    X_train = retrain_df[feature_columns]  # Features used to fit the candidate model.
    y_train = retrain_df[TARGET_COLUMN].astype(int)  # Labels used to fit the candidate model.

    X_valid = valid_df[feature_columns]  # Validation features used by the promotion gate.
    y_valid = valid_df[TARGET_COLUMN].astype(int)  # Validation labels used by the promotion gate.

    candidate_model = _build_model(
        C=float(training_cfg["C"]),  # Regularization strength from config.
        max_iter=int(training_cfg["max_iter"]),  # Iteration budget from config.
        random_state=random_state,  # Shared seed for deterministic retraining.
    )
    candidate_model.fit(X_train, y_train)  # Fit the candidate model on the retraining dataset.
    candidate_metrics = _score_model(candidate_model, X_valid, y_valid)  # Measure candidate performance on validation data.

    current_metrics = {
        "valid_accuracy": -1.0,  # Default sentinel when no current model exists.
        "valid_f1_macro": -1.0,  # Default sentinel when no current model exists.
    }
    current_model = None  # Track currently deployed model if present.
    model_path = Path(paths["model_file"])  # Current deployed model location.
    if model_path.exists():  # Only score current model when it actually exists on disk.
        current_model = joblib.load(model_path)  # Load current deployed model.
        current_metrics = _score_model(current_model, X_valid, y_valid)  # Measure current model on the same validation split.

    promoted, promotion_reason = _decide_promotion(
        current_model=current_model,  # Existing deployed model, if any.
        current_metrics=current_metrics,  # Current model validation metrics.
        candidate_metrics=candidate_metrics,  # Candidate model validation metrics.
    )

    if promoted:  # Replace the deployed model only when the gate passes.
        ensure_parent_dir(paths["model_file"])  # Create model output folder if needed.
        joblib.dump(candidate_model, paths["model_file"])  # Persist candidate as the new deployed model.

    tracking_uri = resolve_mlflow_tracking_uri(params, project_root=project_root)  # Lock runs to one deterministic tracking backend.
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = params.get("mlflow", {}).get("experiment_name", params["project"]["name"])  # Use configured MLflow experiment or project name.
    mlflow.set_experiment(experiment_name)  # Ensure retraining run is logged under the correct experiment.

    with mlflow.start_run(run_name="retrain-logreg"):  # Open an MLflow run for retraining metadata.
        mlflow.log_params(
            {
                "run_type": "retrain",  # Distinguish retraining runs from initial training runs.
                "model": "LogisticRegression",  # Model family used in this run.
                "random_state": random_state,  # Seed used during retraining.
                "C": float(training_cfg["C"]),  # Regularization parameter.
                "max_iter": int(training_cfg["max_iter"]),  # Iteration budget for solver.
                "data_source": data_source,  # Whether live rows were included.
                "base_train_rows": int(len(train_df)),  # Size of original training dataset.
                "live_rows_available": live_rows_count,  # Number of usable live rows discovered.
                "live_rows_min_required": min_new_rows,  # Threshold required before using live rows.
                "retrain_rows_used": int(len(retrain_df)),  # Final row count used to train candidate.
            }
        )
        mlflow.log_metrics(
            {
                "candidate_valid_accuracy": candidate_metrics["valid_accuracy"],  # Candidate validation accuracy.
                "candidate_valid_f1_macro": candidate_metrics["valid_f1_macro"],  # Candidate validation macro F1.
                "current_valid_accuracy": current_metrics["valid_accuracy"],  # Current model validation accuracy.
                "current_valid_f1_macro": current_metrics["valid_f1_macro"],  # Current model validation macro F1.
                "promoted": float(1 if promoted else 0),  # Numeric flag for easier charting in MLflow.
            }
        )
        mlflow.sklearn.log_model(sk_model=candidate_model, artifact_path="candidate_model")  # Always log the candidate model artifact.
        if promoted:  # Log the promoted model file only when deployment actually changed.
            mlflow.log_artifact(paths["model_file"], artifact_path="promoted_model")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),  # Timestamp for the retraining decision.
        "promoted": promoted,  # Whether the candidate replaced the current model.
        "promotion_reason": promotion_reason,  # Short reason explaining the gate outcome.
        "data_source": data_source,  # Dataset source used for retraining.
        "base_train_rows": int(len(train_df)),  # Size of original train split.
        "live_rows_available": live_rows_count,  # Number of pseudo-labeled live rows found.
        "live_rows_min_required": min_new_rows,  # Threshold required to include live rows.
        "retrain_rows_used": int(len(retrain_df)),  # Final number of rows used for candidate training.
        "current_metrics": current_metrics,  # Validation metrics for current deployed model.
        "candidate_metrics": candidate_metrics,  # Validation metrics for candidate model.
        "feature_columns": feature_columns,  # Ordered features used during retraining.
    }

    ensure_parent_dir(report_path)  # Create report output directory if needed.
    with open(report_path, "w", encoding="utf-8") as f:  # Persist retraining decision report.
        json.dump(report, f, indent=2)  # Write pretty-printed JSON for easier inspection.

    print("Retraining complete.")  # Console summary for manual runs.
    print(f"Promotion decision: {promoted} ({promotion_reason})")  # Show the gate result and reason.
    print(f"Report saved to {report_path}")  # Show where the report was written.


if __name__ == "__main__":  # Run retraining when the file is executed directly.
    run_retrain()  # Script entry point.
