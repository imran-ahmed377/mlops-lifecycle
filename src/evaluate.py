from __future__ import annotations

import json  # Save evaluation summary as JSON.

import joblib  # Load the trained model artifact.
import pandas as pd  # Read validation data and build tabular outputs.
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # Compute validation metrics.

from src.common import ensure_parent_dir, load_params  # Shared config/path helpers.

TARGET_COLUMN = "target"  # Name of the label column in validation data.


def run_evaluate() -> None:
    params = load_params()  # Load project configuration from params.yaml.
    paths = params["paths"]  # Resolve configured artifact paths.

    valid_df = pd.read_csv(paths["valid_csv"])  # Load held-out validation dataset.
    model = joblib.load(paths["model_file"])  # Load trained model produced by training stage.

    # Build feature matrix and target vector expected by sklearn metrics.
    feature_columns = [column for column in valid_df.columns if column != TARGET_COLUMN]  # Keep all non-target columns.
    X_valid = valid_df[feature_columns]  # Validation features.
    y_valid = valid_df[TARGET_COLUMN].astype(int)  # Ground-truth validation labels.

    predictions = model.predict(X_valid)  # Generate model predictions on validation split.
    eval_report = {  # Collect summary metrics for monitoring and gating.
        "feature_columns": feature_columns,  # Feature order used during evaluation.
        "valid_rows": int(len(valid_df)),  # Number of evaluated rows.
        "valid_accuracy": float(accuracy_score(y_valid, predictions)),  # Overall classification accuracy.
        "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),  # Class-balanced F1 score.
    }

    ensure_parent_dir(paths["eval_metrics"])  # Ensure output folder exists for metrics JSON.
    with open(paths["eval_metrics"], "w", encoding="utf-8") as f:  # Write evaluation summary to disk.
        json.dump(eval_report, f, indent=2)  # Pretty-print JSON for readability.

    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1, 2])  # Compute confusion matrix with fixed class order.
    matrix_df = pd.DataFrame(
        matrix,  # Matrix values from sklearn.
        index=["actual_0", "actual_1", "actual_2"],  # Row labels represent true classes.
        columns=["pred_0", "pred_1", "pred_2"],  # Column labels represent predicted classes.
    )
    ensure_parent_dir(paths["confusion_matrix"])  # Ensure output folder exists for confusion matrix CSV.
    matrix_df.to_csv(paths["confusion_matrix"])  # Save confusion matrix artifact.

    print("Evaluation complete.")  # Status message for CLI usage.
    print(f"Saved metrics to {paths['eval_metrics']}")  # Show JSON report location.
    print(f"Saved confusion matrix to {paths['confusion_matrix']}")  # Show matrix artifact location.


if __name__ == "__main__":  # Run evaluation when executed directly.
    run_evaluate()  # Script entry point.
