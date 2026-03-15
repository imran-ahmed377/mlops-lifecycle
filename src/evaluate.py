from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.common import ensure_parent_dir, load_params

TARGET_COLUMN = "target"


def run_evaluate() -> None:
    params = load_params()
    paths = params["paths"]

    valid_df = pd.read_csv(paths["valid_csv"])
    model = joblib.load(paths["model_file"])
    
    # Extract feature columns by excluding the target column from the validation dataset
    feature_columns = [column for column in valid_df.columns if column != TARGET_COLUMN]
    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN].astype(int)

    predictions = model.predict(X_valid)
    eval_report = {
        "feature_columns": feature_columns,
        "valid_rows": int(len(valid_df)),
        "valid_accuracy": float(accuracy_score(y_valid, predictions)),
        "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),
    }

    ensure_parent_dir(paths["eval_metrics"])
    with open(paths["eval_metrics"], "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1, 2])
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1", "actual_2"],
        columns=["pred_0", "pred_1", "pred_2"],
    )
    ensure_parent_dir(paths["confusion_matrix"])
    matrix_df.to_csv(paths["confusion_matrix"])

    print("Evaluation complete.")
    print(f"Saved metrics to {paths['eval_metrics']}")
    print(f"Saved confusion matrix to {paths['confusion_matrix']}")


if __name__ == "__main__":
    run_evaluate()
