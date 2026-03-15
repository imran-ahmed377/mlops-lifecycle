from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ensure_parent_dir, load_params

TARGET_COLUMN = "target"
PREDICTED_COLUMN = "predicted_class"


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


def _score_model(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    predictions = model.predict(X_valid)
    return {
        "valid_accuracy": float(accuracy_score(y_valid, predictions)),
        "valid_f1_macro": float(f1_score(y_valid, predictions, average="macro")),
    }


def _load_base_datasets(train_csv: str, valid_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_csv), pd.read_csv(valid_csv)


def _load_labeled_live_rows(live_csv: str, feature_columns: List[str]) -> pd.DataFrame:
    live_path = Path(live_csv)
    if not live_path.exists():
        return pd.DataFrame(columns=feature_columns + [TARGET_COLUMN])

    live_df = pd.read_csv(live_path)
    required_columns = set(feature_columns + [PREDICTED_COLUMN])
    if not required_columns.issubset(set(live_df.columns)):
        return pd.DataFrame(columns=feature_columns + [TARGET_COLUMN])

    labeled_live = live_df[feature_columns + [PREDICTED_COLUMN]].copy()
    labeled_live = labeled_live.rename(columns={PREDICTED_COLUMN: TARGET_COLUMN})
    labeled_live[TARGET_COLUMN] = labeled_live[TARGET_COLUMN].astype(int)
    return labeled_live


def _decide_promotion(
    current_model: Pipeline | None,
    current_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
) -> Tuple[bool, str]:
    if current_model is None:
        return True, "no_current_model"

    if (
        candidate_metrics["valid_accuracy"] >= current_metrics["valid_accuracy"]
        and candidate_metrics["valid_f1_macro"] >= current_metrics["valid_f1_macro"]
    ):
        return True, "candidate_meets_gate"

    return False, "candidate_below_gate"


def run_retrain() -> None:
    params = load_params()
    paths = params["paths"]
    training_cfg = params["training"]["logistic"]
    random_state = int(params["project"]["random_state"])

    retrain_cfg = params.get("retrain", {})
    min_new_rows = int(retrain_cfg.get("min_new_rows", 5))
    report_path = paths.get("retrain_report", "metrics/retrain_report.json")

    train_df, valid_df = _load_base_datasets(paths["train_csv"], paths["valid_csv"])
    feature_columns = [c for c in train_df.columns if c != TARGET_COLUMN]

    labeled_live_df = _load_labeled_live_rows(paths["live_data"], feature_columns)
    live_rows_count = int(len(labeled_live_df))

    if live_rows_count >= min_new_rows:
        retrain_df = pd.concat([train_df, labeled_live_df], ignore_index=True)
        data_source = "train_plus_live"
    else:
        retrain_df = train_df.copy()
        data_source = "train_only"

    X_train = retrain_df[feature_columns]
    y_train = retrain_df[TARGET_COLUMN].astype(int)

    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN].astype(int)

    candidate_model = _build_model(
        C=float(training_cfg["C"]),
        max_iter=int(training_cfg["max_iter"]),
        random_state=random_state,
    )
    candidate_model.fit(X_train, y_train)
    candidate_metrics = _score_model(candidate_model, X_valid, y_valid)

    current_metrics = {
        "valid_accuracy": -1.0,
        "valid_f1_macro": -1.0,
    }
    current_model = None
    model_path = Path(paths["model_file"])
    if model_path.exists():
        current_model = joblib.load(model_path)
        current_metrics = _score_model(current_model, X_valid, y_valid)

    promoted, promotion_reason = _decide_promotion(
        current_model=current_model,
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
    )

    if promoted:
        ensure_parent_dir(paths["model_file"])
        joblib.dump(candidate_model, paths["model_file"])

    experiment_name = params.get("mlflow", {}).get("experiment_name", params["project"]["name"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="retrain-logreg"):
        mlflow.log_params(
            {
                "run_type": "retrain",
                "model": "LogisticRegression",
                "random_state": random_state,
                "C": float(training_cfg["C"]),
                "max_iter": int(training_cfg["max_iter"]),
                "data_source": data_source,
                "base_train_rows": int(len(train_df)),
                "live_rows_available": live_rows_count,
                "live_rows_min_required": min_new_rows,
                "retrain_rows_used": int(len(retrain_df)),
            }
        )
        mlflow.log_metrics(
            {
                "candidate_valid_accuracy": candidate_metrics["valid_accuracy"],
                "candidate_valid_f1_macro": candidate_metrics["valid_f1_macro"],
                "current_valid_accuracy": current_metrics["valid_accuracy"],
                "current_valid_f1_macro": current_metrics["valid_f1_macro"],
                "promoted": float(1 if promoted else 0),
            }
        )
        mlflow.sklearn.log_model(sk_model=candidate_model, artifact_path="candidate_model")
        if promoted:
            mlflow.log_artifact(paths["model_file"], artifact_path="promoted_model")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "data_source": data_source,
        "base_train_rows": int(len(train_df)),
        "live_rows_available": live_rows_count,
        "live_rows_min_required": min_new_rows,
        "retrain_rows_used": int(len(retrain_df)),
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "feature_columns": feature_columns,
    }

    ensure_parent_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Retraining complete.")
    print(f"Promotion decision: {promoted} ({promotion_reason})")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    run_retrain()
