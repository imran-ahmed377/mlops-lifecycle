from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.common import ensure_parent_dir, load_params

"""
This module provides functionality for preparing data for machine learning models.
It includes functions for loading parameters from a YAML file, 
ensuring that necessary directories exist, and building baseline 
statistics for the training data. The main function, run_prepare, 
orchestrates the data preparation process by loading the Iris dataset, 
splitting it into training and validation sets, saving the datasets to CSV files, 
and calculating baseline statistics for the training data. The module uses 
the scikit-learn library for dataset loading and splitting, and the json library 
for saving baseline statistics in a structured format.

"""

def _build_baseline_stats(train_rows: List[Dict[str, float]], feature_columns: List[str]) -> Dict[str, object]:
    summary: Dict[str, Dict[str, float]] = {}

    for col in feature_columns:
        values = [float(row[col]) for row in train_rows]
        count = len(values)
        mean = sum(values) / count if count else 0.0
        variance = sum((v - mean) ** 2 for v in values) / count if count else 0.0
        std = variance ** 0.5

        summary[col] = {
            "mean": mean,
            "std": std,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(train_rows),
        "feature_columns": feature_columns,
        "summary": summary,
    }


def run_prepare() -> None:
    params = load_params()
    paths = params["paths"]
    training_cfg = params["training"]
    random_state = int(params["project"]["random_state"])

    dataset = load_iris(as_frame=True)
    full_df = dataset.frame.copy()
    feature_columns = list(dataset.feature_names)

    ensure_parent_dir(paths["raw_csv"])
    full_df.to_csv(paths["raw_csv"], index=False)

    train_df, valid_df = train_test_split(
        full_df,
        test_size=float(training_cfg["test_size"]),
        random_state=random_state,
        stratify=full_df["target"],
    )

    ensure_parent_dir(paths["train_csv"])
    train_df.to_csv(paths["train_csv"], index=False)

    ensure_parent_dir(paths["valid_csv"])
    valid_df.to_csv(paths["valid_csv"], index=False)

    # Save the feature columns from the training dataset as a reference 
    # for drift detection
    baseline_reference = train_df[feature_columns]
    ensure_parent_dir(paths["baseline_reference"])
    baseline_reference.to_csv(paths["baseline_reference"], index=False)

    baseline_stats = _build_baseline_stats(
        baseline_reference.to_dict(orient="records"),
        feature_columns,
    )
    ensure_parent_dir(paths["baseline_stats"])
    with open(paths["baseline_stats"], "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    print(f"Prepared data with {len(train_df)} train rows and {len(valid_df)} valid rows.")
    print(f"Feature columns: {feature_columns}")


if __name__ == "__main__":
    run_prepare()
