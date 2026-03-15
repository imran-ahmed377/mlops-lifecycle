from __future__ import annotations  # Postpone annotation evaluation for forward compatibility.

import json  # Write baseline statistics to a JSON file.
from datetime import datetime, timezone  # Add UTC timestamps to generated artifacts.
from typing import Dict, List  # Type hints for dictionaries and lists.

from sklearn.datasets import load_iris  # Load the built-in Iris dataset.
from sklearn.model_selection import train_test_split  # Split data into train and validation sets.

from src.common import ensure_parent_dir, load_params  # Shared helpers for config and directory creation.

"""Prepare Iris data artifacts for training, evaluation, and drift monitoring."""


def _build_baseline_stats(train_rows: List[Dict[str, float]], feature_columns: List[str]) -> Dict[str, object]:  # Build per-feature baseline statistics.
    summary: Dict[str, Dict[str, float]] = {}  # Store stats for each feature column.

    for col in feature_columns:  # Process one feature at a time.
        values = [float(row[col]) for row in train_rows]  # Collect all numeric values for this feature.
        count = len(values)  # Count values to support average and variance calculations.
        mean = sum(values) / count if count else 0.0  # Compute mean safely when list is empty.
        variance = sum((v - mean) ** 2 for v in values) / count if count else 0.0  # Compute population variance.
        std = variance ** 0.5  # Convert variance to standard deviation.

        summary[col] = {  # Save computed statistics for this feature.
            "mean": mean,  # Average of feature values.
            "std": std,  # Standard deviation of feature values.
            "min": min(values) if values else 0.0,  # Smallest observed value.
            "max": max(values) if values else 0.0,  # Largest observed value.
        }

    return {  # Return a complete baseline stats payload.
        "created_at_utc": datetime.now(timezone.utc).isoformat(),  # Artifact creation time in UTC.
        "reference_rows": len(train_rows),  # Number of rows used as reference.
        "feature_columns": feature_columns,  # Ordered list of monitored feature names.
        "summary": summary,  # Per-feature numeric summary.
    }


def run_prepare() -> None:  # Run the full data preparation workflow.
    params = load_params()  # Load settings from params.yaml.
    paths = params["paths"]  # Read all input/output file paths.
    training_cfg = params["training"]  # Read training-related configuration values.
    random_state = int(params["project"]["random_state"])  # Seed for reproducible split behavior.

    dataset = load_iris(as_frame=True)  # Load Iris data as pandas-friendly structures.
    full_df = dataset.frame.copy()  # Copy full dataset (features + target).
    feature_columns = list(dataset.feature_names)  # Keep canonical feature names for downstream use.

    ensure_parent_dir(paths["raw_csv"])  # Create parent folder for raw CSV if needed.
    full_df.to_csv(paths["raw_csv"], index=False)  # Save untouched raw dataset snapshot.

    train_df, valid_df = train_test_split(  # Split data into train and validation partitions.
        full_df,  # Input dataset to split.
        test_size=float(training_cfg["test_size"]),  # Fraction assigned to validation set.
        random_state=random_state,  # Use fixed seed for deterministic split.
        stratify=full_df["target"],  # Preserve class distribution across splits.
    )

    ensure_parent_dir(paths["train_csv"])  # Create parent folder for train split file.
    train_df.to_csv(paths["train_csv"], index=False)  # Save training split to CSV.

    ensure_parent_dir(paths["valid_csv"])  # Create parent folder for validation split file.
    valid_df.to_csv(paths["valid_csv"], index=False)  # Save validation split to CSV.

    baseline_reference = train_df[feature_columns]  # Keep feature-only training data for drift baseline.
    ensure_parent_dir(paths["baseline_reference"])  # Create parent folder for baseline reference CSV.
    baseline_reference.to_csv(paths["baseline_reference"], index=False)  # Save baseline reference rows.

    baseline_stats = _build_baseline_stats(  # Compute summary statistics from baseline reference rows.
        baseline_reference.to_dict(orient="records"),  # Convert baseline DataFrame to row dictionaries.
        feature_columns,  # Pass feature names in fixed order.
    )
    ensure_parent_dir(paths["baseline_stats"])  # Create parent folder for baseline stats JSON.
    with open(paths["baseline_stats"], "w", encoding="utf-8") as f:  # Open output JSON file for writing.
        json.dump(baseline_stats, f, indent=2)  # Write pretty-printed baseline stats payload.

    print(f"Prepared data with {len(train_df)} train rows and {len(valid_df)} valid rows.")  # Log split sizes.
    print(f"Feature columns: {feature_columns}")  # Log feature names for visibility.


if __name__ == "__main__":  # Run preparation when executed as a script.
    run_prepare()  # Entry-point call.
