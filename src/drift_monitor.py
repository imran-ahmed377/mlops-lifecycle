from __future__ import annotations

import json  # Save drift results as JSON artifacts.
from datetime import datetime, timezone  # Timestamp drift reports in UTC.
from typing import Dict  # Type hints for nested result dictionaries.

import pandas as pd  # Load baseline and live feature data from CSV files.
from scipy.stats import ks_2samp  # Compare baseline vs live distributions with the KS test.

from src.common import ensure_parent_dir, load_params  # Shared config and directory helpers.


TARGET_COLUMN = "target"  # Label column name; excluded from drift feature comparisons.


def _status_from_score(score: float, warning_ks: float, alert_ks: float) -> str:  # Map a KS score to a human-readable severity.
    if score >= alert_ks:  # Highest threshold means the feature is in alert state.
        return "alert"
    if score >= warning_ks:  # Mid threshold means the feature is drifting but not yet critical.
        return "warning"
    return "ok"  # Below both thresholds means no meaningful drift was detected.


def run_drift_check() -> None:  # Compare live request data against the baseline reference and persist the result.
    params = load_params()  # Load project configuration from params.yaml.
    paths = params["paths"]  # Resolve configured artifact locations.
    drift_cfg = params["drift"]  # Read drift thresholds and window configuration.

    baseline_path = paths["baseline_reference"]  # Training-derived feature reference data.
    live_path = paths["live_data"]  # Logged live prediction requests.
    output_path = paths.get("drift_status", "drift/drift_status.json")  # Output file for the latest drift summary.

    baseline_df = pd.read_csv(baseline_path)  # Load baseline reference rows.
    if TARGET_COLUMN in baseline_df.columns:  # Drop target if it exists so only features are compared.
        baseline_df = baseline_df.drop(columns=[TARGET_COLUMN])

    if not pd.io.common.file_exists(live_path):  # No live traffic has been logged yet.
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),  # Time this status was generated.
            "status": "insufficient_data",  # Drift cannot be computed without live rows.
            "message": "No live request data found yet.",  # Human-readable explanation.
            "live_rows": 0,  # Live dataset is empty because file does not exist.
            "window_size": int(drift_cfg["window_size"]),  # Still report configured comparison window.
            "features": {},  # No per-feature scores are available.
        }
        ensure_parent_dir(output_path)  # Create output directory if needed.
        with open(output_path, "w", encoding="utf-8") as f:  # Persist status for API/UI consumers.
            json.dump(result, f, indent=2)
        print(result["message"])  # Log the reason to the console.
        return  # Stop early because drift analysis is not possible yet.

    live_df = pd.read_csv(live_path)  # Load live request history.
    if "predicted_class" in live_df.columns:  # Remove prediction output so only input features remain.
        live_df = live_df.drop(columns=["predicted_class"])

    feature_columns = [c for c in baseline_df.columns if c in live_df.columns]  # Compare only shared feature columns.
    window_size = int(drift_cfg["window_size"])  # Maximum number of recent live rows to inspect.
    warning_ks = float(drift_cfg["warning_ks"])  # Threshold for warning status.
    alert_ks = float(drift_cfg["alert_ks"])  # Threshold for alert status.
    min_required_rows = 5  # Minimum live sample size before the KS test is considered useful.

    if len(live_df) < min_required_rows:  # Avoid unstable drift decisions from too few samples.
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),  # Time this status was generated.
            "status": "insufficient_data",  # Not enough data for a trustworthy comparison.
            "message": f"Need at least {min_required_rows} live rows, found {len(live_df)}.",  # Explain the shortfall.
            "live_rows": int(len(live_df)),  # Report available live rows.
            "window_size": window_size,  # Report configured comparison window.
            "features": {},  # No per-feature metrics until enough data exists.
        }
        ensure_parent_dir(output_path)  # Create output directory if needed.
        with open(output_path, "w", encoding="utf-8") as f:  # Save status for downstream readers.
            json.dump(result, f, indent=2)
        print(result["message"])  # Log reason to console.
        return  # Stop early because sample size is too small.

    live_window = live_df.tail(window_size)  # Use only the most recent live rows for the comparison window.

    feature_results: Dict[str, Dict[str, float | str]] = {}  # Store KS statistics for each feature.
    overall_status = "ok"  # Start optimistic and escalate if any feature crosses thresholds.

    for col in feature_columns:  # Evaluate drift feature by feature.
        baseline_values = baseline_df[col].astype(float)  # Baseline distribution for this feature.
        live_values = live_window[col].astype(float)  # Recent live distribution for this feature.

        ks_score, p_value = ks_2samp(baseline_values, live_values)  # Compare distributions with a two-sample KS test.
        status = _status_from_score(float(ks_score), warning_ks, alert_ks)  # Convert score to ok/warning/alert.

        feature_results[col] = {
            "ks_stat": float(ks_score),  # Magnitude of distribution difference.
            "p_value": float(p_value),  # Statistical significance estimate from the KS test.
            "status": status,  # Severity label for this feature.
        }

        if status == "alert":  # Any alert-level feature escalates the overall result to alert.
            overall_status = "alert"
        elif status == "warning" and overall_status != "alert":  # Warning applies only if no feature is already alerting.
            overall_status = "warning"

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),  # Time the completed drift check was recorded.
        "status": overall_status,  # Overall status derived from all feature checks.
        "message": "KS drift check completed.",  # Human-readable summary.
        "live_rows": int(len(live_df)),  # Total number of live rows currently stored.
        "window_size": window_size,  # Number of recent rows considered in the comparison.
        "features": feature_results,  # Per-feature drift measurements.
    }

    ensure_parent_dir(output_path)  # Create output directory if needed.
    with open(output_path, "w", encoding="utf-8") as f:  # Write the latest drift status to disk.
        json.dump(result, f, indent=2)

    print(f"Drift check completed with status: {overall_status}")  # Console summary for manual runs.


if __name__ == "__main__":  # Run drift monitoring when executed directly.
    run_drift_check()  # Script entry point.