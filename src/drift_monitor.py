from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from scipy.stats import ks_2samp

from src.common import ensure_parent_dir, load_params


def _status_from_score(score: float, warning_ks: float, alert_ks: float) -> str:
    if score >= alert_ks:
        return "alert"
    if score >= warning_ks:
        return "warning"
    return "ok"


def run_drift_check() -> None:
    params = load_params()
    paths = params["paths"]
    drift_cfg = params["drift"]

    baseline_path = paths["baseline_reference"]
    live_path = paths["live_data"]
    output_path = paths.get("drift_status", "drift/drift_status.json")

    baseline_df = pd.read_csv(baseline_path)
    if TARGET_COLUMN in baseline_df.columns:
        baseline_df = baseline_df.drop(columns=[TARGET_COLUMN])

    if not pd.io.common.file_exists(live_path):
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "message": "No live request data found yet.",
            "live_rows": 0,
            "window_size": int(drift_cfg["window_size"]),
            "features": {},
        }
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(result["message"])
        return

    live_df = pd.read_csv(live_path)
    if "predicted_class" in live_df.columns:
        live_df = live_df.drop(columns=["predicted_class"])

    feature_columns = [c for c in baseline_df.columns if c in live_df.columns]
    window_size = int(drift_cfg["window_size"])
    warning_ks = float(drift_cfg["warning_ks"])
    alert_ks = float(drift_cfg["alert_ks"])
    min_required_rows = 5

    if len(live_df) < min_required_rows:
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "message": f"Need at least {min_required_rows} live rows, found {len(live_df)}.",
            "live_rows": int(len(live_df)),
            "window_size": window_size,
            "features": {},
        }
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(result["message"])
        return

    live_window = live_df.tail(window_size)

    feature_results: Dict[str, Dict[str, float | str]] = {}
    overall_status = "ok"

    for col in feature_columns:
        baseline_values = baseline_df[col].astype(float)
        live_values = live_window[col].astype(float)

        ks_score, p_value = ks_2samp(baseline_values, live_values)
        status = _status_from_score(float(ks_score), warning_ks, alert_ks)

        feature_results[col] = {
            "ks_stat": float(ks_score),
            "p_value": float(p_value),
            "status": status,
        }

        if status == "alert":
            overall_status = "alert"
        elif status == "warning" and overall_status != "alert":
            overall_status = "warning"

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "message": "KS drift check completed.",
        "live_rows": int(len(live_df)),
        "window_size": window_size,
        "features": feature_results,
    }

    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Drift check completed with status: {overall_status}")


TARGET_COLUMN = "target"


if __name__ == "__main__":
    run_drift_check()