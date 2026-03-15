from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.drift_monitor import run_drift_check


class DriftMonitorTests(unittest.TestCase):
    def test_run_drift_check_reports_insufficient_data_for_small_live_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            baseline_path = temp_root / "baseline_reference.csv"
            live_path = temp_root / "live_requests.csv"
            output_path = temp_root / "drift_status.json"

            baseline_df = pd.DataFrame(
                {
                    "sepal length (cm)": [5.1, 4.9, 4.7, 4.6, 5.0],
                    "sepal width (cm)": [3.5, 3.0, 3.2, 3.1, 3.6],
                    "petal length (cm)": [1.4, 1.4, 1.3, 1.5, 1.4],
                    "petal width (cm)": [0.2, 0.2, 0.2, 0.2, 0.2],
                }
            )
            baseline_df.to_csv(baseline_path, index=False)

            live_df = pd.DataFrame(
                {
                    "sepal length (cm)": [6.2, 6.3],
                    "sepal width (cm)": [2.8, 2.9],
                    "petal length (cm)": [4.8, 4.9],
                    "petal width (cm)": [1.8, 1.7],
                    "predicted_class": [2, 2],
                }
            )
            live_df.to_csv(live_path, index=False)

            fake_params = {
                "paths": {
                    "baseline_reference": str(baseline_path),
                    "live_data": str(live_path),
                    "drift_status": str(output_path),
                },
                "drift": {
                    "window_size": 50,
                    "warning_ks": 0.15,
                    "alert_ks": 0.25,
                },
            }

            with patch("src.drift_monitor.load_params", return_value=fake_params):
                run_drift_check()

            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            self.assertEqual(result["status"], "insufficient_data")
            self.assertEqual(result["live_rows"], 2)
            self.assertEqual(result["window_size"], 50)
            self.assertEqual(result["features"], {})
            self.assertIn("Need at least 5 live rows", result["message"])


if __name__ == "__main__":
    unittest.main()