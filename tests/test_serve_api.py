from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import serve_api


class ServeApiTests(unittest.TestCase):
    def test_health_endpoint_reports_ok_and_model_loaded(self) -> None:
        with patch.object(serve_api.service, "model", object()):
            response = serve_api.health()

        self.assertEqual(response, {"status": "ok", "model_loaded": True})

    def test_drift_status_returns_insufficient_data_when_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_status_path = Path(tmpdir) / "missing_drift_status.json"

            with patch.object(serve_api.service, "drift_status_path", return_value=missing_status_path):
                response = serve_api.drift_status()

        self.assertEqual(
            response,
            {
                "status": "insufficient_data",
                "message": "Drift check has not run yet.",
                "features": {},
                "live_rows": 0,
            },
        )

    def test_predict_returns_confidence_and_latency(self) -> None:
        class StubModel:
            classes_ = [0, 1, 2]

            def predict(self, _frame):
                return [1]

            def predict_proba(self, _frame):
                return [[0.05, 0.9, 0.05]]

        payload = serve_api.PredictRequest(
            sepal_length_cm=5.1,
            sepal_width_cm=3.5,
            petal_length_cm=1.4,
            petal_width_cm=0.2,
        )

        with patch.object(serve_api.service, "model", StubModel()):
            with patch.object(serve_api.service, "append_live_row"):
                response = serve_api.predict(payload)

        self.assertEqual(response["predicted_class"], 1)
        self.assertEqual(response["predicted_label"], "versicolor")
        self.assertAlmostEqual(float(response["confidence"]), 0.9, places=6)

        class_probabilities = response.get("class_probabilities", {})
        self.assertIn("setosa", class_probabilities)
        self.assertIn("versicolor", class_probabilities)
        self.assertIn("virginica", class_probabilities)
        self.assertAlmostEqual(float(class_probabilities["versicolor"]), 0.9, places=6)

        self.assertIn("latency_ms", response)
        self.assertGreaterEqual(float(response["latency_ms"]), 0.0)


if __name__ == "__main__":
    unittest.main()