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


if __name__ == "__main__":
    unittest.main()