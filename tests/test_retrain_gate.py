from __future__ import annotations

import unittest

from src.retrain import _decide_promotion


class RetrainGateTests(unittest.TestCase):
    def test_promotes_when_no_current_model_exists(self) -> None:
        promoted, reason = _decide_promotion(
            current_model=None,
            current_metrics={"valid_accuracy": -1.0, "valid_f1_macro": -1.0},
            candidate_metrics={"valid_accuracy": 0.80, "valid_f1_macro": 0.79},
        )

        self.assertTrue(promoted)
        self.assertEqual(reason, "no_current_model")

    def test_promotes_when_candidate_meets_gate(self) -> None:
        promoted, reason = _decide_promotion(
            current_model=object(),
            current_metrics={"valid_accuracy": 0.90, "valid_f1_macro": 0.88},
            candidate_metrics={"valid_accuracy": 0.90, "valid_f1_macro": 0.88},
        )

        self.assertTrue(promoted)
        self.assertEqual(reason, "candidate_meets_gate")

    def test_rejects_when_candidate_is_worse(self) -> None:
        promoted, reason = _decide_promotion(
            current_model=object(),
            current_metrics={"valid_accuracy": 0.93, "valid_f1_macro": 0.93},
            candidate_metrics={"valid_accuracy": 0.92, "valid_f1_macro": 0.93},
        )

        self.assertFalse(promoted)
        self.assertEqual(reason, "candidate_below_gate")


if __name__ == "__main__":
    unittest.main()