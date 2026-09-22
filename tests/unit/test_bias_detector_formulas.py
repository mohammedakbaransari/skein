"""
tests/unit/test_bias_detector_formulas.py
============================================
Hand-computed numeric-correctness tests for
agents.bias_detection.bias_detector.analyse_evaluation_bias — closes the
"formulas exist but are never checked against known expected values" gap
identified in the architecture assessment (roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.bias_detection.bias_detector import analyse_evaluation_bias


def _stat(analysis, supplier_type):
    return next(s for s in analysis.supplier_type_stats if s.supplier_type == supplier_type)


class TestBiasDetectorFormulas(unittest.TestCase):

    def setUp(self):
        self.evaluations = [
            # incumbent, evaluator E1
            {"evaluator_id": "E1", "supplier_type": "incumbent", "objective_score": 80, "subjective_score": 85, "awarded": True},
            {"evaluator_id": "E1", "supplier_type": "incumbent", "objective_score": 90, "subjective_score": 95, "awarded": True},
            # new_entrant, evaluator E1
            {"evaluator_id": "E1", "supplier_type": "new_entrant", "objective_score": 70, "subjective_score": 60, "awarded": False},
            {"evaluator_id": "E1", "supplier_type": "new_entrant", "objective_score": 80, "subjective_score": 70, "awarded": False},
            # diverse_owned, evaluator E2 (only 2 evals -> below the evaluator-metrics threshold of 4)
            {"evaluator_id": "E2", "supplier_type": "diverse_owned", "objective_score": 70, "subjective_score": 70, "awarded": False},
            {"evaluator_id": "E2", "supplier_type": "diverse_owned", "objective_score": 72, "subjective_score": 72, "awarded": False},
        ]
        self.result = analyse_evaluation_bias(self.evaluations)

    def test_total_evaluations(self):
        self.assertEqual(self.result.total_evaluations, 6)

    def test_incumbent_supplier_type_stats(self):
        s = _stat(self.result, "incumbent")
        self.assertEqual(s.count, 2)
        self.assertEqual(s.award_rate_pct, 100.0)
        self.assertEqual(s.avg_objective_score, 85.0)
        self.assertEqual(s.avg_subjective_score, 90.0)
        self.assertEqual(s.subjective_premium, 5.0)

    def test_new_entrant_supplier_type_stats(self):
        s = _stat(self.result, "new_entrant")
        self.assertEqual(s.count, 2)
        self.assertEqual(s.award_rate_pct, 0.0)
        self.assertEqual(s.avg_objective_score, 75.0)
        self.assertEqual(s.avg_subjective_score, 65.0)
        self.assertEqual(s.subjective_premium, -10.0)

    def test_incumbent_vs_new_entrant_deltas(self):
        self.assertEqual(self.result.incumbent_objective_delta, 10.0)
        self.assertEqual(self.result.award_rate_gap_pct, 100.0)

    def test_evaluator_bias_metrics_only_e1_qualifies(self):
        # E2 has only 2 evaluations, below the len(evals) >= 4 threshold.
        self.assertEqual(len(self.result.evaluator_bias_metrics), 1)
        m = self.result.evaluator_bias_metrics[0]
        self.assertEqual(m.evaluator_id, "E1")
        self.assertEqual(m.evaluation_count, 4)
        self.assertEqual(m.incumbent_premium, 5.0)
        self.assertEqual(m.non_incumbent_premium, -10.0)
        self.assertEqual(m.bias_differential, 15.0)

    def test_diverse_suppression_flag_true(self):
        # diverse_owned: avg_objective_score=71.0 (>=65) and award_rate_pct=0.0 (<30)
        self.assertTrue(self.result.diverse_suppression_flag)

    def test_sme_suppression_flag_false_when_no_sme_data(self):
        self.assertFalse(self.result.sme_suppression_flag)

    def test_empty_input_returns_zeroed_analysis(self):
        empty = analyse_evaluation_bias([])
        self.assertEqual(empty.total_evaluations, 0)
        self.assertEqual(empty.supplier_type_stats, ())
        self.assertFalse(empty.diverse_suppression_flag)


if __name__ == "__main__":
    unittest.main()
