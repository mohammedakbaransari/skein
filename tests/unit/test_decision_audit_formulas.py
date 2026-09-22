"""
tests/unit/test_decision_audit_formulas.py
=============================================
Hand-computed numeric-correctness tests for
agents.decision_audit.agent.compute_accountability_metrics (roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.decision_audit.agent import compute_accountability_metrics


class TestDecisionAuditFormulas(unittest.TestCase):

    def setUp(self):
        self.decisions = [
            {"decision_id": "D001", "evaluator_id": "E1", "category": "IT",
             "rationale_logged": False, "human_override": False, "ai_score": 90,
             "factors_weighted": {"price": 0.4}},
            {"decision_id": "D002", "evaluator_id": "E1", "category": "IT",
             "rationale_logged": True, "human_override": True, "ai_score": 70,
             "factors_weighted": {"price": 0.5}},
            {"decision_id": "D003", "evaluator_id": "E1", "category": "Facilities",
             "rationale_logged": False, "human_override": False, "ai_score": 95,
             "factors_weighted": {"price": 0.3}},
            {"decision_id": "D004", "evaluator_id": "E2", "category": "IT",
             "rationale_logged": True, "human_override": False, "ai_score": 60,
             "factors_weighted": {"price": 0.6}},
            {"decision_id": "D005", "evaluator_id": "E2", "category": "Facilities",
             "rationale_logged": False, "human_override": True, "ai_score": 50,
             "factors_weighted": {"price": 0.2}},
        ]
        self.metrics = compute_accountability_metrics(self.decisions)

    def test_overall_totals(self):
        m = self.metrics
        self.assertEqual(m.total_decisions, 5)
        self.assertEqual(m.rationale_gap_count, 3)       # D001, D003, D005
        self.assertEqual(m.rationale_gap_pct, 60.0)
        self.assertEqual(m.override_count, 2)             # D002, D005
        self.assertEqual(m.override_rate_pct, 40.0)

    def test_high_risk_decisions_require_no_rationale_and_high_ai_score(self):
        # D001 (90) and D003 (95) qualify; D005 has no rationale but ai_score=50.
        self.assertEqual(self.metrics.high_risk_decision_ids, ["D001", "D003"])

    def test_evaluator_metrics_only_e1_meets_threshold_of_three(self):
        # E2 has only 2 decisions, below the len(ev_dec) >= 3 threshold.
        self.assertEqual(len(self.metrics.evaluator_metrics), 1)
        e1 = self.metrics.evaluator_metrics[0]
        self.assertEqual(e1.evaluator_id, "E1")
        self.assertEqual(e1.decision_count, 3)
        self.assertEqual(e1.rationale_gap_pct, 66.7)
        self.assertEqual(e1.override_count, 1)
        self.assertEqual(e1.price_weight_variance, 0.0067)

    def test_category_metrics(self):
        cat = self.metrics.category_metrics
        self.assertEqual(cat["IT"]["decisions"], 3)
        self.assertEqual(cat["IT"]["rationale_gap_pct"], 33.3)
        self.assertEqual(cat["IT"]["override_rate_pct"], 33.3)
        self.assertEqual(cat["Facilities"]["decisions"], 2)
        self.assertEqual(cat["Facilities"]["rationale_gap_pct"], 100.0)
        self.assertEqual(cat["Facilities"]["override_rate_pct"], 50.0)

    def test_empty_input_returns_zeroed_metrics(self):
        m = compute_accountability_metrics([])
        self.assertEqual(m.total_decisions, 0)
        self.assertEqual(m.evaluator_metrics, [])
        self.assertEqual(m.category_metrics, {})


if __name__ == "__main__":
    unittest.main()
