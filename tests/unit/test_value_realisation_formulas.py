"""
tests/unit/test_value_realisation_formulas.py
================================================
Hand-computed numeric-correctness tests for
agents.contract_analysis.value_realisation.analyse_savings_portfolio
(roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.contract_analysis.value_realisation import analyse_savings_portfolio


def _profile(profiles, contract_id):
    return next(p for p in profiles if p.contract_id == contract_id)


class TestValueRealisationFormulas(unittest.TestCase):

    def setUp(self):
        self.records = [
            # C001: negotiated 10% -> actual drops to 4% over 4 months (deteriorating, critical)
            {"contract_id": "C001", "category": "IT", "month": "2024-01",
             "negotiated_savings_pct": 10.0, "actual_savings_pct": 10.0,
             "leakage_amount_usd": 1000, "leakage_causes": ["maverick_spend"]},
            {"contract_id": "C001", "category": "IT", "month": "2024-02",
             "negotiated_savings_pct": 10.0, "actual_savings_pct": 8.0,
             "leakage_amount_usd": 1500, "leakage_causes": ["spec_change"]},
            {"contract_id": "C001", "category": "IT", "month": "2024-03",
             "negotiated_savings_pct": 10.0, "actual_savings_pct": 6.0,
             "leakage_amount_usd": 2000, "leakage_causes": ["maverick_spend"]},
            {"contract_id": "C001", "category": "IT", "month": "2024-04",
             "negotiated_savings_pct": 10.0, "actual_savings_pct": 4.0,
             "leakage_amount_usd": 2500, "leakage_causes": ["volume_shortfall"]},
            # C002: negotiated 5% -> actual stays 5% (stable, no leakage)
            {"contract_id": "C002", "category": "Facilities", "month": "2024-01",
             "negotiated_savings_pct": 5.0, "actual_savings_pct": 5.0, "leakage_amount_usd": 0},
            {"contract_id": "C002", "category": "Facilities", "month": "2024-02",
             "negotiated_savings_pct": 5.0, "actual_savings_pct": 5.0, "leakage_amount_usd": 0},
        ]
        self.profiles = analyse_savings_portfolio(self.records)

    def test_c001_leakage_and_trend(self):
        p = _profile(self.profiles, "C001")
        self.assertEqual(p.negotiated_savings_pct, 10.0)
        self.assertEqual(p.actual_savings_pct, 4.0)
        self.assertEqual(p.leakage_pct, 6.0)
        self.assertEqual(p.trend, "Deteriorating")
        self.assertEqual(p.alert_level, "critical")
        self.assertEqual(p.cumulative_leakage_usd, 7000.0)
        self.assertEqual(p.active_causes, ("maverick_spend", "spec_change", "volume_shortfall"))
        self.assertEqual(p.months_tracked, 4)

    def test_c002_stable_no_leakage(self):
        p = _profile(self.profiles, "C002")
        self.assertEqual(p.leakage_pct, 0.0)
        self.assertEqual(p.trend, "Stable")
        self.assertEqual(p.alert_level, "ok")
        self.assertEqual(p.cumulative_leakage_usd, 0.0)

    def test_sorted_by_cumulative_leakage_descending(self):
        self.assertEqual([p.contract_id for p in self.profiles], ["C001", "C002"])


if __name__ == "__main__":
    unittest.main()
