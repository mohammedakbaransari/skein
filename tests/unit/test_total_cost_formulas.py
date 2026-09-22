"""
tests/unit/test_total_cost_formulas.py
=========================================
Hand-computed numeric-correctness tests for
agents.cost_intelligence.total_cost.analyse_tco_portfolio (roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.cost_intelligence.total_cost import analyse_tco_portfolio


def _asset(summary, asset_id):
    return next(a for a in summary.asset_profiles if a.asset_id == asset_id)


class TestTotalCostFormulas(unittest.TestCase):

    def setUp(self):
        self.assets = [
            {"asset_id": "A1", "asset_type": "IT_Hardware",
             "purchase_price_usd": 100_000, "total_tco_usd": 100_000,
             "procurement_decided_on_price_alone": False},
            {"asset_id": "A2", "asset_type": "MRO",
             "purchase_price_usd": 50_000, "total_tco_usd": 250_000,
             "procurement_decided_on_price_alone": True},
            {"asset_id": "A3", "asset_type": "MRO",
             "purchase_price_usd": 20_000, "total_tco_usd": 90_000,
             "procurement_decided_on_price_alone": True},
        ]
        self.summary = analyse_tco_portfolio(self.assets)

    def test_asset_ratios_and_value_at_risk(self):
        a1, a2, a3 = (_asset(self.summary, i) for i in ("A1", "A2", "A3"))
        self.assertEqual(a1.tco_to_price_ratio, 1.0)
        self.assertEqual(a1.lifecycle_value_at_risk_usd, 0.0)
        self.assertEqual(a2.tco_to_price_ratio, 5.0)
        self.assertEqual(a2.lifecycle_value_at_risk_usd, 200_000.0)
        self.assertEqual(a3.tco_to_price_ratio, 4.5)
        self.assertEqual(a3.lifecycle_value_at_risk_usd, 70_000.0)

    def test_portfolio_totals(self):
        s = self.summary
        self.assertEqual(s.total_assets, 3)
        self.assertEqual(s.price_only_count, 2)
        self.assertEqual(s.price_only_pct, 66.7)
        self.assertEqual(s.high_ratio_count, 2)   # ratio > 4.0: A2, A3
        self.assertEqual(s.total_value_at_risk_usd, 270_000.0)
        self.assertEqual(s.avg_tco_to_price_ratio, 3.5)

    def test_category_breakdown(self):
        cat = self.summary.category_breakdown
        self.assertEqual(cat["IT_Hardware"]["count"], 1)
        self.assertEqual(cat["IT_Hardware"]["avg_ratio"], 1.0)
        self.assertEqual(cat["IT_Hardware"]["value_at_risk"], 0.0)
        self.assertEqual(cat["MRO"]["count"], 2)
        self.assertEqual(cat["MRO"]["avg_ratio"], 4.75)
        self.assertEqual(cat["MRO"]["price_only_pct"], 100.0)
        self.assertEqual(cat["MRO"]["value_at_risk"], 270_000.0)

    def test_sorted_by_value_at_risk_descending(self):
        ids = [a.asset_id for a in self.summary.asset_profiles]
        self.assertEqual(ids, ["A2", "A3", "A1"])

    def test_empty_input_returns_zeroed_summary(self):
        s = analyse_tco_portfolio([])
        self.assertEqual(s.total_assets, 0)
        self.assertEqual(s.asset_profiles, ())
        self.assertEqual(s.category_breakdown, {})


if __name__ == "__main__":
    unittest.main()
