"""
tests/unit/test_should_cost_formulas.py
==========================================
Hand-computed numeric-correctness tests for
agents.cost_intelligence.should_cost.compute_commodity_movements
(roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.cost_intelligence.should_cost import compute_commodity_movements


def _movement(model, key):
    return next(m for m in model.commodity_movements if m.commodity_key == key)


class TestShouldCostFormulas(unittest.TestCase):

    def setUp(self):
        self.records = [
            {
                "month": "2024-01",
                "steel_hrc_usd_ton": 1000, "copper_lme_usd_ton": 9000,
                "hdpe_resin_usd_ton": 1500, "labour_index_mfg": 100,
                "energy_index": 100,
            },
            {
                "month": "2024-06",
                "steel_hrc_usd_ton": 850,    # -15.0%  -> High leverage
                "copper_lme_usd_ton": 9000,  #   0.0%  -> None
                "hdpe_resin_usd_ton": 1410,  #  -6.0%  -> Medium leverage
                "labour_index_mfg": 108,     #  +8.0%  -> rising cost warning
                "energy_index": 100,         #   0.0%  -> None
            },
        ]
        self.model = compute_commodity_movements(self.records)

    def test_periods_and_date_range(self):
        self.assertEqual(self.model.periods_analysed, 2)
        self.assertEqual(self.model.date_range, "2024-01 \u2192 2024-06")

    def test_steel_high_leverage(self):
        m = _movement(self.model, "steel_hrc_usd_ton")
        self.assertEqual(m.change_pct, -15.0)
        self.assertEqual(m.leverage_level, "High")

    def test_hdpe_medium_leverage(self):
        m = _movement(self.model, "hdpe_resin_usd_ton")
        self.assertEqual(m.change_pct, -6.0)
        self.assertEqual(m.leverage_level, "Medium")

    def test_copper_no_leverage(self):
        m = _movement(self.model, "copper_lme_usd_ton")
        self.assertEqual(m.change_pct, 0.0)
        self.assertEqual(m.leverage_level, "None")

    def test_leverage_opportunities_below_threshold_only(self):
        keys = [m.commodity_key for m in self.model.leverage_opportunities]
        self.assertEqual(keys, ["steel_hrc_usd_ton", "hdpe_resin_usd_ton"])

    def test_rising_cost_warning_for_labour_index(self):
        self.assertEqual(self.model.rising_cost_warnings, ("Manufacturing Labour Index",))

    def test_insufficient_data_returns_empty_model(self):
        model = compute_commodity_movements([{"month": "2024-01", "steel_hrc_usd_ton": 1000}])
        self.assertEqual(model.periods_analysed, 0)
        self.assertEqual(model.date_range, "insufficient data")
        self.assertEqual(model.commodity_movements, ())


if __name__ == "__main__":
    unittest.main()
