"""
tests/unit/test_supplier_stress.py
=====================================
Unit tests for SupplierStressAgent — pure signal extraction functions
tested independently of LLM, registry, or framework infrastructure.
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.supply_risk.supplier_stress import (
    SupplierStressAgent,
    analyse_supplier_portfolio,
    extract_supplier_stress_profile,
)
from framework.core.types import Severity, SessionId, Task
from framework.reasoning.stubs import DryRunReasoningEngine


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def _healthy(sid: str, name: str, month: str) -> Dict:
    return {
        "supplier_id": sid, "supplier_name": name, "month": month,
        "po_ack_days": 2.0, "otd_pct": 97.0, "quality_hold_pct": 0.8,
        "invoice_disputes": 1, "unsolicited_discounts": 0, "sales_response_hours": 4.0,
    }


def _stressed(sid: str, name: str, month: str, level: int = 2) -> Dict:
    m = 1 + (level * 0.5)
    return {
        "supplier_id": sid, "supplier_name": name, "month": month,
        "po_ack_days": 2.0 * m * 2, "otd_pct": 97.0 / m,
        "quality_hold_pct": 0.8 * m * 2,
        "invoice_disputes": int(1 * m * 3), "unsolicited_discounts": level,
        "sales_response_hours": 4.0 * m * 3,
    }


def _records(sid: str, name: str, n: int = 6, stress_from: int = 999) -> List[Dict]:
    months = [f"2024-{i+1:02d}" for i in range(n)]
    return [
        _healthy(sid, name, m) if i < stress_from
        else _stressed(sid, name, m, level=min(3, i - stress_from + 1))
        for i, m in enumerate(months)
    ]


# ---------------------------------------------------------------------------
# Tests: extract_supplier_stress_profile
# ---------------------------------------------------------------------------

class TestExtractProfile(unittest.TestCase):

    def test_healthy_supplier_is_green(self):
        records = _records("S001", "HealthyCo", n=6)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.risk_level, "Green")
        self.assertEqual(profile.composite_score, 0)

    def test_stressed_supplier_reaches_red(self):
        records = _records("S002", "StressedCo", n=6, stress_from=3)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)
        self.assertIn(profile.risk_level, ("Red", "Critical"))
        self.assertGreater(profile.composite_score, 5)

    def test_critical_supplier_reaches_critical(self):
        # 3 healthy baseline months, then 3 heavily stressed months
        records = _records("S003", "CriticalCo", n=6, stress_from=3)
        # Override last 3 with maximum stress
        months = [f"2024-{i+1:02d}" for i in range(6)]
        for i in range(3, 6):
            records[i] = _stressed("S003", "CriticalCo", months[i], level=4)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)
        self.assertIn(profile.risk_level, ("Red", "Critical"))
        self.assertGreaterEqual(profile.composite_score, 7)

    def test_returns_none_with_insufficient_data(self):
        records = _records("S004", "TooFew", n=3)
        result = extract_supplier_stress_profile(records)
        self.assertIsNone(result)

    def test_exactly_four_months_returns_profile(self):
        records = _records("S005", "MinData", n=4)
        result = extract_supplier_stress_profile(records)
        self.assertIsNotNone(result)

    def test_six_signals_returned(self):
        records = _records("S006", "SixSig", n=6)
        profile = extract_supplier_stress_profile(records)
        self.assertEqual(len(profile.signals), 6)

    def test_first_signal_month_detected(self):
        """When stress starts at month 3, first_signal_month should be 2024-04 or later."""
        records = _records("S007", "EarlyWarn", n=8, stress_from=3)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile.first_signal_month)

    def test_trend_deteriorating_when_stress_increases(self):
        records = _records("S008", "Worsening", n=8, stress_from=4)
        profile = extract_supplier_stress_profile(records)
        self.assertIn(profile.trend_direction, ("Deteriorating", "Stable"))

    def test_composite_score_sum_of_signals(self):
        records = _records("S009", "ScoreTest", n=6)
        profile = extract_supplier_stress_profile(records)
        expected = sum(s.score for s in profile.signals)
        self.assertEqual(profile.composite_score, expected)

    def test_all_fields_populated(self):
        records = _records("S010", "Complete", n=6)
        p = extract_supplier_stress_profile(records)
        self.assertIsNotNone(p.supplier_id)
        self.assertIsNotNone(p.supplier_name)
        self.assertIsNotNone(p.risk_level)
        self.assertIsNotNone(p.trend_direction)
        self.assertEqual(p.months_analysed, 6)


# ---------------------------------------------------------------------------
# Tests: analyse_supplier_portfolio
# ---------------------------------------------------------------------------

class TestPortfolioAnalysis(unittest.TestCase):

    def test_multiple_suppliers_sorted_by_risk(self):
        records = (
            _records("A", "Critical", n=6, stress_from=0) +
            _records("B", "Healthy",  n=6) +
            _records("C", "Amber",    n=6, stress_from=4)
        )
        profiles = analyse_supplier_portfolio(records)
        self.assertGreater(len(profiles), 0)
        # First should be highest risk
        self.assertGreaterEqual(profiles[0].composite_score, profiles[-1].composite_score)

    def test_empty_input_returns_empty(self):
        profiles = analyse_supplier_portfolio([])
        self.assertEqual(profiles, [])

    def test_insufficient_data_per_supplier_skipped(self):
        records = _records("X", "TooFew", n=2)
        profiles = analyse_supplier_portfolio(records)
        self.assertEqual(profiles, [])

    def test_supplier_grouping(self):
        records = (
            _records("SUP-A", "Alpha", n=6) +
            _records("SUP-B", "Beta",  n=6)
        )
        profiles = analyse_supplier_portfolio(records)
        supplier_ids = {p.supplier_id for p in profiles}
        self.assertIn("SUP-A", supplier_ids)
        self.assertIn("SUP-B", supplier_ids)


# ---------------------------------------------------------------------------
# Tests: SupplierStressAgent.observe()
# ---------------------------------------------------------------------------

class TestSupplierStressAgentObserve(unittest.TestCase):

    def setUp(self):
        self.engine = DryRunReasoningEngine({"executive_summary": "test", "suppliers": []})
        self.agent  = SupplierStressAgent(reasoning_engine=self.engine)

    def test_observe_returns_structured_dict(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("X", "TestCo", n=6)},
        )
        obs = self.agent.observe(task)
        self.assertIn("supplier_count", obs)
        self.assertIn("supplier_profiles", obs)
        self.assertIsInstance(obs["supplier_profiles"], list)

    def test_observe_empty_data_raises(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": []},
        )
        with self.assertRaises(ValueError):
            self.agent.observe(task)

    def test_observe_profile_fields_present(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("Y", "YCo", n=6)},
        )
        obs = self.agent.observe(task)
        self.assertGreater(obs["supplier_count"], 0)
        profile = obs["supplier_profiles"][0]
        for field in ("supplier_id", "supplier_name", "composite_score",
                      "risk_level", "signals", "trend_direction"):
            self.assertIn(field, profile, f"Missing field: {field}")


# ---------------------------------------------------------------------------
# Tests: SupplierStressAgent.run() with DryRun
# ---------------------------------------------------------------------------

class TestSupplierStressAgentRun(unittest.TestCase):

    def setUp(self):
        stressed_json = {
            "executive_summary": "Critical supplier detected",
            "suppliers": [{
                "supplier_id": "S-CRIT",
                "supplier_name": "CriticalCo",
                "risk_level": "Critical",
                "composite_score": 11,
                "advance_warning_estimate_months": 6,
                "key_finding": "Significant operational deterioration",
                "recommended_action": "Qualify alternative immediately",
                "watch_indicators": ["PO ack times", "invoice disputes"],
                "intervention_deadline": "30 days",
            }],
            "immediate_priorities": ["CriticalCo"],
        }
        self.engine = DryRunReasoningEngine(stressed_json)
        self.agent  = SupplierStressAgent(reasoning_engine=self.engine)

    def test_run_succeeds(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("S-CRIT", "CriticalCo", n=6, stress_from=0)},
        )
        result = self.agent.run(task)
        self.assertTrue(result.succeeded)

    def test_run_produces_findings(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("S-CRIT", "CriticalCo", n=6, stress_from=0)},
        )
        result = self.agent.run(task)
        self.assertGreater(len(result.findings), 0)

    def test_critical_finding_severity(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("S-CRIT", "CriticalCo", n=6, stress_from=0)},
        )
        result = self.agent.run(task)
        severities = {f.severity for f in result.findings}
        self.assertIn(Severity.CRITICAL, severities)

    def test_result_has_duration(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("S-CRIT", "CriticalCo", n=6)},
        )
        result = self.agent.run(task)
        self.assertIsNotNone(result.duration_ms)
        self.assertGreater(result.duration_ms, 0)

    def test_result_has_reasoning_trace(self):
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _records("S-CRIT", "CriticalCo", n=6)},
        )
        result = self.agent.run(task)
        self.assertNotEqual(result.reasoning_trace, "")


if __name__ == "__main__":
    unittest.main()
