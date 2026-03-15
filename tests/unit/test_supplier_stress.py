"""
tests/unit/test_supplier_stress.py
====================================
Per-agent unit tests for SupplierStressAgent.

Tests the pure signal extraction functions independently of any LLM,
registry, or framework infrastructure.

Test strategy:
  - Unit:       pure functions (extract_supplier_stress_profile)
  - Integration: full agent.run() with DryRunLLMGateway
  - Edge cases:  empty data, single month, boundary composites
"""

from __future__ import annotations

import json
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
from framework.core.types import Severity, Task, SessionId, TaskId
from framework.reasoning.stubs import DryRunReasoningEngine


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def _healthy_record(supplier_id: str, supplier_name: str, month: str) -> Dict:
    return {
        "supplier_id":           supplier_id,
        "supplier_name":         supplier_name,
        "month":                 month,
        "po_ack_days":           2.0,
        "otd_pct":               97.0,
        "quality_hold_pct":      0.8,
        "invoice_disputes":      1,
        "unsolicited_discounts": 0,
        "sales_response_hours":  4.0,
    }


def _stressed_record(supplier_id: str, supplier_name: str, month: str,
                      stress_level: int = 2) -> Dict:
    multiplier = 1 + (stress_level * 0.5)
    return {
        "supplier_id":           supplier_id,
        "supplier_name":         supplier_name,
        "month":                 month,
        "po_ack_days":           2.0 * multiplier * 2,
        "otd_pct":               97.0 / multiplier,
        "quality_hold_pct":      0.8 * multiplier * 2,
        "invoice_disputes":      int(1 * multiplier * 3),
        "unsolicited_discounts": stress_level,
        "sales_response_hours":  4.0 * multiplier * 3,
    }


def _build_records(supplier_id: str, name: str,
                    months: int = 6, stress_from_month: int = 999) -> List[Dict]:
    months_list = [f"2024-{m+1:02d}" for m in range(months)]
    return [
        _healthy_record(supplier_id, name, m)
        if i < stress_from_month
        else _stressed_record(supplier_id, name, m, stress_level=min(3, i - stress_from_month + 1))
        for i, m in enumerate(months_list)
    ]


# ---------------------------------------------------------------------------
# Unit tests — pure signal extraction
# ---------------------------------------------------------------------------

class TestExtractStressProfile(unittest.TestCase):

    def test_healthy_supplier_scores_green(self):
        records = _build_records("S001", "Healthy Co", months=6)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.risk_level, "Green")
        self.assertLessEqual(profile.composite_score, 3)

    def test_stressed_supplier_scores_red_or_critical(self):
        records = _build_records("S002", "Stressed Co", months=6, stress_from_month=2)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)
        self.assertIn(profile.risk_level, ("Red", "Critical"))
        self.assertGreaterEqual(profile.composite_score, 7)

    def test_insufficient_data_returns_none(self):
        records = _build_records("S003", "Short Co", months=3)
        self.assertIsNone(extract_supplier_stress_profile(records))

    def test_empty_records_returns_none(self):
        self.assertIsNone(extract_supplier_stress_profile([]))

    def test_exactly_six_signals_computed(self):
        records = _build_records("S004", "Six Co", months=6)
        profile = extract_supplier_stress_profile(records)
        self.assertEqual(len(profile.signals), 6)

    def test_composite_score_equals_sum_of_signal_scores(self):
        records = _build_records("S005", "Sum Co", months=6, stress_from_month=2)
        profile = extract_supplier_stress_profile(records)
        expected = sum(s.score for s in profile.signals)
        self.assertEqual(profile.composite_score, expected)

    def test_composite_score_within_valid_range(self):
        for sf in [0, 2, 4]:
            records = _build_records(f"S00{sf}", f"Range Co {sf}", months=6,
                                      stress_from_month=sf)
            profile = extract_supplier_stress_profile(records)
            if profile:
                self.assertGreaterEqual(profile.composite_score, 0)
                self.assertLessEqual(profile.composite_score, 12)

    def test_risk_level_boundaries(self):
        # Green: 0-3, Amber: 4-6, Red: 7-9, Critical: 10-12
        records = _build_records("S010", "Boundary", months=6)
        profile = extract_supplier_stress_profile(records)
        composite = profile.composite_score
        if composite <= 3:
            self.assertEqual(profile.risk_level, "Green")
        elif composite <= 6:
            self.assertEqual(profile.risk_level, "Amber")
        elif composite <= 9:
            self.assertEqual(profile.risk_level, "Red")
        else:
            self.assertEqual(profile.risk_level, "Critical")

    def test_first_signal_detected_before_stress_peak(self):
        # Stress starts at month 2 — first signal should be month 2 or 3
        records = _build_records("S006", "Early Co", months=8, stress_from_month=2)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile.first_signal_month)
        # Signal should appear before the last month
        last_month  = int(records[-1]["month"].split("-")[1])
        first_month = int(profile.first_signal_month.split("-")[1])
        self.assertLess(first_month, last_month)

    def test_trend_direction_stable_for_consistent_health(self):
        records = _build_records("S007", "Stable Co", months=8)
        profile = extract_supplier_stress_profile(records)
        self.assertEqual(profile.trend_direction, "Stable")

    def test_signals_are_immutable_tuple(self):
        records = _build_records("S008", "Immutable Co", months=6)
        profile = extract_supplier_stress_profile(records)
        self.assertIsInstance(profile.signals, tuple)

    def test_analyse_portfolio_sorted_descending(self):
        records = (
            _build_records("S001", "Healthy", months=6)
            + _build_records("S002", "Stressed", months=6, stress_from_month=2)
        )
        profiles = analyse_supplier_portfolio(records)
        scores = [p.composite_score for p in profiles]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_analyse_portfolio_handles_mixed_data(self):
        records = (
            _build_records("S001", "Healthy", months=6)
            + _build_records("S002", "Stressed", months=6, stress_from_month=1)
            + _build_records("S003", "Short", months=2)  # excluded (< 4 months)
        )
        profiles = analyse_supplier_portfolio(records)
        supplier_ids = {p.supplier_id for p in profiles}
        self.assertIn("S001", supplier_ids)
        self.assertIn("S002", supplier_ids)
        self.assertNotIn("S003", supplier_ids)  # insufficient data excluded


# ---------------------------------------------------------------------------
# Integration tests — full agent.run() with DryRunLLMGateway
# ---------------------------------------------------------------------------

class TestSupplierStressAgentIntegration(unittest.TestCase):

    def _make_reasoning_engine(self, suppliers=None):
        """Return a DryRunReasoningEngine with a synthetic supplier stress response."""
        payload = {
            "executive_summary": "Integration test result.",
            "suppliers": suppliers or [
                {
                    "supplier_id": "S002",
                    "supplier_name": "Stressed Co",
                    "risk_level": "Critical",
                    "composite_score": 10,
                    "advance_warning_estimate_months": 3,
                    "key_finding": "All 6 signals active",
                    "recommended_action": "Qualify alternative immediately",
                    "watch_indicators": ["PO ack time"],
                    "intervention_deadline": "4 weeks",
                }
            ],
            "immediate_priorities": ["Stressed Co"],
        }
        return DryRunReasoningEngine(synthetic_json=payload)

    def _make_task(self, records):
        return Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": records, "analysis_date": "2024-12"},
            session_id=SessionId.generate(),
        )

    def test_successful_run_returns_findings(self):
        records = (
            _build_records("S001", "Healthy", 6)
            + _build_records("S002", "Stressed", 6, stress_from_month=2)
        )
        agent = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine())  # type: ignore
        task  = self._make_task(records)

        # Use run() (full lifecycle) not execute() directly
        result = agent.run(task)

        self.assertTrue(result.succeeded, f"Agent failed: {result.error}")
        self.assertGreater(len(result.findings), 0)
        self.assertIsNotNone(result.reasoning_trace)

    def test_empty_payload_produces_error_result(self):
        agent = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine())  # type: ignore
        task  = Task.create(
            agent_type="SupplierStressAgent",
            payload={},
            session_id=SessionId.generate(),
        )
        result = agent.run(task)
        self.assertFalse(result.succeeded)
        self.assertIsNotNone(result.error)

    def test_result_is_json_serialisable(self):
        records = _build_records("S001", "Healthy", 6)
        agent   = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine([]))  # type: ignore
        task    = self._make_task(records)
        result  = agent.run(task)
        json_str = result.to_json()
        parsed   = json.loads(json_str)
        self.assertEqual(parsed["agent_name"], "Supplier Financial Stress Early Warning Agent")

    def test_context_payload_not_mutated(self):
        records = _build_records("S001", "Healthy", 6)
        payload = {"transaction_data": records, "analysis_date": "2024-12"}
        import copy
        original = copy.deepcopy(payload)
        agent = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine([]))  # type: ignore
        task  = Task.create("SupplierStressAgent", payload)
        agent.run(task)
        self.assertEqual(payload["transaction_data"], original["transaction_data"])

    def test_findings_have_valid_severity_levels(self):
        records = _build_records("S002", "Stressed", 6, stress_from_month=1)
        agent   = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine())  # type: ignore
        task    = self._make_task(records)
        result  = agent.run(task)
        valid   = {s for s in Severity}
        for finding in result.findings:
            self.assertIn(finding.severity, valid)

    def test_run_id_unique_per_execution(self):
        records = _build_records("S001", "Healthy", 6)
        agent   = SupplierStressAgent(reasoning_engine=self._make_reasoning_engine([]))  # type: ignore
        task1   = self._make_task(records)
        task2   = self._make_task(records)
        r1 = agent.run(task1)
        r2 = agent.run(task2)
        self.assertNotEqual(str(r1.task_id), str(r2.task_id))

    def test_metadata_attributes_present(self):
        self.assertEqual(SupplierStressAgent.METADATA.agent_type, "SupplierStressAgent")
        self.assertIn("supplier_stress_detection",
                      [c.name for c in SupplierStressAgent.METADATA.capabilities])
        self.assertIn("mystery_02", SupplierStressAgent.METADATA.mystery_refs)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestSupplierStressEdgeCases(unittest.TestCase):

    def test_all_zeros_in_records(self):
        records = [
            {"supplier_id": "S_ZERO", "supplier_name": "Zero Co", "month": f"2024-{i+1:02d}",
             "po_ack_days": 0, "otd_pct": 0, "quality_hold_pct": 0,
             "invoice_disputes": 0, "unsolicited_discounts": 0, "sales_response_hours": 0}
            for i in range(6)
        ]
        profile = extract_supplier_stress_profile(records)
        # All zeros — no change from baseline — should score zero
        self.assertIsNotNone(profile)
        self.assertEqual(profile.composite_score, 0)

    def test_exactly_four_months_minimum(self):
        records = _build_records("S_MIN", "Minimum Co", months=4)
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)

    def test_large_portfolio_performance(self):
        """50 suppliers × 12 months should complete in < 2 seconds."""
        import time
        import random
        rng = random.Random(42)
        records = [
            {"supplier_id": f"S{i:03d}", "supplier_name": f"Supplier {i}",
             "month": f"2024-{m+1:02d}",
             "po_ack_days": round(rng.uniform(1.5, 3.0), 1),
             "otd_pct": round(rng.uniform(90, 99), 1),
             "quality_hold_pct": round(rng.uniform(0.3, 1.2), 2),
             "invoice_disputes": rng.randint(0, 2),
             "unsolicited_discounts": 0,
             "sales_response_hours": round(rng.uniform(2, 8), 1)}
            for i in range(50)
            for m in range(12)
        ]
        t0 = time.monotonic()
        profiles = analyse_supplier_portfolio(records)
        elapsed = time.monotonic() - t0
        self.assertEqual(len(profiles), 50)
        self.assertLess(elapsed, 2.0, f"Large portfolio took {elapsed:.2f}s (limit 2s)")

    def test_missing_fields_handled_gracefully(self):
        """Records with missing fields should not crash the extractor."""
        records = [
            {"supplier_id": "S_PARTIAL", "supplier_name": "Partial Co",
             "month": f"2024-{i+1:02d}",
             "po_ack_days": 2.0}  # all other fields missing
            for i in range(6)
        ]
        # Should not raise — missing fields default to 0
        profile = extract_supplier_stress_profile(records)
        self.assertIsNotNone(profile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
