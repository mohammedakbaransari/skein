"""Golden-case evaluation gate for SupplierStressAgent (R22, CI-wired).

Runs `EvaluationHarness` against fixed LLM-shaped reasoning text (no live
LLM call) so a prompt/parsing regression is caught by the normal CI test
run, not only by a separate manual review.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.evaluation import EvaluationHarness, GoldenCase
from framework.core.registry import reset_registry
from framework.core.types import Task
from framework.resilience.retry import reset_circuit_registry
from agents.supply_risk.supplier_stress import SupplierStressAgent


class TestSupplierStressAgentEvaluationGate(unittest.TestCase):
    def setUp(self):
        reset_registry()
        reset_circuit_registry()
        self.agent = SupplierStressAgent()
        self.harness = EvaluationHarness(
            self.agent, lambda: Task.create("SupplierStressAgent", {})
        )

    def test_golden_cases_pass(self):
        cases = [
            GoldenCase(
                name="critical_supplier_escalates",
                observations={},
                reasoning_text=json.dumps({
                    "executive_summary": "One supplier in critical distress.",
                    "suppliers": [{
                        "supplier_id": "S001", "supplier_name": "CritCo",
                        "risk_level": "Critical", "composite_score": 11,
                        "advance_warning_estimate_months": 5,
                        "key_finding": "Severe deterioration",
                        "recommended_action": "Qualify alternative",
                        "watch_indicators": [], "intervention_deadline": "14 days",
                    }],
                    "immediate_priorities": ["S001"],
                }),
                expected_min_findings=2,
                expected_severities=("critical",),
            ),
            GoldenCase(
                name="healthy_supplier_no_action_finding",
                observations={},
                reasoning_text=json.dumps({
                    "executive_summary": "Portfolio healthy.",
                    "suppliers": [{
                        "supplier_id": "S002", "supplier_name": "HealthyCo",
                        "risk_level": "Green", "composite_score": 1,
                    }],
                    "immediate_priorities": [],
                }),
                expected_min_findings=1,
            ),
            GoldenCase(
                name="invalid_llm_json_produces_parse_error_finding",
                observations={},
                reasoning_text="not valid json",
                expected_min_findings=1,
                expected_severities=("high",),
            ),
        ]

        results = self.harness.run(cases)
        failures = [r for r in results if not r.passed]
        self.assertEqual(failures, [], f"golden-case regressions: {failures}")
        self.assertTrue(EvaluationHarness.all_passed(results))


if __name__ == "__main__":
    unittest.main()
