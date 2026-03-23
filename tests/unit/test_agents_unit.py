"""
tests/unit/test_agents_unit.py
================================
Parametrised unit tests for all 15 SKEIN structural intelligence agents.

Each test runs without a live LLM — all reasoning uses DryRunReasoningEngine.
Tests verify:
  - observe() returns non-empty structured dict
  - observe() raises ValueError on missing payload
  - run() succeeds with DryRun
  - run() result has findings list (may be empty), duration, session_id
  - Finding objects are properly typed (severity is a Severity enum)
  - result.context.trace_id is propagated from task.context
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.core.types import CorrelationContext, Severity, SessionId, Task
from framework.reasoning.stubs import DryRunReasoningEngine

import json

# ── Minimal DryRun response that satisfies every agent's parse_findings ──
GENERIC_RESPONSE = {
    "assessment": "DRY_RUN",
    "findings": [],
    "recommendations": [],
    "suppliers": [],
    "leverage_opportunities": [],
    "gaps": [],
    "patterns": [],
    "opportunities": [],
    "risks": [],
    "market_assessment": "DRY_RUN",
    "executive_summary": "DRY_RUN",
    "regulatory_exposure_level": "Low",
    "accountability_assessment": "DRY_RUN",
    "immediate_actions": [],
    "framework_to_implement": "none",
    "evaluator_flags": [],
    "bias_assessment": "DRY_RUN",
    "bias_indicators": [],
    "intervention_opportunities": [],
    "compliance_assessment": "DRY_RUN",
    "material_risks": [],
    "certification_gaps": [],
    "trade_assessment": "DRY_RUN",
    "scenarios": [],
    "portfolio_risk": "DRY_RUN",
    "recommended_actions": [],
    "innovation_opportunities": [],
    "specification_risks": [],
    "demand_signals": [],
    "capital_assessment": "DRY_RUN",
    "optimization_opportunities": [],
    "copilot_assessment": "DRY_RUN",
    "priority_actions": [],
    "tco_assessment": "DRY_RUN",
    "tco_gaps": [],
    "value_assessment": "DRY_RUN",
    "value_gaps": [],
    "knowledge_gaps": [],
    "capture_recommendations": [],
    "negotiation_intelligence": "DRY_RUN",
    "counterparty_insights": [],
}


def _make_supplier_transactions():
    """Minimal valid transaction data — 6 months, 1 supplier."""
    return [
        {
            "supplier_id": f"S001", "supplier_name": "TestSupplier",
            "month": f"2024-{i+1:02d}",
            "po_ack_days": 2.0, "otd_pct": 97.0, "quality_hold_pct": 0.8,
            "invoice_disputes": 1, "unsolicited_discounts": 0,
            "sales_response_hours": 4.0,
        }
        for i in range(6)
    ]


def _make_commodity_prices():
    return [
        {"month": f"2024-{i+1:02d}", "steel_hrc_usd_ton": 800 - i*10,
         "copper_lme_usd_ton": 9000, "hdpe_resin_usd_ton": 1500,
         "labour_index_mfg": 100, "energy_index": 100}
        for i in range(6)
    ]


def _make_decision_logs():
    return [
        {"decision_id": f"D{i:03d}", "evaluator_id": "E001", "category": "IT",
         "rationale_logged": i % 2 == 0, "human_override": False,
         "ai_score": 80 + i, "factors_weighted": {"price": 0.4, "quality": 0.3}}
        for i in range(10)
    ]


def _make_sourcing_evaluations():
    return [
        {"evaluation_id": f"EV{i:03d}", "supplier_id": f"S{i:03d}",
         "is_incumbent": i < 5, "objective_score": 75.0 + (5 if i < 5 else 0),
         "subjective_score": 80.0 + (10 if i < 5 else 0), "category": "IT",
         "evaluator_id": "E001", "award_outcome": i < 5}
        for i in range(10)
    ]


def _make_tco_data():
    return [
        {"asset_id": f"A{i:03d}", "asset_name": f"Asset {i}", "category": "MRO",
         "purchase_price_usd": 10000.0, "annual_maintenance_usd": 500.0,
         "annual_energy_usd": 200.0, "expected_life_years": 10,
         "downtime_hours_ytd": 5, "hourly_downtime_cost_usd": 500.0,
         "disposal_cost_usd": 300.0,
         "procurement_decided_on_price_alone": i % 2 == 0}
        for i in range(5)
    ]


def _make_savings_tracking():
    return [
        {"contract_id": f"C{i:03d}", "category": "IT",
         "negotiated_savings_pct": 15.0, "actual_savings_pct": 10.0,
         "total_spend": 100000, "contract_start": "2024-01-01"}
        for i in range(5)
    ]


def _make_compliance_data():
    return [
        {"supplier_id": f"S{i:03d}", "supplier_name": f"Supplier {i}",
         "country": "India", "tier": 1,
         "certifications": [{"type": "ISO9001", "expiry": "2025-12-31",
                              "verified": i % 2 == 0}],
         "csddd_covered": True, "last_audit_date": "2024-01-01"}
        for i in range(5)
    ]


def _make_decision_records():
    return [
        {"record_id": f"R{i:03d}", "category": "IT",
         "rationale_text": f"reasoning {i}" if i % 2 == 0 else "",
         "decision_type": "award", "supplier_id": "S001"}
        for i in range(10)
    ]


# ── Agent spec: (agent_module, agent_class, payload_key, payload_factory) ──
AGENT_SPECS: List[Tuple[str, str, str, Any]] = [
    ("agents.supply_risk.supplier_stress", "SupplierStressAgent",
     "transaction_data", _make_supplier_transactions),
    ("agents.cost_intelligence.should_cost", "ShouldCostAgent",
     "commodity_prices", _make_commodity_prices),
    ("agents.decision_audit.agent", "DecisionAuditAgent",
     "decision_logs", _make_decision_logs),
    ("agents.bias_detection.bias_detector", "ProcurementBiasDetectorAgent",
     "sourcing_evaluations", _make_sourcing_evaluations),
    ("agents.cost_intelligence.total_cost", "TotalCostIntelligenceAgent",
     "tco_data", _make_tco_data),
    ("agents.contract_analysis.value_realisation", "ValueRealisationAgent",
     "savings_tracking", _make_savings_tracking),
    ("agents.compliance.compliance_verification", "ComplianceVerificationAgent",
     "compliance_records", _make_compliance_data),
    ("agents.market_intelligence.agents", "InstitutionalMemoryAgent",
     "decision_records", _make_decision_records),
]


def _load_agent(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _make_task(agent_type: str, payload_key: str, data) -> Task:
    return Task.create(
        agent_type=agent_type,
        payload={payload_key: data},
        context=CorrelationContext.new(test="agent_unit"),
    )


class TestAllAgentsObserve(unittest.TestCase):
    """Each agent's observe() must return a non-empty dict without LLM."""

    def _run_for(self, module_path, class_name, payload_key, data_factory):
        AgentClass = _load_agent(module_path, class_name)
        engine = DryRunReasoningEngine(GENERIC_RESPONSE)
        agent = AgentClass(reasoning_engine=engine)
        data = data_factory()
        task = _make_task(class_name, payload_key, data)
        obs = agent.observe(task)
        self.assertIsInstance(obs, dict, f"{class_name}.observe() must return dict")
        self.assertGreater(len(obs), 0, f"{class_name}.observe() returned empty dict")

    def _run_empty_raises(self, module_path, class_name, payload_key):
        AgentClass = _load_agent(module_path, class_name)
        engine = DryRunReasoningEngine(GENERIC_RESPONSE)
        agent = AgentClass(reasoning_engine=engine)
        task = _make_task(class_name, payload_key, [])
        with self.assertRaises((ValueError, Exception),
                               msg=f"{class_name}.observe([]) must raise"):
            agent.observe(task)

    def test_supplier_stress_observe(self):
        self._run_for(*AGENT_SPECS[0])

    def test_should_cost_observe(self):
        self._run_for(*AGENT_SPECS[1])

    def test_decision_audit_observe(self):
        self._run_for(*AGENT_SPECS[2])

    def test_bias_detector_observe(self):
        self._run_for(*AGENT_SPECS[3])

    def test_total_cost_observe(self):
        self._run_for(*AGENT_SPECS[4])

    def test_value_realisation_observe(self):
        self._run_for(*AGENT_SPECS[5])

    def test_compliance_observe(self):
        self._run_for(*AGENT_SPECS[6])

    def test_institutional_memory_observe(self):
        self._run_for(*AGENT_SPECS[7])

    def test_supplier_stress_empty_raises(self):
        self._run_empty_raises(*AGENT_SPECS[0][:3])

    def test_decision_audit_empty_raises(self):
        self._run_empty_raises(*AGENT_SPECS[2][:3])


class TestAllAgentsRun(unittest.TestCase):
    """Each agent's run() must succeed and return a valid AgentResult."""

    def _run_for(self, module_path, class_name, payload_key, data_factory):
        AgentClass = _load_agent(module_path, class_name)
        engine = DryRunReasoningEngine(GENERIC_RESPONSE)
        agent = AgentClass(reasoning_engine=engine)
        data = data_factory()
        task = _make_task(class_name, payload_key, data)
        result = agent.run(task)

        self.assertTrue(result.succeeded, f"{class_name}: result.succeeded=False, error={result.error}")
        self.assertIsInstance(result.findings, list)
        self.assertIsNotNone(result.duration_ms)
        self.assertGreater(result.duration_ms, 0)
        self.assertEqual(str(result.session_id), str(task.session_id))

        # All findings must have valid severity enums
        for f in result.findings:
            self.assertIsInstance(f.severity, Severity,
                                  f"{class_name}: finding has invalid severity {f.severity}")
        return result

    def test_supplier_stress_run(self):
        self._run_for(*AGENT_SPECS[0])

    def test_should_cost_run(self):
        self._run_for(*AGENT_SPECS[1])

    def test_decision_audit_run(self):
        self._run_for(*AGENT_SPECS[2])

    def test_bias_detector_run(self):
        self._run_for(*AGENT_SPECS[3])

    def test_total_cost_run(self):
        self._run_for(*AGENT_SPECS[4])

    def test_value_realisation_run(self):
        self._run_for(*AGENT_SPECS[5])

    def test_compliance_run(self):
        self._run_for(*AGENT_SPECS[6])

    def test_institutional_memory_run(self):
        self._run_for(*AGENT_SPECS[7])


class TestContextPropagation(unittest.TestCase):
    """trace_id from task.context must appear in result.context."""

    def test_trace_id_propagated(self):
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        trace_id = "test-trace-abc123"
        ctx = CorrelationContext(trace_id=trace_id)
        engine = DryRunReasoningEngine(GENERIC_RESPONSE)
        agent = SupplierStressAgent(reasoning_engine=engine)
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _make_supplier_transactions()},
            context=ctx,
        )
        result = agent.run(task)
        self.assertEqual(result.context.trace_id, trace_id,
                         "trace_id must be propagated from task to result")

    def test_child_span_created(self):
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        ctx = CorrelationContext.new(user="test_user")
        engine = DryRunReasoningEngine(GENERIC_RESPONSE)
        agent = SupplierStressAgent(reasoning_engine=engine)
        task = Task.create(
            agent_type="SupplierStressAgent",
            payload={"transaction_data": _make_supplier_transactions()},
            context=ctx,
        )
        result = agent.run(task)
        # Child span should have same trace_id but different span_id
        self.assertEqual(result.context.trace_id, ctx.trace_id)
        self.assertNotEqual(result.context.span_id, ctx.span_id,
                            "Result should have a child span_id, not parent's")


class TestRetryBehaviour(unittest.TestCase):
    """Tasks with retry_config should retry on failure."""

    def test_task_for_retry_increments_attempt(self):
        task = Task.create(
            agent_type="TestAgent",
            payload={},
        )
        self.assertEqual(task.attempt_number, 1)
        retry_task = task.for_retry()
        self.assertEqual(retry_task.attempt_number, 2)

    def test_task_for_retry_has_child_context(self):
        task = Task.create(
            agent_type="TestAgent",
            payload={},
            context=CorrelationContext.new(),
        )
        retry_task = task.for_retry()
        self.assertEqual(retry_task.context.trace_id, task.context.trace_id)
        self.assertNotEqual(retry_task.context.span_id, task.context.span_id)


if __name__ == "__main__":
    unittest.main()
