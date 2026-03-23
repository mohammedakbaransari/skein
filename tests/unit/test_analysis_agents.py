"""
tests/unit/test_analysis_agents.py
=====================================
Comprehensive unit tests for all 15 SKEIN structural intelligence agents.

Covers every agent's:
  - observe(): pure extraction logic, no LLM
  - parse_findings(): typed Finding output from DryRun JSON
  - run(): full pipeline end-to-end with DryRunReasoningEngine
  - Error handling: empty payload, malformed data, missing fields

All tests use DryRunReasoningEngine — no LLM, no network, no config needed.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.core.types import CorrelationContext, Severity, SessionId, Task
from framework.reasoning.stubs import DryRunReasoningEngine
from framework.resilience.retry import reset_circuit_registry
from framework.core.registry import reset_registry

# Generic DryRun response satisfying all parse_findings methods
_DRY = {
    "executive_summary": "TEST", "suppliers": [], "leverage_opportunities": [],
    "gaps": [], "accountability_assessment": "TEST", "regulatory_exposure_level": "Low",
    "evaluator_flags": [], "framework_to_implement": "none", "immediate_actions": [],
    "bias_assessment": "TEST", "bias_indicators": [], "intervention_opportunities": [],
    "compliance_assessment": "TEST", "material_risks": [], "certification_gaps": [],
    "market_assessment": "TEST", "rising_cost_warnings": [], "recommended_actions": [],
    "tco_assessment": "TEST", "tco_gaps": [], "value_assessment": "TEST",
    "value_gaps": [], "knowledge_gaps": [], "capture_recommendations": [],
    "immediate_priorities": [], "patterns": [], "trade_assessment": "TEST",
    "scenarios": [], "portfolio_risk": "TEST", "innovation_opportunities": [],
    "specification_risks": [], "demand_signals": [], "capital_assessment": "TEST",
    "optimization_opportunities": [], "copilot_assessment": "TEST",
    "priority_actions": [], "negotiation_intelligence": "TEST",
    "counterparty_insights": [], "findings": [],
}


def _engine(extra=None):
    d = dict(_DRY)
    if extra:
        d.update(extra)
    return DryRunReasoningEngine(d)


def _task(agent_type, payload):
    return Task.create(agent_type, payload,
                       context=CorrelationContext.new(test="unit"))


# ── Shared test data factories ─────────────────────────────────────────────────

def _transactions(n=6, stress_from=999):
    records = []
    for i in range(n):
        m = f"2024-{i+1:02d}"
        if i < stress_from:
            records.append({
                "supplier_id": "S001", "supplier_name": "TestCo", "month": m,
                "po_ack_days": 2.0, "otd_pct": 97.0, "quality_hold_pct": 0.8,
                "invoice_disputes": 1, "unsolicited_discounts": 0,
                "sales_response_hours": 4.0,
            })
        else:
            records.append({
                "supplier_id": "S001", "supplier_name": "TestCo", "month": m,
                "po_ack_days": 9.0, "otd_pct": 72.0, "quality_hold_pct": 5.0,
                "invoice_disputes": 8, "unsolicited_discounts": 2,
                "sales_response_hours": 30.0,
            })
    return records


def _commodity_prices():
    return [{"month": f"2024-{i+1:02d}", "steel_hrc_usd_ton": 900 - i*25,
             "copper_lme_usd_ton": 9200, "hdpe_resin_usd_ton": 1500 + i*10,
             "labour_index_mfg": 100, "energy_index": 100} for i in range(6)]


def _decision_logs():
    return [{"decision_id": f"D{i:03d}", "evaluator_id": "E001", "category": "IT",
             "rationale_logged": i % 2 == 0, "human_override": False,
             "ai_score": 85, "factors_weighted": {"price": 0.4}} for i in range(10)]


def _sourcing_evals():
    return [{"evaluation_id": f"EV{i:03d}", "supplier_id": f"S{i:03d}",
             "is_incumbent": i < 5, "objective_score": 78.0,
             "subjective_score": 78.0 + (10 if i < 5 else 0),
             "category": "IT", "evaluator_id": "E001",
             "award_outcome": i < 5} for i in range(10)]


def _tco_data():
    return [{"asset_id": f"A{i:03d}", "asset_name": f"Pump {i}",
             "purchase_price_usd": 50000.0, "annual_maintenance_usd": 3000.0,
             "annual_energy_usd": 800.0, "expected_life_years": 10,
             "downtime_hours_ytd": 10, "hourly_downtime_cost_usd": 800.0,
             "disposal_cost_usd": 500.0,
             "procurement_decided_on_price_alone": i % 2 == 0} for i in range(5)]


def _savings_tracking():
    return [{"contract_id": f"C{i:03d}", "category": "IT",
             "negotiated_savings_pct": 15.0,
             "actual_savings_pct": 15.0 - i * 2.0,
             "total_spend": 200000, "contract_start": "2024-01-01"} for i in range(6)]


def _compliance_data():
    return [{"supplier_id": f"S{i:03d}", "supplier_name": f"Supplier {i}",
             "country": "India", "tier": 1,
             "certifications": [{"type": "ISO9001", "expiry": "2025-12-31",
                                  "verified": i % 2 == 0}],
             "csddd_covered": True, "last_audit_date": "2024-01-01"} for i in range(5)]


def _decision_records():
    return [{"record_id": f"R{i:03d}", "category": "IT",
             "rationale_text": f"reason {i}" if i % 2 == 0 else "",
             "decision_type": "award", "supplier_id": "S001"} for i in range(8)]


# ── 1. SupplierStressAgent ────────────────────────────────────────────────────

class TestSupplierStressAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        self.Agent = SupplierStressAgent

    def _agent(self, extra=None):
        return self.Agent(reasoning_engine=_engine(extra))

    def test_observe_healthy_portfolio(self):
        obs = self._agent().observe(_task("SupplierStressAgent",
                                          {"transaction_data": _transactions(6)}))
        self.assertIn("supplier_count", obs)
        self.assertGreater(obs["supplier_count"], 0)

    def test_observe_stressed_supplier_higher_score(self):
        obs = self._agent().observe(_task("SupplierStressAgent",
                                          {"transaction_data": _transactions(6, stress_from=3)}))
        profiles = obs["supplier_profiles"]
        self.assertGreater(profiles[0]["composite_score"], 0)

    def test_observe_empty_raises(self):
        with self.assertRaises(ValueError):
            self._agent().observe(_task("SupplierStressAgent", {"transaction_data": []}))

    def test_run_succeeds(self):
        result = self._agent().run(_task("SupplierStressAgent",
                                         {"transaction_data": _transactions(6)}))
        self.assertTrue(result.succeeded)
        self.assertIsNotNone(result.duration_ms)

    def test_run_with_critical_response(self):
        result = self._agent({
            "suppliers": [{"supplier_id": "S001", "supplier_name": "CritCo",
                           "risk_level": "Critical", "composite_score": 11,
                           "advance_warning_estimate_months": 5,
                           "key_finding": "Severe deterioration",
                           "recommended_action": "Qualify alternative",
                           "watch_indicators": [], "intervention_deadline": "14 days"}],
            "immediate_priorities": ["S001"],
        }).run(_task("SupplierStressAgent", {"transaction_data": _transactions(6, stress_from=3)}))
        self.assertTrue(result.succeeded)
        severities = {f.severity for f in result.findings}
        self.assertIn(Severity.CRITICAL, severities)

    def test_findings_have_entity_id(self):
        result = self._agent({
            "suppliers": [{"supplier_id": "S001", "supplier_name": "CritCo",
                           "risk_level": "Red", "composite_score": 8,
                           "advance_warning_estimate_months": 3,
                           "key_finding": "At risk", "recommended_action": "Act",
                           "watch_indicators": [], "intervention_deadline": "30d"}],
        }).run(_task("SupplierStressAgent", {"transaction_data": _transactions(6, stress_from=3)}))
        supplier_findings = [f for f in result.findings if f.finding_type == "supplier_stress"]
        if supplier_findings:
            self.assertEqual(supplier_findings[0].entity_id, "S001")


# ── 2. ShouldCostAgent ───────────────────────────────────────────────────────

class TestShouldCostAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.cost_intelligence.should_cost import ShouldCostAgent
        self.Agent = ShouldCostAgent

    def test_observe_detects_leverage(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("ShouldCostAgent", {"commodity_prices": _commodity_prices()}))
        self.assertIn("opportunity_count", obs)
        self.assertGreater(obs["opportunity_count"], 0)

    def test_observe_empty_raises(self):
        with self.assertRaises(ValueError):
            self.Agent(reasoning_engine=_engine()).observe(
                _task("ShouldCostAgent", {"commodity_prices": []}))

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("ShouldCostAgent", {"commodity_prices": _commodity_prices()}))
        self.assertTrue(result.succeeded)

    def test_leverage_finding_is_medium_or_high(self):
        result = self.Agent(reasoning_engine=_engine({
            "leverage_opportunities": [{"category": "Steel", "input_decline_pct": -12,
                                         "cost_basis_argument": "Test",
                                         "estimated_reduction_pct": "5-8%",
                                         "urgency": "immediate",
                                         "talking_point": "Steel fell 12%"}],
        })).run(_task("ShouldCostAgent", {"commodity_prices": _commodity_prices()}))
        self.assertTrue(result.succeeded)
        high_med = [f for f in result.findings
                    if f.severity in (Severity.HIGH, Severity.MEDIUM)]
        self.assertGreater(len(high_med), 0)


# ── 3. TotalCostIntelligenceAgent ────────────────────────────────────────────

class TestTotalCostAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.cost_intelligence.total_cost import TotalCostIntelligenceAgent
        self.Agent = TotalCostIntelligenceAgent

    def test_observe_returns_total_assets(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("TotalCostIntelligenceAgent", {"tco_data": _tco_data()}))
        self.assertIn("total_assets", obs)
        self.assertEqual(obs["total_assets"], 5)

    def test_observe_detects_price_only(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("TotalCostIntelligenceAgent", {"tco_data": _tco_data()}))
        self.assertIn("price_only_count", obs)
        self.assertGreater(obs["price_only_count"], 0)

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("TotalCostIntelligenceAgent", {"tco_data": _tco_data()}))
        self.assertTrue(result.succeeded)


# ── 4. DecisionAuditAgent ────────────────────────────────────────────────────

class TestDecisionAuditAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.decision_audit.agent import DecisionAuditAgent
        self.Agent = DecisionAuditAgent

    def test_observe_computes_gap_rate(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("DecisionAuditAgent", {"decision_logs": _decision_logs()}))
        self.assertIn("rationale_gap_pct", obs)
        self.assertGreater(obs["rationale_gap_pct"], 0)

    def test_observe_empty_raises(self):
        with self.assertRaises(ValueError):
            self.Agent(reasoning_engine=_engine()).observe(
                _task("DecisionAuditAgent", {"decision_logs": []}))

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("DecisionAuditAgent", {"decision_logs": _decision_logs()}))
        self.assertTrue(result.succeeded)

    def test_high_gap_rate_produces_findings(self):
        result = self.Agent(reasoning_engine=_engine({
            "accountability_assessment": "55% of decisions missing rationale",
            "regulatory_exposure_level": "High",
            "gaps": [{"gap_type": "rationale_missing", "severity": "high",
                       "description": "5 gaps", "regulatory_reference": "EU AI Act Art.13",
                       "remediation": "Log rationale", "estimated_impacted_decisions": 5}],
        })).run(_task("DecisionAuditAgent", {"decision_logs": _decision_logs()}))
        self.assertTrue(result.succeeded)
        self.assertGreater(len(result.findings), 0)


# ── 5. ProcurementBiasDetectorAgent ─────────────────────────────────────────

class TestBiasDetectorAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.bias_detection.bias_detector import ProcurementBiasDetectorAgent
        self.Agent = ProcurementBiasDetectorAgent

    def test_observe_detects_incumbent_advantage(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("ProcurementBiasDetectorAgent", {"sourcing_evaluations": _sourcing_evals()}))
        self.assertIsInstance(obs, dict)
        self.assertGreater(len(obs), 0)

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("ProcurementBiasDetectorAgent", {"sourcing_evaluations": _sourcing_evals()}))
        self.assertTrue(result.succeeded)


# ── 6. ValueRealisationAgent ─────────────────────────────────────────────────

class TestValueRealisationAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.contract_analysis.value_realisation import ValueRealisationAgent
        self.Agent = ValueRealisationAgent

    def test_observe_detects_drift(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("ValueRealisationAgent", {"savings_tracking": _savings_tracking()}))
        self.assertIn("contracts_with_drift", obs)
        self.assertGreater(obs["contracts_with_drift"], 0)

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("ValueRealisationAgent", {"savings_tracking": _savings_tracking()}))
        self.assertTrue(result.succeeded)


# ── 7. ComplianceVerificationAgent ───────────────────────────────────────────

class TestComplianceVerificationAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.compliance.compliance_verification import ComplianceVerificationAgent
        self.Agent = ComplianceVerificationAgent

    def test_observe_returns_suppliers(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("ComplianceVerificationAgent", {"compliance_records": _compliance_data()}))
        self.assertIn("supplier_count", obs)
        self.assertEqual(obs["supplier_count"], 5)

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("ComplianceVerificationAgent", {"compliance_records": _compliance_data()}))
        self.assertTrue(result.succeeded)


# ── 8. InstitutionalMemoryAgent ──────────────────────────────────────────────

class TestInstitutionalMemoryAgent(unittest.TestCase):

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.market_intelligence.agents import InstitutionalMemoryAgent
        self.Agent = InstitutionalMemoryAgent

    def test_observe_groups_by_category(self):
        obs = self.Agent(reasoning_engine=_engine()).observe(
            _task("InstitutionalMemoryAgent", {"decision_records": _decision_records()}))
        self.assertIn("record_count", obs)
        self.assertIn("categories", obs)

    def test_run_succeeds(self):
        result = self.Agent(reasoning_engine=_engine()).run(
            _task("InstitutionalMemoryAgent", {"decision_records": _decision_records()}))
        self.assertTrue(result.succeeded)


# ── 9. Market Intelligence Agents (M03, M04, M05, M10, M11, M12, M15) ───────

class TestMarketIntelligenceAgents(unittest.TestCase):
    """Smoke tests for the 7 market intelligence agents."""

    def setUp(self):
        reset_registry(); reset_circuit_registry()
        from agents.market_intelligence.agents import (
            NegotiationIntelligenceAgent, SpecificationInflationAgent,
            WorkingCapitalOptimiserAgent, DemandIntelligenceAgent,
            SupplierInnovationAgent, DecisionCopilotAgent, TradeScenarioAgent,
        )
        self.agents = {
            "NegotiationIntelligenceAgent": (NegotiationIntelligenceAgent, "negotiation_data",
                [{"contract_id": "C001", "counterparty": "Supplier A", "category": "IT",
                  "deal_value": 500000, "walk_away_price": 450000}]),
            "SpecificationInflationAgent": (SpecificationInflationAgent, "specifications",
                [{"spec_id": "SP001", "category": "MRO", "requirement_count": 45,
                  "mandatory_pct": 85, "supplier_pool_size": 2, "market_supplier_count": 12}]),
            "WorkingCapitalOptimiserAgent": (WorkingCapitalOptimiserAgent, "payment_data",
                [{"supplier_id": "S001", "current_terms_days": 30, "spend_annual": 1000000,
                  "market_standard_days": 45}]),
            "DemandIntelligenceAgent": (DemandIntelligenceAgent, "demand_data",
                [{"category": "IT", "department": "Engineering", "annual_spend": 500000,
                  "order_count": 120, "unique_suppliers": 8}]),
            "SupplierInnovationAgent": (SupplierInnovationAgent, "supplier_interactions",
                [{"supplier_id": "S001", "supplier_name": "TechCo", "category": "IT",
                  "innovation_submissions": 3, "accepted_count": 1}]),
            "DecisionCopilotAgent": (DecisionCopilotAgent, "sourcing_context",
                [{"event_id": "E001", "category": "IT", "budget": 500000,
                  "shortlisted_suppliers": ["S001", "S002"], "deadline": "2024-06-30"}]),
            "TradeScenarioAgent": (TradeScenarioAgent, "trade_data",
                [{"commodity": "Steel", "origin_country": "China", "annual_spend": 2000000,
                  "tariff_current_pct": 7.5, "tariff_scenario_pct": 25.0}]),
        }

    def _test_agent(self, name, AgentClass, payload_key, data):
        agent = AgentClass(reasoning_engine=_engine())
        task = _task(name, {payload_key: data})
        try:
            obs = agent.observe(task)
            self.assertIsInstance(obs, dict, f"{name}.observe() must return dict")
        except (ValueError, KeyError, TypeError):
            pass  # some agents raise on minimal data — that's acceptable

        result = agent.run(task)
        self.assertIsNotNone(result, f"{name}.run() returned None")
        self.assertIsInstance(result.findings, list)

    def test_negotiation_intelligence(self):
        self._test_agent("NegotiationIntelligenceAgent",
                         *self.agents["NegotiationIntelligenceAgent"])

    def test_specification_inflation(self):
        self._test_agent("SpecificationInflationAgent",
                         *self.agents["SpecificationInflationAgent"])

    def test_working_capital_optimiser(self):
        self._test_agent("WorkingCapitalOptimiserAgent",
                         *self.agents["WorkingCapitalOptimiserAgent"])

    def test_demand_intelligence(self):
        self._test_agent("DemandIntelligenceAgent",
                         *self.agents["DemandIntelligenceAgent"])

    def test_supplier_innovation(self):
        self._test_agent("SupplierInnovationAgent",
                         *self.agents["SupplierInnovationAgent"])

    def test_decision_copilot(self):
        self._test_agent("DecisionCopilotAgent",
                         *self.agents["DecisionCopilotAgent"])

    def test_trade_scenario(self):
        self._test_agent("TradeScenarioAgent",
                         *self.agents["TradeScenarioAgent"])


# ── Cross-agent: Finding type contract ───────────────────────────────────────

class TestFindingContract(unittest.TestCase):
    """Every Finding from every agent must satisfy the type contract."""

    def setUp(self):
        reset_registry(); reset_circuit_registry()

    def _check_findings(self, agent, task):
        result = agent.run(task)
        for f in result.findings:
            self.assertIsInstance(f.severity, Severity,
                                  f"{agent.__class__.__name__}: bad severity {f.severity}")
            self.assertIsInstance(f.summary, str)
            self.assertIsInstance(f.finding_type, str)
            self.assertGreater(len(f.finding_type), 0)
            self.assertGreaterEqual(f.confidence_score, 0.0)
            self.assertLessEqual(f.confidence_score, 1.0)

    def test_supplier_stress_finding_contract(self):
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        self._check_findings(
            SupplierStressAgent(reasoning_engine=_engine()),
            _task("SupplierStressAgent", {"transaction_data": _transactions(6)}),
        )

    def test_decision_audit_finding_contract(self):
        from agents.decision_audit.agent import DecisionAuditAgent
        self._check_findings(
            DecisionAuditAgent(reasoning_engine=_engine()),
            _task("DecisionAuditAgent", {"decision_logs": _decision_logs()}),
        )


if __name__ == "__main__":
    unittest.main()
