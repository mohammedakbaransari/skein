"""
tests/unit/test_analysis_agents.py
====================================
Unit tests for the five data-driven agents.

Covers pure signal extraction functions and full dry-run integration
for ShouldCostAgent, ValueRealisationAgent, TotalCostIntelligenceAgent,
ProcurementBiasDetectorAgent, and DecisionAuditAgent.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.cost_intelligence.should_cost import (
    ShouldCostAgent, compute_commodity_movements,
)
from agents.contract_analysis.value_realisation import (
    ValueRealisationAgent, analyse_savings_portfolio,
)
from agents.cost_intelligence.total_cost import (
    TotalCostIntelligenceAgent, analyse_tco_portfolio,
)
from agents.bias_detection.bias_detector import (
    ProcurementBiasDetectorAgent, analyse_evaluation_bias,
)
from agents.decision_audit.agent import (
    DecisionAuditAgent, compute_accountability_metrics,
)
from framework.core.types import Severity, SessionId, Task
from framework.reasoning.stubs import DryRunReasoningEngine


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def _price_records(months: int = 6) -> List[Dict]:
    base = {"steel_hrc_usd_ton": 800, "copper_lme_usd_ton": 8500,
            "hdpe_resin_usd_ton": 1200, "labour_index_mfg": 100, "energy_index": 100}
    # Steel and copper decline over time; labour rises
    records = []
    for m in range(months):
        records.append({
            "month": f"2024-{m+1:02d}",
            "steel_hrc_usd_ton":  round(800 * (1 - 0.02 * m), 2),
            "copper_lme_usd_ton": round(8500 * (1 - 0.015 * m), 2),
            "hdpe_resin_usd_ton": round(1200 * (1 + 0.012 * m), 2),  # +7.2% over 6 months → triggers rising warning
            "labour_index_mfg":   round(100 * (1 + 0.005 * m), 2),
            "energy_index":       round(100 * (1 - 0.008 * m), 2),
        })
    return records


def _savings_records() -> List[Dict]:
    records = []
    # Contract 1: significant leakage
    for m in range(6):
        records.append({
            "contract_id": "CTR-001", "category": "Packaging",
            "month": f"2024-{m+1:02d}",
            "negotiated_savings_pct": 12.0,
            "actual_savings_pct": 12.0 - m * 1.5,
            "leakage_amount_usd": 50000 * m,
            "leakage_causes": ["maverick_spend", "erp_not_updated"] if m > 2 else [],
        })
    # Contract 2: healthy
    for m in range(6):
        records.append({
            "contract_id": "CTR-002", "category": "Logistics",
            "month": f"2024-{m+1:02d}",
            "negotiated_savings_pct": 8.0,
            "actual_savings_pct": 7.8,
            "leakage_amount_usd": 2000,
            "leakage_causes": [],
        })
    return records


def _tco_assets() -> List[Dict]:
    return [
        {"asset_id": "A001", "asset_type": "industrial_pump",
         "purchase_price_usd": 45000, "total_tco_usd": 189000,
         "annual_energy_cost_usd": 8000, "annual_maintenance_cost_usd": 12000,
         "annual_downtime_risk_usd": 15000, "lifecycle_years": 10,
         "procurement_decided_on_price_alone": True},
        {"asset_id": "A002", "asset_type": "compressor",
         "purchase_price_usd": 82000, "total_tco_usd": 205000,
         "annual_energy_cost_usd": 6000, "annual_maintenance_cost_usd": 9000,
         "annual_downtime_risk_usd": 5000, "lifecycle_years": 15,
         "procurement_decided_on_price_alone": False},
        {"asset_id": "A003", "asset_type": "industrial_pump",
         "purchase_price_usd": 38000, "total_tco_usd": 190000,
         "annual_energy_cost_usd": 10000, "annual_maintenance_cost_usd": 14000,
         "annual_downtime_risk_usd": 20000, "lifecycle_years": 10,
         "procurement_decided_on_price_alone": True},
    ]


def _eval_records() -> List[Dict]:
    """
    Each evaluator scores all supplier types — required for bias differential computation.
    5 evaluators × 6 evaluations each (2 incumbent, 2 new_entrant, 2 diverse_owned) = 30 records.
    """
    records = []
    types_cycle = ["incumbent", "new_entrant", "diverse_owned",
                   "incumbent", "new_entrant", "diverse_owned"]
    for ev in range(1, 6):  # 5 evaluators
        for j, stype in enumerate(types_cycle):
            obj  = 72.0 if stype == "incumbent" else 70.0
            subj = 85.0 if stype == "incumbent" else 67.0  # clear incumbent inflation
            idx  = (ev - 1) * 6 + j
            records.append({
                "evaluation_id": f"EVAL-{idx:03d}", "supplier_id": f"S{idx:03d}",
                "supplier_type": stype, "evaluator_id": f"EVA-{ev:02d}",
                "category": "MRO", "objective_score": obj, "subjective_score": subj,
                "combined_score": (obj + subj) / 2,
                "awarded": stype == "incumbent",
                "factors_weighted": {"price": 0.35 + (0.1 if stype == "incumbent" else 0)},
            })
    return records


def _decision_records() -> List[Dict]:
    records = []
    for i in range(20):
        records.append({
            "decision_id": f"DEC-{i:03d}",
            "category": "IT Hardware" if i % 2 == 0 else "Facilities",
            "evaluator_id": f"EVA-0{i % 3 + 1}",
            "ai_score": 75 + (i % 20),
            "human_override": i % 5 == 0,
            "rationale_logged": i % 4 != 0,   # 25% missing rationale
            "awarded": True,
            "factors_weighted": {"price": 0.3 + (0.1 * (i % 3))},
        })
    return records


# ---------------------------------------------------------------------------
# Should-Cost unit tests
# ---------------------------------------------------------------------------

class TestShouldCostAnalysis(unittest.TestCase):

    def test_declining_commodities_detected_as_leverage(self):
        model = compute_commodity_movements(_price_records(6), leverage_threshold_pct=-3.0)
        self.assertGreater(len(model.leverage_opportunities), 0)

    def test_rising_commodity_flagged_as_warning(self):
        model = compute_commodity_movements(_price_records(6))
        # HDPE and labour rise in our test data
        self.assertGreater(len(model.rising_cost_warnings), 0)

    def test_insufficient_data_returns_empty_model(self):
        model = compute_commodity_movements([_price_records(1)[0]])
        self.assertEqual(model.periods_analysed, 0)

    def test_model_is_immutable(self):
        model = compute_commodity_movements(_price_records(6))
        with self.assertRaises((AttributeError, TypeError)):
            model.periods_analysed = 999  # type: ignore

    def test_leverage_level_thresholds(self):
        model = compute_commodity_movements(_price_records(6))
        for m in model.commodity_movements:
            self.assertIn(m.leverage_level, ("High", "Medium", "Low", "None"))


class TestShouldCostAgentIntegration(unittest.TestCase):

    def _engine(self):
        return DryRunReasoningEngine({"market_assessment": "Test result.",
                                       "leverage_opportunities": [
                                           {"category": "Steel", "input_decline_pct": -12.0,
                                            "cost_basis_argument": "Steel down 12%",
                                            "estimated_reduction_pct": "8-10%",
                                            "urgency": "immediate",
                                            "talking_point": "Your steel input costs fell 12%"}],
                                       "rising_cost_warnings": ["HDPE"],
                                       "recommended_actions": ["Review steel contracts"]})

    def test_full_run_succeeds(self):
        agent  = ShouldCostAgent(reasoning_engine=self._engine())
        task   = Task.create("ShouldCostAgent", {"commodity_prices": _price_records()})
        result = agent.run(task)
        self.assertTrue(result.succeeded, result.error)
        self.assertGreater(len(result.findings), 0)

    def test_metadata_correct(self):
        self.assertEqual(ShouldCostAgent.METADATA.agent_type, "ShouldCostAgent")
        self.assertIn("mystery_06", ShouldCostAgent.METADATA.mystery_refs)


# ---------------------------------------------------------------------------
# Value Realisation unit tests
# ---------------------------------------------------------------------------

class TestValueRealisationAnalysis(unittest.TestCase):

    def test_leakage_detected_for_deteriorating_contract(self):
        profiles = analyse_savings_portfolio(_savings_records())
        ctr1 = next(p for p in profiles if p.contract_id == "CTR-001")
        self.assertGreater(ctr1.leakage_pct, 0)
        self.assertIn(ctr1.alert_level, ("critical", "high", "medium"))

    def test_healthy_contract_has_low_alert(self):
        profiles = analyse_savings_portfolio(_savings_records())
        ctr2 = next(p for p in profiles if p.contract_id == "CTR-002")
        self.assertLessEqual(ctr2.leakage_pct, 1.0)

    def test_profiles_sorted_by_leakage_descending(self):
        profiles = analyse_savings_portfolio(_savings_records())
        usd = [p.cumulative_leakage_usd for p in profiles]
        self.assertEqual(usd, sorted(usd, reverse=True))

    def test_trend_computed(self):
        profiles = analyse_savings_portfolio(_savings_records())
        for p in profiles:
            self.assertIn(p.trend, ("Improving", "Stable", "Deteriorating"))

    def test_empty_records_returns_empty(self):
        self.assertEqual(analyse_savings_portfolio([]), [])


class TestValueRealisationAgentIntegration(unittest.TestCase):

    def _engine(self):
        return DryRunReasoningEngine({"portfolio_assessment": "Test.",
                                       "cfo_credibility_risk": "Moderate.",
                                       "interventions": [], "systemic_fixes": [],
                                       "immediate_actions": []})

    def test_full_run_succeeds(self):
        agent  = ValueRealisationAgent(reasoning_engine=self._engine())
        task   = Task.create("ValueRealisationAgent", {"savings_tracking": _savings_records()})
        result = agent.run(task)
        self.assertTrue(result.succeeded, result.error)

    def test_metadata_correct(self):
        self.assertEqual(ValueRealisationAgent.METADATA.agent_type, "ValueRealisationAgent")
        self.assertIn("mystery_11", ValueRealisationAgent.METADATA.mystery_refs)


# ---------------------------------------------------------------------------
# TCO unit tests
# ---------------------------------------------------------------------------

class TestTCOAnalysis(unittest.TestCase):

    def test_price_only_pct_detected(self):
        summary = analyse_tco_portfolio(_tco_assets())
        self.assertGreater(summary.price_only_pct, 0)
        self.assertEqual(summary.price_only_count, 2)  # A001 and A003

    def test_avg_ratio_above_one(self):
        summary = analyse_tco_portfolio(_tco_assets())
        self.assertGreater(summary.avg_tco_to_price_ratio, 1.0)

    def test_at_risk_usd_for_price_only_assets(self):
        summary = analyse_tco_portfolio(_tco_assets())
        self.assertGreater(summary.total_value_at_risk_usd, 0)

    def test_category_breakdown_populated(self):
        summary = analyse_tco_portfolio(_tco_assets())
        self.assertIn("industrial_pump", summary.category_breakdown)

    def test_empty_assets_returns_zero_summary(self):
        summary = analyse_tco_portfolio([])
        self.assertEqual(summary.total_assets, 0)

    def test_profiles_sorted_by_at_risk_descending(self):
        summary = analyse_tco_portfolio(_tco_assets())
        risks = [p.lifecycle_value_at_risk_usd for p in summary.asset_profiles]
        self.assertEqual(risks, sorted(risks, reverse=True))


class TestTCOAgentIntegration(unittest.TestCase):

    def _engine(self):
        return DryRunReasoningEngine({"tco_assessment": "Test.", "financial_impact": "2.3M at risk.",
                                       "findings": [], "process_gaps": [], "immediate_priorities": []})

    def test_full_run_succeeds(self):
        agent  = TotalCostIntelligenceAgent(reasoning_engine=self._engine())
        task   = Task.create("TotalCostIntelligenceAgent", {"tco_data": _tco_assets()})
        result = agent.run(task)
        self.assertTrue(result.succeeded, result.error)

    def test_metadata_correct(self):
        self.assertEqual(TotalCostIntelligenceAgent.METADATA.agent_type, "TotalCostIntelligenceAgent")
        self.assertIn("mystery_14", TotalCostIntelligenceAgent.METADATA.mystery_refs)


# ---------------------------------------------------------------------------
# Bias Detector unit tests
# ---------------------------------------------------------------------------

class TestBiasAnalysis(unittest.TestCase):

    def test_incumbent_award_rate_higher(self):
        result = analyse_evaluation_bias(_eval_records())
        self.assertGreater(result.award_rate_gap_pct, 0)

    def test_incumbent_subjective_premium_positive(self):
        result = analyse_evaluation_bias(_eval_records())
        inc = next((m for m in result.supplier_type_stats if m.supplier_type == "incumbent"), None)
        self.assertIsNotNone(inc)
        self.assertGreater(inc.subjective_premium, 0)

    def test_all_supplier_types_present(self):
        result = analyse_evaluation_bias(_eval_records())
        types  = {m.supplier_type for m in result.supplier_type_stats}
        self.assertIn("incumbent",    types)
        self.assertIn("new_entrant",  types)
        self.assertIn("diverse_owned", types)

    def test_evaluator_bias_metrics_computed(self):
        result = analyse_evaluation_bias(_eval_records())
        self.assertGreater(len(result.evaluator_bias_metrics), 0)

    def test_empty_evaluations_returns_empty(self):
        result = analyse_evaluation_bias([])
        self.assertEqual(result.total_evaluations, 0)


class TestBiasDetectorAgentIntegration(unittest.TestCase):

    def _engine(self):
        return DryRunReasoningEngine({"bias_assessment": "Test.", "incumbent_finding": "73pp premium.",
                                       "diverse_sme_finding": "Suppressed.", "patterns": [],
                                       "evaluator_flags": [], "systemic_risks": [],
                                       "immediate_actions": []})

    def test_full_run_succeeds(self):
        agent  = ProcurementBiasDetectorAgent(reasoning_engine=self._engine())
        task   = Task.create("ProcurementBiasDetectorAgent", {"sourcing_evaluations": _eval_records()})
        result = agent.run(task)
        self.assertTrue(result.succeeded, result.error)

    def test_metadata_correct(self):
        self.assertEqual(ProcurementBiasDetectorAgent.METADATA.agent_type, "ProcurementBiasDetectorAgent")
        self.assertIn("mystery_15", ProcurementBiasDetectorAgent.METADATA.mystery_refs)


# ---------------------------------------------------------------------------
# Decision Audit unit tests
# ---------------------------------------------------------------------------

class TestDecisionAuditAnalysis(unittest.TestCase):

    def test_rationale_gap_detected(self):
        metrics = compute_accountability_metrics(_decision_records())
        self.assertGreater(metrics.rationale_gap_pct, 0)

    def test_override_rate_in_valid_range(self):
        metrics = compute_accountability_metrics(_decision_records())
        self.assertGreaterEqual(metrics.override_rate_pct, 0)
        self.assertLessEqual(metrics.override_rate_pct, 100)

    def test_evaluator_metrics_populated(self):
        metrics = compute_accountability_metrics(_decision_records())
        self.assertGreater(len(metrics.evaluator_metrics), 0)

    def test_empty_returns_zero(self):
        metrics = compute_accountability_metrics([])
        self.assertEqual(metrics.total_decisions, 0)

    def test_high_risk_decisions_identified(self):
        # Decisions with ai_score >= 85 and no rationale should be flagged
        records = _decision_records()
        # Add a high-score, no-rationale decision
        records.append({"decision_id": "DEC-HIGH", "category": "IT",
                         "evaluator_id": "EVA-01", "ai_score": 92,
                         "human_override": False, "rationale_logged": False,
                         "awarded": True, "factors_weighted": {"price": 0.35}})
        metrics = compute_accountability_metrics(records)
        self.assertIn("DEC-HIGH", metrics.high_risk_decision_ids)


class TestDecisionAuditAgentIntegration(unittest.TestCase):

    def _engine(self):
        return DryRunReasoningEngine({"accountability_assessment": "Test.",
                                       "regulatory_exposure_level": "High",
                                       "gaps": [{"gap_type": "rationale_missing",
                                                  "severity": "high",
                                                  "description": "25% missing rationale",
                                                  "regulatory_reference": "EU AI Act Art. 13",
                                                  "remediation": "Mandate rationale capture",
                                                  "estimated_impacted_decisions": 5}],
                                       "evaluator_flags": [], "immediate_actions": [],
                                       "framework_to_implement": "Decision Audit Trail v1"})

    def test_full_run_succeeds(self):
        agent  = DecisionAuditAgent(reasoning_engine=self._engine())
        task   = Task.create("DecisionAuditAgent", {"decision_logs": _decision_records()})
        result = agent.run(task)
        self.assertTrue(result.succeeded, result.error)
        self.assertGreater(len(result.findings), 0)

    def test_governance_finding_has_correct_severity(self):
        agent  = DecisionAuditAgent(reasoning_engine=self._engine())
        task   = Task.create("DecisionAuditAgent", {"decision_logs": _decision_records()})
        result = agent.run(task)
        gov = next((f for f in result.findings if f.finding_type == "governance_assessment"), None)
        self.assertIsNotNone(gov)
        self.assertIn(gov.severity, {Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM})

    def test_accountability_gap_findings_present(self):
        agent  = DecisionAuditAgent(reasoning_engine=self._engine())
        task   = Task.create("DecisionAuditAgent", {"decision_logs": _decision_records()})
        result = agent.run(task)
        gaps = [f for f in result.findings if f.finding_type == "accountability_gap"]
        self.assertGreater(len(gaps), 0)

    def test_metadata_correct(self):
        self.assertEqual(DecisionAuditAgent.METADATA.agent_type, "DecisionAuditAgent")
        self.assertIn("mystery_13", DecisionAuditAgent.METADATA.mystery_refs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
