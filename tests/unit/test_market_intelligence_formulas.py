"""
tests/unit/test_market_intelligence_formulas.py
==================================================
Hand-computed numeric-correctness tests for the pure observe() logic of all
8 agents/market_intelligence/agents.py agents — completes roadmap item P2-2
(the remaining agents deferred in the first P2-2 pass).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.market_intelligence.agents import (
    DecisionCopilotAgent, DemandIntelligenceAgent, InstitutionalMemoryAgent,
    NegotiationIntelligenceAgent, SpecificationInflationAgent,
    SupplierInnovationAgent, TradeScenarioAgent, WorkingCapitalOptimiserAgent,
)
from framework.core.types import Task


def _task(agent_type, payload):
    return Task.create(agent_type, payload)


class TestInstitutionalMemoryAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = InstitutionalMemoryAgent()
        self.payload = {
            "decision_records": [
                {"category": "IT", "rationale_text": "because X"},
                {"category": "IT", "rationale_text": ""},
                {"category": "Facilities", "rationale_text": "reason Y"},
            ]
        }

    def test_grouping_and_rationale_pct(self):
        obs = self.agent.observe(_task("InstitutionalMemoryAgent", self.payload))
        self.assertEqual(obs["record_count"], 3)
        self.assertEqual(obs["categories"], ["IT", "Facilities"])
        self.assertEqual(obs["category_distribution"], {"IT": 2, "Facilities": 1})
        self.assertEqual(obs["has_rationale_pct"], 66.7)
        self.assertEqual(len(obs["sample_decisions"]), 3)

    def test_missing_decision_records_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("InstitutionalMemoryAgent", {}))


class TestNegotiationIntelligenceAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = NegotiationIntelligenceAgent()
        self.payload = {
            "supplier_id": "S1", "supplier_name": "Acme",
            "negotiation_history": [
                {"price_concession_pct": 5}, {"price_concession_pct": 10},
                {"price_concession_pct": None},
            ],
            "supplier_financials": {"revenue": 1000},
            "negotiation_context": {"deal_size": 50000},
        }

    def test_concession_stats(self):
        obs = self.agent.observe(_task("NegotiationIntelligenceAgent", self.payload))
        self.assertEqual(obs["negotiation_count"], 3)
        self.assertEqual(obs["avg_price_concession_pct"], 7.5)
        self.assertEqual(obs["max_price_concession_pct"], 10)
        self.assertTrue(obs["financials_available"])
        self.assertEqual(len(obs["historical_outcomes"]), 3)

    def test_missing_supplier_id_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("NegotiationIntelligenceAgent", {"negotiation_history": []}))


class TestSpecificationInflationAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = SpecificationInflationAgent()
        self.payload = {
            "category": "IT",
            "specification": {"requirements": [
                {"description": "X", "suppliers_qualifying": 1},
                {"description": "Y", "suppliers_qualifying": 5},
                {"description": "Z", "suppliers_qualifying": 2},
            ]},
            "supplier_database": [
                {"can_qualify": True}, {"can_qualify": True}, {"can_qualify": True},
                {"can_qualify": False}, {"can_qualify": False},
            ],
        }

    def test_competitive_pool_and_high_risk_requirements(self):
        obs = self.agent.observe(_task("SpecificationInflationAgent", self.payload))
        self.assertEqual(obs["requirement_count"], 3)
        self.assertEqual(obs["qualifiable_count"], 3)
        self.assertEqual(obs["total_suppliers_db"], 5)
        self.assertEqual(obs["competitive_pool_pct"], 60.0)
        self.assertEqual(
            [r["description"] for r in obs["high_risk_requirements"]], ["X", "Z"]
        )

    def test_missing_specification_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("SpecificationInflationAgent", {"specification": {}}))


class TestWorkingCapitalOptimiserAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = WorkingCapitalOptimiserAgent()
        self.payload = {
            "suppliers_with_terms": [
                {"supplier_id": "S1", "supplier_name": "Acme", "annual_spend_usd": 100_000,
                 "current_payment_terms_days": 45, "health_score": 7, "is_critical": True},
                {"supplier_id": "S2", "supplier_name": "Beta"},  # relies on defaults
            ],
            "treasury_position": {"cash": 5_000_000},
            "scf_facilities": ["facility1"],
        }

    def test_supplier_reshaping_and_defaults(self):
        obs = self.agent.observe(_task("WorkingCapitalOptimiserAgent", self.payload))
        self.assertEqual(obs["supplier_count"], 2)
        self.assertTrue(obs["scf_available"])
        s1, s2 = obs["suppliers"]
        self.assertEqual(s1["annual_spend"], 100_000)
        self.assertEqual(s1["current_terms"], 45)
        self.assertTrue(s1["critical_tier"])
        # S2 has none of the optional fields -> defaults apply
        self.assertEqual(s2["annual_spend"], 0)
        self.assertEqual(s2["current_terms"], 30)
        self.assertEqual(s2["health_score"], 5)
        self.assertFalse(s2["critical_tier"])

    def test_missing_suppliers_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("WorkingCapitalOptimiserAgent", {"suppliers_with_terms": []}))


class TestDemandIntelligenceAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = DemandIntelligenceAgent()
        self.payload = {
            "macro_indicators": [
                {"name": "A", "change_pct": 6.0},
                {"name": "B", "change_pct": -2.0},
                {"name": "C", "change_pct": -7.0},
                {"name": "D", "change_pct": 3.0},
            ],
            "category_mappings": {"cat1": ["x"]},
        }

    def test_significant_moves_threshold_is_absolute_five_pct(self):
        obs = self.agent.observe(_task("DemandIntelligenceAgent", self.payload))
        self.assertEqual(obs["indicator_count"], 4)
        self.assertEqual(obs["significant_moves"], 2)
        self.assertEqual(obs["category_count"], 1)
        self.assertEqual([s["name"] for s in obs["leading_signals"]], ["A", "C"])

    def test_missing_macro_indicators_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("DemandIntelligenceAgent", {"macro_indicators": []}))


class TestSupplierInnovationAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = SupplierInnovationAgent()
        self.payload = {
            "buyer_strategic_agenda": {"priorities": ["AI", "sustainability"]},
            "supplier_innovation_signals": [
                {"supplier_id": "S1", "relevance_score": 0.8},
                {"supplier_id": "S1", "relevance_score": 0.3},   # below 0.6 threshold
                {"supplier_id": "S2", "relevance_score": 0.9},
                {"supplier_id": "S3", "relevance_score": 0.6},   # exactly at threshold -> included
            ],
        }

    def test_relevance_threshold_and_ranking(self):
        obs = self.agent.observe(_task("SupplierInnovationAgent", self.payload))
        self.assertEqual(obs["supplier_count"], 3)   # unique supplier_ids across ALL signals
        self.assertEqual(obs["signal_count"], 4)
        self.assertEqual(obs["relevant_signals"], 3)  # 0.8, 0.9, 0.6 qualify; 0.3 does not
        self.assertEqual(
            [s["supplier_id"] for s in obs["top_signals"]], ["S2", "S1", "S3"]
        )  # sorted by relevance_score descending: 0.9, 0.8, 0.6

    def test_missing_signals_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("SupplierInnovationAgent", {"supplier_innovation_signals": []}))


class TestDecisionCopilotAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = DecisionCopilotAgent()
        self.payload = {
            "user_context": {"role": "category_manager"},
            "pending_alerts": [
                {"id": "A1", "severity": "critical", "type": "t1", "summary": "s1"},
                {"id": "A2", "severity": "high", "type": "t2", "summary": "s2"},
                {"id": "A3", "severity": "high", "type": "t3", "summary": "s3"},
                {"id": "A4", "severity": "medium", "type": "t4", "summary": "s4"},
                {"id": "A5", "severity": "low", "type": "t5", "summary": "s5"},
            ],
        }

    def test_severity_distribution(self):
        obs = self.agent.observe(_task("DecisionCopilotAgent", self.payload))
        self.assertEqual(obs["alert_count"], 5)
        self.assertEqual(
            obs["by_severity"], {"critical": 1, "high": 2, "medium": 1, "low": 1}
        )
        self.assertEqual(len(obs["alert_summaries"]), 5)

    def test_missing_alerts_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("DecisionCopilotAgent", {"pending_alerts": []}))


class TestTradeScenarioAgentObserve(unittest.TestCase):

    def setUp(self):
        self.agent = TradeScenarioAgent()
        self.payload = {
            "sourcing_network": [
                {"supplier_name": "A", "country": "CN", "annual_spend_usd": 500_000, "tariff_exposure_pct": 15},
                {"supplier_name": "B", "country": "US", "annual_spend_usd": 200_000, "tariff_exposure_pct": 5},
                {"supplier_name": "C", "country": "VN", "annual_spend_usd": 300_000, "tariff_exposure_pct": 20},
            ],
            "trade_scenarios": [{"name": "S1", "description": "d1", "probability": 0.3}],
        }

    def test_exposure_threshold_is_strictly_greater_than_ten_pct(self):
        obs = self.agent.observe(_task("TradeScenarioAgent", self.payload))
        self.assertEqual(obs["network_size"], 3)
        self.assertEqual(obs["scenario_count"], 1)
        self.assertEqual(obs["exposed_relationships"], 2)   # A (15%) and C (20%); B (5%) excluded
        self.assertEqual(obs["total_exposed_spend"], 800_000)

    def test_missing_sourcing_network_raises(self):
        with self.assertRaises(ValueError):
            self.agent.observe(_task("TradeScenarioAgent", {"sourcing_network": []}))


if __name__ == "__main__":
    unittest.main()
