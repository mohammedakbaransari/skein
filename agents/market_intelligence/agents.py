"""
agents/market_intelligence/agents.py
======================================
Mysteries 01, 03, 04, 05, 07, 08, 10, 12

Eight agents sharing the market_intelligence subpackage.
Each is a complete StructuralAgent / DecisionAgent implementation.

Design decision: Where several agents share a subpackage, each agent class
is defined in this module and exported individually. For large agents that
will accumulate significant domain logic (e.g. InstitutionalMemoryAgent),
split into a dedicated subpackage as complexity grows.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework.agents.base import DecisionAgent, StructuralAgent
from framework.agents.catalogue import (
    DEMAND_INTELLIGENCE_METADATA,
    DECISION_COPILOT_METADATA,
    INSTITUTIONAL_MEMORY_METADATA,
    NEGOTIATION_INTELLIGENCE_METADATA,
    SPECIFICATION_INFLATION_METADATA,
    SUPPLIER_INNOVATION_METADATA,
    TRADE_SCENARIO_METADATA,
    WORKING_CAPITAL_METADATA,
)
from framework.core.types import Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dry_json(agent_name: str) -> str:
    return json.dumps({
        "assessment": f"{agent_name}: reasoning engine not configured.",
        "findings": [], "recommendations": []
    })


def _call_reasoning(agent, system: str, user: str, obs: Dict, task: Task) -> str:
    if not agent.reasoning:
        return _dry_json(agent.name)
    resp = agent.reasoning.reason(ReasoningRequest(
        system_prompt=system, user_prompt=user,
        observations=obs, strategy=ReasoningStrategy.STRUCTURED,
        session_id=str(task.session_id),
    ))
    return resp.content


# ===========================================================================
# Mystery 01 — Institutional Memory
# ===========================================================================

_IM_SYSTEM = """You are a procurement knowledge management specialist.
You extract and structure the reasoning patterns behind procurement decisions —
not the decisions themselves, but the situational logic that drove them.
Output ONLY valid JSON."""


class InstitutionalMemoryAgent(DecisionAgent):
    """
    Mystery 01 — Institutional Memory Capture Agent.

    Extracts reasoning patterns from decision logs and practitioner narratives.
    Stores patterns in the institutional memory layer for future retrieval.
    The goal: when the expert leaves, the reasoning stays.
    """
    METADATA = INSTITUTIONAL_MEMORY_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        records = task.payload.get("decision_records", [])
        if not records:
            raise ValueError("Payload must contain 'decision_records'")
        # Group decisions by category and supplier to identify patterns
        by_category: Dict[str, List] = {}
        for r in records:
            by_category.setdefault(r.get("category", "Unknown"), []).append(r)
        return {
            "record_count": len(records),
            "categories": list(by_category.keys()),
            "category_distribution": {c: len(rs) for c, rs in by_category.items()},
            "has_rationale_pct": round(
                100 * sum(1 for r in records if r.get("rationale_text")) / len(records), 1
            ) if records else 0.0,
            "sample_decisions": records[:5],  # first 5 for pattern priming
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        user = (
            f"Extract procurement reasoning patterns from {observations['record_count']} "
            f"decision records across {len(observations['categories'])} categories.\n\n"
            f"Categories: {', '.join(observations['categories'][:10])}\n"
            f"Decisions with documented rationale: {observations['has_rationale_pct']}%\n\n"
            f"Sample decisions:\n{json.dumps(observations['sample_decisions'], indent=2)}\n\n"
            "Return JSON:\n"
            '{"patterns": [{"category": "...", "pattern_type": "...", '
            '"reasoning_template": "...", "situational_triggers": [...], '
            '"decision_heuristic": "..."}], '
            '"knowledge_gaps": ["areas with no documented reasoning"], '
            '"capture_recommendations": ["steps to accelerate expert knowledge capture"]}'
        )
        return _call_reasoning(self, _IM_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        patterns = parsed.get("patterns", [])
        # Persist extracted patterns to institutional memory
        for p in patterns:
            key = f"pattern:{p.get('category','?')}:{p.get('pattern_type','?')}"
            self.remember(key, p)
        findings = [self._make_finding(
            finding_type="knowledge_patterns_extracted",
            severity=Severity.INFO,
            summary=f"Extracted {len(patterns)} reasoning patterns from {observations['record_count']} decisions",
            evidence={"patterns_count": len(patterns),
                      "knowledge_gaps": parsed.get("knowledge_gaps", []),
                      "capture_recommendations": parsed.get("capture_recommendations", [])}
        )]
        for kgap in parsed.get("knowledge_gaps", []):
            findings.append(self._make_finding(
                finding_type="knowledge_gap",
                severity=Severity.MEDIUM,
                summary=f"Knowledge gap: {kgap}",
                recommended_action="Schedule expert knowledge capture session",
            ))
        return findings


# ===========================================================================
# Mystery 03 — Negotiation Intelligence
# ===========================================================================

_NI_SYSTEM = """You are a negotiation intelligence analyst with 20 years of
experience in high-value B2B procurement negotiations.

You build counterparty profiles from historical data to inform negotiation
strategy for specific supplier relationships. Your output is specific,
tactical, and grounded in the data provided.

Output ONLY valid JSON."""


class NegotiationIntelligenceAgent(StructuralAgent):
    """
    Mystery 03 — Counterparty Negotiation Intelligence Agent.

    Builds per-supplier negotiation models from concession patterns,
    authority structure, financial position, and historical movement.
    Generates tactics calibrated to this specific counterparty.
    """
    METADATA = NEGOTIATION_INTELLIGENCE_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        supplier_id  = task.payload.get("supplier_id")
        history      = task.payload.get("negotiation_history", [])
        financials   = task.payload.get("supplier_financials", {})
        context      = task.payload.get("negotiation_context", {})
        if not supplier_id:
            raise ValueError("Payload must contain 'supplier_id'")

        # Extract concession pattern from history
        price_concessions = [
            r.get("price_concession_pct", 0) for r in history
            if r.get("price_concession_pct") is not None
        ]
        avg_concession = round(sum(price_concessions) / len(price_concessions), 2) if price_concessions else 0.0

        return {
            "supplier_id":          supplier_id,
            "supplier_name":        task.payload.get("supplier_name", supplier_id),
            "negotiation_count":    len(history),
            "avg_price_concession_pct": avg_concession,
            "max_price_concession_pct": max(price_concessions, default=0.0),
            "financials_available": bool(financials),
            "negotiation_context":  context,
            "historical_outcomes":  history[-3:] if history else [],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        user = (
            f"Build negotiation strategy for {observations['supplier_name']} ({observations['supplier_id']}).\n\n"
            f"Historical data: {observations['negotiation_count']} negotiations\n"
            f"Average price concession: {observations['avg_price_concession_pct']}%\n"
            f"Maximum achieved: {observations['max_price_concession_pct']}%\n"
            f"Current context: {json.dumps(observations['negotiation_context'])}\n\n"
            "Return JSON:\n"
            '{"counterparty_profile": {"authority_structure": "...", "decision_style": "...",'
            '"financial_pressure_indicators": [...]}, '
            '"opening_strategy": "...", "target_position": "...", "walkaway_indicators": [...],'
            '"tactical_moves": [{"situation": "...", "tactic": "...", "rationale": "..."}],'
            '"risk_factors": ["..."], "confidence_level": "high|medium|low"}'
        )
        return _call_reasoning(self, _NI_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        conf_sev = {"high": Severity.INFO, "medium": Severity.LOW, "low": Severity.MEDIUM}
        return [self._make_finding(
            finding_type="negotiation_strategy",
            severity=conf_sev.get(parsed.get("confidence_level", "medium"), Severity.LOW),
            entity_id=observations.get("supplier_id"),
            entity_name=observations.get("supplier_name"),
            summary=parsed.get("opening_strategy", ""),
            detail=parsed.get("target_position", ""),
            evidence={"counterparty_profile": parsed.get("counterparty_profile", {}),
                      "tactical_moves": parsed.get("tactical_moves", []),
                      "risk_factors": parsed.get("risk_factors", [])},
        )]


# ===========================================================================
# Mystery 04 — Specification Inflation
# ===========================================================================

_SI_SYSTEM = """You are a procurement specification analyst specialising in
competitive market design.

You identify specification requirements that artificially limit competitive
tension — requirements that, as written, qualify only one or two suppliers
globally. Your job is to surface these before the specification is locked,
not after the RFP has produced a non-competitive result.

Output ONLY valid JSON."""


class SpecificationInflationAgent(StructuralAgent):
    """
    Mystery 04 — Specification Inflation Trap Agent.

    Operates upstream in specification workflows to detect requirements
    that unnecessarily restrict the qualified supplier pool.
    """
    METADATA = SPECIFICATION_INFLATION_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        spec       = task.payload.get("specification", {})
        category   = task.payload.get("category", "Unknown")
        supplier_db = task.payload.get("supplier_database", [])
        if not spec:
            raise ValueError("Payload must contain 'specification'")
        requirements = spec.get("requirements", [])
        qualifiable  = [s for s in supplier_db if s.get("can_qualify", False)]
        return {
            "category":            category,
            "requirement_count":   len(requirements),
            "requirements":        requirements,
            "total_suppliers_db":  len(supplier_db),
            "qualifiable_count":   len(qualifiable),
            "competitive_pool_pct": round(100 * len(qualifiable) / max(len(supplier_db), 1), 1),
            "high_risk_requirements": [
                r for r in requirements if r.get("suppliers_qualifying", 999) <= 2
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        req_block = "\n".join(
            f"  - {r.get('description', 'unknown')}: qualifies {r.get('suppliers_qualifying','?')} suppliers"
            for r in observations.get("requirements", [])
        ) or "  No requirements provided"
        user = (
            f"Analyse specification for {observations['category']}.\n"
            f"Total suppliers in database: {observations['total_suppliers_db']}\n"
            f"Qualifiable under current spec: {observations['qualifiable_count']} "
            f"({observations['competitive_pool_pct']}%)\n\n"
            f"REQUIREMENTS:\n{req_block}\n\n"
            "Return JSON:\n"
            '{"competitive_assessment": "...", "inflated_requirements": '
            '[{"requirement": "...", "inflation_type": "proprietary_interface|'
            'unnecessary_certification|over_specified_tolerance|incumbent_reference",'
            '"suppliers_qualifying": 0, "recommended_alternative": "..."}],'
            '"estimated_competitive_pool_increase": "...", '
            '"overall_risk": "critical|high|medium|low"}'
        )
        return _call_reasoning(self, _SI_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        risk_map = {"critical": Severity.CRITICAL, "high": Severity.HIGH,
                    "medium": Severity.MEDIUM, "low": Severity.LOW}
        findings = [self._make_finding(
            finding_type="specification_assessment",
            severity=risk_map.get(parsed.get("overall_risk", "medium"), Severity.MEDIUM),
            entity_name=observations.get("category"),
            summary=parsed.get("competitive_assessment", ""),
            evidence={"competitive_pool_pct": observations.get("competitive_pool_pct"),
                      "pool_increase_estimate": parsed.get("estimated_competitive_pool_increase")},
        )]
        for req in parsed.get("inflated_requirements", []):
            findings.append(self._make_finding(
                finding_type="inflated_requirement",
                severity=Severity.HIGH if req.get("suppliers_qualifying", 99) <= 1 else Severity.MEDIUM,
                summary=f"Specification restricts to {req.get('suppliers_qualifying','?')} suppliers: {req.get('requirement','')}",
                recommended_action=req.get("recommended_alternative", ""),
                evidence={"inflation_type": req.get("inflation_type")},
            ))
        return findings


# ===========================================================================
# Mystery 05 — Working Capital Triangle
# ===========================================================================

_WC_SYSTEM = """You are a procurement-finance integration specialist focused
on working capital optimisation.

You treat payment terms, supplier financial health, and treasury position
as one continuous optimisation problem. Your recommendations are specific,
quantified, and account for the real cost of supplier disruption.

Output ONLY valid JSON."""


class WorkingCapitalOptimiserAgent(DecisionAgent):
    """
    Mystery 05 — Working Capital Triangle Optimisation Agent.

    Computes the optimal payment terms given current supplier health signals
    and buyer treasury position — making the trade-off calculation that
    procurement and treasury currently make separately, if at all.
    """
    METADATA = WORKING_CAPITAL_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        suppliers       = task.payload.get("suppliers_with_terms", [])
        treasury_pos    = task.payload.get("treasury_position", {})
        scf_available   = task.payload.get("scf_facilities", [])
        if not suppliers:
            raise ValueError("Payload must contain 'suppliers_with_terms'")
        return {
            "supplier_count":        len(suppliers),
            "treasury_position":     treasury_pos,
            "scf_available":         bool(scf_available),
            "scf_facilities":        scf_available,
            "suppliers": [{
                "supplier_id":    s.get("supplier_id"),
                "supplier_name":  s.get("supplier_name"),
                "annual_spend":   s.get("annual_spend_usd", 0),
                "current_terms":  s.get("current_payment_terms_days", 30),
                "health_score":   s.get("health_score", 5),
                "critical_tier":  s.get("is_critical", False),
            } for s in suppliers],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        supp_block = "\n".join(
            f"  {s['supplier_name']}: ${s['annual_spend']:,.0f}/yr | "
            f"terms {s['current_terms']}d | health {s['health_score']}/10 | "
            f"critical: {s['critical_tier']}"
            for s in observations.get("suppliers", [])
        )
        user = (
            f"Optimise working capital triangle for {observations['supplier_count']} suppliers.\n"
            f"Treasury position: {json.dumps(observations['treasury_position'])}\n"
            f"SCF facilities available: {observations['scf_available']}\n\n"
            f"SUPPLIERS:\n{supp_block}\n\n"
            "Return JSON:\n"
            '{"portfolio_assessment": "...", "recommendations": '
            '[{"supplier_id": "...", "current_terms_days": 0, "recommended_terms_days": 0,'
            '"rationale": "...", "cost_of_extension_usd": 0, '
            '"cost_of_supplier_loss_usd": 0, "net_benefit_usd": 0,'
            '"scf_eligible": false, "urgency": "immediate|normal"}],'
            '"total_working_capital_impact_usd": 0}'
        )
        return _call_reasoning(self, _WC_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        findings = [self._make_finding(
            finding_type="working_capital_assessment",
            severity=Severity.MEDIUM,
            summary=parsed.get("portfolio_assessment", ""),
            evidence={"total_impact_usd": parsed.get("total_working_capital_impact_usd", 0)},
        )]
        for rec in parsed.get("recommendations", []):
            urgency_sev = Severity.HIGH if rec.get("urgency") == "immediate" else Severity.LOW
            findings.append(self._make_finding(
                finding_type="payment_terms_optimisation",
                severity=urgency_sev,
                entity_id=rec.get("supplier_id"),
                summary=f"Terms: {rec.get('current_terms_days')}d → {rec.get('recommended_terms_days')}d | Net benefit: ${rec.get('net_benefit_usd',0):,.0f}",
                recommended_action=rec.get("rationale", ""),
                evidence={"cost_extension": rec.get("cost_of_extension_usd"),
                          "cost_of_loss": rec.get("cost_of_supplier_loss_usd"),
                          "scf_eligible": rec.get("scf_eligible")},
            ))
        return findings


# ===========================================================================
# Mystery 07 — Pre-Signal Demand Intelligence
# ===========================================================================

_DI_SYSTEM = """You are a strategic procurement analyst specialising in
leading-indicator demand intelligence.

You identify exogenous signals that precede demand materialisation in the
order book by 3-6 months — enabling procurement to position before competitors
have the same signal. Output ONLY valid JSON."""


class DemandIntelligenceAgent(StructuralAgent):
    """
    Mystery 07 — Pre-Signal Demand Intelligence Agent.

    Monitors exogenous leading indicators to generate procurement positioning
    recommendations 3-6 months before demand appears in the order book.
    """
    METADATA = DEMAND_INTELLIGENCE_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        indicators = task.payload.get("macro_indicators", [])
        categories = task.payload.get("category_mappings", {})
        if not indicators:
            raise ValueError("Payload must contain 'macro_indicators'")
        # Find indicators that moved significantly
        signals = [
            ind for ind in indicators
            if abs(ind.get("change_pct", 0)) >= 5.0
        ]
        return {
            "indicator_count":   len(indicators),
            "significant_moves": len(signals),
            "category_count":    len(categories),
            "leading_signals":   signals[:10],
            "category_mappings": categories,
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        sig_block = "\n".join(
            f"  {s.get('name','?')}: {s.get('change_pct',0):+.1f}% — "
            f"relates to: {', '.join(s.get('related_categories', []))}"
            for s in observations.get("leading_signals", [])
        ) or "  No significant signals"
        user = (
            f"Analyse {observations['indicator_count']} macro indicators for demand intelligence.\n"
            f"Significant moves: {observations['significant_moves']}\n\n"
            f"LEADING SIGNALS:\n{sig_block}\n\n"
            "Return JSON:\n"
            '{"demand_outlook": "...", "positioning_recommendations": '
            '[{"category": "...", "signal": "...", "direction": "increase|decrease",'
            '"confidence": "high|medium|low", "lead_time_months": 0,'
            '"recommended_action": "..."}], '
            '"urgent_positions": ["categories to act on this month"]}'
        )
        return _call_reasoning(self, _DI_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        conf_sev = {"high": Severity.HIGH, "medium": Severity.MEDIUM, "low": Severity.LOW}
        findings = [self._make_finding(
            finding_type="demand_intelligence_outlook",
            severity=Severity.INFO,
            summary=parsed.get("demand_outlook", ""),
            evidence={"urgent_positions": parsed.get("urgent_positions", [])},
        )]
        for rec in parsed.get("positioning_recommendations", []):
            findings.append(self._make_finding(
                finding_type="demand_positioning",
                severity=conf_sev.get(rec.get("confidence", "medium"), Severity.MEDIUM),
                entity_name=rec.get("category"),
                summary=f"{rec.get('direction','?').title()} demand signal: {rec.get('signal','')}",
                recommended_action=rec.get("recommended_action", ""),
                evidence={"lead_time_months": rec.get("lead_time_months"),
                          "direction": rec.get("direction")},
            ))
        return findings


# ===========================================================================
# Mystery 08 — Supplier Innovation
# ===========================================================================

_SI2_SYSTEM = """You are a supplier innovation intelligence analyst.

You surface relevant supplier R&D and innovation capability — from patents,
product announcements, and capability signals — before suppliers present it
to the buying organisation. You cross-reference supplier innovation against
the buyer's product roadmap and strategic agenda.

Output ONLY valid JSON."""


class SupplierInnovationAgent(StructuralAgent):
    """
    Mystery 08 — Supplier Innovation Intelligence Agent.

    Monitors patent filings, product announcements, and R&D hiring across
    the supplier base and surfaces relevant innovation before suppliers
    think to bring it.
    """
    METADATA = SUPPLIER_INNOVATION_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        supplier_signals = task.payload.get("supplier_innovation_signals", [])
        buyer_agenda     = task.payload.get("buyer_strategic_agenda", {})
        if not supplier_signals:
            raise ValueError("Payload must contain 'supplier_innovation_signals'")
        relevant = [s for s in supplier_signals if s.get("relevance_score", 0) >= 0.6]
        return {
            "supplier_count":     len({s.get("supplier_id") for s in supplier_signals}),
            "signal_count":       len(supplier_signals),
            "relevant_signals":   len(relevant),
            "buyer_priorities":   buyer_agenda.get("priorities", []),
            "top_signals":        sorted(relevant, key=lambda s: s.get("relevance_score", 0), reverse=True)[:8],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        sig_block = "\n".join(
            f"  {s.get('supplier_name','?')}: {s.get('signal_type','?')} — "
            f"{s.get('description','')} (relevance: {s.get('relevance_score',0):.0%})"
            for s in observations.get("top_signals", [])
        ) or "  No highly relevant signals"
        user = (
            f"Identify supplier innovation opportunities from {observations['signal_count']} signals.\n"
            f"Buyer priorities: {', '.join(observations['buyer_priorities'])}\n\n"
            f"TOP RELEVANT SIGNALS:\n{sig_block}\n\n"
            "Return JSON:\n"
            '{"innovation_summary": "...", "engagement_opportunities": '
            '[{"supplier_id": "...", "supplier_name": "...", "innovation_area": "...",'
            '"buyer_relevance": "...", "recommended_engagement": "...",'
            '"potential_value": "..."}], '
            '"missed_opportunities_risk": "...", "suggested_rfis": [...]}'
        )
        return _call_reasoning(self, _SI2_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        findings = [self._make_finding(
            finding_type="innovation_landscape",
            severity=Severity.INFO,
            summary=parsed.get("innovation_summary", ""),
            evidence={"missed_risk": parsed.get("missed_opportunities_risk"),
                      "suggested_rfis": parsed.get("suggested_rfis", [])},
        )]
        for opp in parsed.get("engagement_opportunities", []):
            findings.append(self._make_finding(
                finding_type="innovation_opportunity",
                severity=Severity.MEDIUM,
                entity_id=opp.get("supplier_id"), entity_name=opp.get("supplier_name"),
                summary=f"{opp.get('innovation_area','?')}: {opp.get('buyer_relevance','')}",
                recommended_action=opp.get("recommended_engagement", ""),
                evidence={"potential_value": opp.get("potential_value")},
            ))
        return findings


# ===========================================================================
# Mystery 10 — Decision Co-Pilot (Cognitive Load)
# ===========================================================================

_CP_SYSTEM = """You are a procurement decision prioritisation assistant.

You review pending alerts and recommendations and identify which ones genuinely
require human judgment this week. You provide the context needed to decide well
and defer or auto-handle everything that does not require human attention.

Output ONLY valid JSON."""


class DecisionCopilotAgent(StructuralAgent):
    """
    Mystery 10 — Category Manager Decision Co-Pilot.

    Prioritises AI-generated outputs for human review. Surfaces decisions
    that require human judgment, provides context, and defers or handles
    everything else. Reduces cognitive load rather than adding to it.
    """
    METADATA = DECISION_COPILOT_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        alerts     = task.payload.get("pending_alerts", [])
        context    = task.payload.get("user_context", {})
        if not alerts:
            raise ValueError("Payload must contain 'pending_alerts'")
        return {
            "alert_count":       len(alerts),
            "by_severity":       {
                "critical": sum(1 for a in alerts if a.get("severity") == "critical"),
                "high":     sum(1 for a in alerts if a.get("severity") == "high"),
                "medium":   sum(1 for a in alerts if a.get("severity") == "medium"),
                "low":      sum(1 for a in alerts if a.get("severity") == "low"),
            },
            "user_context":      context,
            "alert_summaries":   [{"id": a.get("id"), "severity": a.get("severity"),
                                    "type": a.get("type"), "summary": a.get("summary", "")}
                                   for a in alerts],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        alerts_block = "\n".join(
            f"  [{a['severity']}] {a['type']}: {a['summary']}"
            for a in observations.get("alert_summaries", [])
        )
        user = (
            f"Prioritise {observations['alert_count']} pending procurement alerts.\n"
            f"Severity distribution: {json.dumps(observations['by_severity'])}\n"
            f"User context: {json.dumps(observations['user_context'])}\n\n"
            f"ALERTS:\n{alerts_block}\n\n"
            "Return JSON:\n"
            '{"this_week_decisions": [{"alert_id": "...", "why_human_needed": "...",'
            '"decision_context": "...", "options": ["option A", "option B"]}],'
            '"auto_handled": [{"alert_id": "...", "action_taken": "..."}],'
            '"deferred": [{"alert_id": "...", "defer_until": "..."}],'
            '"cognitive_load_reduction_pct": 0}'
        )
        return _call_reasoning(self, _CP_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        human_decisions = parsed.get("this_week_decisions", [])
        findings = [self._make_finding(
            finding_type="decision_queue",
            severity=Severity.INFO,
            summary=f"{len(human_decisions)} decisions require human judgment this week",
            evidence={"cognitive_load_reduction_pct": parsed.get("cognitive_load_reduction_pct"),
                      "auto_handled_count": len(parsed.get("auto_handled", [])),
                      "deferred_count":     len(parsed.get("deferred", []))},
        )]
        for dec in human_decisions:
            findings.append(self._make_finding(
                finding_type="human_decision_required",
                severity=Severity.HIGH,
                entity_id=dec.get("alert_id"),
                summary=dec.get("why_human_needed", ""),
                detail=dec.get("decision_context", ""),
                evidence={"options": dec.get("options", [])},
            ))
        return findings


# ===========================================================================
# Mystery 12 — Trade Policy Scenario Intelligence
# ===========================================================================

_TS_SYSTEM = """You are a geopolitical trade risk and scenario planning specialist
for enterprise procurement.

You maintain a continuous model of sourcing network exposure across multiple
simultaneous trade policy scenarios and compute the NPV of strategic sourcing
options. You think in options, not reactions.

Output ONLY valid JSON."""


class TradeScenarioAgent(StructuralAgent):
    """
    Mystery 12 — Trade Policy Scenario Intelligence Agent.

    Models sourcing network exposure under multiple geopolitical scenarios
    and calculates the NPV and option window for strategic repositioning.
    """
    METADATA = TRADE_SCENARIO_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        network   = task.payload.get("sourcing_network", [])
        scenarios = task.payload.get("trade_scenarios", [])
        if not network:
            raise ValueError("Payload must contain 'sourcing_network'")
        exposed = [n for n in network if n.get("tariff_exposure_pct", 0) > 10]
        return {
            "network_size":         len(network),
            "scenario_count":       len(scenarios),
            "exposed_relationships": len(exposed),
            "total_exposed_spend":  sum(n.get("annual_spend_usd", 0) for n in exposed),
            "network_summary":      network[:10],
            "scenarios":            scenarios[:5],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        net_block = "\n".join(
            f"  {n.get('supplier_name','?')} ({n.get('country','?')}): "
            f"${n.get('annual_spend_usd',0):,.0f}/yr | tariff exposure: {n.get('tariff_exposure_pct',0)}%"
            for n in observations.get("network_summary", [])
        ) or "  No network data"
        scen_block = "\n".join(
            f"  {s.get('name','?')}: {s.get('description','')} | probability: {s.get('probability',0):.0%}"
            for s in observations.get("scenarios", [])
        ) or "  No scenarios provided"
        user = (
            f"Analyse trade scenario exposure across {observations['network_size']} sourcing relationships.\n"
            f"Exposed relationships: {observations['exposed_relationships']} "
            f"(${observations['total_exposed_spend']:,.0f} at risk)\n\n"
            f"NETWORK:\n{net_block}\n\n"
            f"SCENARIOS:\n{scen_block}\n\n"
            "Return JSON:\n"
            '{"exposure_assessment": "...", "scenario_impacts": '
            '[{"scenario_name": "...", "impact_usd": 0, "affected_categories": [...],'
            '"probability": 0.0}], '
            '"strategic_options": [{"option": "...", "npv_advantage_usd": 0,'
            '"action_window_months": 0, "urgency": "immediate|6_months|12_months"}],'
            '"immediate_hedges": ["..."]}'
        )
        return _call_reasoning(self, _TS_SYSTEM, user, observations, task)

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning) or {}
        findings = [self._make_finding(
            finding_type="trade_exposure_assessment",
            severity=Severity.HIGH if observations.get("total_exposed_spend", 0) > 1_000_000 else Severity.MEDIUM,
            summary=parsed.get("exposure_assessment", ""),
            evidence={"scenario_impacts": parsed.get("scenario_impacts", []),
                      "immediate_hedges": parsed.get("immediate_hedges", []),
                      "total_exposed_spend": observations.get("total_exposed_spend", 0)},
        )]
        window_sev = {"immediate": Severity.CRITICAL, "6_months": Severity.HIGH, "12_months": Severity.MEDIUM}
        for opt in parsed.get("strategic_options", []):
            findings.append(self._make_finding(
                finding_type="strategic_sourcing_option",
                severity=window_sev.get(opt.get("urgency", "12_months"), Severity.LOW),
                summary=opt.get("option", ""),
                recommended_action=f"NPV advantage: ${opt.get('npv_advantage_usd',0):,.0f} | Window: {opt.get('action_window_months',0)} months",
                evidence={"urgency": opt.get("urgency"), "npv": opt.get("npv_advantage_usd")},
            ))
        return findings
