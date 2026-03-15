"""
agents/contract_analysis/value_realisation.py
===============================================
Mystery 11 — The Value Realisation Black Hole

PROBLEM:
  20-40% of negotiated savings never reach the P&L. No current system
  continuously tracks the gap, identifies causes, and intervenes while
  there is still time to recover the value.

ROLE: Contract performance and savings realisation monitor
AUTHORITY: RECOMMEND
MYSTERY REF: mystery_11
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework.agents.base import StructuralAgent
from framework.agents.catalogue import VALUE_REALISATION_METADATA
from framework.core.types import Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeakageProfile:
    """Savings tracking profile for one contract."""
    contract_id:              str
    category:                 str
    negotiated_savings_pct:   float
    actual_savings_pct:       float
    leakage_pct:              float
    cumulative_leakage_usd:   float
    trend:                    str    # Improving | Stable | Deteriorating
    active_causes:            tuple
    months_tracked:           int
    alert_level:              str    # critical | high | medium | low | ok


# ---------------------------------------------------------------------------
# Pure analysis
# ---------------------------------------------------------------------------

_CAUSE_LABELS: Dict[str, str] = {
    "spec_change":      "Post-award specification changes",
    "maverick_spend":   "Departments buying off-contract",
    "erp_not_updated":  "ERP approved supplier list not updated",
    "volume_shortfall": "Volume commitments not met",
}


def _compute_trend(series: List[float]) -> str:
    if len(series) < 2:
        return "Stable"
    mid  = len(series) // 2
    first_half  = sum(series[:mid]) / mid
    second_half = sum(series[mid:]) / (len(series) - mid)
    if second_half < first_half - 0.5:  return "Deteriorating"
    if second_half > first_half + 0.5:  return "Improving"
    return "Stable"


def _alert_level(leakage_pct: float, trend: str) -> str:
    if leakage_pct >= 5 or (leakage_pct >= 3 and trend == "Deteriorating"):
        return "critical"
    if leakage_pct >= 3: return "high"
    if leakage_pct >= 1: return "medium"
    if leakage_pct > 0:  return "low"
    return "ok"


def analyse_savings_portfolio(records: List[Dict]) -> List[LeakageProfile]:
    """
    Group records by contract and compute leakage profiles.
    Pure function — thread-safe.
    """
    by_contract: Dict[str, List[Dict]] = {}
    for r in records:
        by_contract.setdefault(r.get("contract_id", "UNKNOWN"), []).append(r)

    profiles: List[LeakageProfile] = []
    for cid, rows in by_contract.items():
        rows = sorted(rows, key=lambda r: r.get("month", ""))
        negotiated    = rows[0].get("negotiated_savings_pct", 0.0)
        actual_series = [r.get("actual_savings_pct", 0.0) for r in rows]
        latest_actual = actual_series[-1] if actual_series else 0.0
        leakage_pct   = round(negotiated - latest_actual, 2)
        cumulative    = sum(r.get("leakage_amount_usd", 0.0) for r in rows)
        trend         = _compute_trend(actual_series)
        causes: List[str] = []
        for row in rows:
            for c in row.get("leakage_causes", []):
                if c not in causes:
                    causes.append(c)
        profiles.append(LeakageProfile(
            contract_id=cid,
            category=rows[0].get("category", "Unknown"),
            negotiated_savings_pct=negotiated,
            actual_savings_pct=latest_actual,
            leakage_pct=leakage_pct,
            cumulative_leakage_usd=round(cumulative, 2),
            trend=trend,
            active_causes=tuple(causes),
            months_tracked=len(rows),
            alert_level=_alert_level(leakage_pct, trend),
        ))
    return sorted(profiles, key=lambda p: p.cumulative_leakage_usd, reverse=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a procurement value realisation expert focused on
closing the gap between negotiated savings and P&L delivery.

You provide specific, named, time-bound interventions. You are honest with the
CFO risk: if the gap is credibility-threatening, you say so.

Output ONLY valid JSON. No preamble."""


def _build_prompt(obs: Dict[str, Any]) -> str:
    contract_block = "\n".join(
        f"  {p['contract_id']} ({p['category']}): "
        f"negotiated {p['negotiated_pct']}% → actual {p['actual_pct']}% "
        f"[leakage {p['leakage_pct']}pp | ${p['leakage_usd']:,.0f} | "
        f"trend: {p['trend']} | causes: {', '.join(p['causes']) or 'none'}]"
        for p in obs.get("contract_summaries", [])
    )
    return f"""Analyse savings leakage across {obs['contract_count']} contracts.
Total cumulative leakage: ${obs['total_leakage_usd']:,.0f}

CONTRACTS:
{contract_block}

Return JSON:
{{
  "portfolio_assessment": "2-3 sentences on severity and pattern",
  "cfo_credibility_risk": "one sentence on what this means for the procurement function",
  "interventions": [
    {{
      "contract_id": "...",
      "category": "...",
      "leakage_usd": 0,
      "primary_cause": "...",
      "intervention": "specific corrective action with named owner and deadline",
      "recovery_potential_pct": "% of leaked savings recoverable",
      "urgency": "immediate|this_week|this_quarter"
    }}
  ],
  "systemic_fixes": ["process or system changes to prevent recurrence"],
  "immediate_actions": ["what to do today"]
}}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ValueRealisationAgent(StructuralAgent):
    """
    Structural Intelligence Agent — Mystery 11: Value Realisation.

    Continuously monitors the gap between negotiated savings commitments
    and actual P&L capture. Identifies specific leakage causes and
    recommends targeted interventions while there is still time to recover.
    """

    METADATA = VALUE_REALISATION_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        records = task.payload.get("savings_tracking", [])
        if not records:
            raise ValueError("Payload must contain 'savings_tracking'")
        profiles = analyse_savings_portfolio(records)
        total = sum(p.cumulative_leakage_usd for p in profiles)
        drift_count = sum(1 for p in profiles if p.leakage_pct > 2.0)
        return {
            "contract_count":       len(profiles),
            "total_leakage_usd":    total,
            "contracts_with_drift": drift_count,
            "contract_summaries": [
                {"contract_id": p.contract_id, "category": p.category,
                 "negotiated_pct": p.negotiated_savings_pct,
                 "actual_pct": p.actual_savings_pct,
                 "leakage_pct": p.leakage_pct, "leakage_usd": p.cumulative_leakage_usd,
                 "trend": p.trend, "alert_level": p.alert_level,
                 "causes": list(p.active_causes)}
                for p in profiles
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if not self.reasoning:
            return json.dumps({"portfolio_assessment": "Reasoning engine not configured.",
                               "cfo_credibility_risk": "", "interventions": [],
                               "systemic_fixes": [], "immediate_actions": []})
        resp = self.reasoning.reason(ReasoningRequest(
            system_prompt=_SYSTEM_PROMPT, user_prompt=_build_prompt(observations),
            observations=observations, strategy=ReasoningStrategy.STRUCTURED,
            session_id=str(task.session_id),
        ))
        return resp.content

    def parse_findings(self, observations: Dict[str, Any], reasoning: str,
                       task: Task) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning)
        if not parsed:
            return [self._make_finding("parse_error", Severity.HIGH,
                                       "LLM did not return valid JSON", reasoning[:300])]
        findings = [self._make_finding(
            finding_type="portfolio_assessment",
            severity=Severity.HIGH if observations.get("contracts_with_drift", 0) > 2 else Severity.MEDIUM,
            summary=parsed.get("portfolio_assessment", ""),
            detail=parsed.get("cfo_credibility_risk", ""),
            evidence={"total_leakage_usd": observations.get("total_leakage_usd", 0),
                      "systemic_fixes": parsed.get("systemic_fixes", []),
                      "immediate_actions": parsed.get("immediate_actions", [])},
        )]
        urgency_map = {"immediate": Severity.CRITICAL, "this_week": Severity.HIGH,
                       "this_quarter": Severity.MEDIUM}
        for iv in parsed.get("interventions", []):
            findings.append(self._make_finding(
                finding_type="leakage_intervention",
                severity=urgency_map.get(iv.get("urgency", "this_quarter"), Severity.MEDIUM),
                entity_id=iv.get("contract_id"), entity_name=iv.get("category"),
                summary=f"${iv.get('leakage_usd', 0):,.0f} leakage — {iv.get('primary_cause', '')}",
                recommended_action=iv.get("intervention", ""),
                evidence={"recovery_potential_pct": iv.get("recovery_potential_pct"),
                          "urgency": iv.get("urgency")},
            ))
        return findings
