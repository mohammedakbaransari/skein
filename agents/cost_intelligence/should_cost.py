"""
agents/cost_intelligence/should_cost.py
========================================
Mystery 06 — The Should-Cost Intelligence Gap

PROBLEM:
  Should-cost modelling is available only to the largest enterprises for
  their top 20-30 spend categories. For the rest of the portfolio, buyers
  negotiate blind. This agent closes that gap by maintaining continuous
  cost models from commodity price indices across every category.

ROLE: Procurement cost intelligence analyst
AUTHORITY: RECOMMEND
MYSTERY REF: mystery_06
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

from framework.agents.base import StructuralAgent
from framework.agents.catalogue import SHOULD_COST_METADATA
from framework.core.types import Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommodityMovement:
    """Price movement for one commodity over the analysis window."""
    commodity_key:   str
    display_name:    str
    from_value:      float
    to_value:        float
    change_pct:      float
    periods:         int
    leverage_level:  str    # High | Medium | Low | None


@dataclass(frozen=True)
class ShouldCostModel:
    """Should-cost snapshot for the full portfolio."""
    periods_analysed:        int
    date_range:              str
    commodity_movements:     Tuple[CommodityMovement, ...]
    leverage_opportunities:  Tuple[CommodityMovement, ...]
    rising_cost_warnings:    Tuple[str, ...]


# ---------------------------------------------------------------------------
# Pure analysis functions
# ---------------------------------------------------------------------------

_COMMODITY_NAMES: Dict[str, str] = {
    "steel_hrc_usd_ton":  "Steel HRC (USD/ton)",
    "copper_lme_usd_ton": "Copper LME (USD/ton)",
    "hdpe_resin_usd_ton": "HDPE Resin (USD/ton)",
    "labour_index_mfg":   "Manufacturing Labour Index",
    "energy_index":       "Energy Index",
}


def compute_commodity_movements(
    price_records: List[Dict],
    leverage_threshold_pct: float = -3.0,
) -> ShouldCostModel:
    """
    Compute price movements and identify negotiation leverage.

    Pure function — no I/O, no side effects. Thread-safe.

    Args:
        price_records:        Monthly commodity price dicts, sorted ascending.
        leverage_threshold_pct: % decline that qualifies as leverage opportunity.

    Returns:
        ShouldCostModel with movements, leverage opps, and rising cost warnings.
    """
    if len(price_records) < 2:
        return ShouldCostModel(0, "insufficient data", (), (), ())

    sorted_records = sorted(price_records, key=lambda r: r.get("month", ""))
    oldest = sorted_records[0]
    latest = sorted_records[-1]

    movements: List[CommodityMovement] = []
    for key, display_name in _COMMODITY_NAMES.items():
        old_val = oldest.get(key)
        new_val = latest.get(key)
        if old_val is None or new_val is None or old_val == 0:
            continue
        change_pct = round(((new_val - old_val) / abs(old_val)) * 100, 1)
        leverage = (
            "High"   if change_pct < -10
            else "Medium" if change_pct < -5
            else "Low"    if change_pct < leverage_threshold_pct
            else "None"
        )
        movements.append(CommodityMovement(
            commodity_key=key,
            display_name=display_name,
            from_value=round(old_val, 2),
            to_value=round(new_val, 2),
            change_pct=change_pct,
            periods=len(sorted_records),
            leverage_level=leverage,
        ))

    leverage_opps = tuple(m for m in movements if m.change_pct < leverage_threshold_pct)
    rising        = tuple(m.display_name for m in movements if m.change_pct > 5.0)

    date_range = f"{sorted_records[0].get('month','?')} → {sorted_records[-1].get('month','?')}"
    return ShouldCostModel(
        periods_analysed=len(sorted_records),
        date_range=date_range,
        commodity_movements=tuple(movements),
        leverage_opportunities=leverage_opps,
        rising_cost_warnings=rising,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a strategic procurement cost intelligence analyst with
expertise in should-cost modelling and commodity cost dynamics.

Your purpose is to identify where supplier price positions lack cost basis —
where commodity input costs have fallen but buyer prices have not. Your findings
are specific, quantified, and time-bounded.

Output ONLY valid JSON. No preamble or markdown fences."""


def _build_prompt(obs: Dict[str, Any]) -> str:
    movements = "\n".join(
        f"  {m['display_name']}: {m['from_value']} → {m['to_value']} "
        f"({m['change_pct']:+.1f}% over {m['periods']} months)"
        for m in obs.get("commodity_movements", [])
    ) or "  No movement data"
    leverage = "\n".join(
        f"  {m['display_name']}: {m['change_pct']:+.1f}% | Leverage: {m['leverage_level']}"
        for m in obs.get("leverage_opportunities", [])
    ) or "  No material declines detected"
    return f"""Analyse should-cost intelligence for {obs.get('periods_analysed', 0)} months.

ALL COMMODITY MOVEMENTS:
{movements}

NEGOTIATION LEVERAGE OPPORTUNITIES (input costs declined):
{leverage}

RISING COST WARNINGS: {', '.join(obs.get('rising_cost_warnings', [])) or 'none'}

Return JSON:
{{
  "market_assessment": "2-3 sentences on the current cost environment",
  "leverage_opportunities": [
    {{
      "category": "specific category name",
      "input_decline_pct": 0.0,
      "cost_basis_argument": "specific claim the buyer should make",
      "estimated_reduction_pct": "achievable range e.g. 5-8%",
      "urgency": "immediate|within_quarter|within_year",
      "talking_point": "verbatim argument the buyer should open with"
    }}
  ],
  "rising_cost_warnings": ["categories to watch for supplier price increase requests"],
  "recommended_actions": ["ordered action list"]
}}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ShouldCostAgent(StructuralAgent):
    """
    Structural Intelligence Agent — Mystery 06: Should-Cost Intelligence.

    Analyses commodity price movements to surface categories where suppliers
    have no cost basis for their current pricing — restoring the information
    balance that should-cost modelling provides for tier-one categories to
    the entire spend portfolio.
    """

    METADATA = SHOULD_COST_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        prices = task.payload.get("commodity_prices", [])
        if not prices:
            raise ValueError("Payload must contain 'commodity_prices'")
        model = compute_commodity_movements(prices)
        return {
            "periods_analysed":    model.periods_analysed,
            "date_range":          model.date_range,
            "opportunity_count":   len(model.leverage_opportunities),
            "commodity_movements": [
                {"display_name": m.display_name, "from_value": m.from_value,
                 "to_value": m.to_value, "change_pct": m.change_pct,
                 "periods": m.periods, "leverage_level": m.leverage_level}
                for m in model.commodity_movements
            ],
            "leverage_opportunities": [
                {"display_name": m.display_name, "change_pct": m.change_pct,
                 "leverage_level": m.leverage_level}
                for m in model.leverage_opportunities
            ],
            "rising_cost_warnings": list(model.rising_cost_warnings),
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if not self.reasoning:
            return json.dumps({"market_assessment": "Reasoning engine not configured.",
                               "leverage_opportunities": [], "rising_cost_warnings": [],
                               "recommended_actions": []})
        resp = self.reasoning.reason(ReasoningRequest(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_prompt(observations),
            observations=observations,
            strategy=ReasoningStrategy.STRUCTURED,
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
            finding_type="market_assessment",
            severity=Severity.INFO,
            summary=parsed.get("market_assessment", ""),
            evidence={"recommended_actions": parsed.get("recommended_actions", []),
                      "rising_warnings": parsed.get("rising_cost_warnings", []),
                      "opportunity_count": observations.get("opportunity_count", 0)},
        )]
        urgency_map = {"immediate": Severity.HIGH, "within_quarter": Severity.MEDIUM,
                       "within_year": Severity.LOW}
        for opp in parsed.get("leverage_opportunities", []):
            findings.append(self._make_finding(
                finding_type="should_cost_leverage",
                severity=urgency_map.get(opp.get("urgency", "within_year"), Severity.LOW),
                entity_name=opp.get("category"),
                summary=opp.get("cost_basis_argument", ""),
                detail=opp.get("talking_point", ""),
                recommended_action=f"Estimated reduction: {opp.get('estimated_reduction_pct', '')} | Urgency: {opp.get('urgency', '')}",
                evidence={"input_decline_pct": opp.get("input_decline_pct")},
            ))
        return findings
