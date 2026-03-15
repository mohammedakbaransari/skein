"""
agents/cost_intelligence/total_cost.py
========================================
Mystery 14 — The Total Cost Blindspot

PROBLEM:
  Procurement optimises for invoice price. That number represents 25-40%
  of the true lifecycle cost of capital equipment. No AI procurement tool
  computes the full lifecycle cost profile at the moment of sourcing.

ROLE: TCO analyst for capital and complex service categories
AUTHORITY: RECOMMEND
MYSTERY REF: mystery_14
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework.agents.base import ProcurementAgent
from framework.agents.catalogue import TOTAL_COST_METADATA
from framework.core.types import Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetTCOProfile:
    asset_id:                    str
    asset_type:                  str
    purchase_price_usd:          float
    total_tco_usd:               float
    tco_to_price_ratio:          float
    annual_energy_cost_usd:      float
    annual_maintenance_cost_usd: float
    annual_downtime_risk_usd:    float
    lifecycle_years:             int
    sourced_on_price_alone:      bool
    lifecycle_value_at_risk_usd: float


@dataclass(frozen=True)
class TCOPortfolioSummary:
    total_assets:            int
    price_only_count:        int
    price_only_pct:          float
    high_ratio_count:        int
    total_value_at_risk_usd: float
    avg_tco_to_price_ratio:  float
    asset_profiles:          tuple
    category_breakdown:      dict


# ---------------------------------------------------------------------------
# Pure analysis
# ---------------------------------------------------------------------------

def analyse_tco_portfolio(assets: List[Dict]) -> TCOPortfolioSummary:
    """Compute TCO profiles across all assets. Pure function — thread-safe."""
    if not assets:
        return TCOPortfolioSummary(0, 0, 0.0, 0, 0.0, 0.0, (), {})

    profiles = []
    for a in assets:
        price  = a.get("purchase_price_usd", 0.0)
        tco    = a.get("total_tco_usd", price)
        ratio  = round(tco / price, 2) if price > 0 else 1.0
        po     = bool(a.get("procurement_decided_on_price_alone", False))
        at_risk = round(tco - price, 2) if po and ratio > 3.0 else 0.0
        profiles.append(AssetTCOProfile(
            asset_id=a.get("asset_id", "UNKNOWN"),
            asset_type=a.get("asset_type", "unknown"),
            purchase_price_usd=price, total_tco_usd=tco,
            tco_to_price_ratio=ratio,
            annual_energy_cost_usd=a.get("annual_energy_cost_usd", 0.0),
            annual_maintenance_cost_usd=a.get("annual_maintenance_cost_usd", 0.0),
            annual_downtime_risk_usd=a.get("annual_downtime_risk_usd", 0.0),
            lifecycle_years=a.get("lifecycle_years", 10),
            sourced_on_price_alone=po,
            lifecycle_value_at_risk_usd=at_risk,
        ))

    price_only = sum(1 for p in profiles if p.sourced_on_price_alone)
    high_ratio = sum(1 for p in profiles if p.tco_to_price_ratio > 4.0)
    total_risk = sum(p.lifecycle_value_at_risk_usd for p in profiles)
    avg_ratio  = sum(p.tco_to_price_ratio for p in profiles) / len(profiles)

    by_type: Dict[str, list] = {}
    for p in profiles:
        by_type.setdefault(p.asset_type, []).append(p)
    cat = {
        t: {"count": len(items),
            "avg_ratio": round(sum(p.tco_to_price_ratio for p in items) / len(items), 2),
            "price_only_pct": round(100 * sum(1 for p in items if p.sourced_on_price_alone) / len(items), 1),
            "value_at_risk": round(sum(p.lifecycle_value_at_risk_usd for p in items), 2)}
        for t, items in by_type.items()
    }

    return TCOPortfolioSummary(
        total_assets=len(profiles),
        price_only_count=price_only,
        price_only_pct=round(100 * price_only / len(profiles), 1),
        high_ratio_count=high_ratio,
        total_value_at_risk_usd=round(total_risk, 2),
        avg_tco_to_price_ratio=round(avg_ratio, 2),
        asset_profiles=tuple(sorted(profiles, key=lambda p: p.lifecycle_value_at_risk_usd, reverse=True)),
        category_breakdown=cat,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a strategic procurement analyst specialising in
total cost of ownership and lifecycle cost optimisation.

You identify and quantify the financial impact of price-only procurement decisions.
Be specific. Use dollar amounts. Recommend concrete actions with owners.

Output ONLY valid JSON. No preamble."""


def _build_prompt(obs: Dict[str, Any]) -> str:
    cat_block = "\n".join(
        f"  {t}: avg ratio {s['avg_ratio']}× | {s['price_only_pct']}% price-only | "
        f"${s['value_at_risk']:,.0f} at risk"
        for t, s in obs.get("category_breakdown", {}).items()
    )
    top = "\n".join(
        f"  {a['asset_id']} ({a['asset_type']}): "
        f"${a['purchase_price_usd']:,.0f} → TCO ${a['total_tco_usd']:,.0f} "
        f"({a['tco_to_price_ratio']}×) | at risk: ${a['lifecycle_value_at_risk_usd']:,.0f}"
        for a in obs.get("top_at_risk_assets", [])
    ) or "  No high-risk assets"
    return f"""Analyse TCO blindspot across {obs['total_assets']} assets.
Price-only sourcing: {obs['price_only_count']} ({obs['price_only_pct']}%)
High TCO/price ratio (>4×): {obs['high_ratio_count']}
Average TCO/price ratio: {obs['avg_tco_to_price_ratio']}×
Total lifecycle value at risk: ${obs['total_value_at_risk_usd']:,.0f}

BY CATEGORY:
{cat_block}

TOP AT-RISK ASSETS:
{top}

Return JSON:
{{
  "tco_assessment": "2-3 sentences on scale and pattern of the blindspot",
  "financial_impact": "portfolio-level USD impact quantification",
  "findings": [
    {{
      "finding_type": "price_only_bias|tco_ratio_extreme|category_systematic",
      "severity": "critical|high|medium",
      "description": "specific finding with data",
      "financial_implication": "USD impact",
      "recommended_action": "action with owner and timeline"
    }}
  ],
  "process_gaps": ["specific process changes to embed TCO at decision time"],
  "immediate_priorities": ["what to action this week"]
}}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TotalCostIntelligenceAgent(ProcurementAgent):
    """
    Procurement Intelligence Agent — Mystery 14: Total Cost Blindspot.

    Identifies where procurement is optimising invoice price instead of
    true lifecycle cost, and quantifies the financial impact of that
    systematic decision pattern across the asset portfolio.
    """

    METADATA = TOTAL_COST_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        assets = task.payload.get("tco_data", [])
        if not assets:
            raise ValueError("Payload must contain 'tco_data'")
        summary = analyse_tco_portfolio(assets)
        return {
            "total_assets":           summary.total_assets,
            "price_only_count":       summary.price_only_count,
            "price_only_pct":         summary.price_only_pct,
            "high_ratio_count":       summary.high_ratio_count,
            "total_value_at_risk_usd": summary.total_value_at_risk_usd,
            "avg_tco_to_price_ratio": summary.avg_tco_to_price_ratio,
            "category_breakdown":     summary.category_breakdown,
            "top_at_risk_assets": [
                {"asset_id": a.asset_id, "asset_type": a.asset_type,
                 "purchase_price_usd": a.purchase_price_usd,
                 "total_tco_usd": a.total_tco_usd,
                 "tco_to_price_ratio": a.tco_to_price_ratio,
                 "lifecycle_value_at_risk_usd": a.lifecycle_value_at_risk_usd,
                 "sourced_on_price_alone": a.sourced_on_price_alone}
                for a in summary.asset_profiles[:6]
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if not self.reasoning:
            return json.dumps({"tco_assessment": "Reasoning engine not configured.",
                               "financial_impact": "", "findings": [],
                               "process_gaps": [], "immediate_priorities": []})
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
            finding_type="tco_portfolio_assessment",
            severity=Severity.HIGH if observations.get("price_only_pct", 0) > 50 else Severity.MEDIUM,
            summary=parsed.get("tco_assessment", ""),
            detail=parsed.get("financial_impact", ""),
            evidence={"process_gaps": parsed.get("process_gaps", []),
                      "immediate_priorities": parsed.get("immediate_priorities", []),
                      "total_value_at_risk_usd": observations.get("total_value_at_risk_usd", 0)},
        )]
        sev_map = {"critical": Severity.CRITICAL, "high": Severity.HIGH, "medium": Severity.MEDIUM}
        for f in parsed.get("findings", []):
            findings.append(self._make_finding(
                finding_type="tco_finding",
                severity=sev_map.get(f.get("severity", "medium"), Severity.MEDIUM),
                summary=f.get("description", ""), detail=f.get("financial_implication", ""),
                recommended_action=f.get("recommended_action", ""),
                evidence={"finding_type": f.get("finding_type")},
            ))
        return findings
