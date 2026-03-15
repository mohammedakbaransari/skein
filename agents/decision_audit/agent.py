"""
agents/decision_audit/agent.py
================================
Mystery 13 — Procurement Decision Accountability Agent

ROLE:
  The accountability watchdog for AI-assisted procurement decisions.
  55-65% of AI-assisted procurement decisions have no recorded rationale
  (ISACA 2025). This agent surfaces that gap and its regulatory exposure.

EXTENDS: DecisionAgent (it monitors other agents' decision records,
  so it inherits the authority and escalation machinery)

DATA SOURCES:
  - Decision logs (AI recommendations, human actions, rationale flags)
  - Evaluator records (who evaluated, factor weights, override history)
  - Governance logger output (execution records)

OUTPUTS:
  - Accountability gap rate (% decisions without rationale)
  - Evaluator consistency metrics
  - High-risk decision set (high confidence + no rationale)
  - Regulatory exposure assessment (EU AI Act, OECD principles)
  - Specific remediation recommendations

DECISION AUTHORITY: ESCALATE (accountability failures need senior attention)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.agents.base import DecisionAgent
from framework.agents.catalogue import DECISION_AUDIT_METADATA
from framework.core.types import Finding, Severity, Task


# ---------------------------------------------------------------------------
# Domain metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluatorMetrics:
    evaluator_id:           str
    decision_count:         int
    rationale_gap_pct:      float
    override_count:         int
    price_weight_variance:  float


@dataclass(frozen=True)
class AccountabilityMetrics:
    total_decisions:          int
    rationale_gap_count:      int
    rationale_gap_pct:        float
    override_count:           int
    override_rate_pct:        float
    high_risk_decision_ids:   List[str]
    evaluator_metrics:        List[EvaluatorMetrics]
    category_metrics:         Dict[str, Dict]


# ---------------------------------------------------------------------------
# Analysis logic — pure functions
# ---------------------------------------------------------------------------

def compute_accountability_metrics(decisions: List[Dict]) -> AccountabilityMetrics:
    """
    Analyse decision logs for accountability gaps.
    Pure function — thread-safe.
    """
    if not decisions:
        return AccountabilityMetrics(0, 0, 0.0, 0, 0.0, [], [], {})

    total = len(decisions)
    no_rationale = [d for d in decisions if not d.get("rationale_logged")]
    overrides    = [d for d in decisions if d.get("human_override")]
    high_risk    = [
        d["decision_id"] for d in no_rationale
        if d.get("ai_score", 0) >= 85
    ][:10]

    # Per-evaluator
    by_eval: Dict[str, List[Dict]] = {}
    for d in decisions:
        by_eval.setdefault(d.get("evaluator_id", "UNKNOWN"), []).append(d)

    ev_metrics: List[EvaluatorMetrics] = []
    for eid, ev_dec in by_eval.items():
        if len(ev_dec) < 3:
            continue
        pw = [d.get("factors_weighted", {}).get("price", 0.0) for d in ev_dec]
        avg_w = sum(pw) / len(pw)
        var   = sum((w - avg_w) ** 2 for w in pw) / len(pw)
        ev_metrics.append(EvaluatorMetrics(
            evaluator_id=eid,
            decision_count=len(ev_dec),
            rationale_gap_pct=round(
                100 * sum(1 for d in ev_dec if not d.get("rationale_logged")) / len(ev_dec), 1
            ),
            override_count=sum(1 for d in ev_dec if d.get("human_override")),
            price_weight_variance=round(var, 4),
        ))

    # Per-category
    by_cat: Dict[str, List[Dict]] = {}
    for d in decisions:
        by_cat.setdefault(d.get("category", "Unknown"), []).append(d)
    cat_metrics = {
        cat: {
            "decisions":         len(ds),
            "rationale_gap_pct": round(100 * sum(1 for d in ds if not d.get("rationale_logged")) / len(ds), 1),
            "override_rate_pct": round(100 * sum(1 for d in ds if d.get("human_override")) / len(ds), 1),
        }
        for cat, ds in by_cat.items()
    }

    return AccountabilityMetrics(
        total_decisions=total,
        rationale_gap_count=len(no_rationale),
        rationale_gap_pct=round(100 * len(no_rationale) / total, 1),
        override_count=len(overrides),
        override_rate_pct=round(100 * len(overrides) / total, 1),
        high_risk_decision_ids=high_risk,
        evaluator_metrics=ev_metrics,
        category_metrics=cat_metrics,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a procurement governance and AI accountability specialist.

You analyse decision records to identify accountability gaps that would
fail scrutiny under:
  - EU AI Act (high-risk AI in supplier evaluation — Art. 13, 17)
  - OECD AI Principles (transparency, accountability)
  - Internal audit standards (decision traceability)
  - Legal challenge (contract award disputes)

You identify structural patterns. You are not judging individual decisions.
You quantify exposure and prescribe specific remediations with owners.

Output ONLY valid JSON. No preamble."""


def build_user_prompt(obs: Dict[str, Any]) -> str:
    ev_block = "\n".join(
        f"  {m['evaluator_id']}: {m['decision_count']} decisions | "
        f"{m['rationale_gap_pct']}% no rationale | "
        f"{m['override_count']} overrides | "
        f"price-weight variance {m['price_weight_variance']:.4f}"
        for m in obs.get("evaluator_metrics", [])
    ) or "  Insufficient evaluator data"

    cat_block = "\n".join(
        f"  {cat}: {s['decisions']} decisions | "
        f"{s['rationale_gap_pct']}% no rationale | "
        f"{s['override_rate_pct']}% overrides"
        for cat, s in obs.get("category_metrics", {}).items()
    ) or "  No category data"

    return f"""Analyse procurement decision accountability.
Total decisions: {obs['total_decisions']}
Missing rationale: {obs['rationale_gap_count']} ({obs['rationale_gap_pct']}%)
Override rate: {obs['override_count']} ({obs['override_rate_pct']}%)
High-risk (score≥85 + no rationale): {', '.join(obs.get('high_risk_decision_ids', [])) or 'none'}

EVALUATOR PATTERNS:
{ev_block}

CATEGORY PATTERNS:
{cat_block}

Return JSON:
{{
  "accountability_assessment": "2-3 sentences on overall governance risk",
  "regulatory_exposure_level": "Critical|High|Medium|Low",
  "gaps": [
    {{
      "gap_type": "rationale_missing|override_undocumented|evaluator_inconsistency",
      "severity": "critical|high|medium",
      "description": "specific gap with data evidence",
      "regulatory_reference": "specific regulation or principle",
      "remediation": "action with owner and deadline",
      "estimated_impacted_decisions": 0
    }}
  ],
  "evaluator_flags": [
    {{
      "evaluator_id": "...",
      "concern": "...",
      "recommended_action": "..."
    }}
  ],
  "immediate_actions": ["ordered action list for this week"],
  "framework_to_implement": "name and brief description"
}}"""


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class DecisionAuditAgent(DecisionAgent):
    """
    Structural Intelligence Agent — Mystery 13: Decision Accountability.

    Inherits from DecisionAgent so its own outputs carry formal
    decision records (meta-accountability — the accountability agent
    itself is accountable).
    """

    METADATA = DECISION_AUDIT_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        decisions = task.payload.get("decision_logs", [])
        if not decisions:
            raise ValueError("Payload must contain 'decision_logs'")

        metrics = compute_accountability_metrics(decisions)
        self._log.info(
            "[agent=%s] %d decisions: %.1f%% missing rationale, %.1f%% override",
            self.agent_id, metrics.total_decisions,
            metrics.rationale_gap_pct, metrics.override_rate_pct,
        )
        return {
            "total_decisions":       metrics.total_decisions,
            "rationale_gap_count":   metrics.rationale_gap_count,
            "rationale_gap_pct":     metrics.rationale_gap_pct,
            "override_count":        metrics.override_count,
            "override_rate_pct":     metrics.override_rate_pct,
            "high_risk_decision_ids": metrics.high_risk_decision_ids,
            "evaluator_metrics": [
                {
                    "evaluator_id":          m.evaluator_id,
                    "decision_count":        m.decision_count,
                    "rationale_gap_pct":     m.rationale_gap_pct,
                    "override_count":        m.override_count,
                    "price_weight_variance": m.price_weight_variance,
                }
                for m in metrics.evaluator_metrics
            ],
            "category_metrics": metrics.category_metrics,
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if self.reasoning:
            from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy
            request = ReasoningRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(observations),
                observations=observations,
                strategy=ReasoningStrategy.STRUCTURED,
                session_id=str(task.session_id),
            )
            return self.reasoning.reason(request).content
        return json.dumps({
            "accountability_assessment": "Reasoning engine not configured.",
            "regulatory_exposure_level": "Unknown",
            "gaps": [], "evaluator_flags": [],
            "immediate_actions": [], "framework_to_implement": "",
        })

    def parse_findings(
        self,
        observations: Dict[str, Any],
        reasoning: str,
        task: Task,
    ) -> List[Finding]:
        parsed = self._parse_llm_json(reasoning)
        if not parsed:
            return [self._make_finding("parse_error", Severity.HIGH,
                                       "LLM did not return valid JSON", reasoning[:400])]

        findings: List[Finding] = []
        exp_levels = {"Critical": Severity.CRITICAL, "High": Severity.HIGH,
                      "Medium": Severity.MEDIUM, "Low": Severity.LOW}

        findings.append(self._make_finding(
            finding_type="governance_assessment",
            severity=exp_levels.get(
                parsed.get("regulatory_exposure_level", "Medium"), Severity.MEDIUM
            ),
            summary=parsed.get("accountability_assessment", ""),
            evidence={
                "framework_to_implement": parsed.get("framework_to_implement"),
                "immediate_actions":      parsed.get("immediate_actions", []),
                "rationale_gap_pct":      observations.get("rationale_gap_pct", 0),
            },
        ))

        gap_severities = {"critical": Severity.CRITICAL, "high": Severity.HIGH,
                          "medium": Severity.MEDIUM}
        for gap in parsed.get("gaps", []):
            findings.append(self._make_finding(
                finding_type="accountability_gap",
                severity=gap_severities.get(gap.get("severity", "medium"), Severity.MEDIUM),
                summary=gap.get("description", ""),
                detail=gap.get("regulatory_reference", ""),
                recommended_action=gap.get("remediation", ""),
                evidence={
                    "gap_type": gap.get("gap_type"),
                    "impacted_decisions": gap.get("estimated_impacted_decisions", 0),
                },
            ))

        for flag in parsed.get("evaluator_flags", []):
            findings.append(self._make_finding(
                finding_type="evaluator_concern",
                severity=Severity.MEDIUM,
                entity_id=flag.get("evaluator_id"),
                summary=flag.get("concern", ""),
                recommended_action=flag.get("recommended_action", ""),
            ))

        return findings
