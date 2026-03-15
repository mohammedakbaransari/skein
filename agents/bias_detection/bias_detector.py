"""
agents/bias_detection/bias_detector.py
========================================
Mystery 15 — The Incumbent Advantage Bias Nobody Is Measuring

PROBLEM:
  The incumbent wins at a rate that objective performance cannot explain.
  AI trained on historical award data amplifies this bias invisibly.
  Diverse-owned and SME suppliers are systematically suppressed relative
  to their objective scores.

ROLE: Procurement evaluation integrity analyst
AUTHORITY: ESCALATE (structural bias needs leadership attention)
MYSTERY REF: mystery_15
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
from framework.agents.catalogue import BIAS_DETECTOR_METADATA
from framework.core.types import DecisionAuthority, Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupplierTypeStats:
    supplier_type:        str
    count:                int
    award_rate_pct:       float
    avg_objective_score:  float
    avg_subjective_score: float
    subjective_premium:   float    # subj − obj; positive = inflated


@dataclass(frozen=True)
class EvaluatorBias:
    evaluator_id:             str
    evaluation_count:         int
    incumbent_premium:        float   # avg (subj-obj) for incumbents
    non_incumbent_premium:    float
    bias_differential:        float   # incumbent_premium - non_incumbent_premium


@dataclass(frozen=True)
class BiasAnalysis:
    total_evaluations:           int
    supplier_type_stats:         Tuple[SupplierTypeStats, ...]
    incumbent_objective_delta:   float   # near 0 = fair; large = incumbents genuinely better
    award_rate_gap_pct:          float   # incumbent rate − new entrant rate
    evaluator_bias_metrics:      Tuple[EvaluatorBias, ...]
    diverse_suppression_flag:    bool
    sme_suppression_flag:        bool


# ---------------------------------------------------------------------------
# Pure analysis
# ---------------------------------------------------------------------------

def analyse_evaluation_bias(evaluations: List[Dict]) -> BiasAnalysis:
    """Compute comprehensive bias metrics. Pure function — thread-safe."""
    if not evaluations:
        return BiasAnalysis(0, (), 0.0, 0.0, (), False, False)

    by_type: Dict[str, List[Dict]] = {}
    for e in evaluations:
        by_type.setdefault(e.get("supplier_type", "unknown"), []).append(e)

    type_stats = []
    for stype, items in by_type.items():
        awarded  = [e for e in items if e.get("awarded")]
        avg_obj  = sum(e.get("objective_score",  0) for e in items) / len(items)
        avg_subj = sum(e.get("subjective_score", 0) for e in items) / len(items)
        type_stats.append(SupplierTypeStats(
            supplier_type=stype,
            count=len(items),
            award_rate_pct=round(100 * len(awarded) / len(items), 1),
            avg_objective_score=round(avg_obj, 1),
            avg_subjective_score=round(avg_subj, 1),
            subjective_premium=round(avg_subj - avg_obj, 2),
        ))

    inc = next((m for m in type_stats if m.supplier_type == "incumbent"), None)
    new = next((m for m in type_stats if m.supplier_type == "new_entrant"), None)
    obj_delta = round(inc.avg_objective_score - new.avg_objective_score, 1) if inc and new else 0.0
    award_gap = round(inc.award_rate_pct    - new.award_rate_pct,    1) if inc and new else 0.0

    # Per-evaluator bias
    by_eval: Dict[str, List[Dict]] = {}
    for e in evaluations:
        by_eval.setdefault(e.get("evaluator_id", "UNKNOWN"), []).append(e)

    ev_metrics = []
    for eid, evals in by_eval.items():
        if len(evals) < 4: continue
        inc_e   = [e for e in evals if e.get("supplier_type") == "incumbent"]
        non_inc = [e for e in evals if e.get("supplier_type") != "incumbent"]
        if not inc_e or not non_inc: continue
        def _prem(rs): return sum(r.get("subjective_score",0)-r.get("objective_score",0) for r in rs) / len(rs)
        ip = round(_prem(inc_e),   2)
        np = round(_prem(non_inc), 2)
        ev_metrics.append(EvaluatorBias(
            evaluator_id=eid, evaluation_count=len(evals),
            incumbent_premium=ip, non_incumbent_premium=np,
            bias_differential=round(ip - np, 2),
        ))

    div_m = next((m for m in type_stats if m.supplier_type == "diverse_owned"), None)
    sme_m = next((m for m in type_stats if m.supplier_type == "sme"),           None)
    return BiasAnalysis(
        total_evaluations=len(evaluations),
        supplier_type_stats=tuple(sorted(type_stats, key=lambda m: m.award_rate_pct, reverse=True)),
        incumbent_objective_delta=obj_delta,
        award_rate_gap_pct=award_gap,
        evaluator_bias_metrics=tuple(sorted(ev_metrics, key=lambda m: m.bias_differential, reverse=True)),
        diverse_suppression_flag=bool(div_m and div_m.avg_objective_score >= 65 and div_m.award_rate_pct < 30),
        sme_suppression_flag=bool(sme_m      and sme_m.avg_objective_score      >= 65 and sme_m.award_rate_pct      < 30),
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a procurement evaluation integrity analyst specialising
in fair competition and bias detection.

You surface structural patterns that reduce competitive tension, create systemic
unfairness, and embed historical bias into AI training data. You are precise about
what the data shows. You do not speculate beyond the evidence.

Output ONLY valid JSON. No preamble."""


def _build_prompt(obs: Dict[str, Any]) -> str:
    type_block = "\n".join(
        f"  {m['supplier_type']}: {m['count']} evals | award {m['award_rate_pct']}% | "
        f"obj {m['avg_objective_score']} | subj {m['avg_subjective_score']} | premium {m['subjective_premium']:+.1f}"
        for m in obs.get("supplier_type_stats", [])
    )
    ev_block = "\n".join(
        f"  {m['evaluator_id']}: differential {m['bias_differential']:+.2f} "
        f"(inc {m['incumbent_premium']:+.2f} vs non-inc {m['non_incumbent_premium']:+.2f})"
        for m in obs.get("evaluator_bias_metrics", [])[:5]
    ) or "  Insufficient evaluator data"
    return f"""Analyse procurement evaluation bias across {obs['total_evaluations']} evaluations.

SUPPLIER TYPE PERFORMANCE:
{type_block}

INCUMBENT ADVANTAGE:
  Objective score delta (incumbent − new entrant): {obs['incumbent_objective_delta']:+.1f}
  Award rate gap (incumbent − new entrant): {obs['award_rate_gap_pct']:+.1f} pp

SUPPRESSION FLAGS:
  Diverse-owned: {obs.get('diverse_suppression_flag')}
  SME: {obs.get('sme_suppression_flag')}

EVALUATOR BIAS (top):
{ev_block}

Return JSON:
{{
  "bias_assessment": "2-3 sentences on pattern and significance",
  "incumbent_finding": "specific quantified incumbent advantage finding",
  "diverse_sme_finding": "specific finding on diverse/SME outcomes",
  "patterns": [
    {{
      "pattern_type": "incumbent_advantage|diverse_suppression|sme_suppression|evaluator_inconsistency",
      "severity": "critical|high|medium",
      "description": "specific pattern with evidence",
      "business_impact": "competitive and cost implications",
      "remediation": "specific corrective action"
    }}
  ],
  "evaluator_flags": [
    {{
      "evaluator_id": "...",
      "differential": 0.0,
      "concern": "...",
      "recommended_action": "..."
    }}
  ],
  "systemic_risks": ["risks if patterns continue"],
  "immediate_actions": ["this week"]
}}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProcurementBiasDetectorAgent(StructuralAgent):
    """
    Structural Intelligence Agent — Mystery 15: Incumbent Advantage Bias.

    Surfaces systematic bias patterns that are invisible in individual
    decisions but statistically significant in aggregate. Prevents AI
    systems trained on historically biased award data from amplifying
    those patterns at scale.
    """

    METADATA = BIAS_DETECTOR_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        evals = task.payload.get("sourcing_evaluations", [])
        if not evals:
            raise ValueError("Payload must contain 'sourcing_evaluations'")
        result = analyse_evaluation_bias(evals)
        return {
            "total_evaluations":         result.total_evaluations,
            "incumbent_objective_delta": result.incumbent_objective_delta,
            "award_rate_gap_pct":        result.award_rate_gap_pct,
            "diverse_suppression_flag":  result.diverse_suppression_flag,
            "sme_suppression_flag":      result.sme_suppression_flag,
            "supplier_type_stats": [
                {"supplier_type": m.supplier_type, "count": m.count,
                 "award_rate_pct": m.award_rate_pct,
                 "avg_objective_score": m.avg_objective_score,
                 "avg_subjective_score": m.avg_subjective_score,
                 "subjective_premium": m.subjective_premium}
                for m in result.supplier_type_stats
            ],
            "evaluator_bias_metrics": [
                {"evaluator_id": m.evaluator_id, "evaluation_count": m.evaluation_count,
                 "incumbent_premium": m.incumbent_premium,
                 "non_incumbent_premium": m.non_incumbent_premium,
                 "bias_differential": m.bias_differential}
                for m in result.evaluator_bias_metrics
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if not self.reasoning:
            return json.dumps({"bias_assessment": "Reasoning engine not configured.",
                               "incumbent_finding": "", "diverse_sme_finding": "",
                               "patterns": [], "evaluator_flags": [],
                               "systemic_risks": [], "immediate_actions": []})
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
            finding_type="bias_portfolio_assessment",
            severity=Severity.HIGH,
            summary=parsed.get("bias_assessment", ""),
            evidence={"incumbent_finding": parsed.get("incumbent_finding"),
                      "diverse_sme_finding": parsed.get("diverse_sme_finding"),
                      "systemic_risks": parsed.get("systemic_risks", []),
                      "immediate_actions": parsed.get("immediate_actions", []),
                      "award_rate_gap_pct": observations.get("award_rate_gap_pct", 0)},
        )]
        sev_map = {"critical": Severity.CRITICAL, "high": Severity.HIGH, "medium": Severity.MEDIUM}
        for p in parsed.get("patterns", []):
            findings.append(self._make_finding(
                finding_type="bias_pattern",
                severity=sev_map.get(p.get("severity", "medium"), Severity.MEDIUM),
                summary=p.get("description", ""), detail=p.get("business_impact", ""),
                recommended_action=p.get("remediation", ""),
                evidence={"pattern_type": p.get("pattern_type")},
            ))
        for flag in parsed.get("evaluator_flags", []):
            diff = abs(flag.get("differential", 0))
            findings.append(self._make_finding(
                finding_type="evaluator_bias_flag",
                severity=Severity.HIGH if diff > 5 else Severity.MEDIUM,
                entity_id=flag.get("evaluator_id"),
                summary=flag.get("concern", ""),
                recommended_action=flag.get("recommended_action", ""),
                evidence={"bias_differential": flag.get("differential")},
            ))
        return findings
