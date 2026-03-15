"""
agents/supply_risk/supplier_stress.py
=======================================
Mystery 02 — Supplier Stress Signal

Full production implementation of SupplierStressAgent.
Inherits from ProcurementAgent — observe/reason/parse_findings pipeline.

ROLE:
  Early warning system for supplier financial or operational distress.
  Reads signals your own ERP already contains, 6–9 months ahead of any
  external intelligence source.

DATA SOURCES:
  - ERP transaction data (PO acknowledgements, goods receipts)
  - Quality management records (holds, inspection failures)
  - AP records (invoice disputes, payment patterns)
  - CRM/SRM records (sales rep interaction logs)

OUTPUTS:
  - SupplierStressProfile per supplier (composite 0-12 score)
  - Risk-level classification: Green / Amber / Red / Critical
  - First-signal detection (earliest month showing stress)
  - Recommended actions with intervention deadlines
  - Advance warning estimate in months

DECISION AUTHORITY: RECOMMEND (qualify_alternative) | ESCALATE (Critical)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.agents.base import ProcurementAgent
from framework.agents.catalogue import SUPPLIER_STRESS_METADATA
from framework.core.types import (
    AgentCapability, AgentMetadata, DecisionAuthority,
    Finding, Severity, Task,
)


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalScore:
    """Score for one behavioural signal."""
    name:            str
    score:           int          # 0=normal | 1=watch | 2=alert
    baseline_value:  float
    current_value:   float
    change_pct:      float
    interpretation:  str


@dataclass(frozen=True)
class SupplierStressProfile:
    """Complete stress analysis for one supplier."""
    supplier_id:       str
    supplier_name:     str
    composite_score:   int          # 0-12
    risk_level:        str          # Green|Amber|Red|Critical
    trend_direction:   str          # Improving|Stable|Deteriorating
    first_signal_month: Optional[str]
    months_analysed:   int
    signals:           Tuple[SignalScore, ...]
    advance_warning_months: int


# ---------------------------------------------------------------------------
# Signal extraction — pure functions (no LLM, no I/O)
# ---------------------------------------------------------------------------

_SIX_SIGNALS_SPEC = [
    # (field, display_name, warn_pct, alert_pct, higher_is_worse)
    ("po_ack_days",           "PO Acknowledgement Time (days)",  20, 50,  True),
    ("otd_pct",               "On-Time Delivery Rate (%)",         3,  8, False),
    ("quality_hold_pct",      "Quality Hold Rate (%)",            30, 80,  True),
    ("invoice_disputes",      "Invoice Dispute Count",            50,150,  True),
    ("unsolicited_discounts", "Unsolicited Discounts",            50,200,  True),
    ("sales_response_hours",  "Sales Response Time (hours)",      40,100,  True),
]

_FIRST_SIGNAL_THRESHOLDS = {
    "po_ack_days":           lambda v: v > 4.5,
    "otd_pct":               lambda v: v < 88,
    "quality_hold_pct":      lambda v: v > 2.5,
    "invoice_disputes":      lambda v: v > 3,
    "unsolicited_discounts": lambda v: v > 0,
    "sales_response_hours":  lambda v: v > 18,
}


def _avg_field(records: List[Dict], field: str) -> float:
    values = [r[field] for r in records if r.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _score_one_signal(
    current: float, baseline: float,
    warn_pct: float, alert_pct: float,
    higher_is_worse: bool,
) -> Tuple[int, float]:
    """Return (score, raw_change_pct)."""
    if baseline == 0:
        return 0, 0.0
    raw_pct = ((current - baseline) / abs(baseline)) * 100.0
    change  = raw_pct if higher_is_worse else -raw_pct
    score   = 2 if change >= alert_pct else 1 if change >= warn_pct else 0
    return score, round(raw_pct, 1)


def extract_supplier_stress_profile(
    monthly_records: List[Dict],
) -> Optional[SupplierStressProfile]:
    """
    Compute a SupplierStressProfile from monthly transaction records.

    Pure function — no I/O, no side effects. Thread-safe.

    Requires: >= 4 months of data.
    Baseline = first 3 months; Current = last 3 months.
    """
    if len(monthly_records) < 4:
        return None

    records   = sorted(monthly_records, key=lambda r: r["month"])
    baseline  = records[:3]
    current   = records[-3:]
    supplier_id   = records[0]["supplier_id"]
    supplier_name = records[0]["supplier_name"]

    signals: List[SignalScore] = []
    for field, display, warn, alert, higher in _SIX_SIGNALS_SPEC:
        b_val = _avg_field(baseline, field)
        c_val = _avg_field(current, field)
        sc, raw_pct = _score_one_signal(c_val, b_val, warn, alert, higher)
        signals.append(SignalScore(
            name=display,
            score=sc,
            baseline_value=round(b_val, 2),
            current_value=round(c_val, 2),
            change_pct=raw_pct,
            interpretation={
                0: f"{display}: within normal range",
                1: f"{display}: marginal deterioration ({raw_pct:+.1f}%)",
                2: f"{display}: significant deterioration ({raw_pct:+.1f}%) — action needed",
            }[sc],
        ))

    composite = sum(s.score for s in signals)
    risk_level = (
        "Critical" if composite >= 10
        else "Red"    if composite >= 7
        else "Amber"  if composite >= 4
        else "Green"
    )

    # Trend via early vs late half
    mid = len(records) // 2
    def _half_score(recs):
        total = 0
        for fld, _, w, a, hiw in _SIX_SIGNALS_SPEC[:3]:
            b = _avg_field(records[:3], fld)
            c = _avg_field(recs, fld)
            sc, _ = _score_one_signal(c, b, w, a, hiw)
            total += sc
        return total

    early_score = _half_score(records[:mid])
    late_score  = _half_score(records[mid:])
    trend = (
        "Deteriorating" if late_score > early_score + 1
        else "Improving" if early_score > late_score + 1
        else "Stable"
    )

    # First signal month
    first_signal_month = None
    for r in records:
        if any(thresh(r.get(fld, 0)) for fld, thresh in _FIRST_SIGNAL_THRESHOLDS.items()):
            first_signal_month = r["month"]
            break

    # Estimate advance warning
    advance_warning = 0
    if first_signal_month and risk_level in ("Red", "Critical"):
        try:
            last_month = int(records[-1]["month"].split("-")[1])
            first_month = int(first_signal_month.split("-")[1])
            advance_warning = max(0, last_month - first_month)
        except (ValueError, IndexError):
            advance_warning = 0

    return SupplierStressProfile(
        supplier_id=supplier_id,
        supplier_name=supplier_name,
        composite_score=composite,
        risk_level=risk_level,
        trend_direction=trend,
        first_signal_month=first_signal_month,
        months_analysed=len(records),
        signals=tuple(signals),
        advance_warning_months=advance_warning,
    )


def analyse_supplier_portfolio(records: List[Dict]) -> List[SupplierStressProfile]:
    """Group records by supplier and extract profiles. Thread-safe."""
    by_supplier: Dict[str, List[Dict]] = {}
    for r in records:
        by_supplier.setdefault(r.get("supplier_id", "UNKNOWN"), []).append(r)

    profiles = [
        p for p in (
            extract_supplier_stress_profile(recs)
            for recs in by_supplier.values()
        )
        if p is not None
    ]
    return sorted(profiles, key=lambda p: p.composite_score, reverse=True)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior procurement risk analyst with 20 years of experience
detecting supplier financial distress from operational behaviour signals.

Your analysis style:
- Reason from observed patterns, never from external assumptions
- Distinguish transient noise from systematic trend
- Provide specific intervention timelines, not vague recommendations
- Estimate advance warning windows based on signal emergence dates

Output ONLY valid JSON with no preamble or markdown fences."""


def build_user_prompt(observations: Dict[str, Any]) -> str:
    profiles = observations.get("supplier_profiles", [])
    signal_blocks = []
    for p in profiles:
        sigs = "\n".join(
            f"    [{s['score']}/2] {s['name']}: "
            f"{s['baseline_value']} → {s['current_value']} ({s['change_pct']:+.1f}%) | "
            f"{s['interpretation']}"
            for s in p["signals"]
        )
        signal_blocks.append(
            f"SUPPLIER: {p['supplier_name']} ({p['supplier_id']})\n"
            f"  Risk: {p['risk_level']} | Score: {p['composite_score']}/12 | "
            f"Trend: {p['trend_direction']} | "
            f"First signal: {p.get('first_signal_month', 'none')} | "
            f"Active: {p['active_signals']}/6\n"
            f"  SIGNALS:\n{sigs}"
        )

    return f"""Analyse supplier stress signals across {observations['supplier_count']} suppliers.
Analysis date: {observations['analysis_date']}
Data window: {observations['months_of_data']} months

{chr(10).join(signal_blocks)}

Return JSON:
{{
  "executive_summary": "2-3 sentences on overall portfolio risk",
  "suppliers": [
    {{
      "supplier_id": "...",
      "supplier_name": "...",
      "risk_level": "Green|Amber|Red|Critical",
      "composite_score": 0,
      "advance_warning_estimate_months": 0,
      "key_finding": "most important procurement implication",
      "recommended_action": "specific action with timing",
      "watch_indicators": ["what to monitor in next 30 days"],
      "intervention_deadline": "how long before this becomes unmanageable"
    }}
  ],
  "immediate_priorities": ["ordered suppliers needing action this week"]
}}
Order by risk (Critical first)."""


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

class SupplierStressAgent(ProcurementAgent):
    """
    Procurement Intelligence Agent — Mystery 02: Supplier Stress Signal.

    Implements ProcurementAgent.observe/reason/parse_findings.
    Registered in the global agent registry via METADATA class attribute.
    """

    METADATA = SUPPLIER_STRESS_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        """
        Extract and score all six signals from transaction data.

        Payload keys:
            transaction_data (List[Dict]): Monthly supplier transaction records.
            analysis_date (str):          Label for the analysis period.
        """
        records = task.payload.get("transaction_data", [])
        if not records:
            raise ValueError("Payload must contain 'transaction_data'")

        profiles = analyse_supplier_portfolio(records)
        self._log.info(
            "[agent=%s] Scored %d supplier profiles",
            self.agent_id, len(profiles),
        )

        return {
            "supplier_count":    len(profiles),
            "analysis_date":     task.payload.get("analysis_date", "unspecified"),
            "months_of_data":    profiles[0].months_analysed if profiles else 0,
            "supplier_profiles": [
                {
                    "supplier_id":        p.supplier_id,
                    "supplier_name":      p.supplier_name,
                    "composite_score":    p.composite_score,
                    "risk_level":         p.risk_level,
                    "trend_direction":    p.trend_direction,
                    "first_signal_month": p.first_signal_month,
                    "active_signals":     sum(1 for s in p.signals if s.score > 0),
                    "advance_warning_months": p.advance_warning_months,
                    "signals": [
                        {
                            "name":             s.name,
                            "score":            s.score,
                            "baseline_value":   s.baseline_value,
                            "current_value":    s.current_value,
                            "change_pct":       s.change_pct,
                            "interpretation":   s.interpretation,
                        }
                        for s in p.signals
                    ],
                }
                for p in profiles
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        """Call the reasoning engine (LLM or configured strategy)."""
        if self.reasoning:
            from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy
            request = ReasoningRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(observations),
                observations=observations,
                strategy=ReasoningStrategy.STRUCTURED,
                session_id=str(task.session_id),
            )
            resp = self.reasoning.reason(request)
            return resp.content
        else:
            # No reasoning engine — return structured stub for testing
            return json.dumps({
                "executive_summary": "Reasoning engine not configured.",
                "suppliers": [],
                "immediate_priorities": [],
            })

    def parse_findings(
        self,
        observations: Dict[str, Any],
        reasoning: str,
        task: Task,
    ) -> List[Finding]:
        """Parse LLM JSON output into typed Finding objects."""
        parsed = self._parse_llm_json(reasoning)
        if not parsed:
            return [self._make_finding(
                finding_type="parse_error",
                severity=Severity.HIGH,
                summary="LLM did not return valid JSON",
                detail=reasoning[:400],
            )]

        findings: List[Finding] = []

        # Portfolio-level finding
        if summary := parsed.get("executive_summary"):
            findings.append(self._make_finding(
                finding_type="portfolio_summary",
                severity=Severity.INFO,
                summary=summary,
                metadata={"immediate_priorities": parsed.get("immediate_priorities", [])},
            ))

        # Per-supplier findings
        risk_to_severity = {
            "Critical": Severity.CRITICAL,
            "Red":       Severity.HIGH,
            "Amber":     Severity.MEDIUM,
            "Green":     Severity.LOW,
        }
        for supplier in parsed.get("suppliers", []):
            risk    = supplier.get("risk_level", "Unknown")
            severity = risk_to_severity.get(risk, Severity.INFO)
            if risk in ("Green",):
                continue  # No action finding for healthy suppliers

            findings.append(self._make_finding(
                finding_type="supplier_stress",
                severity=severity,
                entity_id=supplier.get("supplier_id"),
                entity_name=supplier.get("supplier_name"),
                summary=supplier.get("key_finding", ""),
                detail=(
                    f"Composite score: {supplier.get('composite_score',0)}/12. "
                    f"Advance warning: {supplier.get('advance_warning_estimate_months',0)} months."
                ),
                recommended_action=supplier.get("recommended_action", ""),
                evidence={
                    "composite_score":       supplier.get("composite_score"),
                    "advance_warning":       supplier.get("advance_warning_estimate_months"),
                    "intervention_deadline": supplier.get("intervention_deadline"),
                    "watch_indicators":      supplier.get("watch_indicators", []),
                },
                confidence=0.85 if risk == "Critical" else 0.75,
            ))

        return findings

    def _make_finding(self, finding_type, severity, summary, detail="",
                      entity_id=None, entity_name=None, recommended_action="",
                      evidence=None, confidence=1.0, metadata=None):
        """Override to attach metadata to Finding."""
        finding = super()._make_finding(
            finding_type, severity, summary, detail,
            entity_id, entity_name, recommended_action, evidence, confidence,
        )
        if metadata:
            # Finding is frozen — build new with metadata in evidence
            from dataclasses import replace
            existing = dict(finding.evidence)
            existing.update(metadata)
            return Finding(
                finding_id=finding.finding_id,
                finding_type=finding.finding_type,
                severity=finding.severity,
                entity_id=finding.entity_id,
                entity_name=finding.entity_name,
                summary=finding.summary,
                detail=finding.detail,
                evidence=existing,
                recommended_action=finding.recommended_action,
                decision_authority=finding.decision_authority,
                confidence_score=finding.confidence_score,
                tags=finding.tags,
            )
        return finding
