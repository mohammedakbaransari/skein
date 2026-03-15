"""
agents/compliance/compliance_verification.py
=============================================
Mystery 09 — The Compliance Theatre Problem

PROBLEM:
  Enterprises collect certifications. Nobody verifies what is actually
  happening in the supply chains those certificates attest to. The EU CSDDD,
  CSRD, US forced labour legislation, and equivalent regulations are shifting
  legal liability to the buying organisation for conditions throughout their
  supply chain — including Tier 2, 3, and 4 levels.

ROLE: Supply chain compliance verification analyst
AUTHORITY: ESCALATE (regulatory exposure requires leadership attention)
MYSTERY REF: mystery_09
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

from framework.agents.base import ProcurementAgent
from framework.agents.catalogue import COMPLIANCE_VERIFICATION_METADATA
from framework.core.types import Finding, Severity, Task
from framework.reasoning.engine import ReasoningRequest, ReasoningStrategy


@dataclass(frozen=True)
class SupplierComplianceProfile:
    supplier_id:          str
    supplier_name:        str
    certifications_held:  int
    certifications_valid: int
    certifications_at_risk: int
    last_verified:        Optional[str]
    discrepancy_signals:  tuple
    risk_tier:            str    # critical | high | medium | low


def analyse_compliance_portfolio(records: List[Dict]) -> List[SupplierComplianceProfile]:
    """Assess compliance posture per supplier. Pure function — thread-safe."""
    profiles = []
    for r in records:
        certs     = r.get("certifications", [])
        valid     = [c for c in certs if c.get("status") == "verified"]
        at_risk   = [c for c in certs if c.get("status") in ("unverified", "expired", "disputed")]
        signals   = r.get("discrepancy_signals", [])
        risk_tier = (
            "critical" if len(at_risk) > 2 or len(signals) >= 3
            else "high"    if len(at_risk) >= 1 or len(signals) >= 1
            else "medium"  if len(certs) > 0 and not valid
            else "low"
        )
        profiles.append(SupplierComplianceProfile(
            supplier_id=r.get("supplier_id", "UNKNOWN"),
            supplier_name=r.get("supplier_name", "Unknown"),
            certifications_held=len(certs),
            certifications_valid=len(valid),
            certifications_at_risk=len(at_risk),
            last_verified=r.get("last_verification_date"),
            discrepancy_signals=tuple(signals),
            risk_tier=risk_tier,
        ))
    return sorted(profiles, key=lambda p: {"critical": 0, "high": 1, "medium": 2, "low": 3}[p.risk_tier])


_SYSTEM_PROMPT = """You are a supply chain compliance specialist with expertise
in EU CSDDD, CSRD, US forced labour legislation, and ISO supply chain due diligence.

You surface the gap between certification on paper and reality in operations.
Your findings are regulatory-exposure-focused: what is the legal and reputational
risk, and what must be done before a regulator or journalist does it first.

Output ONLY valid JSON. No preamble."""


def _build_prompt(obs: Dict[str, Any]) -> str:
    supp_block = "\n".join(
        f"  {s['supplier_name']} ({s['supplier_id']}): "
        f"{s['certifications_held']} certs ({s['certifications_valid']} verified, "
        f"{s['certifications_at_risk']} at risk) | signals: {s['discrepancy_signal_count']} | "
        f"risk: {s['risk_tier']}"
        for s in obs.get("supplier_summaries", [])
    ) or "  No supplier data"
    return f"""Analyse compliance posture across {obs['supplier_count']} suppliers.
Critical risk suppliers: {obs['critical_count']}
High risk: {obs['high_count']}
Total certifications at risk: {obs['total_at_risk_certs']}

SUPPLIERS:
{supp_block}

Return JSON:
{{
  "compliance_assessment": "2-3 sentences on overall regulatory exposure",
  "regulatory_exposure_level": "Critical|High|Medium|Low",
  "findings": [
    {{
      "supplier_id": "...",
      "supplier_name": "...",
      "risk_tier": "critical|high|medium",
      "specific_gap": "what is claimed vs what is verifiable",
      "applicable_regulation": "CSDDD Art. X | CSRD | US CBP | etc.",
      "verification_action": "specific step to triangulate this certification",
      "timeline": "how urgently this must be addressed"
    }}
  ],
  "systemic_gaps": ["process-level compliance gaps"],
  "immediate_escalations": ["suppliers requiring board-level attention this week"]
}}"""


class ComplianceVerificationAgent(ProcurementAgent):
    """
    Procurement Intelligence Agent — Mystery 09: Compliance Theatre.

    Surfaces the gap between certification claims and observable supply
    chain reality. Focuses on regulatory exposure under EU CSDDD/CSRD
    and equivalent legislation.
    """

    METADATA = COMPLIANCE_VERIFICATION_METADATA

    def observe(self, task: Task) -> Dict[str, Any]:
        records = task.payload.get("compliance_records", [])
        if not records:
            raise ValueError("Payload must contain 'compliance_records'")
        profiles = analyse_compliance_portfolio(records)
        return {
            "supplier_count":      len(profiles),
            "critical_count":      sum(1 for p in profiles if p.risk_tier == "critical"),
            "high_count":          sum(1 for p in profiles if p.risk_tier == "high"),
            "total_at_risk_certs": sum(p.certifications_at_risk for p in profiles),
            "supplier_summaries": [
                {"supplier_id": p.supplier_id, "supplier_name": p.supplier_name,
                 "certifications_held": p.certifications_held,
                 "certifications_valid": p.certifications_valid,
                 "certifications_at_risk": p.certifications_at_risk,
                 "discrepancy_signal_count": len(p.discrepancy_signals),
                 "risk_tier": p.risk_tier}
                for p in profiles
            ],
        }

    def reason(self, observations: Dict[str, Any], task: Task) -> str:
        if not self.reasoning:
            return json.dumps({"compliance_assessment": "Reasoning engine not configured.",
                               "regulatory_exposure_level": "Unknown",
                               "findings": [], "systemic_gaps": [],
                               "immediate_escalations": []})
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
        exp_map = {"Critical": Severity.CRITICAL, "High": Severity.HIGH,
                   "Medium": Severity.MEDIUM, "Low": Severity.LOW}
        findings = [self._make_finding(
            finding_type="compliance_assessment",
            severity=exp_map.get(parsed.get("regulatory_exposure_level", "Medium"), Severity.MEDIUM),
            summary=parsed.get("compliance_assessment", ""),
            evidence={"systemic_gaps": parsed.get("systemic_gaps", []),
                      "immediate_escalations": parsed.get("immediate_escalations", [])},
        )]
        tier_map = {"critical": Severity.CRITICAL, "high": Severity.HIGH, "medium": Severity.MEDIUM}
        for f in parsed.get("findings", []):
            findings.append(self._make_finding(
                finding_type="compliance_gap",
                severity=tier_map.get(f.get("risk_tier", "medium"), Severity.MEDIUM),
                entity_id=f.get("supplier_id"), entity_name=f.get("supplier_name"),
                summary=f.get("specific_gap", ""),
                detail=f.get("applicable_regulation", ""),
                recommended_action=f.get("verification_action", ""),
                evidence={"timeline": f.get("timeline")},
            ))
        return findings
