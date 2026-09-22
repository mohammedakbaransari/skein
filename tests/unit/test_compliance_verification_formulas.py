"""
tests/unit/test_compliance_verification_formulas.py
======================================================
Hand-computed numeric-correctness tests for
agents.compliance.compliance_verification.analyse_compliance_portfolio
(roadmap item P2-2).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.compliance.compliance_verification import analyse_compliance_portfolio


def _profile(profiles, supplier_id):
    return next(p for p in profiles if p.supplier_id == supplier_id)


class TestComplianceVerificationFormulas(unittest.TestCase):

    def setUp(self):
        self.records = [
            {"supplier_id": "S1", "supplier_name": "LowRisk Co",
             "certifications": [{"status": "verified"}, {"status": "verified"}],
             "discrepancy_signals": []},
            {"supplier_id": "S2", "supplier_name": "CriticalCo",
             "certifications": [{"status": "unverified"}, {"status": "unverified"}, {"status": "unverified"}],
             "discrepancy_signals": []},
            {"supplier_id": "S3", "supplier_name": "SignalCo",
             "certifications": [{"status": "verified"}],
             "discrepancy_signals": ["media_report"]},
            {"supplier_id": "S4", "supplier_name": "PendingCo",
             "certifications": [{"status": "pending"}],
             "discrepancy_signals": []},
        ]
        self.profiles = analyse_compliance_portfolio(self.records)

    def test_s1_low_risk(self):
        p = _profile(self.profiles, "S1")
        self.assertEqual(p.certifications_held, 2)
        self.assertEqual(p.certifications_valid, 2)
        self.assertEqual(p.certifications_at_risk, 0)
        self.assertEqual(p.risk_tier, "low")

    def test_s2_critical_more_than_two_at_risk(self):
        p = _profile(self.profiles, "S2")
        self.assertEqual(p.certifications_at_risk, 3)
        self.assertEqual(p.risk_tier, "critical")

    def test_s3_high_due_to_discrepancy_signal(self):
        p = _profile(self.profiles, "S3")
        self.assertEqual(p.certifications_at_risk, 0)
        self.assertEqual(p.risk_tier, "high")

    def test_s4_medium_certs_present_but_none_verified(self):
        p = _profile(self.profiles, "S4")
        self.assertEqual(p.certifications_valid, 0)
        self.assertEqual(p.certifications_at_risk, 0)
        self.assertEqual(p.risk_tier, "medium")

    def test_sorted_by_risk_tier_severity(self):
        ids = [p.supplier_id for p in self.profiles]
        self.assertEqual(ids, ["S2", "S3", "S4", "S1"])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(analyse_compliance_portfolio([]), [])


if __name__ == "__main__":
    unittest.main()
