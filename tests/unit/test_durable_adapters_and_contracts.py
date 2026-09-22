import tempfile
import unittest
from pathlib import Path

from framework.adapters.audit import InMemoryAuditSink
from framework.adapters.identity import Principal
from framework.billing.jsonl_ledger import JsonlUsageLedger
from framework.findings.jsonl_feedback import JsonlFeedbackStore
from framework.findings.jsonl_store import JsonlFindingsStore
from framework.findings.store import FindingRecord
from framework.api.openapi import ROUTES, build_openapi_spec
from framework.governance.logger import GovernanceLogger
from framework.multitenancy.isolation import tenant_scoped_path


class TestJsonlPersistenceAdapters(unittest.TestCase):
    def test_findings_usage_and_feedback_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            finding_path = Path(directory) / "findings.jsonl"
            findings = JsonlFindingsStore(finding_path)
            findings.add(FindingRecord(
                finding_id="f1", tenant_id="acme", task_id="t1", agent_type="A",
                severity="high", finding_type="risk", summary="issue", entity_id=None,
                confidence_score=0.7,
            ))
            self.assertEqual(len(JsonlFindingsStore(finding_path)), 1)

            usage_path = Path(directory) / "usage.jsonl"
            usage = JsonlUsageLedger(usage_path)
            usage.record("acme", 100, "t1")
            self.assertEqual(JsonlUsageLedger(usage_path).total_tokens("acme"), 100)

            feedback_path = Path(directory) / "feedback.jsonl"
            feedback = JsonlFeedbackStore(feedback_path)
            feedback.record("f1", "A", 0.7, True, "reviewer")
            self.assertEqual(len(JsonlFeedbackStore(feedback_path)), 1)


class TestTenantGovernanceAndOpenApi(unittest.TestCase):
    def test_governance_logger_for_tenant_uses_safe_scoped_path(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = GovernanceLogger.for_tenant(directory, "acme")
            logger.audit("test", {"tenant_id": "acme"})
            self.assertTrue((Path(directory) / "tenant_acme" / "audit.jsonl").exists())
            with self.assertRaises(ValueError):
                GovernanceLogger.for_tenant(directory, "../other")

    def test_openapi_declares_workflow_routes(self):
        spec = build_openapi_spec()
        for method, path in (("POST", "/v1/workflows"), ("POST", "/v1/workflows/async")):
            self.assertIn(path, spec["paths"])
            self.assertIn(method.lower(), spec["paths"][path])


if __name__ == "__main__":
    unittest.main()
