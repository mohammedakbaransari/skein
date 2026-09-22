import json
import sys
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.adapters.identity.oidc import OIDCClaimsError, OIDCIdentityAdapter
from framework.adapters.audit.http_sink import AuditDeliveryError, HttpAuditSink
from framework.adapters.audit.interfaces import AuditEvent
from framework.multitenancy.scoped_memory import TenantScopedMemoryStore
from framework.memory.store import WorkingMemory
from framework.api.webhooks import WebhookDispatcher
from framework.agents.evaluation import EvaluationHarness, GoldenCase
from framework.core.types import AgentMetadata, Severity, Task
from framework.agents.base import StructuralAgent


def _future_ts(seconds: int = 3600) -> int:
    return int((datetime.now(timezone.utc) + timedelta(seconds=seconds)).timestamp())


class TestOIDCIdentityAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = OIDCIdentityAdapter(
            trusted_issuers=["https://issuer.example"],
            accepted_audiences=["skein-api"],
        )

    def test_valid_claims_map_to_principal(self):
        principal = self.adapter.normalize({
            "iss": "https://issuer.example", "aud": "skein-api",
            "sub": "user-1", "tid": "acme", "roles": ["reviewer"],
            "exp": _future_ts(),
        })
        self.assertEqual(principal.tenant_id, "acme")
        self.assertEqual(principal.roles, ("reviewer",))

    def test_untrusted_issuer_is_rejected(self):
        with self.assertRaises(OIDCClaimsError):
            self.adapter.normalize({
                "iss": "https://evil.example", "aud": "skein-api",
                "sub": "user-1", "tid": "acme", "exp": _future_ts(),
            })

    def test_expired_token_is_rejected(self):
        with self.assertRaises(OIDCClaimsError):
            self.adapter.normalize({
                "iss": "https://issuer.example", "aud": "skein-api",
                "sub": "user-1", "tid": "acme", "exp": _future_ts(-3600),
            })

    def test_missing_tenant_claim_is_rejected(self):
        with self.assertRaises(OIDCClaimsError):
            self.adapter.normalize({
                "iss": "https://issuer.example", "aud": "skein-api",
                "sub": "user-1", "exp": _future_ts(),
            })


class _EchoHandler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _EchoHandler.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestHttpAuditSink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_event_is_delivered_as_json_post(self):
        _EchoHandler.received.clear()
        sink = HttpAuditSink(f"http://127.0.0.1:{self.port}/ingest")
        sink.emit(AuditEvent("evt-1", "task.completed", "acme", "completed", "success"))
        self.assertEqual(len(_EchoHandler.received), 1)
        self.assertEqual(_EchoHandler.received[0]["tenant_id"], "acme")

    def test_unreachable_endpoint_raises(self):
        sink = HttpAuditSink("http://127.0.0.1:1/ingest", timeout_seconds=1)
        with self.assertRaises(AuditDeliveryError):
            sink.emit(AuditEvent("evt-1", "task.completed", "acme", "completed", "success"))


class TestTenantScopedMemoryStore(unittest.TestCase):
    def test_isolates_two_tenants_sharing_one_backing_store(self):
        backing = WorkingMemory()
        acme = TenantScopedMemoryStore(backing, "acme")
        globex = TenantScopedMemoryStore(backing, "globex")
        acme.set("k", "acme-value")
        globex.set("k", "globex-value")
        self.assertEqual(acme.get("k"), "acme-value")
        self.assertEqual(globex.get("k"), "globex-value")
        self.assertEqual(acme.keys(), ["k"])


class TestWebhookDispatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_dispatches_only_findings_meeting_severity_threshold(self):
        _EchoHandler.received.clear()
        dispatcher = WebhookDispatcher()
        dispatcher.subscribe("acme", f"http://127.0.0.1:{self.port}/hook", min_severity="high")
        dispatcher.dispatch_finding("acme", {"finding_id": "f1", "severity": "low"})
        dispatcher.dispatch_finding("acme", {"finding_id": "f2", "severity": "critical"})
        self.assertEqual(len(_EchoHandler.received), 1)
        self.assertEqual(_EchoHandler.received[0]["finding_id"], "f2")

    def test_delivery_failure_is_recorded_not_raised(self):
        dispatcher = WebhookDispatcher(timeout_seconds=1)
        dispatcher.subscribe("acme", "http://127.0.0.1:1/hook", min_severity="info")
        dispatcher.dispatch_finding("acme", {"finding_id": "f3", "severity": "info"})
        records = dispatcher.deliveries("acme")
        self.assertFalse(records[0].succeeded)


class _EvalAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_EvalAgent", display_name="Eval", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task):
        return {}

    def reason(self, observations, task):
        return "{}"

    def parse_findings(self, observations, reasoning, task):
        if observations.get("risk") == "high":
            return [self._make_finding("signal", Severity.CRITICAL, "elevated risk")]
        return [self._make_finding("signal", Severity.LOW, "nominal")]


class TestEvaluationHarness(unittest.TestCase):
    def test_golden_cases_pass_and_fail_correctly(self):
        agent = _EvalAgent()
        harness = EvaluationHarness(agent, lambda: Task.create("_EvalAgent", {}))
        results = harness.run([
            GoldenCase("high_risk", {"risk": "high"}, "", expected_severities=("critical",)),
            GoldenCase("nominal", {"risk": "low"}, "", expected_severities=("critical",)),
        ])
        self.assertTrue(results[0].passed)
        self.assertFalse(results[1].passed)
        self.assertFalse(EvaluationHarness.all_passed(results))


if __name__ == "__main__":
    unittest.main()
