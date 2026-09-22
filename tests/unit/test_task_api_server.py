"""
tests/unit/test_task_api_server.py
=====================================
Integration tests for the task-submission API (framework/api/server.py) —
starts a real HTTP server on an ephemeral port and issues real HTTP
requests via urllib (no extra test dependencies).
"""

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.api.server import start_task_api_server, stop_task_api_server
from framework.auth.api_keys import ApiKeyStore, generate_api_key
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, Severity, Task
from framework.multitenancy.context import TenantContext, TenantRegistry
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.resilience.retry import reset_circuit_registry
from framework.resilience.pool import PoolExhaustedError


class _ApiEchoAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_ApiEchoAgent", display_name="ApiEcho", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
        input_schema={
            "type": "object",
            "required": ["msg"],
            "properties": {"msg": {"type": "string", "minLength": 1}},
        },
    )

    def observe(self, task: Task):
        return {"echo": task.payload.get("msg", "")}

    def reason(self, obs, task):
        return json.dumps({"echo": obs["echo"]})

    def parse_findings(self, obs, reasoning, task):
        return [self._make_finding("echo", Severity.INFO, f"echo:{obs['echo']}")]


def _post(port: int, path: str, body: dict, headers: dict = None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    return _request_with_connect_retry(req)


def _request_with_connect_retry(req, attempts: int = 5):
    """Retry only transient connection-level failures (e.g. this stdlib
    single-threaded HTTPServer occasionally aborting an in-flight read
    under rapid sequential test load) — never retries an actual server
    response (HTTPError)."""
    import http.client
    import time
    last_exc = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
        except (urllib.error.URLError, ConnectionError, http.client.HTTPException, OSError) as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(0.05 * (attempt + 1))
    raise last_exc


class TestTaskAPIServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_ApiEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry
        )
        assert cls.port, "task API server failed to bind an ephemeral port"

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_valid_submission_with_known_tenant_succeeds(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "acme",
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["succeeded"])
        self.assertEqual(len(body["findings"]), 1)

    def test_missing_tenant_id_is_rejected(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"},
        })
        self.assertEqual(status, 400)
        self.assertIn("tenant_id", body["error"])

    def test_blank_tenant_id_is_rejected(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "",
        })
        self.assertEqual(status, 400)

    def test_unknown_tenant_id_is_rejected(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "does-not-exist",
        })
        self.assertEqual(status, 400)
        self.assertIn("unknown tenant_id", body["error"])

    def test_unknown_agent_type_is_rejected(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_NoSuchAgent", "payload": {}, "tenant_id": "acme",
        })
        self.assertEqual(status, 400)

    def test_missing_payload_is_rejected(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "tenant_id": "acme",
        })
        self.assertEqual(status, 400)

    def test_payload_schema_error_is_rejected_with_field_path(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": 42}, "tenant_id": "acme",
        })
        self.assertEqual(status, 400)
        self.assertIn("$.payload.msg", body["error"])

    def test_oversized_request_body_is_rejected_with_413(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "x" * 1_100_000},
            "tenant_id": "acme",
        })
        self.assertEqual(status, 413)

    def test_unknown_route_returns_404(self):
        status, body = _post(self.port, "/v1/nope", {})
        self.assertEqual(status, 404)


class TestTaskAPIServerWithAuth(unittest.TestCase):
    """Auth-enabled mode: an ApiKeyStore with keys in it makes the API
    require a valid key and derive tenant_id from it, per
    framework/auth/api_keys.py."""

    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_ApiEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.tenant_registry.register(TenantContext(tenant_id="globex", catalog="tenant_globex"))
        cls.key_store = ApiKeyStore()
        cls.acme_key = generate_api_key()
        cls.key_store.register("acme", cls.acme_key)
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, api_key_store=cls.key_store,
            agent_registry=cls.registry,
        )
        assert cls.port, "task API server failed to bind an ephemeral port"

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_valid_key_without_body_tenant_id_succeeds(self):
        status, body = _post(
            self.port, "/v1/tasks",
            {"agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}},
            headers={"Authorization": f"Bearer {self.acme_key}"},
        )
        self.assertEqual(status, 200)

        self.assertTrue(body["succeeded"])

    def test_valid_key_with_matching_body_tenant_id_succeeds(self):
        status, body = _post(
            self.port, "/v1/tasks",
            {"agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}, "tenant_id": "acme"},
            headers={"Authorization": f"Bearer {self.acme_key}"},
        )
        self.assertEqual(status, 200)

    def test_missing_key_is_rejected_with_401(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}, "tenant_id": "acme",
        })
        self.assertEqual(status, 401)

    def test_invalid_key_is_rejected_with_401(self):
        status, body = _post(
            self.port, "/v1/tasks",
            {"agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}},
            headers={"Authorization": "Bearer skein_not-a-real-key"},
        )
        self.assertEqual(status, 401)

    def test_valid_key_acting_as_different_tenant_is_rejected_with_403(self):
        """The authZ check: acme's key cannot be used to submit work as globex."""
        status, body = _post(
            self.port, "/v1/tasks",
            {"agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}, "tenant_id": "globex"},
            headers={"Authorization": f"Bearer {self.acme_key}"},
        )
        self.assertEqual(status, 403)

    def test_x_api_key_header_also_accepted(self):
        status, body = _post(
            self.port, "/v1/tasks",
            {"agent_type": "_ApiEchoAgent", "payload": {"msg": "hi"}},
            headers={"X-API-Key": self.acme_key},
        )
        self.assertEqual(status, 200)


class TestTaskAPIServerBackPressure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class ExhaustedOrchestrator:
            def run_task(self, task):
                raise PoolExhaustedError(task.agent_type, 0.5)

        cls.port = start_task_api_server(
            ExhaustedOrchestrator(), TenantRegistry(), port=0, max_request_body_bytes=1000
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_pool_exhaustion_returns_retryable_429(self):
        data = json.dumps({
            "agent_type": "_ApiEchoAgent", "payload": {}, "tenant_id": "acme",
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/tasks", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(raised.exception.code, 429)
        self.assertEqual(raised.exception.headers["Retry-After"], "1")


def _get(port: int, path: str, headers: dict = None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers=headers or {}, method="GET",
    )
    return _request_with_connect_retry(req)


class TestTaskAPIServerFindingsAndReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        from framework.findings.store import FindingsStore
        from framework.findings.review import ReviewWorkflow

        cls.registry = AgentRegistry()
        cls.registry.register_class(_ApiEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.findings_store = FindingsStore()
        cls.review_workflow = ReviewWorkflow()
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry,
            findings_store=cls.findings_store, review_workflow=cls.review_workflow,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_submitted_task_findings_are_queryable(self):
        status, _ = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "acme",
        })
        self.assertEqual(status, 200)
        status, body = _get(self.port, "/v1/findings?tenant_id=acme")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["count"], 1)

    def test_findings_query_requires_tenant_id(self):
        status, body = _get(self.port, "/v1/findings")
        self.assertEqual(status, 400)

    def test_review_transition_endpoint_updates_state(self):
        status, body = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "acme",
        })
        finding_id = body["findings"][0]["finding_id"]
        status, body = _post(self.port, f"/v1/findings/{finding_id}/review", {
            "state": "in_review", "assignee": "alice",
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "in_review")
        self.assertEqual(body["assignee"], "alice")

    def test_invalid_review_transition_returns_400(self):
        status, body = _post(self.port, "/v1/findings/does-not-exist/review", {
            "state": "actioned",
        })
        self.assertEqual(status, 400)


class TestTaskAPIServerAsync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        from framework.api.jobs import JobStore

        cls.registry = AgentRegistry()
        cls.registry.register_class(_ApiEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.job_store = JobStore()
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry,
            job_store=cls.job_store,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_async_submission_returns_202_then_polls_to_completion(self):
        status, body = _post(self.port, "/v1/tasks/async", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "acme",
        })
        self.assertEqual(status, 202)
        job_id = body["job_id"]

        import time
        for _ in range(20):
            status, body = _get(self.port, f"/v1/tasks/async/{job_id}")
            if status == 200:
                break
            time.sleep(0.05)
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "succeeded")
        self.assertTrue(body["succeeded"])

    def test_poll_unknown_job_returns_404(self):
        status, body = _get(self.port, "/v1/tasks/async/does-not-exist")
        self.assertEqual(status, 404)


class TestTaskAPIServerWebhooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        from framework.api.webhooks import WebhookDispatcher
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading as _threading
        import json as _json

        class _Hook(BaseHTTPRequestHandler):
            received = []

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                _Hook.received.append(_json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        cls.hook_server = HTTPServer(("127.0.0.1", 0), _Hook)
        cls.hook_thread = _threading.Thread(target=cls.hook_server.serve_forever, daemon=True)
        cls.hook_thread.start()
        cls.hook = _Hook

        cls.registry = AgentRegistry()
        cls.registry.register_class(_ApiEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.dispatcher = WebhookDispatcher()
        cls.dispatcher.subscribe("acme", f"http://127.0.0.1:{cls.hook_server.server_address[1]}/hook", min_severity="info")
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry,
            webhook_dispatcher=cls.dispatcher,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()
        cls.hook_server.shutdown()

    def test_submission_triggers_webhook_delivery(self):
        self.hook.received.clear()
        status, _ = _post(self.port, "/v1/tasks", {
            "agent_type": "_ApiEchoAgent", "payload": {"msg": "hello"}, "tenant_id": "acme",
        })
        self.assertEqual(status, 200)
        self.assertEqual(len(self.hook.received), 1)


if __name__ == "__main__":
    unittest.main()
