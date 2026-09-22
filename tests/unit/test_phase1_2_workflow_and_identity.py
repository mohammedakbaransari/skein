import base64
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
from framework.adapters.identity.oidc import OIDCIdentityAdapter
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, Severity, Task
from framework.multitenancy.context import TenantContext, TenantRegistry
from framework.multitenancy.isolation import tenant_scoped_path
from framework.multitenancy.scoped_logging import get_tenant_logger
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.resilience.retry import reset_circuit_registry


class _WfEchoAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_WfEchoAgent", display_name="WfEcho", description="Test",
        version="1.0", capabilities=(),
    )

    def observe(self, task: Task):
        return {"echo": task.payload.get("msg", "")}

    def reason(self, obs, task):
        return json.dumps(obs)

    def parse_findings(self, obs, reasoning, task):
        return [self._make_finding("echo", Severity.INFO, f"echo:{obs['echo']}")]


def _post(port, path, body, headers=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestWorkflowSubmissionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_WfEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        from framework.api.jobs import JobStore
        cls.job_store = JobStore()
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry,
            job_store=cls.job_store,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def test_sync_workflow_runs_dependent_steps(self):
        status, body = _post(self.port, "/v1/workflows", {
            "name": "wf1", "tenant_id": "acme",
            "steps": [
                {"agent_type": "_WfEchoAgent", "payload": {"msg": "first"}},
                {"agent_type": "_WfEchoAgent", "payload": {"msg": "second"}, "depends_on": [0]},
            ],
        })
        self.assertEqual(status, 200)
        self.assertTrue(body["succeeded"])
        self.assertEqual(len(body["task_results"]), 2)

    def test_workflow_with_invalid_dependency_returns_400(self):
        status, body = _post(self.port, "/v1/workflows", {
            "name": "wf2", "tenant_id": "acme",
            "steps": [{"agent_type": "_WfEchoAgent", "payload": {"msg": "x"}, "depends_on": [0]}],
        })
        self.assertEqual(status, 400)

    def test_async_workflow_submission_polls_to_completion(self):
        status, body = _post(self.port, "/v1/workflows/async", {
            "name": "wf3", "tenant_id": "acme",
            "steps": [{"agent_type": "_WfEchoAgent", "payload": {"msg": "hi"}}],
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
        self.assertIn("task_results", body)


class TestTrustedClaimsPrincipalPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_registry()
        reset_circuit_registry()
        cls.registry = AgentRegistry()
        cls.registry.register_class(_WfEchoAgent)
        cls.orch = TaskOrchestrator(cls.registry, config=None)
        cls.tenant_registry = TenantRegistry()
        cls.tenant_registry.register(TenantContext(tenant_id="acme", catalog="tenant_acme"))
        cls.adapter = OIDCIdentityAdapter(
            trusted_issuers=["https://issuer.example"], accepted_audiences=["skein-api"],
        )
        cls.port = start_task_api_server(
            cls.orch, cls.tenant_registry, port=0, agent_registry=cls.registry,
            identity_adapter=cls.adapter,
        )

    @classmethod
    def tearDownClass(cls):
        stop_task_api_server()

    def _claims_header(self, **overrides):
        import time
        claims = {
            "iss": "https://issuer.example", "aud": "skein-api",
            "sub": "user-1", "tid": "acme", "roles": ["reviewer"],
            "exp": int(time.time()) + 3600,
        }
        claims.update(overrides)
        return base64.b64encode(json.dumps(claims).encode()).decode()

    def test_valid_claims_header_succeeds(self):
        status, body = _post(
            self.port, "/v1/tasks", {"agent_type": "_WfEchoAgent", "payload": {"msg": "hi"}, "tenant_id": "acme"},
            headers={"X-Principal-Claims": self._claims_header()},
        )
        self.assertEqual(status, 200)

    def test_untrusted_issuer_claims_rejected_with_401(self):
        status, body = _post(
            self.port, "/v1/tasks", {"agent_type": "_WfEchoAgent", "payload": {"msg": "hi"}, "tenant_id": "acme"},
            headers={"X-Principal-Claims": self._claims_header(iss="https://evil.example")},
        )
        self.assertEqual(status, 401)


class TestIsolationPathAndLoggingHelpers(unittest.TestCase):
    def test_tenant_scoped_path_rejects_traversal(self):
        with self.assertRaises(ValueError):
            tenant_scoped_path("/logs", "../etc")
        scoped = tenant_scoped_path("/logs", "acme")
        self.assertTrue(scoped.endswith("tenant_acme"))

    def test_tenant_logger_tags_records_with_tenant_id(self):
        import logging
        records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = get_tenant_logger("test.tenant.logger", "acme")
        logger.logger.addHandler(_CaptureHandler())
        logger.logger.setLevel(logging.INFO)
        logger.info("hello")
        self.assertEqual(records[-1].tenant_id, "acme")


if __name__ == "__main__":
    unittest.main()
