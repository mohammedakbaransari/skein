"""
framework/api/server.py
==========================
Minimal task-submission HTTP API — closes the gap identified in the
architecture assessment (§6/§15/§33): scripts/server.py previously started
only the health/metrics server, with no way to actually submit a task to
the running orchestrator over the network.

Deliberately built on the stdlib http.server, the same way
framework/observability/health.py is, to avoid adding a web-framework
dependency to the minimal core (pyyaml/requests only) for a v1 API.

Endpoints:
  POST /v1/tasks   Submit a single task, run it synchronously, return the
                   AgentResult as JSON.

TENANT IDENTITY: EXPLICIT, AND AUTHENTICATED WHEN AN ApiKeyStore IS SET
=========================================================================
Every request must resolve to a non-empty tenant_id — a missing one is a
400, never a silent single-tenant fallback.

Two modes, chosen per-server by whether an ApiKeyStore with any keys in
it was passed to start_task_api_server():

  - No ApiKeyStore configured (or an empty one): tenant_id must be given
    explicitly in the request body. If a TenantRegistry has any tenants
    registered, it must also be one of them (400 otherwise) — a
    structural allowlist check, not authentication; it stops typos and
    requests for unprovisioned tenants, it does NOT verify the caller is
    actually allowed to act as the tenant_id they claim.

  - ApiKeyStore configured with at least one key: every request must
    present a valid key (`Authorization: Bearer <key>` or `X-API-Key`
    header), or the request is rejected with 401 before the body is even
    parsed. The AUTHENTICATED key's tenant becomes the tenant_id for the
    request; a body-supplied tenant_id is only accepted if it matches the
    authenticated tenant (403 otherwise — this is what stops one tenant's
    credentials being used to act as another tenant). See
    framework/auth/api_keys.py.

Either way: this API still expects a trusted network path to it (a
reverse proxy/gateway terminating TLS, etc.) — API-key auth here answers
"which tenant is this," it does not replace transport security.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from framework.core.types import SessionId, Task, TenantId
from framework.multitenancy.context import TenantRegistry
from framework.auth.api_keys import (
    ApiKeyStore, AuthContext, AuthenticationError, AuthorizationError,
    authorize_tenant_match,
)
from framework.core.schema import validate_json_schema, SchemaValidationError
from framework.resilience.pool import PoolExhaustedError
from framework.billing.ledger import QuotaExceededError
from framework.adapters.workflow import InMemoryWorkflowEngine, WorkflowDefinition
from framework.security.authorization import AuthorizationPolicy, RoleAuthorizationError
from urllib.parse import urlparse, parse_qs

log = logging.getLogger(__name__)


class TaskAPIError(Exception):
    """Raised for any request that should produce a 4xx response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def _estimate_tokens(payload: Dict[str, Any], requested: Any = None) -> int:
    if requested is not None:
        try:
            return max(1, int(requested))
        except (TypeError, ValueError) as exc:
            raise TaskAPIError(400, "estimated_tokens must be an integer") from exc
    return max(1, len(json.dumps(payload, default=str)) // 4)


def _parse_task_request(
    body: Dict[str, Any],
    tenant_registry: Optional[TenantRegistry],
    auth: Optional[AuthContext] = None,
    agent_registry=None,
    principal=None,
) -> Task:
    agent_type = body.get("agent_type")
    if not agent_type or not isinstance(agent_type, str):
        raise TaskAPIError(400, "'agent_type' is required and must be a string")

    payload = body.get("payload")
    if payload is None or not isinstance(payload, dict):
        raise TaskAPIError(400, "'payload' is required and must be a JSON object")

    if agent_registry is not None:
        try:
            schema = agent_registry.get_metadata(agent_type).input_schema
        except KeyError:
            schema = {}
        if schema:
            try:
                validate_json_schema(payload, schema, path="$.payload")
            except SchemaValidationError as exc:
                raise TaskAPIError(400, f"payload schema validation failed: {exc}") from exc

    body_tenant_id = body.get("tenant_id")

    if auth is not None:
        # Authenticated request: tenant identity comes from the API key,
        # not the client-editable request body. A body tenant_id is only
        # accepted if it agrees with the authenticated tenant — this is
        # the actual authorization check (one tenant's key cannot be used
        # to act as another tenant), raised as AuthorizationError (403).
        tenant_id = authorize_tenant_match(auth, body_tenant_id)
    else:
        # No API key store configured for this server — same explicit,
        # caller-supplied tenant_id contract as before auth existed.
        if not body_tenant_id or not isinstance(body_tenant_id, str):
            raise TaskAPIError(400, "'tenant_id' is required and must be a non-empty string")
        tenant_id = body_tenant_id

    if tenant_registry is not None and len(tenant_registry) > 0 and tenant_id not in tenant_registry:
        raise TaskAPIError(400, f"unknown tenant_id {tenant_id!r} — not registered")

    kwargs: Dict[str, Any] = {"tenant_id": TenantId(tenant_id)}
    if principal is not None:
        kwargs["principal_id"] = principal.subject_id
        kwargs["principal_roles"] = tuple(principal.roles)
    session_id = body.get("session_id")
    if session_id:
        kwargs["session_id"] = SessionId(str(session_id))
    if "priority" in body:
        kwargs["priority"] = int(body["priority"])
    if "timeout_seconds" in body:
        kwargs["timeout_seconds"] = int(body["timeout_seconds"])

    return Task.create(agent_type=agent_type, payload=payload, **kwargs)


def _parse_workflow_request(
    body: Dict[str, Any],
    tenant_registry: Optional[TenantRegistry],
    auth: Optional[AuthContext] = None,
    agent_registry=None,
):
    """Build a `Workflow` from a JSON DAG description (R7 remainder):
    {"name": "...", "steps": [{"agent_type": "...", "payload": {...},
    "depends_on": [<earlier step index>, ...]}], "tenant_id": "..."}."""
    from framework.orchestration.orchestrator import Workflow
    from framework.core.types import SessionId

    steps = body.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TaskAPIError(400, "'steps' is required and must be a non-empty array")

    session_id = SessionId.generate()
    tasks = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TaskAPIError(400, f"steps[{index}] must be a JSON object")
        step_body = {**step, "tenant_id": step.get("tenant_id", body.get("tenant_id"))}
        task = _parse_task_request(step_body, tenant_registry, auth, agent_registry)
        task.session_id = session_id
        for dep_index in step.get("depends_on", []):
            if not isinstance(dep_index, int) or not (0 <= dep_index < index):
                raise TaskAPIError(400, f"steps[{index}].depends_on references an invalid earlier step")
            task.depends_on.append(tasks[dep_index].task_id)
        tasks.append(task)

    return Workflow(
        workflow_id=f"wf-{tasks[0].task_id.value}", name=body.get("name", "workflow"),
        session_id=session_id, tasks=tasks,
        max_workers=int(body.get("max_workers", 4)),
        timeout_seconds=int(body.get("timeout_seconds", 600)),
    )


def _workflow_result_to_dict(result) -> Dict[str, Any]:
    return {
        "workflow_id": result.workflow_id, "workflow_name": result.workflow_name,
        "succeeded": result.succeeded, "failed_tasks": result.failed_tasks,
        "cancelled_tasks": result.cancelled_tasks, "timed_out_tasks": result.timed_out_tasks,
        "duration_ms": result.duration_ms,
        "task_results": {k: v.to_dict() for k, v in result.task_results.items()},
    }


class _TaskAPIHandler(BaseHTTPRequestHandler):
    """Routes POST /v1/tasks to the orchestrator injected on the server."""

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/v1/tasks":
            self._handle_submit_task()
        elif path == "/v1/tasks/async":
            self._handle_submit_task_async()
        elif path == "/v1/workflows":
            self._handle_submit_workflow()
        elif path == "/v1/workflows/async":
            self._handle_submit_workflow_async()
        elif path.startswith("/v1/findings/") and path.endswith("/review"):
            self._handle_review_transition(path)
        else:
            self._respond(404, {"error": f"no such route: POST {path}"})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/findings":
            self._handle_query_findings(parse_qs(parsed.query))
        elif parsed.path.startswith("/v1/tasks/async/"):
            self._handle_poll_async_task(parsed.path.rsplit("/", 1)[-1])
        else:
            self._respond(404, {"error": f"no such route: GET {parsed.path}"})

    def _handle_submit_task(self) -> None:
        try:
            auth = self._authenticate_if_required()
            principal = self._extract_principal_if_configured()
            body = self._read_json_body()
            task = _parse_task_request(body, self.server.tenant_registry, auth, self.server.agent_registry, principal)  # type: ignore[attr-defined]
            quota_enforcer = getattr(self.server, "quota_enforcer", None)
            usage_ledger = getattr(self.server, "usage_ledger", None)
            estimated_tokens = _estimate_tokens(task.payload, body.get("estimated_tokens"))
            if quota_enforcer is not None and task.tenant_id:
                quota_enforcer.check(str(task.tenant_id), estimated_tokens)
            result = self.server.orchestrator.run_task(task)  # type: ignore[attr-defined]
            if usage_ledger is not None and task.tenant_id:
                usage_ledger.record(
                    str(task.tenant_id), result.llm_tokens_used or estimated_tokens,
                    str(task.task_id),
                )
            findings_store = getattr(self.server, "findings_store", None)
            if findings_store is not None and task.tenant_id:
                findings_store.add_result(str(task.tenant_id), str(task.task_id), task.agent_type, result)
            webhook_dispatcher = getattr(self.server, "webhook_dispatcher", None)
            if webhook_dispatcher is not None and task.tenant_id:
                for finding in result.to_dict().get("findings", []):
                    webhook_dispatcher.dispatch_finding(str(task.tenant_id), finding)
            self._respond(200 if result.succeeded else 422, result.to_dict())
        except AuthenticationError as exc:
            self._respond(401, {"error": str(exc)})
        except AuthorizationError as exc:
            self._respond(403, {"error": str(exc)})
        except PoolExhaustedError as exc:
            self._respond(429, {"error": str(exc)}, {"Retry-After": "1"})
        except QuotaExceededError as exc:
            self._respond(429, {"error": str(exc)}, {"Retry-After": "60"})
        except TaskAPIError as exc:
            self._respond(exc.status_code, {"error": exc.message})
        except KeyError as exc:
            # Unknown agent_type — AgentRegistry raises KeyError, not
            # something this API should ever surface as a 500.
            self._respond(400, {"error": str(exc)})
        except Exception as exc:
            from framework.security.controls import get_security_enforcer
            log.error("[task-api] Unhandled error submitting task: %s", get_security_enforcer().redact(str(exc)), exc_info=True)
            self._respond(500, {"error": "internal error processing task"})

    def _handle_submit_task_async(self) -> None:
        job_store = getattr(self.server, "job_store", None)
        if job_store is None:
            self._respond(404, {"error": "async job store not configured on this server"})
            return
        try:
            auth = self._authenticate_if_required()
            principal = self._extract_principal_if_configured()
            body = self._read_json_body()
            task = _parse_task_request(body, self.server.tenant_registry, auth, self.server.agent_registry, principal)  # type: ignore[attr-defined]
            job = job_store.submit(self.server.orchestrator, task)  # type: ignore[attr-defined]
            self._respond(202, {"job_id": job.job_id, "state": job.state.value})
        except AuthenticationError as exc:
            self._respond(401, {"error": str(exc)})
        except AuthorizationError as exc:
            self._respond(403, {"error": str(exc)})
        except TaskAPIError as exc:
            self._respond(exc.status_code, {"error": exc.message})

    def _handle_submit_workflow(self) -> None:
        try:
            auth = self._authenticate_if_required()
            body = self._read_json_body()
            workflow = _parse_workflow_request(body, self.server.tenant_registry, auth, self.server.agent_registry)  # type: ignore[attr-defined]
            quota_enforcer = getattr(self.server, "quota_enforcer", None)
            if quota_enforcer and workflow.tasks:
                estimated = sum(_estimate_tokens(t.payload) for t in workflow.tasks)
                tenant_id = str(workflow.tasks[0].tenant_id) if workflow.tasks[0].tenant_id else ""
                if tenant_id:
                    quota_enforcer.check(tenant_id, estimated)
            result = self._run_workflow_via_engine(workflow)
            self._respond(200 if result.succeeded else 422, _workflow_result_to_dict(result))
        except AuthenticationError as exc:
            self._respond(401, {"error": str(exc)})
        except AuthorizationError as exc:
            self._respond(403, {"error": str(exc)})
        except TaskAPIError as exc:
            self._respond(exc.status_code, {"error": exc.message})
        except QuotaExceededError as exc:
            self._respond(429, {"error": str(exc)}, {"Retry-After": "60"})
        except (KeyError, ValueError) as exc:
            self._respond(400, {"error": str(exc)})

    def _handle_submit_workflow_async(self) -> None:
        job_store = getattr(self.server, "job_store", None)
        if job_store is None:
            self._respond(404, {"error": "async job store not configured on this server"})
            return
        try:
            auth = self._authenticate_if_required()
            body = self._read_json_body()
            workflow = _parse_workflow_request(body, self.server.tenant_registry, auth, self.server.agent_registry)  # type: ignore[attr-defined]
            quota_enforcer = getattr(self.server, "quota_enforcer", None)
            if quota_enforcer and workflow.tasks:
                estimated = sum(_estimate_tokens(t.payload) for t in workflow.tasks)
                tenant_id = str(workflow.tasks[0].tenant_id) if workflow.tasks[0].tenant_id else ""
                if tenant_id:
                    quota_enforcer.check(tenant_id, estimated)
            job = job_store.submit_workflow(self.server.orchestrator, workflow)  # type: ignore[attr-defined]
            self._respond(202, {"job_id": job.job_id, "state": job.state.value})
        except AuthenticationError as exc:
            self._respond(401, {"error": str(exc)})
        except AuthorizationError as exc:
            self._respond(403, {"error": str(exc)})
        except TaskAPIError as exc:
            self._respond(exc.status_code, {"error": exc.message})
        except QuotaExceededError as exc:
            self._respond(429, {"error": str(exc)}, {"Retry-After": "60"})
        except (KeyError, ValueError) as exc:
            self._respond(400, {"error": str(exc)})

    def _handle_poll_async_task(self, job_id: str) -> None:
        job_store = getattr(self.server, "job_store", None)
        if job_store is None:
            self._respond(404, {"error": "async job store not configured on this server"})
            return
        from framework.api.jobs import JobState
        job = job_store.get(job_id)
        if job is None:
            self._respond(404, {"error": f"no such job {job_id!r}"})
            return
        if job.state in (JobState.PENDING, JobState.RUNNING):
            self._respond(202, {"job_id": job.job_id, "state": job.state.value})
            return
        if job.result is not None and hasattr(job.result, "to_dict"):
            payload = job.result.to_dict()
        elif job.result is not None:
            payload = _workflow_result_to_dict(job.result)
        else:
            payload = {"error": job.error}
        payload["state"] = job.state.value
        self._respond(200, payload)

    def _handle_query_findings(self, query: Dict[str, list]) -> None:
        findings_store = getattr(self.server, "findings_store", None)
        if findings_store is None:
            self._respond(404, {"error": "findings store not configured on this server"})
            return
        try:
            auth = self._authenticate_if_required()
        except AuthenticationError as exc:
            self._respond(401, {"error": str(exc)})
            return
        tenant_id = auth.tenant_id if auth is not None else (query.get("tenant_id", [None])[0])
        if not tenant_id:
            self._respond(400, {"error": "'tenant_id' is required"})
            return
        if auth is not None and query.get("tenant_id", [tenant_id])[0] != tenant_id:
            self._respond(403, {"error": "cannot query findings for a different tenant"})
            return
        severity = query.get("severity", [None])[0]
        since = query.get("since", [None])[0]
        records = findings_store.query(tenant_id=tenant_id, severity=severity, since=since)
        self._respond(200, {"findings": [r.to_dict() for r in records], "count": len(records)})

    def _handle_review_transition(self, path: str) -> None:
        review_workflow = getattr(self.server, "review_workflow", None)
        if review_workflow is None:
            self._respond(404, {"error": "review workflow not configured on this server"})
            return
        finding_id = path.split("/")[3]
        try:
            principal = self._extract_principal_if_configured()
            AuthorizationPolicy().require_any_role(principal, {"reviewer", "admin"}, "finding review transition")
            body = self._read_json_body()
            from framework.findings.review import ReviewState
            new_state = ReviewState(body.get("state", ""))
            review = review_workflow.transition(
                finding_id, new_state,
                assignee=body.get("assignee", ""), comment=body.get("comment", ""),
            )
            self._respond(200, {
                "finding_id": review.finding_id, "state": review.state.value,
                "assignee": review.assignee, "comment": review.comment,
                "updated_at": review.updated_at,
            })
        except RoleAuthorizationError as exc:
            self._respond(403, {"error": str(exc)})
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
        except TaskAPIError as exc:
            self._respond(exc.status_code, {"error": exc.message})

    def _run_workflow_via_engine(self, workflow):
        """Route workflow execution through the WorkflowEngine adapter (R0) —
        the durable-workflow interface is always the enforced execution
        path, even while the concrete engine remains in-memory."""
        engine = getattr(self.server, "workflow_engine", None) or InMemoryWorkflowEngine(self.server.orchestrator)
        definition = WorkflowDefinition(
            workflow_id=workflow.workflow_id, name=workflow.name, tasks=workflow.tasks,
            session_id=workflow.session_id, max_workers=workflow.max_workers,
            timeout_seconds=workflow.timeout_seconds, cancel_on_failure=workflow.cancel_on_failure,
        )
        return engine.start(definition).result

    def _authenticate_if_required(self) -> Optional[AuthContext]:
        """Returns None (auth disabled) when no keys are provisioned on this
        server — same opt-in-enforcement pattern as SecurityEnforcer (P0-3)
        and the TenantRegistry membership check: safe default for
        single-tenant/local/test use, real and enforced once configured."""
        key_store: Optional[ApiKeyStore] = getattr(self.server, "api_key_store", None)
        if key_store is None or len(key_store) == 0:
            return None
        return key_store.authenticate(self._extract_api_key())

    def _extract_api_key(self) -> Optional[str]:
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer "):].strip()
        return self.headers.get("X-API-Key")

    def _extract_principal_if_configured(self):
        """Normalize a trusted, already-validated claims header into a
        `Principal` when an `identity_adapter` (R12) is configured on this
        server — additive to tenant auth, never replaces it. The claims
        must already be validated upstream (a gateway/sidecar terminating
        OIDC); this adapter performs issuer/audience/expiry checks only,
        no signature verification (see `OIDCIdentityAdapter` docstring)."""
        adapter = getattr(self.server, "identity_adapter", None)
        raw_claims = self.headers.get("X-Principal-Claims")
        if adapter is None or not raw_claims:
            return None
        import base64
        try:
            claims = json.loads(base64.b64decode(raw_claims))
            return adapter.normalize(claims)
        except Exception as exc:
            raise TaskAPIError(401, f"invalid principal claims: {exc}") from exc

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise TaskAPIError(400, "Content-Length must be an integer") from exc
        if length <= 0:
            raise TaskAPIError(400, "request body is required")
        if length > self.server.max_request_body_bytes:  # type: ignore[attr-defined]
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(65_536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            raise TaskAPIError(413, "request body exceeds configured limit")
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TaskAPIError(400, f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise TaskAPIError(400, "request body must be a JSON object")
        return body

    def _respond(self, code: int, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        pass  # structured logging is used instead of the default access log


class TaskAPIServer(HTTPServer):
    """HTTPServer subclass carrying the dependencies the handler needs."""

    def __init__(self, address, handler_cls, orchestrator, tenant_registry, api_key_store=None, agent_registry=None, max_request_body_bytes=1_048_576, findings_store=None, review_workflow=None, job_store=None, webhook_dispatcher=None, identity_adapter=None, usage_ledger=None, quota_enforcer=None) -> None:
        super().__init__(address, handler_cls)
        self.orchestrator    = orchestrator
        self.tenant_registry = tenant_registry
        self.api_key_store   = api_key_store
        self.agent_registry  = agent_registry
        self.max_request_body_bytes = max_request_body_bytes
        self.findings_store  = findings_store
        self.review_workflow = review_workflow
        self.job_store        = job_store
        self.webhook_dispatcher = webhook_dispatcher
        self.identity_adapter = identity_adapter
        self.usage_ledger = usage_ledger
        self.quota_enforcer = quota_enforcer


_server_thread: Optional[threading.Thread] = None
_server_instance: Optional[TaskAPIServer] = None
_server_lock = threading.Lock()


def start_task_api_server(
    orchestrator,
    tenant_registry: Optional[TenantRegistry] = None,
    port: int = 8081,
    host: str = "0.0.0.0",
    api_key_store: Optional[ApiKeyStore] = None,
    agent_registry=None,
    max_request_body_bytes: int = 1_048_576,
    findings_store=None,
    review_workflow=None,
    job_store=None,
    webhook_dispatcher=None,
    identity_adapter=None,
    usage_ledger=None,
    quota_enforcer=None,
) -> Optional[int]:
    """
    Start the task-submission API in a background daemon thread.

    Safe to call multiple times — only starts one server. Mirrors
    framework.observability.health.start_health_server's lifecycle pattern.

    Returns the bound port (useful when port=0 is used to get an ephemeral
    port in tests), or None if binding failed.
    """
    global _server_thread, _server_instance

    with _server_lock:
        if _server_thread and _server_thread.is_alive():
            log.debug("[task-api] Server already running")
            return _server_instance.server_address[1] if _server_instance else None

        try:
            _server_instance = TaskAPIServer(
                (host, port), _TaskAPIHandler, orchestrator, tenant_registry, api_key_store,
                agent_registry, max_request_body_bytes, findings_store, review_workflow, job_store,
                webhook_dispatcher, identity_adapter, usage_ledger, quota_enforcer,
            )
        except OSError as exc:
            log.error("[task-api] Could not bind to %s:%d — %s", host, port, exc)
            return None

        bound_port = _server_instance.server_address[1]

        def _serve() -> None:
            log.info("[task-api] Server listening on %s:%d", host, bound_port)
            try:
                _server_instance.serve_forever()
            except Exception as exc:
                log.error("[task-api] Server error: %s", exc)

        _server_thread = threading.Thread(target=_serve, name="skein-task-api", daemon=True)
        _server_thread.start()
        return bound_port


def stop_task_api_server() -> None:
    """Gracefully stop the task API server. Idempotent."""
    global _server_instance, _server_thread
    thread = None
    with _server_lock:
        if _server_instance:
            _server_instance.shutdown()
            _server_instance.server_close()
            log.info("[task-api] Server stopped")
        thread = _server_thread
        _server_instance = None
        _server_thread = None
    if thread and thread is not threading.current_thread():
        thread.join(timeout=2.0)
