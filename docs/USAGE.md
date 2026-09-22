# Usage Guide

Practical, current instructions for running SKEIN. For what each piece actually is under the hood, see [ARCHITECTURE.md](ARCHITECTURE.md). For security defaults, see [SECURITY.md](SECURITY.md).

## 1. Install

```bash
git clone https://github.com/mohammedakbaransari/skein
cd skein
pip install pyyaml requests   # core only
```

## 2. Run a single agent directly (no server, no LLM)

```python
from agents.supply_risk.supplier_stress import SupplierStressAgent
from framework.core.types import Task
from framework.reasoning.stubs import DryRunReasoningEngine

agent  = SupplierStressAgent(reasoning_engine=DryRunReasoningEngine())
task   = Task.create("SupplierStressAgent", {"transaction_data": your_data})
result = agent.run(task)

for finding in result.findings:
    print(f"[{finding.severity.value.upper()}] {finding.summary}")
```

## 3. Run the production server

```bash
python -m scripts.server --config config/config.yaml
```

This starts **two** HTTP servers:

| Server | Default port | Env var | Purpose |
|---|---|---|---|
| Health/metrics | 8080 | `SKEIN_HEALTH_PORT` | `/health`, `/ready`, `/metrics`, `/status` — unauthenticated, cluster-internal only |
| Platform API | 8081 | `SKEIN_TASK_API_PORT` | Tasks, workflows, jobs, findings, and review endpoints |

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/metrics
```

## 4. Submit a task over HTTP

Every request requires an explicit tenant identity — either in the body or (once API keys are configured) via the authenticated key. See [SECURITY.md](SECURITY.md) for the two modes.

**No auth configured** (dev/single-tenant — `tenant_id` in the body is trusted as-is):

```bash
curl -X POST http://localhost:8081/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
        "agent_type": "SupplierStressAgent",
        "tenant_id": "acme",
        "payload": {"transaction_data": [...]}
      }'
```

**With API-key auth configured** (`SKEIN_API_KEYS` set — tenant identity comes from the key, `tenant_id` in the body is optional and cross-checked if present):

```bash
curl -X POST http://localhost:8081/v1/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACME_API_KEY" \
  -d '{
        "agent_type": "SupplierStressAgent",
        "payload": {"transaction_data": [...]}
      }'
```

Response is the agent's `AgentResult` as JSON (`200` on success, `422` if the agent ran but failed its own validation, `400`/`401`/`403` for request-level rejections — see the table below).

| Status | Meaning |
|---|---|
| 200 | Task ran and `succeeded: true` |
| 400 | Malformed request — missing `agent_type`/`payload`/`tenant_id`, unknown `agent_type`, or unregistered `tenant_id` |
| 401 | API-key auth is enabled and no valid key was presented |
| 403 | A valid key was presented, but for a different tenant than the request claims |
| 404 | Unknown route |
| 413 | Request body exceeds `security.max_request_body_bytes` |
| 429 | Agent-pool back-pressure or projected token quota exceeded; inspect `Retry-After` |
| 422 | Task ran but the agent itself reported failure (e.g. missing required payload field) |
| 500 | Unexpected server error |

### Async tasks

```bash
curl -X POST http://localhost:8081/v1/tasks/async \
  -H "Content-Type: application/json" \
  -d '{"agent_type":"SupplierStressAgent","tenant_id":"acme","payload":{"transaction_data":[...]}}'

curl http://localhost:8081/v1/tasks/async/<job_id>
```

The first call returns `202` with a `job_id`. Polling returns `202` while pending/running and `200` when complete. `JobStore` is process-local and does not survive a restart.

### Workflows over HTTP

```bash
curl -X POST http://localhost:8081/v1/workflows/async \
  -H "Content-Type: application/json" \
  -d '{
    "name":"quarterly-review",
    "tenant_id":"acme",
    "steps":[
      {"agent_type":"SupplierStressAgent","payload":{"transaction_data":[...]}},
      {"agent_type":"DecisionAuditAgent","payload":{"decision_logs":[...]},"depends_on":[0]}
    ]
  }'
```

Use `/v1/workflows` for synchronous execution. `depends_on` contains zero-based indexes of earlier steps.

### Findings and review

```bash
curl "http://localhost:8081/v1/findings?tenant_id=acme&severity=high"

curl -X POST http://localhost:8081/v1/findings/<finding_id>/review \
  -H "Content-Type: application/json" \
  -d '{"state":"in_review","assignee":"buyer@example.com"}'
```

When a principal is supplied by a trusted identity gateway, review transitions require the `reviewer` or `admin` role. Findings support `tenant_id`, `severity`, and `since` filters.

## 5. Provisioning a tenant (development/testing)

There is no provisioning CLI yet — register tenants and API keys programmatically at startup, or directly in a script:

```python
from framework.multitenancy.context import get_tenant_registry, TenantContext
from framework.auth.api_keys import ApiKeyStore, generate_api_key

get_tenant_registry().register(TenantContext(tenant_id="acme", catalog="tenant_acme"))

store = ApiKeyStore()
raw_key = generate_api_key()          # give this to the tenant exactly once
store.register("acme", raw_key)
```

`SKEIN_API_KEYS='{"acme": "<raw_key>"}'` achieves the same for `scripts/server.py` at process startup. None of this creates the actual Databricks catalog — that must already exist (see [ARCHITECTURE.md §4](ARCHITECTURE.md#4-multi-tenancy-model)).

## 6. Run a multi-agent workflow (direct Python API)

The same workflow model is available through HTTP and the direct Python API:

```python
from framework.core.registry import get_registry
from framework.orchestration.orchestrator import TaskOrchestrator, WorkflowBuilder
from framework.core.types import SessionId

registry = get_registry()
registry.register_class(SupplierStressAgent)
registry.register_class(DecisionAuditAgent)

orch = TaskOrchestrator(registry, config=None)
sid  = SessionId.generate()

workflow = (
    WorkflowBuilder("quarterly-review")
    .session(sid)
    .step("SupplierStressAgent", {"transaction_data": transactions})
    .then("DecisionAuditAgent",  {"decision_logs":    decisions})
    .build()
)

result = orch.run_workflow(workflow)
print(f"Succeeded: {result.succeeded}")
print(f"Timed out: {result.timed_out_tasks}")
print(f"Findings:  {len(result.all_findings)}")
```

## 7. Configuration reference

`config/config.yaml` is the source of truth; every key has an environment-variable override (see the comments in the file itself). Key sections:

- `llm` — provider (`ollama`/`anthropic`/`openai`/`azure`), model, retry/timeout tuning.
- `reasoning` — strategy, optional LangChain/LangGraph/CrewAI integration flags.
- `resilience` — circuit breaker and retry tuning.
- `orchestration` — worker/timeout defaults.
- `agent_pool` — pool min/max size, acquire timeout.
- `memory` — `WorkingMemory` max entries, optional `InstitutionalMemory` file path.
- `governance` — local JSONL log directory.
- `secrets` — adapter selection (`environment`, `file`, or `vault`) and provider settings.
- `security` — see [SECURITY.md](SECURITY.md) for what each flag actually enforces.
- `observability` — health port, log level/format.
- `agent` — dry-run mode, verbose logging, trace directory.

## 8. Running tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The current baseline is **417 tests**. They require no live LLM, Vault, IdP, SIEM, or Databricks cluster; protocol adapters use local doubles/servers.

```bash
python -m unittest tests.chaos.test_failure_injection -v
python -m unittest tests.load.test_stress_load -v
python -m compileall -q .
```

## 9. Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker, Kubernetes, and Helm instructions.
