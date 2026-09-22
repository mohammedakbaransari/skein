# Changelog

This project does not yet follow a formal release/versioning scheme (see [README.md](README.md) — it is an independent research project accompanying a paper). This changelog records the substantial remediation work performed against the findings in [docs/SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md](docs/SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md), in the order it was done. Each entry links to the assessment section with full evidence, test results, and known limitations.

## Enterprise embedding implementation (2026-09-16)

- Added provider-neutral contracts and adapters for workflow execution, secrets, identity, audit transport, and local durable storage.
- Added synchronous/asynchronous task and workflow APIs, job polling, findings query, review transitions, webhooks, and an OpenAPI 3.0.3 contract.
- Added calibrated confidence profiles, reviewer approval/provenance, groundedness checks, input/output schema validation, and golden-case evaluation gates.
- Added logical tenant isolation helpers, tenant-scoped memory/log/path/governance support, `TenantPolicy`, audited tenant configuration, and capability-driven pool sizing.
- Added OIDC claim normalization, RS256/JWKS signature verification, principal propagation, and RBAC enforcement for finding-review transitions. Live IdP integration remains deployment-specific.
- Added environment, file, and Vault KV v2 secrets adapters plus config-driven startup selection. Vault is protocol-tested against a local mock, not a live cluster.
- Added findings, feedback, usage, and billing foundations with durable local JSONL adapters, token quotas, CSV export, and reconciliation reporting.
- Added dead-letter capture/replay, chaos/failure-injection tests, Bandit/pip-audit CI gates, and load-test gating.
- Current verification baseline: **417 tests pass**; `python -m compileall -q .` succeeds. Ten consecutive full-suite runs passed after stabilizing transient stdlib HTTP test connections.

Remaining external validation: live Databricks/Unity Catalog, real IdP/JWKS retrieval, live Vault, confirmed SIEM/OTLP endpoint, production database reconciliation, and multi-region DR drills.

## Governance and security fixes (P0)

- **Fixed governance hash-chain integrity** under concurrent writers and across process restarts — both were empirically proven broken, then fixed by moving chain state to a per-path, lock-guarded class-level dict, seeded from the file's last valid hash on construction. (§28)
- **Parameterized all Databricks SQL** in `platform/databricks/adapter.py` — values now pass through `args={}` instead of f-string interpolation. (§28, tests: `test_databricks_adapter.py`)
- **Implemented the `security:` config block for real** — input length/depth validation, PII redaction, rate limiting, all disabled by default and opt-in via config. New module: `framework/security/controls.py`. (§28)

## Reliability fixes (P1)

- **Orchestrator workflow timeout no longer raises an unhandled exception** — outstanding tasks are marked `TaskStatus.TIMEOUT` and reported in `WorkflowResult.timed_out_tasks` instead. (§29)
- **Built a real Helm chart** (`deploy/helm/Chart.yaml` + `templates/`) — previously only `values.yaml` existed with no chart scaffolding. Validated with `helm lint`/`helm template`. (§29)
- **Replaced the broken CI workflow** (referenced a nonexistent `environment.yml`) with a working pip-based `.github/workflows/ci.yml` (test matrix + Helm lint). (§29)

## Test coverage and mitigations (P2)

- **Added hand-computed-value numeric-correctness tests for all 15 agents'** deterministic `observe()` formulas — previously untested at the value level. (§30)
- **Added prompt-injection mitigation** — every LLM request is now delimited and defanged automatically via `framework/reasoning/engine.py`'s `_harden_request()`, backed by `framework/security/controls.py::neutralize_prompt_injection`/`wrap_untrusted_data`. (§31)

## Institutional memory and retry idempotency (P3)

- **`InstitutionalMemoryAgent` now actually recalls precedent** instead of only writing it — `observe()` reads back a per-category pattern index before reasoning. (§32)
- **Added `Task.idempotency_key`**, stable across `for_retry()` (unlike `task_id`), as an opt-in tool for retry-safe agent side effects. A companion test empirically confirmed the underlying risk (naive retries duplicate side effects) before demonstrating the fix. (§32)

## Multi-tenant foundation

- **`TenantId` + `Task.tenant_id`**, and `framework/multitenancy/context.py` (`TenantContext`, `TenantRegistry`) — physical per-tenant isolation (dedicated Delta Lake catalog per tenant), routing only, not provisioning. (§33)
- **`DeltaGovernanceStore`** (new) — tenant-partitioned, hash-chained governance log on Delta, reusing the same chain algorithm as the local-file backend via a new shared `framework/governance/hashchain.py`. (§33)
- **Tenant-scoped rate limiting** in `BaseAgent.run()`. (§33)

## Task-submission API

- **`POST /v1/tasks`** (`framework/api/server.py`) — the first way to submit work to a running SKEIN server over the network; previously only the health/metrics server existed. `tenant_id` is mandatory on every request. Live-verified against a real running server, not just unit tests. (§34)

## AuthN/authZ

- **`framework/auth/api_keys.py`** — `ApiKeyStore` (hash-only storage, constant-time comparison) and `authorize_tenant_match()`. Wired into the task API: 401 for missing/invalid keys, 403 for a valid key used to act as a different tenant. Opt-in via `SKEIN_API_KEYS`. (§35)

## Per-tenant storage routing

- **`framework/multitenancy/resolver.py::TenantStoreResolver`** — resolves and caches per-tenant Delta stores; wired into `BaseAgent.run()` via a temporary, request-scoped swap of `memory`/`governance` (restored afterward). Also fixed a previously-undiscovered bug where `InstitutionalMemory` was constructed in `scripts/server.py` but never actually injected into any agent. (§36)

## Documentation

- Added `docs/ARCHITECTURE.md`, `docs/SECURITY.md`, `docs/USAGE.md`, `docs/DEPLOYMENT.md`, and this changelog.
- Updated `README.md` and `CONTRIBUTING.md` for the current test count, task API, and multi-tenancy/security model.

---

Historical evidence for the original remediation is in [docs/SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md](docs/SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md) §28–§36. It is an archived 312-test snapshot. Current limitations are maintained in the enterprise roadmap and living architecture/security/deployment guides; notably, no live Databricks, IdP, Vault, SIEM, production database, or DR validation is claimed.
