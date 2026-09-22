# Security Posture

This document states plainly what SKEIN's security controls do, how to turn them on, and — just as importantly — what they do **not** do. For full evidence and defect history, see [SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md](SKEIN-Second-Pass-Architecture-and-Technical-Assessment.md).

**Verified baseline (2026-09-16):** 417 tests pass, including authZ, tenant-boundary, schema, groundedness, secrets, JWKS, quota, chaos, and API tests.

## 1. Threat model this covers

SKEIN's security work targets a **multi-tenant, service-to-service deployment**: a backend task-submission API called by trusted internal services or gateways on behalf of ~200 customers, not a public, browser-facing consumer product.

## 2. Controls, and how to enable them

All of the following are **disabled by default**. Nothing changes behavior for library/test usage unless explicitly configured — this was a deliberate design choice so adding a control never silently breaks an existing deployment.

| Control | Module | Enable via |
|---|---|---|
| Input length / JSON-depth validation | `framework/security/controls.py` | `security.enable_input_sanitisation: true` in `config.yaml` |
| PII redaction in logs | `framework/security/controls.py` | `security.enable_pii_redaction: true` |
| Rate limiting (per-tenant when `tenant_id` is set, else per agent type) | `framework/security/controls.py` | `security.rate_limit_requests_per_minute: <N>` |
| Prompt-injection mitigation | `framework/reasoning/engine.py` | Always on — delimits and defangs every LLM prompt automatically |
| API-key authentication | `framework/auth/api_keys.py` | `SKEIN_API_KEYS` env var (JSON: `{"tenant_id": "raw_key"}`) |
| API-key authorization (tenant match) | `framework/auth/api_keys.py` | Automatic once any keys are configured |
| OIDC claim normalization + issuer/audience/expiry validation | `framework/adapters/identity/oidc.py` | Configure an identity adapter behind a trusted gateway |
| RS256/JWKS signature verification | `framework/adapters/identity/jwks.py` | Verify compact JWT against trusted JWKS before normalization |
| RBAC review authorization | `framework/security/authorization.py` | `reviewer` or `admin` role when a principal is supplied |
| Payload and LLM-output JSON Schema validation | `framework/core/schema.py` | Agent metadata / `ReasoningRequest.output_schema` |
| Groundedness warnings | `framework/agents/base.py` | Always on after finding parsing |
| Token quota | `framework/billing/ledger.py` | Configure per-tenant `TokenQuotaEnforcer` quota |
| Secrets adapters | `framework/adapters/secrets/` | `secrets.provider`: `environment`, `file`, or `vault` |
| Tenant identifier validation (SQL/identifier injection) | `framework/multitenancy/context.py` | Always on — `TenantContext` rejects anything outside `^[A-Za-z_][A-Za-z0-9_]*$` |
| Parameterized Databricks SQL | `platform/databricks/adapter.py` | Always on — values pass through `args={}`, never string-interpolated |

## 3. What "API-key auth" actually guarantees

Once `SKEIN_API_KEYS` has at least one entry:

- Every `POST /v1/tasks` request must present `Authorization: Bearer <key>` or `X-API-Key: <key>`, or it gets **401**.
- The authenticated key's tenant becomes the tenant identity for the request. If the request body *also* names a `tenant_id`, it must match, or the request gets **403** — this is what stops tenant A's key from being used to act as tenant B.
- Keys are stored as SHA-256 hashes only; the raw key is never retained after `ApiKeyStore.register()` returns; comparison is constant-time (`hmac.compare_digest`).

## 4. What it does NOT guarantee — read this before deploying

- **No turnkey IdP integration.** OIDC mapping and JWKS signature verification exist, but token acquisition, JWKS refresh/caching, gateway header stripping, logout/session policy, and live IdP validation remain deployment responsibilities. Never expose `X-Principal-Claims` directly to untrusted clients.
- **No transport security included.** The task API and health/metrics endpoints run over plain HTTP. Put a TLS-terminating reverse proxy/gateway in front of anything not on a fully private network.
- **`/health`, `/ready`, `/metrics`, `/status` are completely unauthenticated.** Never expose that port outside a private cluster network.
- **Secrets adapters are not production certification.** Environment/file/Vault KV v2 adapters and config selection exist. The Vault adapter is protocol-tested against a local mock only; live Vault/KMS access controls, rotation policy, audit, and availability must be validated before go-live.
- **Tenant validation is an allowlist, not identity verification.** `TenantRegistry` membership checks stop typos and requests for unprovisioned tenants; they do not, by themselves, prove a caller is who they claim to be — that's what the API-key check above is for. If you run the task API *without* any keys configured, tenant_id is fully caller-asserted and unauthenticated.
- **Prompt-injection mitigation is defence-in-depth, not a guarantee.** Regex-based neutralization cannot catch every adversarial phrasing, and no LLM is guaranteed to honor a "treat this as data" instruction.
- **No data-exfiltration controls.** Nothing currently stops an LLM from being tricked into embedding secrets from its context into its output.
- **CI scanning is a gate, not an assurance claim.** CI runs Bandit and `pip-audit`, plus chaos/load tests. Container image scanning and organization-specific policy remain deployment responsibilities.
- **No compliance claims are made or implied** (SOC 2, ISO 27001, GDPR, HIPAA, etc.) — none of this has been assessed against any formal standard.

## 5. Multi-tenant data isolation

Logical isolation is the default product profile, enforced through tenant auth, scoped memory/path/logging helpers, findings/usage filters, and tenant configuration. Physical isolation remains available through dedicated Delta catalogs (`TenantContext`/`TenantStoreResolver`) for regulated deployments. Provisioning those catalogs is outside this repository. Some legacy call sites are not yet automatically wrapped by the logical-isolation helpers; deployment composition must use the scoped adapters consistently.

## 6. Reporting a concern

This is an independent research project (see [README.md](../README.md)). Open a GitHub issue for anything you believe is a security defect; do not open a public issue for anything you believe is actively exploitable in a live deployment you don't control — there is no dedicated security contact/process beyond standard GitHub issue triage.
