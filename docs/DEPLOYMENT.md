# Deployment Guide

Covers Docker, Kubernetes, and Helm. Deployment manifests render and lint; no live cluster, Vault, IdP, SIEM, Databricks, or DR exercise is claimed.

## Docker

```bash
docker build -f deploy/docker/Dockerfile -t skein-framework:latest .

docker run -p 8080:8080 -p 8081:8081 \
  -e LLM_PROVIDER=anthropic \
  -e LLM_API_KEY=sk-ant-... \
  -e SKEIN_API_KEYS='{"acme":"<raw_key>"}' \
  skein-framework:latest
```

Or via compose:

```bash
docker-compose -f deploy/docker/docker-compose.yml up
```

The image is a multi-stage build (builder + minimal non-root runtime), with an import-validation step at build time (fails the build if a core module has a syntax error).

## Kubernetes (raw manifest)

```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl rollout status deployment/skein-agents -n skein
```

Includes Deployment, HorizontalPodAutoscaler (2–10 replicas), PodDisruptionBudget, Service, ConfigMap, and a minimal ServiceAccount. Health/readiness/startup probes are wired to `/health`/`/ready`.

**Read this before relying on HPA:** `WorkingMemory`, `JobStore`, `ReviewWorkflow`, webhook subscriptions, and the default workflow engine are process-local. Local JSONL adapters persist findings/usage/feedback on one volume but do not coordinate concurrent pods. Horizontally scaled deployments must select shared production stores and, if restart recovery is required, a durable workflow adapter.

## Helm

```bash
helm lint ./deploy/helm
helm install skein ./deploy/helm --create-namespace
helm upgrade skein ./deploy/helm
```

`deploy/helm/values.yaml` controls image, replica count, autoscaling, resources, probes, LLM provider selection (via a Kubernetes secret — `llm.secretName`/`llm.secretKey`), and persistence for governance logs. Validated with `helm lint`/`helm template` against a rendered chart (all resources render correctly, conditional toggles for PVC/HPA/PDB work) — **not validated against a live cluster** in this repository.

## Environment variables reference

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `anthropic` \| `openai` \| `azure` |
| `LLM_MODEL` | `llama3.1` | Model name |
| `LLM_API_KEY` | — | Provider API key (never commit; env/secret only) |
| `SKEIN_MAX_WORKERS` | 4 | Orchestrator worker pool size |
| `SKEIN_GOVERNANCE_LOG_DIR` | `logs/governance` | Local JSONL governance log directory |
| `SKEIN_HEALTH_PORT` | 8080 | Health/metrics server port |
| `SKEIN_TASK_API_PORT` | 8081 | Task-submission API port |
| `SKEIN_API_KEYS` | — | JSON `{"tenant_id": "raw_key"}` — bootstrap-only, see [SECURITY.md](SECURITY.md) |
| `SKEIN_SECRETS_PROVIDER` | `environment` | `environment` \| `file` \| `vault` |
| `SKEIN_SECRETS_FILE_PATH` | — | JSON secrets file when using the file adapter |
| `VAULT_ADDR` / `VAULT_TOKEN` | — | Vault KV v2 endpoint/credential when using the Vault adapter |
| `SKEIN_CONFIDENCE_CALIBRATION_PATH` | — | Versioned per-agent confidence profile file |
| `SKEIN_LOG_LEVEL` | `INFO` | Log level |
| `SKEIN_LOG_JSON` | `true` | Structured JSON logs vs. human-readable |
| `AGENT_DRY_RUN` | `false` | Use `DryRunReasoningEngine` instead of a live LLM |

## What is NOT provided

- **Provisioning.** Nothing here creates a Databricks catalog, ADLS container, or the Kubernetes namespace's surrounding cluster infrastructure (ingress, cert-manager, etc.).
- **TLS termination.** Put a real ingress/reverse proxy in front of both the health and task-API ports if they're ever reachable outside a fully private network.
- **Certified secrets integration.** Environment, file, and Vault KV v2 adapters exist; Vault was tested against a local mock, not a live cluster. No cloud KMS adapter is included.
- **Turnkey identity/SIEM.** JWKS verification and generic HTTP audit delivery exist, but live IdP/JWKS refresh and target SIEM contracts are deployment work.
- **Production database/queue.** JSONL persistence is a durable local reference, not a horizontally shared billing/findings database.
- **CI/CD deployment automation.** CI runs tests, chaos/load gates, Bandit, `pip-audit`, and Helm validation; it does not deploy anywhere.
- **Disaster recovery.** No measured RTO/RPO or multi-region failover drill has been completed.
