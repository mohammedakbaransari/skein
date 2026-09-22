"""
scripts/server.py
==================
SKEIN production server entry point.

Wires together:
  - All 15 agent registrations
  - LLM provider from config/environment
  - ReasoningEngine with retry and circuit breaker
  - WorkingMemory + InstitutionalMemory
  - GovernanceLogger
  - AgentPoolManager with configured pool sizes
  - Health/readiness/metrics HTTP server on SKEIN_HEALTH_PORT
  - Structured JSON logging

Usage:
  python3 -m scripts.server                    # default config.yaml
  python3 -m scripts.server --config /path/to/config.yaml
  python3 -m scripts.server --dry-run          # DryRunReasoningEngine

Environment overrides (all config.yaml settings):
  LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
  SKEIN_MAX_WORKERS, SKEIN_GOVERNANCE_LOG_DIR
  SKEIN_HEALTH_PORT (default: 8080)
  SKEIN_LOG_LEVEL (default: INFO)
  SKEIN_LOG_JSON (default: true)
  AGENT_DRY_RUN (default: false)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure repo root is on path when run as module
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from framework.observability.logging import setup_logging
from framework.observability.health import (
    start_health_server, mark_ready, register_readiness_check,
)
from framework.observability.metrics import get_metrics
from framework.core.registry import get_registry, reset_registry
from framework.memory.store import WorkingMemory, InstitutionalMemory
from framework.security.controls import configure_security
from framework.security.controls import register_secret
from framework.adapters.secrets import EnvironmentSecretsProvider, build_secrets_provider
from framework.agents.confidence import JsonCalibrationStore
from framework.multitenancy.context import get_tenant_registry
from framework.multitenancy.resolver import TenantStoreResolver
from framework.auth.api_keys import ApiKeyStore
from framework.api.server import start_task_api_server, stop_task_api_server
from framework.api.jobs import JobStore
from framework.api.webhooks import WebhookDispatcher
from framework.governance.logger import GovernanceLogger
from framework.resilience.retry import RetryConfig, get_circuit_registry
from framework.resilience.pool import AgentPoolManager, PoolConfig
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.findings.jsonl_store import JsonlFindingsStore
from framework.findings.review import ReviewWorkflow
from framework.billing.jsonl_ledger import JsonlUsageLedger
from framework.billing.ledger import TokenQuotaEnforcer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def build_api_key_store(secrets_provider=None) -> ApiKeyStore:
    """Provision task-API keys from SKEIN_API_KEYS (JSON: {tenant_id: raw_key}).

    Bootstrap-only mechanism, not a secrets-manager integration — same
    scope boundary as tenant storage provisioning (framework/multitenancy):
    this loads keys, it doesn't generate/rotate/store them securely at
    rest. Absent or empty means the task API's auth check is disabled
    (see framework/api/server.py's opt-in-enforcement pattern).
    """
    store = ApiKeyStore()
    provider = secrets_provider or EnvironmentSecretsProvider()
    try:
        raw = provider.get_secret("SKEIN_API_KEYS")
        register_secret(raw)
    except KeyError:
        raw = ""
    if not raw:
        return store
    import json as _json
    try:
        mapping = _json.loads(raw)
        for tenant_id, key in mapping.items():
            store.register(tenant_id, key)
            register_secret(key)
        log.info("[server] Loaded %d API key(s) from SKEIN_API_KEYS", len(store))
    except Exception as exc:
        log.error("[server] Failed to parse SKEIN_API_KEYS — task API auth disabled: %s", exc)
    return store


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and overlay environment variables."""
    import yaml
    config = {}
    try:
        with open(path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("Config file not found: %s — using defaults", path)

    # Environment overrides
    llm = config.setdefault("llm", {})
    llm["provider"]        = os.environ.get("LLM_PROVIDER",     llm.get("provider", "ollama"))
    llm["model"]           = os.environ.get("LLM_MODEL",        llm.get("model", "llama3.1"))
    llm["api_key"]         = os.environ.get("LLM_API_KEY",      llm.get("api_key"))
    llm["base_url"]        = os.environ.get("LLM_BASE_URL",     llm.get("base_url", "http://localhost:11434"))
    llm["temperature"]     = float(os.environ.get("LLM_TEMPERATURE", llm.get("temperature", 0.1)))
    llm["max_tokens"]      = int(os.environ.get("LLM_MAX_TOKENS", llm.get("max_tokens", 2048)))
    llm["timeout_seconds"] = int(os.environ.get("LLM_TIMEOUT", llm.get("timeout_seconds", 120)))

    orch = config.setdefault("orchestration", {})
    orch["max_workers"]    = int(os.environ.get("SKEIN_MAX_WORKERS", orch.get("max_workers", 4)))

    gov = config.setdefault("governance", {})
    gov["log_dir"]         = os.environ.get("SKEIN_GOVERNANCE_LOG_DIR", gov.get("log_dir", "logs/governance"))

    agent = config.setdefault("agent", {})
    agent["dry_run"]       = os.environ.get("AGENT_DRY_RUN", str(agent.get("dry_run", False))).lower() == "true"

    return config


# ---------------------------------------------------------------------------
# LLM gateway factory
# ---------------------------------------------------------------------------

def build_reasoning_engine(config: dict, dry_run: bool = False, secrets_provider=None):
    """Build the appropriate ReasoningEngine from config."""
    from framework.reasoning.engine import ReasoningEngine

    if dry_run:
        from framework.reasoning.stubs import DryRunReasoningEngine
        log.info("[server] Using DryRunReasoningEngine")
        return DryRunReasoningEngine()

    llm_cfg = config["llm"]
    if not llm_cfg.get("api_key") and secrets_provider is not None:
        try:
            llm_cfg["api_key"] = secrets_provider.get_secret("LLM_API_KEY")
            register_secret(llm_cfg["api_key"])
        except KeyError:
            pass
    register_secret(llm_cfg.get("api_key", ""))
    provider = llm_cfg["provider"]

    # Build a simple gateway object that provides .complete()
    class _Gateway:
        def __init__(self, provider, model, api_key, base_url, temperature, max_tokens, timeout):
            self.provider    = provider
            self.model       = model
            self.api_key     = api_key
            self.base_url    = base_url
            self.temperature = temperature
            self.max_tokens  = max_tokens
            self.timeout     = timeout

        def complete(self, system_prompt: str, user_prompt: str, session_id: str = ""):
            class Response:
                def __init__(self, content, model, input_tokens=0, output_tokens=0):
                    self.content       = content
                    self.model         = model
                    self.input_tokens  = input_tokens
                    self.output_tokens = output_tokens

            if self.provider == "ollama":
                import urllib.request, json as _json
                payload = _json.dumps({
                    "model": self.model, "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                }).encode()
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = _json.loads(resp.read())
                content = data.get("message", {}).get("content", "")
                return Response(content, self.model)

            elif self.provider == "anthropic":
                import anthropic as ant
                client = ant.Anthropic(api_key=self.api_key)
                resp = client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = resp.content[0].text if resp.content else ""
                return Response(content, self.model,
                                resp.usage.input_tokens, resp.usage.output_tokens)

            elif self.provider in ("openai", "azure"):
                import openai
                client = openai.OpenAI(api_key=self.api_key,
                                       base_url=self.base_url if self.provider == "azure" else None)
                resp = client.chat.completions.create(
                    model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content or ""
                return Response(content, self.model,
                                resp.usage.prompt_tokens, resp.usage.completion_tokens)
            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}")

    gateway = _Gateway(
        provider=provider,
        model=llm_cfg["model"],
        api_key=llm_cfg.get("api_key"),
        base_url=llm_cfg.get("base_url", ""),
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 2048),
        timeout=llm_cfg.get("timeout_seconds", 120),
    )
    if provider in {"anthropic", "openai", "azure"} and not llm_cfg.get("api_key"):
        raise RuntimeError(f"LLM_API_KEY is required for provider '{provider}'")
    retry = RetryConfig(
        max_attempts=llm_cfg.get("max_retries", 3),
        initial_delay_s=llm_cfg.get("retry_backoff_seconds", 2.0),
        max_delay_s=30.0,
    )
    engine = ReasoningEngine.native(gateway, retry_config=retry)
    log.info("[server] ReasoningEngine: provider=%s model=%s", provider, llm_cfg["model"])
    return engine


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------

def register_all_agents(registry) -> int:
    """Register all 15 SKEIN agents. Returns count registered."""
    from agents.supply_risk.supplier_stress import SupplierStressAgent
    from agents.decision_audit.agent import DecisionAuditAgent
    from agents.cost_intelligence.should_cost import ShouldCostAgent
    from agents.cost_intelligence.total_cost import TotalCostIntelligenceAgent
    from agents.contract_analysis.value_realisation import ValueRealisationAgent
    from agents.bias_detection.bias_detector import ProcurementBiasDetectorAgent
    from agents.compliance.compliance_verification import ComplianceVerificationAgent
    from agents.market_intelligence.agents import (
        InstitutionalMemoryAgent, NegotiationIntelligenceAgent,
        SpecificationInflationAgent, WorkingCapitalOptimiserAgent,
        DemandIntelligenceAgent, SupplierInnovationAgent,
        DecisionCopilotAgent, TradeScenarioAgent,
    )
    agents = [
        SupplierStressAgent, DecisionAuditAgent, ShouldCostAgent,
        TotalCostIntelligenceAgent, ValueRealisationAgent,
        ProcurementBiasDetectorAgent, ComplianceVerificationAgent,
        InstitutionalMemoryAgent, NegotiationIntelligenceAgent,
        SpecificationInflationAgent, WorkingCapitalOptimiserAgent,
        DemandIntelligenceAgent, SupplierInnovationAgent,
        DecisionCopilotAgent, TradeScenarioAgent,
    ]
    for cls in agents:
        registry.register_class(cls)
    log.info("[server] Registered %d agents", len(agents))
    return len(agents)


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------

def run_server(config_path: str = "config/config.yaml", dry_run: bool = False) -> None:
    config = load_config(config_path)
    dry_run = dry_run or config["agent"].get("dry_run", False)

    # Logging
    log_level = os.environ.get("SKEIN_LOG_LEVEL", "INFO")
    log_json  = os.environ.get("SKEIN_LOG_JSON", "true").lower() == "true"
    setup_logging(level=log_level, json_output=log_json)

    log.info("[server] SKEIN starting — provider=%s dry_run=%s",
             config["llm"]["provider"], dry_run)

    # Health server (starts before anything else — gives Kubernetes liveness early)
    health_port = int(os.environ.get("SKEIN_HEALTH_PORT", 8080))
    start_health_server(port=health_port)
    log.info("[server] Health server on port %d", health_port)

    # Framework components
    reset_registry()
    configure_security(config.get("security", {}))
    secrets_provider = build_secrets_provider(config.get("secrets", {}))
    calibration_path = os.environ.get(
        "SKEIN_CONFIDENCE_CALIBRATION_PATH",
        config.get("agent", {}).get("confidence_calibration_path"),
    )
    confidence_store = JsonCalibrationStore(calibration_path) if calibration_path else None
    registry    = get_registry()
    n_agents    = register_all_agents(registry)
    reasoning   = build_reasoning_engine(config, dry_run=dry_run, secrets_provider=secrets_provider)
    gov_dir     = config["governance"]["log_dir"]
    governance  = GovernanceLogger(gov_dir)
    findings_store = JsonlFindingsStore(config.get("governance", {}).get("findings_path", "data/findings.jsonl"))
    review_workflow = ReviewWorkflow()
    job_store = JobStore(max_workers=config["orchestration"].get("max_workers", 4))
    webhook_dispatcher = WebhookDispatcher()
    usage_ledger = JsonlUsageLedger(config.get("governance", {}).get("usage_path", "data/usage.jsonl"))
    quota_enforcer = TokenQuotaEnforcer(usage_ledger)
    job_store.configure_usage_ledger(usage_ledger)
    working_mem = WorkingMemory(
        max_entries=config.get("memory", {}).get("working_memory_max_entries", 50_000)
    )

    inst_path = config.get("memory", {}).get("institutional_memory_path")
    inst_mem  = InstitutionalMemory(storage_path=inst_path)

    # Per-tenant Delta-backed memory/governance routing (physical
    # isolation, see architecture assessment §33/§36). Falls back to each
    # agent's own default below when a task has no tenant_id or an
    # unregistered one — single-tenant/dev deployments are unaffected.
    tenant_resolver = TenantStoreResolver(get_tenant_registry())

    # Inject dependencies into every agent instance
    orig_create = registry.create_instance
    def factory(agent_type, cfg, **kwargs):
        inst = orig_create(agent_type, cfg, **kwargs)
        inst.reasoning  = reasoning
        # InstitutionalMemoryAgent gets the persistent, cross-restart
        # store; every other agent gets the in-process WorkingMemory.
        # Previously `inst_mem` was constructed but never actually
        # injected into any agent (architecture assessment §32).
        inst.memory     = inst_mem if agent_type == "InstitutionalMemoryAgent" else working_mem
        inst.governance = governance
        inst.confidence_calibration_store = confidence_store
        inst.tenant_store_resolver = tenant_resolver.resolve
        return inst
    registry.create_instance = factory

    # Pool manager
    pool_cfg = PoolConfig(min_size=1, max_size=config["orchestration"]["max_workers"])
    pool_mgr = AgentPoolManager(registry, config=None, default_pool=pool_cfg)

    # Orchestrator
    orch = TaskOrchestrator(registry, config=None, pool_manager=pool_mgr)

    # Task-submission API — every request must carry an explicit tenant_id
    # (see framework/api/server.py); TenantRegistry is empty by default in
    # this entry point (no provisioning wiring yet — see architecture
    # assessment §33), so the registry-membership check is skipped until
    # tenants are actually registered somewhere upstream of this call.
    task_api_port = int(os.environ.get("SKEIN_TASK_API_PORT", 8081))
    api_key_store = build_api_key_store(secrets_provider)
    if len(api_key_store) == 0:
        log.warning("[server] No SKEIN_API_KEYS configured — task API authentication is DISABLED")
    bound_task_api_port = start_task_api_server(
        orch, get_tenant_registry(), port=task_api_port, api_key_store=api_key_store,
        agent_registry=registry,
        max_request_body_bytes=config.get("security", {}).get("max_request_body_bytes", 1_048_576),
        findings_store=findings_store,
        review_workflow=review_workflow,
        job_store=job_store,
        webhook_dispatcher=webhook_dispatcher,
        usage_ledger=usage_ledger,
        quota_enforcer=quota_enforcer,
    )
    if bound_task_api_port:
        log.info("[server] Task-submission API on port %d", bound_task_api_port)

    # Register readiness check
    def check_agents():
        n = len(registry)
        return n >= 15, f"{n}/15 agents registered"
    register_readiness_check("agents", check_agents)

    # Metrics
    metrics = get_metrics()
    metrics.agent_run_started("_server_init")

    # Mark ready
    mark_ready()
    log.info("[server] SKEIN ready — %d agents, health port %d", n_agents, health_port)

    # Graceful shutdown
    def handle_shutdown(signum, frame):
        log.info("[server] Received signal %d — shutting down", signum)
        from framework.observability.health import mark_not_ready, stop_health_server
        mark_not_ready()
        time.sleep(5)  # Allow in-flight requests to drain
        pool_mgr.shutdown_all()
        stop_task_api_server()
        stop_health_server()
        log.info("[server] Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT,  handle_shutdown)

    log.info("[server] Serving — press Ctrl+C to stop")
    # Keep alive
    while True:
        time.sleep(30)
        log.debug("[server] Heartbeat — agents=%d", len(registry))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKEIN Agent Server")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()
    run_server(config_path=args.config, dry_run=args.dry_run)
