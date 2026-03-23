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
from framework.governance.logger import GovernanceLogger
from framework.resilience.retry import RetryConfig, get_circuit_registry
from framework.resilience.pool import AgentPoolManager, PoolConfig
from framework.orchestration.orchestrator import TaskOrchestrator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

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

def build_reasoning_engine(config: dict, dry_run: bool = False):
    """Build the appropriate ReasoningEngine from config."""
    from framework.reasoning.engine import ReasoningEngine

    if dry_run:
        from framework.reasoning.stubs import DryRunReasoningEngine
        log.info("[server] Using DryRunReasoningEngine")
        return DryRunReasoningEngine()

    llm_cfg = config["llm"]
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
    registry    = get_registry()
    n_agents    = register_all_agents(registry)
    reasoning   = build_reasoning_engine(config, dry_run=dry_run)
    gov_dir     = config["governance"]["log_dir"]
    governance  = GovernanceLogger(gov_dir)
    working_mem = WorkingMemory(
        max_entries=config.get("memory", {}).get("working_memory_max_entries", 50_000)
    )

    inst_path = config.get("memory", {}).get("institutional_memory_path")
    inst_mem  = InstitutionalMemory(storage_path=inst_path)

    # Inject dependencies into every agent instance
    orig_create = registry.create_instance
    def factory(agent_type, cfg, **kwargs):
        inst = orig_create(agent_type, cfg, **kwargs)
        inst.reasoning  = reasoning
        inst.memory     = working_mem
        inst.governance = governance
        return inst
    registry.create_instance = factory

    # Pool manager
    pool_cfg = PoolConfig(min_size=1, max_size=config["orchestration"]["max_workers"])
    pool_mgr = AgentPoolManager(registry, config=None, default_pool=pool_cfg)

    # Orchestrator
    orch = TaskOrchestrator(registry, config=None, pool_manager=pool_mgr)

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
