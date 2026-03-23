"""
framework/observability/metrics.py
====================================
Prometheus-compatible metrics for the SKEIN framework.

Collected metrics:
  skein_agent_runs_total         counter   (agent_type, status)
  skein_agent_duration_ms        histogram (agent_type)
  skein_agent_findings_total     counter   (agent_type, severity)
  skein_llm_calls_total          counter   (provider, status)
  skein_llm_duration_ms          histogram (provider)
  skein_llm_tokens_total         counter   (provider, direction)
  skein_circuit_breaker_state    gauge     (circuit_name, state)
  skein_retry_attempts_total     counter   (function, attempt_number)
  skein_memory_entries           gauge     (tier)
  skein_workflow_duration_ms     histogram (workflow_name, status)
  skein_agent_pool_size          gauge     (agent_type, pool_state)

DESIGN
======
Uses a thin wrapper around prometheus_client when available,
with a no-op fallback when prometheus_client is not installed.
This keeps the framework functional in environments without Prometheus.

The MetricsRegistry is a process-level singleton.
All collectors are registered at module import time.

USAGE
=====
    from framework.observability.metrics import get_metrics

    m = get_metrics()
    m.agent_run_started("SupplierStressAgent")
    m.agent_run_finished("SupplierStressAgent", succeeded=True, duration_ms=1250.0)
    m.llm_call_recorded("anthropic", succeeded=True, duration_ms=900.0, tokens=800)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Fallback no-op metric objects
# ---------------------------------------------------------------------------

class _NoOpCounter:
    def inc(self, amount: float = 1, **labels: str) -> None:
        pass
    def labels(self, **kwargs: str) -> "_NoOpCounter":
        return self

class _NoOpGauge:
    def set(self, value: float, **labels: str) -> None:
        pass
    def inc(self, amount: float = 1) -> None:
        pass
    def dec(self, amount: float = 1) -> None:
        pass
    def labels(self, **kwargs: str) -> "_NoOpGauge":
        return self

class _NoOpHistogram:
    def observe(self, value: float, **labels: str) -> None:
        pass
    def labels(self, **kwargs: str) -> "_NoOpHistogram":
        return self
    def time(self):
        import contextlib
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Prometheus metrics (with graceful fallback)
# ---------------------------------------------------------------------------

def _try_prometheus():
    """Return prometheus_client module or None."""
    try:
        import prometheus_client as prom
        return prom
    except ImportError:
        return None


class MetricsRegistry:
    """
    Holds all SKEIN Prometheus metrics.
    Falls back to no-op collectors when prometheus_client is not installed.

    Thread-safe: metrics objects are created once and reused concurrently.
    """

    def __init__(self, namespace: str = "skein") -> None:
        self._ns   = namespace
        self._prom = _try_prometheus()
        self._lock = threading.Lock()
        self._initialised = False
        self._metrics: Dict[str, Any] = {}
        self._in_memory_stats: Dict[str, int] = {}

    def _init(self) -> None:
        """Lazy initialisation — avoids import-time side effects."""
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            prom = self._prom
            ns   = self._ns

            if prom:
                self._metrics = {
                    "agent_runs":     prom.Counter(
                        f"{ns}_agent_runs_total",
                        "Total agent task executions",
                        ["agent_type", "status"],
                    ),
                    "agent_duration": prom.Histogram(
                        f"{ns}_agent_duration_ms",
                        "Agent task execution duration in milliseconds",
                        ["agent_type"],
                        buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
                    ),
                    "findings":       prom.Counter(
                        f"{ns}_agent_findings_total",
                        "Total findings produced",
                        ["agent_type", "severity"],
                    ),
                    "llm_calls":      prom.Counter(
                        f"{ns}_llm_calls_total",
                        "Total LLM calls",
                        ["provider", "status"],
                    ),
                    "llm_duration":   prom.Histogram(
                        f"{ns}_llm_duration_ms",
                        "LLM call duration in milliseconds",
                        ["provider"],
                        buckets=[100, 250, 500, 1000, 2000, 5000, 15000, 30000, 60000],
                    ),
                    "llm_tokens":     prom.Counter(
                        f"{ns}_llm_tokens_total",
                        "Total LLM tokens",
                        ["provider", "direction"],
                    ),
                    "circuit_state":  prom.Gauge(
                        f"{ns}_circuit_breaker_open",
                        "Circuit breaker state (1=open, 0=closed/half-open)",
                        ["circuit_name"],
                    ),
                    "retry_attempts": prom.Counter(
                        f"{ns}_retry_attempts_total",
                        "Total retry attempts by function",
                        ["function_name", "attempt_number"],
                    ),
                    "memory_entries": prom.Gauge(
                        f"{ns}_memory_entries",
                        "Current entries in memory tier",
                        ["tier"],
                    ),
                    "workflow_duration": prom.Histogram(
                        f"{ns}_workflow_duration_ms",
                        "Workflow execution duration in milliseconds",
                        ["workflow_name", "status"],
                        buckets=[100, 500, 1000, 5000, 15000, 60000, 300000],
                    ),
                    "pool_size":      prom.Gauge(
                        f"{ns}_agent_pool_size",
                        "Agent pool size",
                        ["agent_type", "state"],
                    ),
                }
            else:
                self._metrics = {
                    "agent_runs":       _NoOpCounter(),
                    "agent_duration":   _NoOpHistogram(),
                    "findings":         _NoOpCounter(),
                    "llm_calls":        _NoOpCounter(),
                    "llm_duration":     _NoOpHistogram(),
                    "llm_tokens":       _NoOpCounter(),
                    "circuit_state":    _NoOpGauge(),
                    "retry_attempts":   _NoOpCounter(),
                    "memory_entries":   _NoOpGauge(),
                    "workflow_duration":_NoOpHistogram(),
                    "pool_size":        _NoOpGauge(),
                }
            self._initialised = True

    # ------------------------------------------------------------------
    # High-level recording helpers
    # ------------------------------------------------------------------

    def agent_run_started(self, agent_type: str) -> None:
        self._init()
        self._in_memory_stats[f"running:{agent_type}"] = (
            self._in_memory_stats.get(f"running:{agent_type}", 0) + 1
        )

    def agent_run_finished(
        self,
        agent_type:  str,
        succeeded:   bool,
        duration_ms: float,
        findings:    Optional[Dict[str, int]] = None,
    ) -> None:
        self._init()
        status = "success" if succeeded else "failure"
        self._metrics["agent_runs"].labels(agent_type=agent_type, status=status).inc()
        self._metrics["agent_duration"].labels(agent_type=agent_type).observe(duration_ms)
        self._in_memory_stats[f"total:{agent_type}"] = (
            self._in_memory_stats.get(f"total:{agent_type}", 0) + 1
        )
        running = self._in_memory_stats.get(f"running:{agent_type}", 1)
        self._in_memory_stats[f"running:{agent_type}"] = max(0, running - 1)
        if findings:
            for severity, count in findings.items():
                self._metrics["findings"].labels(
                    agent_type=agent_type, severity=severity
                ).inc(count)

    def llm_call_recorded(
        self,
        provider:    str,
        succeeded:   bool,
        duration_ms: float,
        tokens:      int = 0,
    ) -> None:
        self._init()
        status = "success" if succeeded else "failure"
        self._metrics["llm_calls"].labels(provider=provider, status=status).inc()
        self._metrics["llm_duration"].labels(provider=provider).observe(duration_ms)
        if tokens:
            self._metrics["llm_tokens"].labels(
                provider=provider, direction="total"
            ).inc(tokens)

    def circuit_state_updated(self, circuit_name: str, is_open: bool) -> None:
        self._init()
        self._metrics["circuit_state"].labels(circuit_name=circuit_name).set(
            1 if is_open else 0
        )

    def retry_recorded(self, function_name: str, attempt: int) -> None:
        self._init()
        self._metrics["retry_attempts"].labels(
            function_name=function_name, attempt_number=str(attempt)
        ).inc()

    def memory_updated(self, tier: str, count: int) -> None:
        self._init()
        self._metrics["memory_entries"].labels(tier=tier).set(count)

    def workflow_finished(
        self,
        workflow_name: str,
        succeeded:     bool,
        duration_ms:   float,
    ) -> None:
        self._init()
        status = "success" if succeeded else "failure"
        self._metrics["workflow_duration"].labels(
            workflow_name=workflow_name, status=status
        ).observe(duration_ms)

    def pool_size_updated(self, agent_type: str, idle: int, active: int) -> None:
        self._init()
        self._metrics["pool_size"].labels(agent_type=agent_type, state="idle").set(idle)
        self._metrics["pool_size"].labels(agent_type=agent_type, state="active").set(active)

    # ------------------------------------------------------------------
    # HTTP metrics endpoint helper
    # ------------------------------------------------------------------

    def generate_latest(self) -> bytes:
        """Return Prometheus text format for /metrics endpoint."""
        if self._prom:
            return self._prom.generate_latest()
        summary = "# SKEIN metrics (prometheus_client not installed)\n"
        for k, v in self._in_memory_stats.items():
            summary += f"skein_{k} {v}\n"
        return summary.encode()

    def in_memory_summary(self) -> Dict[str, Any]:
        """Return compact in-memory stats for /health endpoint."""
        return dict(self._in_memory_stats)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_metrics: Optional[MetricsRegistry] = None
_global_lock = threading.Lock()


def get_metrics() -> MetricsRegistry:
    """Return process-level MetricsRegistry (lazy singleton)."""
    global _global_metrics
    if _global_metrics is None:
        with _global_lock:
            if _global_metrics is None:
                _global_metrics = MetricsRegistry()
    return _global_metrics
