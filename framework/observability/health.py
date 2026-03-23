"""
framework/observability/health.py
===================================
HTTP health, readiness, and metrics endpoints for Kubernetes probes
and Prometheus scraping.

Endpoints:
  GET /health    Liveness probe — is the process alive?
  GET /ready     Readiness probe — can it serve requests?
  GET /metrics   Prometheus metrics text format
  GET /status    Human-readable JSON status summary

The server runs in a background daemon thread.
Start it once at application startup:
    from framework.observability.health import start_health_server
    start_health_server(port=8080)

KUBERNETES PROBE CONFIGURATION (in deployment.yaml):
  livenessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 10
    periodSeconds: 15

  readinessProbe:
    httpGet:
      path: /ready
      port: 8080
    initialDelaySeconds: 5
    periodSeconds: 10
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

# Global ready flag — set to True by application code after startup
_READY = threading.Event()
_START_TIME = time.monotonic()

# Registered readiness checks — callables that return (ok: bool, detail: str)
_READINESS_CHECKS: Dict[str, Callable[[], tuple]] = {}
_CHECK_LOCK = threading.Lock()


def register_readiness_check(name: str, check_fn: Callable[[], tuple]) -> None:
    """
    Register a readiness check function.

    check_fn must return (ok: bool, detail: str).
    All registered checks must pass for /ready to return 200.

    Example:
        def check_registry() -> tuple:
            ok = get_registry().live_count() > 0
            return ok, "agents registered" if ok else "no agents registered"

        register_readiness_check("registry", check_registry)
    """
    with _CHECK_LOCK:
        _READINESS_CHECKS[name] = check_fn


def mark_ready() -> None:
    """Signal that the application has completed startup."""
    _READY.set()
    log.info("[health] Application marked ready")


def mark_not_ready() -> None:
    """Signal that the application is temporarily not ready (e.g., draining)."""
    _READY.clear()
    log.warning("[health] Application marked NOT ready")


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for health endpoints."""

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/health":
            self._handle_health()
        elif path == "/ready":
            self._handle_ready()
        elif path == "/metrics":
            self._handle_metrics()
        elif path == "/status":
            self._handle_status()
        else:
            self._respond(404, b"Not Found\n", "text/plain")

    def _handle_health(self) -> None:
        """Liveness: is the process running?"""
        body = json.dumps({
            "status":   "alive",
            "uptime_s": round(time.monotonic() - _START_TIME, 1),
        }).encode()
        self._respond(200, body)

    def _handle_ready(self) -> None:
        """Readiness: can the process serve requests?"""
        if not _READY.is_set():
            body = json.dumps({"status": "not_ready", "reason": "startup incomplete"}).encode()
            self._respond(503, body)
            return

        failures = {}
        with _CHECK_LOCK:
            checks = dict(_READINESS_CHECKS)

        for name, fn in checks.items():
            try:
                ok, detail = fn()
                if not ok:
                    failures[name] = detail
            except Exception as exc:
                failures[name] = str(exc)

        if failures:
            body = json.dumps({"status": "not_ready", "failures": failures}).encode()
            self._respond(503, body)
        else:
            body = json.dumps({
                "status":  "ready",
                "checks":  list(checks.keys()),
                "uptime_s": round(time.monotonic() - _START_TIME, 1),
            }).encode()
            self._respond(200, body)

    def _handle_metrics(self) -> None:
        """Prometheus scrape endpoint."""
        from framework.observability.metrics import get_metrics
        data = get_metrics().generate_latest()
        self._respond(200, data, "text/plain; version=0.0.4; charset=utf-8")

    def _handle_status(self) -> None:
        """Human-readable status summary."""
        from framework.observability.metrics import get_metrics
        from framework.core.registry import get_registry
        from framework.resilience.retry import get_circuit_registry

        summary: Dict[str, Any] = {
            "uptime_s":       round(time.monotonic() - _START_TIME, 1),
            "ready":          _READY.is_set(),
            "metrics":        get_metrics().in_memory_summary(),
            "circuit_breakers": get_circuit_registry().all_snapshots(),
        }
        try:
            reg = get_registry()
            summary["registry"] = {
                "registered_classes": len(reg),
                "live_instances":     len(reg._instances),
            }
        except Exception:
            pass

        body = json.dumps(summary, indent=2, default=str).encode()
        self._respond(200, body)

    def _respond(
        self,
        code:         int,
        body:         bytes,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        """Suppress default access log — we use structured logging."""
        pass


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server_thread: Optional[threading.Thread] = None
_server_instance: Optional[HTTPServer] = None


def start_health_server(port: int = 8080, host: str = "0.0.0.0") -> None:
    """
    Start the health HTTP server in a background daemon thread.

    Safe to call multiple times — only starts one server.
    The daemon thread exits automatically when the main process exits.

    Args:
        port: TCP port to listen on (default 8080).
        host: Bind address (default 0.0.0.0).
    """
    global _server_thread, _server_instance

    if _server_thread and _server_thread.is_alive():
        log.debug("[health] Server already running on port %d", port)
        return

    try:
        _server_instance = HTTPServer((host, port), _HealthHandler)
    except OSError as exc:
        log.error("[health] Could not bind to %s:%d — %s", host, port, exc)
        return

    def _serve() -> None:
        log.info("[health] Server listening on %s:%d", host, port)
        try:
            _server_instance.serve_forever()
        except Exception as exc:
            log.error("[health] Server error: %s", exc)

    _server_thread = threading.Thread(target=_serve, name="skein-health", daemon=True)
    _server_thread.start()


def stop_health_server() -> None:
    """Gracefully stop the health server. Idempotent."""
    global _server_instance
    if _server_instance:
        _server_instance.shutdown()
        log.info("[health] Server stopped")
