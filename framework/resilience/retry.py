"""
framework/resilience/retry.py
==============================
Retry policy with exponential backoff + jitter, and Circuit Breaker pattern.

RETRY POLICY
============
Implements full exponential backoff with jitter:
    delay = min(initial * factor^attempt, max_delay) * (1 ± jitter)

Full jitter (AWS blog recommendation) prevents thundering-herd when many
agents retry simultaneously after a shared LLM provider outage.

CIRCUIT BREAKER
===============
State machine: CLOSED → OPEN → HALF_OPEN → CLOSED

    CLOSED:    Normal operation. Tracks failures in a sliding window.
               Opens when failure_threshold is crossed.

    OPEN:      Rejects all calls immediately (fast-fail).
               Transitions to HALF_OPEN after recovery_timeout_s.

    HALF_OPEN: Allows one probe call through.
               Success → CLOSED. Failure → OPEN (reset timeout).

One circuit breaker per named resource (e.g., "anthropic", "openai", "ollama").
Stored in a process-level registry for sharing across agents.

THREAD SAFETY
=============
RetryExecutor is stateless — safe for concurrent use.
CircuitBreaker uses threading.Lock — all state transitions are atomic.
CircuitBreakerRegistry uses threading.RLock.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

from framework.core.types import CircuitState, RetryConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected by an open circuit breaker."""

    def __init__(self, circuit_name: str, recovery_at: float) -> None:
        self.circuit_name = circuit_name
        self.recovery_at  = recovery_at
        seconds_remaining = max(0.0, recovery_at - time.monotonic())
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. "
            f"Recovery probe in {seconds_remaining:.1f}s."
        )


class MaxRetriesExceeded(RuntimeError):
    """Raised after all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts   = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} attempt(s) failed. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )


# ---------------------------------------------------------------------------
# Retry executor — stateless, thread-safe
# ---------------------------------------------------------------------------

class RetryExecutor:
    """
    Executes a callable with exponential backoff retry.

    Thread-safe: carries no per-instance mutable state.
    All retry state is local to each execute() call.

    Usage:
        executor = RetryExecutor(RetryConfig.conservative())
        result   = executor.execute(my_function, arg1, arg2)
    """

    def __init__(self, config: RetryConfig) -> None:
        self._cfg = config

    def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute fn(*args, **kwargs) with retry on failure.

        Returns the first successful result.
        Raises MaxRetriesExceeded if all attempts fail.
        """
        last_exc: Optional[Exception] = None
        cfg = self._cfg

        for attempt in range(1, cfg.max_attempts + 1):
            try:
                result = fn(*args, **kwargs)
                if attempt > 1:
                    log.info(
                        "[retry] %s succeeded on attempt %d/%d",
                        getattr(fn, "__name__", str(fn)), attempt, cfg.max_attempts,
                    )
                return result

            except Exception as exc:
                # Check if this error type is retryable
                if cfg.retryable_errors and not isinstance(exc, cfg.retryable_errors):
                    log.debug(
                        "[retry] %s: %s is not retryable — raising immediately",
                        getattr(fn, "__name__", "fn"), type(exc).__name__,
                    )
                    raise

                last_exc = exc

                if attempt >= cfg.max_attempts:
                    break

                delay = self._compute_delay(attempt)
                log.warning(
                    "[retry] %s failed (attempt %d/%d): %s — retrying in %.2fs",
                    getattr(fn, "__name__", "fn"),
                    attempt, cfg.max_attempts, exc, delay,
                )
                time.sleep(delay)

        raise MaxRetriesExceeded(cfg.max_attempts, last_exc or RuntimeError("unknown"))

    def _compute_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter."""
        base  = min(
            self._cfg.initial_delay_s * (self._cfg.backoff_factor ** (attempt - 1)),
            self._cfg.max_delay_s,
        )
        jitter = base * self._cfg.jitter_factor
        return base + random.uniform(-jitter, jitter)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Thread-safe circuit breaker for one named resource.

    State machine:
        CLOSED    → OPEN      when failures >= threshold in window
        OPEN      → HALF_OPEN when recovery_timeout_s elapsed
        HALF_OPEN → CLOSED    on one successful probe
        HALF_OPEN → OPEN      on probe failure

    All state transitions are atomic under self._lock.

    Metrics exposed via .snapshot() for Prometheus collection.
    """

    def __init__(
        self,
        name: str,
        failure_threshold:   int   = 5,
        recovery_timeout_s:  float = 30.0,
        half_open_max_calls: int   = 1,
        window_size:         int   = 10,
    ) -> None:
        self.name = name
        self._failure_threshold   = failure_threshold
        self._recovery_timeout_s  = recovery_timeout_s
        self._half_open_max_calls = half_open_max_calls
        self._window_size         = window_size

        self._lock                = threading.Lock()
        self._state               = CircuitState.CLOSED
        self._failure_count       = 0
        self._success_count       = 0
        self._total_calls         = 0
        self._rejected_calls      = 0
        self._last_failure_time   = 0.0
        self._open_since          = 0.0
        self._half_open_calls     = 0
        self._recent_results: list = []  # sliding window: True=success, False=failure

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute fn() through the circuit breaker.

        Raises CircuitOpenError immediately if circuit is OPEN.
        Records success/failure and transitions state accordingly.
        """
        with self._lock:
            state = self._get_state()
            if state == CircuitState.OPEN:
                self._rejected_calls += 1
                raise CircuitOpenError(self.name, self._open_since + self._recovery_timeout_s)
            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    self._rejected_calls += 1
                    raise CircuitOpenError(self.name, time.monotonic() + 1.0)
                self._half_open_calls += 1
            self._total_calls += 1

        try:
            result = fn(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def __enter__(self) -> "CircuitBreaker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    # ------------------------------------------------------------------
    # State management (called under lock)
    # ------------------------------------------------------------------

    def _get_state(self) -> CircuitState:
        """Compute current state, auto-transitioning OPEN→HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._open_since >= self._recovery_timeout_s:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def _record_success(self) -> None:
        with self._lock:
            self._success_count += 1
            self._recent_results.append(True)
            if len(self._recent_results) > self._window_size:
                self._recent_results.pop(0)
            if self._state == CircuitState.HALF_OPEN:
                log.info("[circuit:%s] Probe succeeded → CLOSED", self.name)
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._recent_results.append(False)
            if len(self._recent_results) > self._window_size:
                self._recent_results.pop(0)

            window_failures = self._recent_results.count(False)

            if self._state == CircuitState.HALF_OPEN:
                log.warning("[circuit:%s] Probe failed → OPEN", self.name)
                self._transition_to(CircuitState.OPEN)
            elif (self._state == CircuitState.CLOSED
                  and window_failures >= self._failure_threshold):
                log.error(
                    "[circuit:%s] %d/%d failures in window → OPEN",
                    self.name, window_failures, self._window_size,
                )
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Apply state transition and reset counters."""
        old_state = self._state
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._open_since = time.monotonic()
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._recent_results.clear()
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        log.info(
            "[circuit:%s] %s → %s",
            self.name, old_state.value, new_state.value,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return current metrics dict for Prometheus/health endpoints."""
        with self._lock:
            window_fail_rate = (
                self._recent_results.count(False) / len(self._recent_results)
                if self._recent_results else 0.0
            )
            return {
                "name":             self.name,
                "state":            self._state.value,
                "total_calls":      self._total_calls,
                "success_count":    self._success_count,
                "failure_count":    self._failure_count,
                "rejected_calls":   self._rejected_calls,
                "window_fail_rate": round(window_fail_rate, 3),
                "open_since":       self._open_since if self._state == CircuitState.OPEN else None,
            }

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._get_state()


# ---------------------------------------------------------------------------
# Circuit Breaker Registry — process-level singleton
# ---------------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Process-level registry of named circuit breakers.

    One circuit breaker per LLM provider or external service.
    Shared across all agents in the same process.
    Thread-safe: RLock on all access.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold:  int   = 5,
        recovery_timeout_s: float = 30.0,
    ) -> CircuitBreaker:
        """Return existing breaker or create one with given settings."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout_s=recovery_timeout_s,
                )
                log.info("[circuit_registry] Created breaker '%s'", name)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        with self._lock:
            return self._breakers.get(name)

    def all_snapshots(self) -> Dict[str, Dict[str, Any]]:
        """Return snapshot dict for all registered breakers."""
        with self._lock:
            return {name: cb.snapshot() for name, cb in self._breakers.items()}

    def reset(self, name: str) -> bool:
        """Manually reset a breaker to CLOSED (operations use only)."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name]._transition_to(CircuitState.CLOSED)
                log.warning("[circuit_registry] Manual reset of '%s' to CLOSED", name)
                return True
            return False


# ---------------------------------------------------------------------------
# Global registry singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[CircuitBreakerRegistry] = None
_global_registry_lock = threading.Lock()


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Return process-level circuit breaker registry (lazy singleton)."""
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = CircuitBreakerRegistry()
    return _global_registry


def reset_circuit_registry() -> None:
    """Reset global registry. Tests only."""
    global _global_registry
    with _global_registry_lock:
        _global_registry = None
