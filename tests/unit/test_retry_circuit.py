"""
tests/unit/test_retry_circuit.py
==================================
Unit tests for RetryExecutor and CircuitBreaker.

Tests:
  - Exponential backoff delay computation
  - Retry succeeds on second attempt
  - MaxRetriesExceeded after all attempts
  - Only retryable error types are retried
  - Circuit breaker opens after threshold failures
  - Circuit breaker transitions: OPEN → HALF_OPEN → CLOSED
  - Fast-fail while circuit is OPEN
  - Manual reset of circuit breaker
  - Thread safety: concurrent callers respect circuit state
"""

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.core.types import CircuitState, RetryConfig
from framework.resilience.retry import (
    CircuitBreaker, CircuitBreakerRegistry,
    CircuitOpenError, MaxRetriesExceeded, RetryExecutor,
    reset_circuit_registry,
)


class TestRetryExecutor(unittest.TestCase):

    def setUp(self):
        reset_circuit_registry()

    def test_succeeds_on_first_attempt(self):
        executor = RetryExecutor(RetryConfig(max_attempts=3, initial_delay_s=0.0))
        call_count = [0]

        def fn():
            call_count[0] += 1
            return "ok"

        result = executor.execute(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 1)

    def test_retries_on_failure_and_succeeds(self):
        executor = RetryExecutor(RetryConfig(
            max_attempts=3, initial_delay_s=0.0, backoff_factor=1.0, jitter_factor=0.0
        ))
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return "success"

        result = executor.execute(fn)
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 3)

    def test_raises_max_retries_exceeded(self):
        executor = RetryExecutor(RetryConfig(
            max_attempts=3, initial_delay_s=0.0, backoff_factor=1.0, jitter_factor=0.0
        ))

        def always_fails():
            raise ValueError("permanent")

        with self.assertRaises(MaxRetriesExceeded) as ctx:
            executor.execute(always_fails)
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertIsInstance(ctx.exception.last_error, ValueError)

    def test_non_retryable_error_raises_immediately(self):
        executor = RetryExecutor(RetryConfig(
            max_attempts=5, initial_delay_s=0.0,
            retryable_errors=(ConnectionError,),
        ))
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise TypeError("not retryable")

        with self.assertRaises(TypeError):
            executor.execute(fn)
        self.assertEqual(call_count[0], 1)  # No retries

    def test_retryable_error_type_is_retried(self):
        executor = RetryExecutor(RetryConfig(
            max_attempts=3, initial_delay_s=0.0,
            retryable_errors=(ConnectionError,),
        ))
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("retryable")
            return "done"

        result = executor.execute(fn)
        self.assertEqual(result, "done")
        self.assertEqual(call_count[0], 3)

    def test_delay_computation_increases(self):
        executor = RetryExecutor(RetryConfig(
            initial_delay_s=1.0, backoff_factor=2.0,
            max_delay_s=100.0, jitter_factor=0.0,
        ))
        delays = [executor._compute_delay(i) for i in range(1, 5)]
        # 1.0, 2.0, 4.0, 8.0
        self.assertAlmostEqual(delays[0], 1.0, places=3)
        self.assertAlmostEqual(delays[1], 2.0, places=3)
        self.assertAlmostEqual(delays[2], 4.0, places=3)
        self.assertAlmostEqual(delays[3], 8.0, places=3)

    def test_delay_capped_at_max(self):
        executor = RetryExecutor(RetryConfig(
            initial_delay_s=10.0, backoff_factor=10.0,
            max_delay_s=30.0, jitter_factor=0.0,
        ))
        delay = executor._compute_delay(5)
        self.assertLessEqual(delay, 30.0 * 1.01)  # slight tolerance

    def test_jitter_adds_randomness(self):
        executor = RetryExecutor(RetryConfig(
            initial_delay_s=10.0, backoff_factor=1.0,
            max_delay_s=100.0, jitter_factor=0.5,
        ))
        delays = {executor._compute_delay(1) for _ in range(20)}
        # With 50% jitter on 10s, should get values between 5 and 15
        self.assertGreater(len(delays), 1)
        for d in delays:
            self.assertGreater(d, 4.0)
            self.assertLess(d, 16.0)


class TestCircuitBreaker(unittest.TestCase):

    def setUp(self):
        reset_circuit_registry()

    def _make_breaker(self, threshold=3, recovery=0.05):
        return CircuitBreaker(
            name="test",
            failure_threshold=threshold,
            recovery_timeout_s=recovery,
            window_size=10,
        )

    def test_starts_closed(self):
        cb = self._make_breaker()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_successful_call_stays_closed(self):
        cb = self._make_breaker()
        result = cb.call(lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_opens_after_threshold_failures(self):
        cb = self._make_breaker(threshold=3)
        for _ in range(3):
            with self.assertRaises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_fast_fails_when_open(self):
        cb = self._make_breaker(threshold=3, recovery=60.0)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        # Now open — subsequent calls should fast-fail
        with self.assertRaises(CircuitOpenError):
            cb.call(lambda: "should not execute")

    def test_transitions_to_half_open_after_timeout(self):
        cb = self._make_breaker(threshold=3, recovery=0.05)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        self.assertEqual(cb.state, CircuitState.OPEN)
        time.sleep(0.1)
        # Next call should probe (half-open state)
        self.assertEqual(cb._get_state(), CircuitState.HALF_OPEN)

    def test_closes_after_successful_probe(self):
        cb = self._make_breaker(threshold=3, recovery=0.05)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        time.sleep(0.1)
        # Successful probe → CLOSED
        result = cb.call(lambda: "recovered")
        self.assertEqual(result, "recovered")
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_reopens_on_failed_probe(self):
        cb = self._make_breaker(threshold=3, recovery=0.05)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        time.sleep(0.1)
        # Failed probe → back to OPEN
        with self.assertRaises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("probe failed")))
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_snapshot_returns_metrics(self):
        cb = self._make_breaker()
        cb.call(lambda: "ok")
        snap = cb.snapshot()
        self.assertEqual(snap["state"], "closed")
        self.assertEqual(snap["total_calls"], 1)
        self.assertEqual(snap["success_count"], 1)
        self.assertEqual(snap["failure_count"], 0)

    def test_thread_safety_concurrent_failures(self):
        """Multiple threads failing simultaneously should open circuit exactly once."""
        cb = self._make_breaker(threshold=5, recovery=60.0)
        failures = []
        rejects  = []

        def worker():
            for _ in range(10):
                try:
                    cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
                except CircuitOpenError:
                    rejects.append(1)
                except ValueError:
                    failures.append(1)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        # Circuit should be open
        self.assertEqual(cb.state, CircuitState.OPEN)
        # At least threshold failures before it opened
        self.assertGreaterEqual(len(failures), 5)
        # Some calls were rejected
        self.assertGreater(len(rejects), 0)


class TestCircuitBreakerRegistry(unittest.TestCase):

    def setUp(self):
        reset_circuit_registry()

    def test_get_or_create_returns_same_instance(self):
        from framework.resilience.retry import get_circuit_registry
        reg = get_circuit_registry()
        cb1 = reg.get_or_create("my_service")
        cb2 = reg.get_or_create("my_service")
        self.assertIs(cb1, cb2)

    def test_different_names_are_independent(self):
        from framework.resilience.retry import get_circuit_registry
        reg = get_circuit_registry()
        cb1 = reg.get_or_create("service_a")
        cb2 = reg.get_or_create("service_b")
        self.assertIsNot(cb1, cb2)

    def test_manual_reset(self):
        from framework.resilience.retry import get_circuit_registry
        reg = get_circuit_registry()
        cb  = reg.get_or_create("reset_test", failure_threshold=2)
        # Force open
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        self.assertEqual(cb.state, CircuitState.OPEN)
        # Manual reset
        reg.reset("reset_test")
        self.assertEqual(cb.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
