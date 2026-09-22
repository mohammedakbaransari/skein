"""
tests/unit/test_retry_idempotency.py
=======================================
Empirically tests retry-idempotency behavior (roadmap item P3-2 / R15).

Confirms the previously "theoretical, not proven" risk is real: an agent
that performs a side effect (e.g. a memory write) unconditionally inside
parse_findings() and then fails will have that side effect duplicated when
TaskOrchestrator retries the task, because for_retry() re-executes the full
observe/reason/parse_findings pipeline from scratch.

Also demonstrates the fix this phase adds: Task.idempotency_key is stable
across for_retry() (unlike task_id, which is regenerated per attempt), so
an agent that keys its side-effect guard on idempotency_key instead of
task_id does NOT duplicate the side effect across retries. This is an
opt-in tool for agents, not an automatic framework-wide guarantee — see
the first test below, which shows a naive agent is still unsafe by default.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import AgentMetadata, RetryConfig, SessionId, Severity, Task
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator

_FAST_RETRY = RetryConfig(max_attempts=2, initial_delay_s=0.01, backoff_factor=1.0,
                           max_delay_s=0.01, jitter_factor=0.0)


class _NaiveSideEffectAgent(StructuralAgent):
    """Ignores idempotency_key; duplicates its side effect on every retry."""
    METADATA = AgentMetadata(
        agent_type="_NaiveSideEffect", display_name="NaiveSideEffect", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )
    _failed_once: Dict[str, bool] = {}

    def observe(self, task: Task) -> Dict[str, Any]:
        return {"marker": task.payload["marker"]}

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"ok": True})

    def parse_findings(self, obs, reasoning, task) -> List:
        marker = obs["marker"]
        log_key = f"naive_side_effects:{marker}"
        existing = self.recall(log_key, session_id=task.session_id) or []
        self.remember(log_key, existing + [task.task_id.value], session_id=task.session_id)

        if not _NaiveSideEffectAgent._failed_once.get(marker):
            _NaiveSideEffectAgent._failed_once[marker] = True
            raise RuntimeError("Simulated crash after memory write")
        return [self._make_finding("ok", Severity.INFO, "succeeded on retry")]


class _IdempotentSideEffectAgent(StructuralAgent):
    """Guards its side effect using task.idempotency_key (stable across retries)."""
    METADATA = AgentMetadata(
        agent_type="_IdempotentSideEffect", display_name="IdempotentSideEffect", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )
    _failed_once: Dict[str, bool] = {}

    def observe(self, task: Task) -> Dict[str, Any]:
        return {"marker": task.payload["marker"], "idempotency_key": task.idempotency_key}

    def reason(self, obs: Dict, task: Task) -> str:
        return json.dumps({"ok": True})

    def parse_findings(self, obs, reasoning, task) -> List:
        marker, key = obs["marker"], obs["idempotency_key"]
        applied_key = f"idempotent_applied:{marker}"
        applied = self.recall(applied_key, session_id=task.session_id) or []
        if key not in applied:
            log_key = f"idempotent_side_effects:{marker}"
            existing = self.recall(log_key, session_id=task.session_id) or []
            self.remember(log_key, existing + [key], session_id=task.session_id)
            self.remember(applied_key, applied + [key], session_id=task.session_id)

        if not _IdempotentSideEffectAgent._failed_once.get(marker):
            _IdempotentSideEffectAgent._failed_once[marker] = True
            raise RuntimeError("Simulated crash after memory write")
        return [self._make_finding("ok", Severity.INFO, "succeeded on retry")]


class TestRetryIdempotency(unittest.TestCase):

    def setUp(self):
        reset_registry()
        self.reg = AgentRegistry()
        self.reg.register_class(_NaiveSideEffectAgent)
        self.reg.register_class(_IdempotentSideEffectAgent)
        self.memory = WorkingMemory()
        self.orch = TaskOrchestrator(self.reg, config=None)
        _NaiveSideEffectAgent._failed_once.clear()
        _IdempotentSideEffectAgent._failed_once.clear()

    def test_naive_agent_duplicates_side_effect_across_retry(self):
        """Confirms R15 empirically: retries are NOT idempotent by default."""
        self.reg.create_instance("_NaiveSideEffect", config=None, memory=self.memory)
        session_id = SessionId.generate()
        task = Task.create("_NaiveSideEffect", {"marker": "naive-1"},
                            session_id=session_id, retry_config=_FAST_RETRY)

        result = self.orch.run_task(task)

        self.assertTrue(result.succeeded)  # succeeded on the retry attempt
        effects = self.memory.get("naive_side_effects:naive-1", session_id=session_id)
        self.assertEqual(len(effects), 2, "side effect was duplicated across the retry")

    def test_idempotent_agent_avoids_duplicate_using_idempotency_key(self):
        """Task.idempotency_key (stable across for_retry) lets an agent opt
        into retry-safe side effects — this is the fix this phase adds."""
        self.reg.create_instance("_IdempotentSideEffect", config=None, memory=self.memory)
        session_id = SessionId.generate()
        task = Task.create("_IdempotentSideEffect", {"marker": "idem-1"},
                            session_id=session_id, retry_config=_FAST_RETRY)

        result = self.orch.run_task(task)

        self.assertTrue(result.succeeded)
        effects = self.memory.get("idempotent_side_effects:idem-1", session_id=session_id)
        self.assertEqual(len(effects), 1, "idempotency_key should have prevented duplication")

    def test_idempotency_key_is_stable_across_for_retry(self):
        task = Task.create("_IdempotentSideEffect", {"marker": "x"})
        original_key = task.idempotency_key
        retried = task.for_retry()
        self.assertEqual(retried.idempotency_key, original_key)
        self.assertNotEqual(retried.task_id.value, task.task_id.value)

    def test_idempotency_key_defaults_to_own_task_id_when_not_retried(self):
        task = Task.create("_IdempotentSideEffect", {"marker": "x"})
        self.assertEqual(task.idempotency_key, task.task_id.value)


if __name__ == "__main__":
    unittest.main()
