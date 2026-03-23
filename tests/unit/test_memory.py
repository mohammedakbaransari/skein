"""
tests/unit/test_memory.py
===========================
Unit tests for memory layer — session isolation, LRU eviction, TTL,
concurrent access, and ContextMemory namespace isolation.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.core.types import AgentId, SessionId
from framework.memory.store import ContextMemory, InstitutionalMemory, WorkingMemory


class TestWorkingMemoryBasic(unittest.TestCase):

    def setUp(self):
        self.mem = WorkingMemory(max_entries=100)
        self.s1  = SessionId("session-A")
        self.s2  = SessionId("session-B")

    def test_set_and_get(self):
        self.mem.set("key1", "value1", session_id=self.s1)
        self.assertEqual(self.mem.get("key1", session_id=self.s1), "value1")

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.mem.get("nonexistent", session_id=self.s1))

    def test_session_isolation(self):
        """Keys stored in session A are NOT visible in session B."""
        self.mem.set("shared_key", "session_a_value", session_id=self.s1)
        result = self.mem.get("shared_key", session_id=self.s2)
        self.assertIsNone(result, "Session B should not see Session A's data")

    def test_same_key_different_sessions(self):
        """Same key in different sessions stores independent values."""
        self.mem.set("k", "value_a", session_id=self.s1)
        self.mem.set("k", "value_b", session_id=self.s2)
        self.assertEqual(self.mem.get("k", session_id=self.s1), "value_a")
        self.assertEqual(self.mem.get("k", session_id=self.s2), "value_b")

    def test_delete(self):
        self.mem.set("del_key", "val", session_id=self.s1)
        self.mem.delete("del_key", session_id=self.s1)
        self.assertIsNone(self.mem.get("del_key", session_id=self.s1))

    def test_keys_list(self):
        self.mem.set("a", 1, session_id=self.s1)
        self.mem.set("b", 2, session_id=self.s1)
        self.mem.set("c", 3, session_id=self.s2)
        keys = self.mem.keys(session_id=self.s1)
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertNotIn("c", keys)

    def test_global_namespace(self):
        self.mem.set("global_key", "global_val")
        self.assertEqual(self.mem.get("global_key"), "global_val")
        # Not visible in a session
        self.assertIsNone(self.mem.get("global_key", session_id=self.s1))

    def test_overwrite(self):
        self.mem.set("k", "v1", session_id=self.s1)
        self.mem.set("k", "v2", session_id=self.s1)
        self.assertEqual(self.mem.get("k", session_id=self.s1), "v2")

    def test_get_or_default(self):
        result = self.mem.get_or_default("missing", "default_val", session_id=self.s1)
        self.assertEqual(result, "default_val")


class TestWorkingMemoryTTL(unittest.TestCase):

    def setUp(self):
        self.mem = WorkingMemory(max_entries=100)
        self.s1  = SessionId("session-ttl")

    def test_ttl_expiry(self):
        self.mem.set("ttl_key", "expires", session_id=self.s1, ttl_seconds=0.05)
        self.assertEqual(self.mem.get("ttl_key", session_id=self.s1), "expires")
        time.sleep(0.1)
        self.assertIsNone(self.mem.get("ttl_key", session_id=self.s1))

    def test_no_ttl_persists(self):
        self.mem.set("persist_key", "forever", session_id=self.s1)
        time.sleep(0.05)
        self.assertEqual(self.mem.get("persist_key", session_id=self.s1), "forever")

    def test_expired_key_not_in_keys_list(self):
        self.mem.set("exp", "val", session_id=self.s1, ttl_seconds=0.05)
        time.sleep(0.1)
        keys = self.mem.keys(session_id=self.s1)
        self.assertNotIn("exp", keys)


class TestWorkingMemoryLRU(unittest.TestCase):

    def test_lru_eviction_keeps_recently_used(self):
        mem = WorkingMemory(max_entries=5)
        sid = SessionId("evict-test")
        for i in range(5):
            mem.set(f"key{i}", i, session_id=sid)
        # Access key0 to make it recently used
        mem.get("key0", session_id=sid)
        # Add one more — should evict key1 (oldest not accessed)
        mem.set("key5", 5, session_id=sid)
        total = mem.total_entries
        self.assertLessEqual(total, 5)
        # key0 should still be there (recently used)
        self.assertIsNotNone(mem.get("key0", session_id=sid))

    def test_total_entries_does_not_exceed_max(self):
        mem = WorkingMemory(max_entries=10)
        sid = SessionId("capacity-test")
        for i in range(50):
            mem.set(f"k{i}", i, session_id=sid)
        self.assertLessEqual(mem.total_entries, 10)

    def test_clear_session(self):
        mem = WorkingMemory(max_entries=100)
        sid = SessionId("clear-test")
        for i in range(5):
            mem.set(f"k{i}", i, session_id=sid)
        self.assertEqual(len(mem.keys(session_id=sid)), 5)
        mem.clear_session(sid)
        self.assertEqual(len(mem.keys(session_id=sid)), 0)


class TestWorkingMemoryConcurrency(unittest.TestCase):

    def test_concurrent_writes_different_sessions(self):
        """50 threads writing to different sessions should not interfere."""
        mem = WorkingMemory(max_entries=10_000)
        errors = []

        def write_session(n):
            sid = SessionId(f"session-{n}")
            for i in range(20):
                try:
                    mem.set(f"k{i}", f"s{n}_v{i}", session_id=sid)
                    v = mem.get(f"k{i}", session_id=sid)
                    if v != f"s{n}_v{i}":
                        errors.append(f"session {n}: expected s{n}_v{i} got {v}")
                except Exception as exc:
                    errors.append(str(exc))

        threads = [threading.Thread(target=write_session, args=(n,)) for n in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(errors, [], f"Concurrency errors: {errors[:3]}")

    def test_concurrent_reads_do_not_block(self):
        """Reads must not block each other under shared RLock."""
        mem = WorkingMemory(max_entries=1000)
        sid = SessionId("concurrent-reads")
        for i in range(10):
            mem.set(f"k{i}", i, session_id=sid)

        results = []
        def read():
            for i in range(10):
                v = mem.get(f"k{i}", session_id=sid)
                results.append(v)

        threads = [threading.Thread(target=read) for _ in range(20)]
        start = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.monotonic() - start

        self.assertEqual(len(results), 200)
        self.assertLess(elapsed, 2.0, "Concurrent reads took too long")


class TestContextMemory(unittest.TestCase):

    def test_namespace_isolation(self):
        """Two ContextMemory instances with different workflow_ids should not see each other's keys."""
        wm = WorkingMemory(max_entries=100)
        sid = SessionId("ctx-test")
        cm1 = ContextMemory(wm, "workflow-1")
        cm2 = ContextMemory(wm, "workflow-2")

        cm1.set("key", "val_wf1", session_id=sid)
        cm2.set("key", "val_wf2", session_id=sid)

        self.assertEqual(cm1.get("key", session_id=sid), "val_wf1")
        self.assertEqual(cm2.get("key", session_id=sid), "val_wf2")

    def test_keys_scoped_to_workflow(self):
        wm = WorkingMemory(max_entries=100)
        sid = SessionId("ctx-keys")
        cm1 = ContextMemory(wm, "wf-alpha")
        cm2 = ContextMemory(wm, "wf-beta")

        cm1.set("only_in_alpha", "v", session_id=sid)
        cm2.set("only_in_beta", "v", session_id=sid)

        alpha_keys = cm1.keys(session_id=sid)
        beta_keys  = cm2.keys(session_id=sid)

        self.assertIn("only_in_alpha", alpha_keys)
        self.assertNotIn("only_in_beta", alpha_keys)
        self.assertIn("only_in_beta", beta_keys)
        self.assertNotIn("only_in_alpha", beta_keys)


class TestInstitutionalMemoryInProcess(unittest.TestCase):

    def test_basic_set_get(self):
        mem = InstitutionalMemory(storage_path=None)
        mem.set("pattern:negotiation", {"heuristic": "always anchor high"})
        v = mem.get("pattern:negotiation")
        self.assertEqual(v["heuristic"], "always anchor high")

    def test_delete(self):
        mem = InstitutionalMemory(storage_path=None)
        mem.set("del_me", "value")
        mem.delete("del_me")
        self.assertIsNone(mem.get("del_me"))

    def test_keys_returns_all(self):
        mem = InstitutionalMemory(storage_path=None)
        mem.set("a", 1)
        mem.set("b", 2)
        self.assertIn("a", mem.keys())
        self.assertIn("b", mem.keys())

    def test_atomic_update(self):
        mem = InstitutionalMemory(storage_path=None)
        mem.set("counter", 0)
        results = []
        def increment(current):
            return (current or 0) + 1

        threads = [
            threading.Thread(target=lambda: results.append(mem.update("counter", increment)))
            for _ in range(10)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        final = mem.get("counter")
        self.assertEqual(final, 10, f"Expected 10 after 10 increments, got {final}")


if __name__ == "__main__":
    unittest.main()
