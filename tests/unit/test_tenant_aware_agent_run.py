"""
tests/unit/test_tenant_aware_agent_run.py
============================================
Proves BaseAgent.run() correctly routes to per-tenant memory/governance
when a tenant_store_resolver is configured, and always restores the
agent's default memory/governance afterward — the swap only ever affects
the single run() call in progress.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.agents.base import StructuralAgent
from framework.core.types import AgentMetadata, SessionId, Severity, Task
from framework.memory.store import WorkingMemory
from framework.reasoning.stubs import DryRunReasoningEngine


class _FakeGovernance:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def record_execution(self, agent_id, agent_type, task, result):
        self.calls.append((self.name, str(task.task_id)))


class _RememberingAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_RememberingAgent", display_name="Remembering", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task):
        return {"msg": task.payload.get("msg", "")}

    def reason(self, obs, task):
        return json.dumps({"msg": obs["msg"]})

    def parse_findings(self, obs, reasoning, task):
        self.remember("last_msg", obs["msg"], session_id=task.session_id)
        return [self._make_finding("echo", Severity.INFO, obs["msg"])]


class TestTenantAwareAgentRun(unittest.TestCase):

    def setUp(self):
        self.default_memory = WorkingMemory()
        self.default_governance = _FakeGovernance("default")
        self.tenant_memory = WorkingMemory()
        self.tenant_governance = _FakeGovernance("tenant-acme")

        self.agent = _RememberingAgent(
            memory=self.default_memory,
            governance_logger=self.default_governance,
            reasoning_engine=DryRunReasoningEngine(),
        )
        self.agent.tenant_store_resolver = lambda tenant_id: (
            (self.tenant_memory, self.tenant_governance) if tenant_id == "acme" else None
        )

    def test_no_tenant_uses_default_stores(self):
        session_id = SessionId.generate()
        task = Task.create("_RememberingAgent", {"msg": "hello"}, session_id=session_id)

        result = self.agent.run(task)

        self.assertTrue(result.succeeded)
        self.assertEqual(self.default_memory.get("last_msg", session_id=session_id), "hello")
        self.assertIsNone(self.tenant_memory.get("last_msg", session_id=session_id))
        self.assertEqual(len(self.default_governance.calls), 1)
        self.assertEqual(len(self.tenant_governance.calls), 0)
        # Restored afterward
        self.assertIs(self.agent.memory, self.default_memory)
        self.assertIs(self.agent.governance, self.default_governance)

    def test_known_tenant_routes_to_tenant_stores(self):
        from framework.core.types import TenantId
        session_id = SessionId.generate()
        task = Task.create(
            "_RememberingAgent", {"msg": "hi acme"},
            session_id=session_id, tenant_id=TenantId("acme"),
        )

        result = self.agent.run(task)

        self.assertTrue(result.succeeded)
        self.assertEqual(self.tenant_memory.get("last_msg", session_id=session_id), "hi acme")
        self.assertIsNone(self.default_memory.get("last_msg", session_id=session_id))
        self.assertEqual(len(self.tenant_governance.calls), 1)
        self.assertEqual(len(self.default_governance.calls), 0)
        # Restored afterward — the swap only applies during this one run()
        self.assertIs(self.agent.memory, self.default_memory)
        self.assertIs(self.agent.governance, self.default_governance)

    def test_unregistered_tenant_falls_back_to_default_stores(self):
        from framework.core.types import TenantId
        session_id = SessionId.generate()
        task = Task.create(
            "_RememberingAgent", {"msg": "hi nobody"},
            session_id=session_id, tenant_id=TenantId("does-not-exist"),
        )

        result = self.agent.run(task)

        self.assertTrue(result.succeeded)
        self.assertEqual(self.default_memory.get("last_msg", session_id=session_id), "hi nobody")
        self.assertEqual(len(self.default_governance.calls), 1)

    def test_second_run_for_different_tenant_does_not_see_first_tenants_data(self):
        from framework.core.types import TenantId
        s1 = SessionId.generate()
        self.agent.run(Task.create("_RememberingAgent", {"msg": "tenant-msg"},
                                    session_id=s1, tenant_id=TenantId("acme")))
        s2 = SessionId.generate()
        self.agent.run(Task.create("_RememberingAgent", {"msg": "no-tenant-msg"}, session_id=s2))

        self.assertEqual(self.tenant_memory.get("last_msg", session_id=s1), "tenant-msg")
        self.assertEqual(self.default_memory.get("last_msg", session_id=s2), "no-tenant-msg")


if __name__ == "__main__":
    unittest.main()
