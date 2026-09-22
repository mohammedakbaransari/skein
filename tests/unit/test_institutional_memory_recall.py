"""
tests/unit/test_institutional_memory_recall.py
=================================================
Proves InstitutionalMemoryAgent actually reads back ("recalls") previously
captured patterns before generating new findings, closing roadmap item
P3-1 ("institutional memory is write-only in practice").
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.market_intelligence.agents import InstitutionalMemoryAgent
from framework.core.types import SessionId, Task
from framework.memory.store import WorkingMemory
from framework.reasoning.stubs import DryRunReasoningEngine


def _decision_records(category="IT"):
    return [
        {"category": category, "rationale_text": "Consolidated vendor to reduce overhead"},
        {"category": category, "rationale_text": ""},
    ]


class TestInstitutionalMemoryRecall(unittest.TestCase):

    def setUp(self):
        self.memory = WorkingMemory(max_entries=1000)
        self.engine = DryRunReasoningEngine(synthetic_json={
            "patterns": [{
                "category": "IT",
                "pattern_type": "vendor_consolidation",
                "reasoning_template": "Consolidate to a single vendor above volume threshold",
                "situational_triggers": ["multiple_vendors_same_spec"],
                "decision_heuristic": "Prefer single vendor when annual volume > $500k",
            }],
            "knowledge_gaps": [],
            "capture_recommendations": [],
        })
        self.agent = InstitutionalMemoryAgent(memory=self.memory, reasoning_engine=self.engine)
        self.session_id = SessionId.generate()

    def test_first_run_has_no_precedent(self):
        task = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()},
                            session_id=self.session_id)
        result = self.agent.run(task)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.observations["precedent_patterns"], [])
        main_finding = result.findings[0]
        self.assertEqual(main_finding.evidence["precedent_patterns_used"], 0)

    def test_second_run_recalls_pattern_captured_by_first_run(self):
        task1 = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()},
                             session_id=self.session_id)
        self.agent.run(task1)

        # Same session: the pattern captured above must now be recalled.
        task2 = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()},
                             session_id=self.session_id)
        result2 = self.agent.run(task2)

        self.assertTrue(result2.succeeded)
        precedent = result2.observations["precedent_patterns"]
        self.assertEqual(len(precedent), 1)
        self.assertEqual(precedent[0]["pattern_type"], "vendor_consolidation")

        main_finding = result2.findings[0]
        self.assertEqual(main_finding.evidence["precedent_patterns_used"], 1)
        self.assertIn("precedent pattern", main_finding.summary)

    def test_different_session_does_not_see_precedent(self):
        task1 = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()},
                             session_id=self.session_id)
        self.agent.run(task1)

        other_session = SessionId.generate()
        task2 = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()},
                             session_id=other_session)
        result2 = self.agent.run(task2)

        self.assertEqual(result2.observations["precedent_patterns"], [])

    def test_no_memory_store_configured_is_still_safe(self):
        agent = InstitutionalMemoryAgent(memory=None, reasoning_engine=self.engine)
        task = Task.create("InstitutionalMemoryAgent", {"decision_records": _decision_records()})
        result = agent.run(task)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.observations["precedent_patterns"], [])


if __name__ == "__main__":
    unittest.main()
