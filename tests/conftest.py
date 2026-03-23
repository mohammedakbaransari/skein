"""
tests/conftest.py
==================
Shared pytest fixtures and test configuration for the SKEIN test suite.

All tests run without a live LLM — DryRunReasoningEngine is used throughout.
Registry and circuit registry are reset between test classes to prevent
state bleed across tests.

Available fixtures:
  registry          fresh AgentRegistry with no agents registered
  memory            WorkingMemory(max_entries=1000) for testing
  governance(tmp)   GovernanceLogger writing to a temp directory
  orchestrator      TaskOrchestrator backed by the registry fixture
  dry_engine        DryRunReasoningEngine with generic procurement response
  session_id        a fresh SessionId for each test
  trace_context     a fresh CorrelationContext for each test
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure repo root is importable when running via pytest from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from framework.core.registry import AgentRegistry, reset_registry
from framework.core.types import CorrelationContext, SessionId
from framework.governance.logger import GovernanceLogger
from framework.memory.store import WorkingMemory
from framework.orchestration.orchestrator import TaskOrchestrator
from framework.reasoning.stubs import DryRunReasoningEngine
from framework.resilience.retry import reset_circuit_registry

# Generic DryRun response that satisfies all agents' parse_findings methods
GENERIC_DRY_RESPONSE = {
    "executive_summary": "DRY_RUN_TEST",
    "assessment": "DRY_RUN_TEST",
    "suppliers": [],
    "leverage_opportunities": [],
    "gaps": [],
    "patterns": [],
    "immediate_actions": [],
    "recommended_actions": [],
    "rising_cost_warnings": [],
    "market_assessment": "DRY_RUN_TEST",
    "accountability_assessment": "DRY_RUN_TEST",
    "regulatory_exposure_level": "Low",
    "evaluator_flags": [],
    "framework_to_implement": "none",
    "bias_assessment": "DRY_RUN_TEST",
    "bias_indicators": [],
    "intervention_opportunities": [],
    "compliance_assessment": "DRY_RUN_TEST",
    "material_risks": [],
    "certification_gaps": [],
    "tco_assessment": "DRY_RUN_TEST",
    "tco_gaps": [],
    "value_assessment": "DRY_RUN_TEST",
    "value_gaps": [],
    "knowledge_gaps": [],
    "capture_recommendations": [],
    "immediate_priorities": [],
    "trade_assessment": "DRY_RUN_TEST",
    "scenarios": [],
    "portfolio_risk": "DRY_RUN_TEST",
    "innovation_opportunities": [],
    "specification_risks": [],
    "demand_signals": [],
    "capital_assessment": "DRY_RUN_TEST",
    "optimization_opportunities": [],
    "copilot_assessment": "DRY_RUN_TEST",
    "priority_actions": [],
    "negotiation_intelligence": "DRY_RUN_TEST",
    "counterparty_insights": [],
}


# ---------------------------------------------------------------------------
# pytest fixtures (used when running with pytest)
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.fixture(autouse=True)
    def _reset_globals():
        """Reset global singletons before each test to prevent state bleed."""
        reset_registry()
        reset_circuit_registry()
        yield
        reset_registry()
        reset_circuit_registry()

    @pytest.fixture
    def registry():
        return AgentRegistry()

    @pytest.fixture
    def memory():
        return WorkingMemory(max_entries=1000)

    @pytest.fixture
    def governance(tmp_path):
        return GovernanceLogger(str(tmp_path))

    @pytest.fixture
    def orchestrator(registry):
        return TaskOrchestrator(registry, config=None)

    @pytest.fixture
    def dry_engine():
        return DryRunReasoningEngine(GENERIC_DRY_RESPONSE)

    @pytest.fixture
    def session_id():
        return SessionId.generate()

    @pytest.fixture
    def trace_context():
        return CorrelationContext.new(test="pytest")

except ImportError:
    # pytest not installed — fixtures are no-ops, tests use unittest directly
    pass


# ---------------------------------------------------------------------------
# unittest setUp helpers (used by unittest.TestCase subclasses)
# ---------------------------------------------------------------------------

def reset_all():
    """Call in unittest setUp() to ensure clean state."""
    reset_registry()
    reset_circuit_registry()


def make_dry_engine(extra: dict = None) -> DryRunReasoningEngine:
    """Return a DryRunReasoningEngine with generic response + optional overrides."""
    response = dict(GENERIC_DRY_RESPONSE)
    if extra:
        response.update(extra)
    return DryRunReasoningEngine(response)
