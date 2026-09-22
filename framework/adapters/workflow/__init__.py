"""Portable workflow contracts and the in-memory reference adapter."""

from framework.adapters.workflow.in_memory import InMemoryWorkflowEngine
from framework.adapters.workflow.interfaces import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowRun,
    WorkflowState,
)

__all__ = [
    "InMemoryWorkflowEngine",
    "WorkflowDefinition",
    "WorkflowEngine",
    "WorkflowRun",
    "WorkflowState",
]