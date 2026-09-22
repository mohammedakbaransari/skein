"""Portable audit event contract and local sink."""

from framework.adapters.audit.interfaces import AuditEvent, AuditSink
from framework.adapters.audit.local import InMemoryAuditSink

__all__ = ["AuditEvent", "AuditSink", "InMemoryAuditSink"]