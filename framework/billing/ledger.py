"""Billing-grade, token/cost-based per-tenant usage ledger (R17, R18)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QuotaExceededError(RuntimeError):
    """Raised when a tenant's token/cost spend exceeds its configured quota."""


@dataclass(frozen=True)
class UsageEntry:
    tenant_id: str
    tokens: int
    cost_usd: float
    task_id: str
    recorded_at: str = field(default_factory=_now)


class UsageLedger:
    """Thread-safe, durable-shaped usage ledger — reconcilable per tenant.

    Deployment-scale storage (a real database) can replace the in-memory
    list behind the same `record`/`total_tokens`/`entries` interface.
    """

    def __init__(self, usd_per_1k_tokens: float = 0.01) -> None:
        self._lock = threading.RLock()
        self._entries: List[UsageEntry] = []
        self._usd_per_1k_tokens = usd_per_1k_tokens

    def record(self, tenant_id: str, tokens: int, task_id: str) -> UsageEntry:
        entry = UsageEntry(
            tenant_id=tenant_id, tokens=tokens, task_id=task_id,
            cost_usd=round(tokens / 1000 * self._usd_per_1k_tokens, 6),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def total_tokens(self, tenant_id: str, since: Optional[str] = None) -> int:
        return sum(e.tokens for e in self.entries(tenant_id, since))

    def total_cost_usd(self, tenant_id: str, since: Optional[str] = None) -> float:
        return round(sum(e.cost_usd for e in self.entries(tenant_id, since)), 6)

    def entries(self, tenant_id: Optional[str] = None, since: Optional[str] = None) -> List[UsageEntry]:
        with self._lock:
            records = list(self._entries)
        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]
        if since is not None:
            records = [r for r in records if r.recorded_at >= since]
        return records

    def export_csv(self, tenant_id: Optional[str] = None, since: Optional[str] = None) -> str:
        """Invoicing-ready CSV export (R18) — one row per usage entry."""
        import csv
        import io
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["tenant_id", "task_id", "tokens", "cost_usd", "recorded_at"])
        for entry in self.entries(tenant_id, since):
            writer.writerow([entry.tenant_id, entry.task_id, entry.tokens, entry.cost_usd, entry.recorded_at])
        return buffer.getvalue()

    def reconciliation_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Per-tenant integrity check: recomputed totals vs. stored entries (R18)."""
        entries = self.entries(tenant_id)
        recomputed_cost = round(sum(round(e.tokens / 1000 * self._usd_per_1k_tokens, 6) for e in entries), 6)
        stored_cost = round(sum(e.cost_usd for e in entries), 6)
        return {
            "tenant_id": tenant_id,
            "entry_count": len(entries),
            "total_tokens": sum(e.tokens for e in entries),
            "stored_cost_usd": stored_cost,
            "recomputed_cost_usd": recomputed_cost,
            "reconciled": stored_cost == recomputed_cost,
        }


class TokenQuotaEnforcer:
    """Per-tenant token quota — a real spend limit, not just a request count."""

    def __init__(self, ledger: UsageLedger, quotas: Optional[Dict[str, int]] = None) -> None:
        self._ledger = ledger
        self._quotas = dict(quotas or {})

    def set_quota(self, tenant_id: str, max_tokens_per_window: int) -> None:
        self._quotas[tenant_id] = max_tokens_per_window

    def check(self, tenant_id: str, additional_tokens: int, since: Optional[str] = None) -> None:
        quota = self._quotas.get(tenant_id)
        if quota is None:
            return
        projected = self._ledger.total_tokens(tenant_id, since) + additional_tokens
        if projected > quota:
            raise QuotaExceededError(
                f"tenant {tenant_id!r} would exceed token quota "
                f"({projected} > {quota})"
            )
