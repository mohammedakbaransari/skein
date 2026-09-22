"""
framework/governance/hashchain.py
====================================
Backend-agnostic hash-chain math, extracted from HashChainedWriter so the
same audit-chain algorithm (and its concurrency/restart-safety fixes) can
back multiple persistence targets — local JSONL files
(framework/governance/logger.py::HashChainedWriter) and, at multi-tenant
scale, a Delta table per tenant (platform/databricks/adapter.py::
DeltaGovernanceStore) — without duplicating or drifting the chain logic
between them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable

GENESIS = "GENESIS"


def compute_chained_entry(record: Dict[str, Any], prev_hash: str) -> Dict[str, Any]:
    """Return a new dict with prev_hash/hash fields set for this record.

    Callers must serialise the returned dict themselves (JSONL line, Delta
    row, etc.) — this function only computes the chain fields.
    """
    entry = {**record, "prev_hash": prev_hash}
    raw = json.dumps(entry, default=str, sort_keys=True, ensure_ascii=False)
    entry_hash = hashlib.sha256(raw.encode()).hexdigest()[:24]
    entry["hash"] = entry_hash
    return entry


def verify_chained_entries(entries: Iterable[Dict[str, Any]]) -> bool:
    """Verify a sequence of already-persisted entries (in append order).

    Each entry must contain 'prev_hash' and 'hash' as they were stored.
    Returns True if the chain is internally consistent (no missing,
    reordered, or tampered entries), False otherwise.
    """
    prev_hash = GENESIS
    for entry in entries:
        entry = dict(entry)
        stored_prev = entry.get("prev_hash", "")
        stored_hash = entry.pop("hash", "")
        raw = json.dumps(
            {**entry, "prev_hash": stored_prev},
            default=str, sort_keys=True, ensure_ascii=False,
        )
        expected_hash = hashlib.sha256(raw.encode()).hexdigest()[:24]
        if stored_prev != prev_hash:
            return False
        if stored_hash and expected_hash != stored_hash:
            return False
        prev_hash = stored_hash
    return True
