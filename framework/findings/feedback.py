"""Human feedback loop feeding the confidence evaluation harness (R22)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FeedbackRecord:
    finding_id: str
    agent_type: str
    raw_confidence: float
    correct: bool
    reviewer: str
    notes: str = ""
    recorded_at: str = field(default_factory=_now)


class FeedbackStore:
    """Thread-safe (finding, human verdict) pairs for calibration/eval harnesses."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: List[FeedbackRecord] = []

    def record(
        self, finding_id: str, agent_type: str, raw_confidence: float,
        correct: bool, reviewer: str, notes: str = "",
    ) -> FeedbackRecord:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        entry = FeedbackRecord(
            finding_id=finding_id, agent_type=agent_type, raw_confidence=raw_confidence,
            correct=correct, reviewer=reviewer, notes=notes,
        )
        with self._lock:
            self._records.append(entry)
        return entry

    def for_agent(self, agent_type: str) -> List[FeedbackRecord]:
        with self._lock:
            return [r for r in self._records if r.agent_type == agent_type]

    def to_confidence_samples(self, agent_type: str) -> List["ConfidenceSample"]:
        from framework.agents.confidence import ConfidenceSample
        return [
            ConfidenceSample(raw_score=r.raw_confidence, correct=r.correct)
            for r in self.for_agent(agent_type)
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
