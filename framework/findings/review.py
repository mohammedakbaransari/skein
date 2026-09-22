"""Finding/decision review lifecycle state machine (R21)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewState(Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ACTIONED = "actioned"
    DISMISSED = "dismissed"


_ALLOWED_TRANSITIONS = {
    ReviewState.OPEN: {ReviewState.IN_REVIEW, ReviewState.DISMISSED},
    ReviewState.IN_REVIEW: {ReviewState.ACTIONED, ReviewState.DISMISSED},
    ReviewState.ACTIONED: set(),
    ReviewState.DISMISSED: set(),
}


@dataclass
class FindingReview:
    finding_id: str
    state: ReviewState = ReviewState.OPEN
    assignee: str = ""
    comment: str = ""
    updated_at: str = field(default_factory=_now)


class ReviewWorkflow:
    """Thread-safe finding review state machine, tracked by finding_id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reviews: Dict[str, FindingReview] = {}

    def get_or_open(self, finding_id: str) -> FindingReview:
        with self._lock:
            return self._reviews.setdefault(finding_id, FindingReview(finding_id=finding_id))

    def transition(
        self, finding_id: str, new_state: ReviewState,
        assignee: str = "", comment: str = "",
    ) -> FindingReview:
        with self._lock:
            review = self.get_or_open(finding_id)
            if new_state not in _ALLOWED_TRANSITIONS[review.state]:
                raise ValueError(
                    f"cannot transition finding {finding_id!r} from "
                    f"{review.state.value!r} to {new_state.value!r}"
                )
            review.state = new_state
            if assignee:
                review.assignee = assignee
            if comment:
                review.comment = comment
            review.updated_at = _now()
            return review

    def get(self, finding_id: str) -> Optional[FindingReview]:
        with self._lock:
            return self._reviews.get(finding_id)
