"""Webhook delivery for CRITICAL/HIGH findings (capability in Section 3, R7).

Lets a customer register a URL to receive findings above a severity
threshold instead of polling `GET /v1/findings`. Delivery failures are
recorded, never silently dropped or retried indefinitely.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WebhookSubscription:
    tenant_id: str
    url: str
    min_severity: str = "high"


@dataclass(frozen=True)
class WebhookDeliveryRecord:
    tenant_id: str
    url: str
    finding_id: str
    succeeded: bool
    error: Optional[str] = None
    delivered_at: str = field(default_factory=_now)


class WebhookDispatcher:
    """Thread-safe webhook registry + best-effort synchronous delivery."""

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, List[WebhookSubscription]] = {}
        self._deliveries: List[WebhookDeliveryRecord] = []
        self._timeout_seconds = timeout_seconds

    def subscribe(self, tenant_id: str, url: str, min_severity: str = "high") -> None:
        if min_severity not in _SEVERITY_RANK:
            raise ValueError(f"unknown severity {min_severity!r}")
        with self._lock:
            self._subscriptions.setdefault(tenant_id, []).append(
                WebhookSubscription(tenant_id=tenant_id, url=url, min_severity=min_severity)
            )

    def dispatch_finding(self, tenant_id: str, finding: dict) -> List[WebhookDeliveryRecord]:
        with self._lock:
            subscriptions = list(self._subscriptions.get(tenant_id, []))
        records: List[WebhookDeliveryRecord] = []
        finding_rank = _SEVERITY_RANK.get(finding.get("severity", "info"), 0)
        for subscription in subscriptions:
            if finding_rank < _SEVERITY_RANK[subscription.min_severity]:
                continue
            records.append(self._deliver(subscription, finding))
        return records

    def _deliver(self, subscription: WebhookSubscription, finding: dict) -> WebhookDeliveryRecord:
        body = json.dumps(finding, default=str).encode()
        request = urllib.request.Request(
            subscription.url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                succeeded = response.status < 300
                error = None if succeeded else f"status {response.status}"
        except urllib.error.URLError as exc:
            succeeded, error = False, str(exc)
        record = WebhookDeliveryRecord(
            tenant_id=subscription.tenant_id, url=subscription.url,
            finding_id=finding.get("finding_id", ""), succeeded=succeeded, error=error,
        )
        with self._lock:
            self._deliveries.append(record)
        if not succeeded:
            log.warning("Webhook delivery failed for tenant=%s url=%s: %s",
                        subscription.tenant_id, subscription.url, error)
        return record

    def deliveries(self, tenant_id: Optional[str] = None) -> List[WebhookDeliveryRecord]:
        with self._lock:
            records = list(self._deliveries)
        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]
        return records
