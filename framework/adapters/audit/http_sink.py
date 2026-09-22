"""HTTP-transport AuditSink adapter (R15).

Delivers the vendor-neutral `AuditEvent` contract as a JSON POST to a
configurable endpoint — the same protocol shape used by OTLP/HTTP,
Splunk HEC, and most SIEM collectors. Testable against a local HTTP
server without any live vendor credentials; the concrete SIEM/OTLP
endpoint is a deployment configuration value, not a code change.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Dict, Optional

from framework.adapters.audit.interfaces import AuditEvent

log = logging.getLogger(__name__)


class AuditDeliveryError(RuntimeError):
    """Raised when an audit event could not be delivered to the sink."""


class HttpAuditSink:
    """POSTs each AuditEvent as JSON to `endpoint_url`.

    Failures raise (never silently drop) so callers can decide on
    retry/buffer policy — delivery guarantees are a deployment concern
    (see roadmap R15: "do not claim SIEM compliance until confirmed").
    """

    def __init__(self, endpoint_url: str, headers: Optional[Dict[str, str]] = None, timeout_seconds: float = 5.0) -> None:
        self._endpoint_url = endpoint_url
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout_seconds = timeout_seconds

    def emit(self, event: AuditEvent) -> None:
        body = json.dumps(event.to_dict(), default=str).encode()
        request = urllib.request.Request(
            self._endpoint_url, data=body, headers=self._headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                if response.status >= 300:
                    raise AuditDeliveryError(f"audit sink returned status {response.status}")
        except urllib.error.URLError as exc:
            raise AuditDeliveryError(f"audit sink delivery failed: {exc}") from exc
