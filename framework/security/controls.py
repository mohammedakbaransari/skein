"""
framework/security/controls.py
================================
Real enforcement for the config.yaml `security:` block (input sanitisation,
PII redaction, rate limiting). Previously these config keys existed but no
code read or enforced them.

Disabled by default (safe for library/test usage — see SecurityConfig
defaults); `scripts/server.py` wires this up from config.yaml at startup via
configure_security(). Call configure_security({}) to reset to disabled.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


class InputValidationError(ValueError):
    """Raised when a task payload violates the configured input limits."""


class RateLimitExceededError(RuntimeError):
    """Raised when a caller exceeds the configured rate limit."""


@dataclass(frozen=True)
class SecurityConfig:
    enable_input_sanitisation: bool = False
    enable_pii_redaction:      bool = False
    max_input_length:          int = 50_000
    max_output_length:         int = 20_000
    max_json_depth:             int = 10
    max_request_body_bytes:     int = 1_048_576
    rate_limit_requests_per_minute: int = 0   # 0 = disabled

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SecurityConfig":
        data = data or {}
        defaults = cls()
        return cls(
            enable_input_sanitisation=bool(data.get("enable_input_sanitisation", defaults.enable_input_sanitisation)),
            enable_pii_redaction=bool(data.get("enable_pii_redaction", defaults.enable_pii_redaction)),
            max_input_length=int(data.get("max_input_length", defaults.max_input_length)),
            max_output_length=int(data.get("max_output_length", defaults.max_output_length)),
            max_json_depth=int(data.get("max_json_depth", defaults.max_json_depth)),
            max_request_body_bytes=int(data.get("max_request_body_bytes", defaults.max_request_body_bytes)),
            rate_limit_requests_per_minute=int(data.get("rate_limit_requests_per_minute", defaults.rate_limit_requests_per_minute)),
        )


def _depth(obj: Any, current: int = 0) -> int:
    if current > 1000:
        return current  # guard against pathological/cyclic structures
    if isinstance(obj, dict) and obj:
        return max(_depth(v, current + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)) and obj:
        return max(_depth(v, current + 1) for v in obj)
    return current


_PII_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),                       # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                              # SSN-like
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),                            # card-like
    re.compile(r"\b\+?\d{1,3}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),  # phone
)


def redact_pii(text: str) -> str:
    """Replace common PII patterns (email/SSN/card/phone) with '[REDACTED]'."""
    if not text:
        return text
    redacted = text
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


# ---------------------------------------------------------------------------
# Prompt-injection mitigation
# ---------------------------------------------------------------------------
# Agents interpolate free-text fields from task.payload (rationale text,
# negotiation transcripts, discrepancy signals, etc.) directly into LLM
# prompts. These heuristics are a defence-in-depth measure, not a guarantee:
# they defang common imperative injection phrasing and visually/structurally
# separate untrusted content from the surrounding instructions, applied by
# ReasoningEngine to every request regardless of which agent built it.

_INJECTION_PATTERNS = (
    re.compile(r"(?im)^\s*(system|assistant|user)\s*:\s*"),
    re.compile(r"(?i)ignore (all )?(the )?(previous|prior|above) instructions"),
    re.compile(r"(?i)disregard (the )?(above|previous|prior)( instructions)?"),
    re.compile(r"(?i)\byou are now\b"),
    re.compile(r"(?i)new instructions\s*:"),
)

_registered_secrets: set[str] = set()
_secret_lock = threading.Lock()


def register_secret(secret: str) -> None:
    if secret:
        with _secret_lock:
            _registered_secrets.add(secret)


def redact_secrets(text: str) -> str:
    if not isinstance(text, str):
        return text
    with _secret_lock:
        secrets = sorted(_registered_secrets, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, "[SECRET_REDACTED]")
    return text


def neutralize_prompt_injection(text: str) -> str:
    """Defang common prompt-injection phrasing before it reaches an LLM prompt."""
    if not text:
        return text
    neutralized = text
    for pattern in _INJECTION_PATTERNS:
        neutralized = pattern.sub("[neutralized] ", neutralized)
    return neutralized.replace("```", "'''")


def wrap_untrusted_data(text: str, label: str = "DATA") -> str:
    """Delimit untrusted text so an injection attempt inside it is
    structurally separated from the surrounding trusted instructions."""
    safe_label = re.sub(r"[^A-Za-z0-9_]", "_", label).upper() or "DATA"
    body = neutralize_prompt_injection(text)
    return (
        f"<<<BEGIN_UNTRUSTED_{safe_label}>>>\n"
        f"{body}\n"
        f"<<<END_UNTRUSTED_{safe_label}>>>"
    )


class _RateLimiter:
    """Thread-safe per-key sliding-window limiter. limit<=0 disables checks."""

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._lock = threading.Lock()
        self._hits: Dict[str, list] = {}

    def allow(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits.setdefault(key, [])
            while hits and hits[0] < window_start:
                hits.pop(0)
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


class SecurityEnforcer:
    """Facade used by BaseAgent.run() and the structured logger."""

    def __init__(self, config: SecurityConfig) -> None:
        self._cfg = config
        self._limiter = _RateLimiter(config.rate_limit_requests_per_minute)

    @property
    def config(self) -> SecurityConfig:
        return self._cfg

    def check_payload(self, payload: Dict[str, Any]) -> None:
        if not self._cfg.enable_input_sanitisation:
            return
        import json as _json
        try:
            serialised = _json.dumps(payload, default=str)
        except Exception:
            serialised = str(payload)
        if len(serialised) > self._cfg.max_input_length:
            raise InputValidationError(
                f"Task payload length {len(serialised)} exceeds "
                f"max_input_length={self._cfg.max_input_length}"
            )
        depth = _depth(payload)
        if depth > self._cfg.max_json_depth:
            raise InputValidationError(
                f"Task payload nesting depth {depth} exceeds "
                f"max_json_depth={self._cfg.max_json_depth}"
            )

    def check_rate_limit(self, key: str) -> None:
        if self._cfg.rate_limit_requests_per_minute <= 0:
            return
        if not self._limiter.allow(key):
            raise RateLimitExceededError(
                f"Rate limit exceeded for '{key}' "
                f"({self._cfg.rate_limit_requests_per_minute}/min)"
            )

    def redact(self, text: str) -> str:
        redacted = redact_secrets(text)
        if self._cfg.enable_pii_redaction:
            redacted = redact_pii(redacted)
        return redacted


# ---------------------------------------------------------------------------
# Process-level singleton (mirrors get_metrics()/get_registry() convention)
# ---------------------------------------------------------------------------

_enforcer: Optional[SecurityEnforcer] = None
_enforcer_lock = threading.Lock()


def get_security_enforcer() -> SecurityEnforcer:
    global _enforcer
    if _enforcer is None:
        with _enforcer_lock:
            if _enforcer is None:
                _enforcer = SecurityEnforcer(SecurityConfig())
    return _enforcer


def configure_security(config: Optional[Dict[str, Any]]) -> SecurityEnforcer:
    """Rebuild the global enforcer from a `security:` config dict.
    Pass {} or None to reset to the disabled defaults (tests use this)."""
    global _enforcer
    with _enforcer_lock:
        _enforcer = SecurityEnforcer(SecurityConfig.from_dict(config))
    return _enforcer
