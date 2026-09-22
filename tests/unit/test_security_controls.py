"""
tests/unit/test_security_controls.py
=======================================
Regression tests for the security enforcement wired to config.yaml's
`security:` block (P0-3 fix): input length/depth validation, PII redaction,
and rate limiting. Disabled by default; explicitly enabled per test via
configure_security(), and reset afterwards so state does not leak into
other test modules (the enforcer is a process-level singleton).
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.security.controls import (
    InputValidationError, RateLimitExceededError, SecurityConfig,
    configure_security, get_security_enforcer, redact_pii,
)
from framework.agents.base import StructuralAgent
from framework.core.types import AgentMetadata, Finding, Severity, Task


class _EchoAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_SecEchoAgent", display_name="SecEcho", description="Test",
        version="0.1.0", capabilities=(), tags=("test",),
    )

    def observe(self, task):
        return {"echo": task.payload.get("msg", "")}

    def reason(self, obs, task):
        return json.dumps({"echo": obs["echo"]})

    def parse_findings(self, obs, reasoning, task):
        return [self._make_finding("echo", Severity.INFO, "ok")]


class TestSecurityDisabledByDefault(unittest.TestCase):
    """Library/test usage must be unaffected unless explicitly configured."""

    def tearDown(self):
        configure_security({})

    def test_default_enforcer_is_disabled(self):
        configure_security({})
        cfg = get_security_enforcer().config
        self.assertFalse(cfg.enable_input_sanitisation)
        self.assertFalse(cfg.enable_pii_redaction)
        self.assertEqual(cfg.rate_limit_requests_per_minute, 0)


class TestInputValidation(unittest.TestCase):

    def tearDown(self):
        configure_security({})

    def test_oversized_payload_rejected_when_enabled(self):
        configure_security({"enable_input_sanitisation": True, "max_input_length": 20})
        enforcer = get_security_enforcer()
        with self.assertRaises(InputValidationError):
            enforcer.check_payload({"data": "x" * 100})

    def test_deeply_nested_payload_rejected_when_enabled(self):
        configure_security({"enable_input_sanitisation": True, "max_json_depth": 2})
        enforcer = get_security_enforcer()
        nested = {"a": {"b": {"c": {"d": 1}}}}
        with self.assertRaises(InputValidationError):
            enforcer.check_payload(nested)

    def test_valid_payload_passes_when_enabled(self):
        configure_security({"enable_input_sanitisation": True, "max_input_length": 1000, "max_json_depth": 10})
        get_security_enforcer().check_payload({"data": "small"})

    def test_disabled_by_default_never_raises(self):
        configure_security({})
        get_security_enforcer().check_payload({"data": "x" * 1_000_000})

    def test_agent_run_fails_gracefully_on_oversized_payload(self):
        configure_security({"enable_input_sanitisation": True, "max_input_length": 10})
        agent = _EchoAgent()
        task = Task.create("_SecEchoAgent", {"msg": "way more than ten characters"})
        result = agent.run(task)
        self.assertFalse(result.succeeded)
        self.assertIn("max_input_length", result.error)


class TestPiiRedaction(unittest.TestCase):

    def tearDown(self):
        configure_security({})

    def test_redact_pii_masks_email_and_phone(self):
        text = "Contact john.doe@example.com or +1 415-555-0134 for details."
        redacted = redact_pii(text)
        self.assertNotIn("john.doe@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_redact_noop_when_disabled(self):
        configure_security({"enable_pii_redaction": False})
        enforcer = get_security_enforcer()
        text = "email me at a@b.com"
        self.assertEqual(enforcer.redact(text), text)

    def test_redact_active_when_enabled(self):
        configure_security({"enable_pii_redaction": True})
        enforcer = get_security_enforcer()
        self.assertNotIn("a@b.com", enforcer.redact("email me at a@b.com"))


class TestRateLimiting(unittest.TestCase):

    def tearDown(self):
        configure_security({})

    def test_rate_limit_blocks_after_threshold(self):
        configure_security({"rate_limit_requests_per_minute": 2})
        enforcer = get_security_enforcer()
        enforcer.check_rate_limit("k")
        enforcer.check_rate_limit("k")
        with self.assertRaises(RateLimitExceededError):
            enforcer.check_rate_limit("k")

    def test_rate_limit_disabled_when_zero(self):
        configure_security({"rate_limit_requests_per_minute": 0})
        enforcer = get_security_enforcer()
        for _ in range(50):
            enforcer.check_rate_limit("k")

    def test_rate_limit_keys_independent(self):
        configure_security({"rate_limit_requests_per_minute": 1})
        enforcer = get_security_enforcer()
        enforcer.check_rate_limit("agent-a")
        enforcer.check_rate_limit("agent-b")  # different key, not blocked

    def test_agent_run_rate_limits_per_tenant_not_globally(self):
        """Multi-tenant: BaseAgent.run() keys the rate limit on task.tenant_id
        when present, so one noisy tenant cannot exhaust another tenant's
        quota (see framework/agents/base.py rate_limit_key)."""
        from framework.core.types import TenantId
        configure_security({"rate_limit_requests_per_minute": 1})
        agent = _EchoAgent()

        tenant_a = TenantId("tenant-a")
        tenant_b = TenantId("tenant-b")
        r1 = agent.run(Task.create("_SecEchoAgent", {"msg": "x"}, tenant_id=tenant_a))
        r2 = agent.run(Task.create("_SecEchoAgent", {"msg": "x"}, tenant_id=tenant_b))
        r3 = agent.run(Task.create("_SecEchoAgent", {"msg": "x"}, tenant_id=tenant_a))

        self.assertTrue(r1.succeeded)
        self.assertTrue(r2.succeeded, "a different tenant must not be blocked by tenant-a's quota")
        self.assertFalse(r3.succeeded, "tenant-a's second request within the window must be blocked")


if __name__ == "__main__":
    unittest.main()
