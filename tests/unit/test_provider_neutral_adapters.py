import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.adapters.audit import AuditEvent, InMemoryAuditSink
from framework.adapters.identity import Principal
from framework.adapters.secrets import EnvironmentSecretsProvider
from framework.adapters.workflow import (
    InMemoryWorkflowEngine,
    WorkflowDefinition,
    WorkflowState,
)
from framework.core.types import SessionId
from framework.core.schema import SchemaValidationError, validate_json_schema
from framework.agents.base import StructuralAgent
from framework.agents.confidence import (
    CalibrationProfile, ConfidenceSample, ConfidenceScorer, JsonCalibrationStore,
)
from framework.core.types import AgentMetadata, Severity, Task
from framework.security.controls import configure_security, register_secret, redact_secrets
from framework.reasoning.engine import ReasoningEngine, ReasoningRequest, ReasoningResponse
from framework.core.types import ReasoningStrategy
from framework.multitenancy.policy import TenantPolicy


class TestAuditContracts(unittest.TestCase):
    def test_event_serializes_versioned_portable_fields(self):
        event = AuditEvent(
            event_id="evt-1",
            event_type="workflow.task.completed",
            tenant_id="acme",
            action="completed",
            result="success",
        )
        payload = event.to_dict()
        self.assertEqual(payload["event_version"], "1.0")
        self.assertEqual(payload["tenant_id"], "acme")
        self.assertIn("timestamp", payload)

    def test_in_memory_sink_preserves_event_order(self):
        sink = InMemoryAuditSink()
        first = AuditEvent("evt-1", "task.started", "acme", "started", "success")
        second = AuditEvent("evt-2", "task.completed", "acme", "completed", "success")
        sink.emit(first)
        sink.emit(second)
        self.assertEqual(sink.events(), [first, second])


class TestEnvironmentSecretsProvider(unittest.TestCase):
    def test_missing_secret_fails_closed(self):
        os.environ.pop("SKEIN_TEST_SECRET", None)
        with self.assertRaises(KeyError):
            EnvironmentSecretsProvider().get_secret("SKEIN_TEST_SECRET")

    def test_secret_metadata_does_not_expose_value(self):
        os.environ["SKEIN_TEST_SECRET"] = "sensitive-value"
        try:
            metadata = EnvironmentSecretsProvider().get_secret_metadata("SKEIN_TEST_SECRET")
            self.assertEqual(metadata.name, "SKEIN_TEST_SECRET")
            self.assertNotIn("sensitive-value", repr(metadata))
        finally:
            os.environ.pop("SKEIN_TEST_SECRET", None)

    def test_environment_rotation_is_explicitly_unsupported(self):
        with self.assertRaises(NotImplementedError):
            EnvironmentSecretsProvider().rotate_secret("SKEIN_TEST_SECRET")


class TestTenantPolicy(unittest.TestCase):
    def test_logical_is_the_default_profile(self):
        self.assertEqual(TenantPolicy().isolation_profile, "logical")

    def test_region_must_be_allowed(self):
        with self.assertRaises(ValueError):
            TenantPolicy(allowed_regions=["eu"], processing_region="us")

    def test_cross_region_backup_requires_permission(self):
        with self.assertRaises(ValueError):
            TenantPolicy(storage_region="eu", backup_region="us")


class TestIdentityContract(unittest.TestCase):
    def test_principal_has_provider_neutral_identity_fields(self):
        principal = Principal(
            subject_id="user-1",
            tenant_id="acme",
            issuer="https://issuer.example",
            authentication_method="oidc",
            token_expiry=datetime.now(timezone.utc),
            roles=("reviewer",),
        )
        self.assertEqual(principal.tenant_id, "acme")
        self.assertEqual(principal.roles, ("reviewer",))


class TestInMemoryWorkflowEngine(unittest.TestCase):
    def test_delegates_to_existing_orchestrator_and_tracks_status(self):
        result = SimpleNamespace(succeeded=True)

        class FakeOrchestrator:
            def __init__(self):
                self.received = None

            def run_workflow(self, workflow):
                self.received = workflow
                return result

        orchestrator = FakeOrchestrator()
        engine = InMemoryWorkflowEngine(orchestrator)
        definition = WorkflowDefinition(
            workflow_id="wf-1",
            name="test",
            tasks=[],
            session_id=SessionId.generate(),
        )

        run = engine.start(definition)

        self.assertIsNotNone(orchestrator.received)
        self.assertEqual(run.state, WorkflowState.SUCCEEDED)
        self.assertIs(engine.get_status(run.run_id), run)

    def test_unknown_run_is_rejected(self):
        engine = InMemoryWorkflowEngine(object())
        with self.assertRaises(KeyError):
            engine.get_status("missing")


class _ConfidenceAgent(StructuralAgent):
    METADATA = AgentMetadata(
        agent_type="_ConfidenceAgent", display_name="Confidence", description="Test",
        version="1.0", capabilities=(), input_schema={
            "type": "object", "required": ["amount"],
            "properties": {"amount": {"type": "number", "minimum": 0}},
        },
    )

    def observe(self, task):
        return task.payload

    def reason(self, observations, task):
        return "{}"

    def parse_findings(self, observations, reasoning, task):
        return [self._make_finding(
            "signal", Severity.INFO, "A sufficiently detailed finding summary.",
            detail="Evidence is complete and directly supports this recommendation.",
            entity_id="supplier-1", evidence={"amount": observations["amount"]},
        )]


class TestPhase0TrustControls(unittest.TestCase):
    def tearDown(self):
        configure_security({})

    def test_schema_validator_reports_field_path(self):
        schema = {"type": "object", "required": ["amount"], "properties": {"amount": {"type": "number"}}}
        with self.assertRaisesRegex(SchemaValidationError, r"\$\.amount: expected number"):
            validate_json_schema({"amount": "bad"}, schema)

    def test_confidence_uses_evidence_and_completeness_signals(self):
        agent = _ConfidenceAgent()
        low = agent._make_finding("signal", Severity.INFO, "short")
        high = agent._make_finding(
            "signal", Severity.INFO, "A sufficiently detailed finding summary.",
            detail="Evidence is complete and directly supports this recommendation.",
            entity_id="supplier-1", entity_name="Supplier One",
            evidence={"amount": 10, "currency": "USD", "source": "invoice"},
        )
        self.assertLess(low.confidence_score, high.confidence_score)
        self.assertLessEqual(high.confidence_score, 1.0)

    def test_confidence_calibration_uses_labeled_sample(self):
        empirical = ConfidenceScorer.calibrate([
            ConfidenceSample(0.9, True),
            ConfidenceSample(0.6, False),
            ConfidenceSample(0.8, True),
        ])
        calibrated = ConfidenceScorer.apply_calibration(0.9, empirical)
        self.assertEqual(empirical, 0.667)
        self.assertLess(calibrated, 0.9)

    def test_calibration_store_round_trips_by_agent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = JsonCalibrationStore(Path(directory) / "calibration.json")
            store.put("SupplierStressAgent", 0.667)
            self.assertEqual(store.get("SupplierStressAgent"), 0.667)
            self.assertIsNone(store.get("UnknownAgent"))

    def test_runtime_finding_uses_agent_calibration_profile(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = JsonCalibrationStore(Path(directory) / "calibration.json")
            store.put("_ConfidenceAgent", 0.4)
            agent = _ConfidenceAgent()
            agent.confidence_calibration_store = store
            finding = agent._make_finding("signal", Severity.INFO, "short")
            self.assertEqual(finding.confidence_score, 0.425)

    def test_versioned_calibration_profile_round_trips_provenance(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = JsonCalibrationStore(Path(directory) / "calibration.json")
            profile = ConfidenceScorer.build_profile(
                "SupplierStressAgent",
                [ConfidenceSample(0.8, True), ConfidenceSample(0.7, False)],
                "golden-v1",
            )
            store.put_profile(profile)
            loaded = store.get_profile("SupplierStressAgent")
            self.assertIsInstance(loaded, CalibrationProfile)
            self.assertEqual(loaded.dataset_version, "golden-v1")
            self.assertEqual(loaded.sample_count, 2)

    def test_unapproved_profile_is_not_used_until_approved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = JsonCalibrationStore(Path(directory) / "calibration.json")
            profile = ConfidenceScorer.build_profile(
                "_ConfidenceAgent", [ConfidenceSample(0.8, True)], "golden-v1"
            )
            store.put_profile(profile)
            self.assertIsNone(store.get("_ConfidenceAgent"))
            approved = store.approve("_ConfidenceAgent", "reviewer-1")
            self.assertTrue(approved.approved)
            self.assertEqual(store.get("_ConfidenceAgent"), 1.0)

    def test_nested_schema_constraints_report_precise_path(self):
        schema = {
            "type": "object", "required": ["items"],
            "properties": {"items": {"type": "array", "minItems": 1,
                "items": {"type": "object", "required": ["id"],
                    "properties": {"id": {"type": "string"}}}}},
        }
        with self.assertRaisesRegex(SchemaValidationError, r"\$\.items: item count must be >= 1"):
            validate_json_schema({"items": []}, schema)

    def test_schema_alternatives_and_string_constraints(self):
        schema = {
            "oneOf": [
                {"type": "string", "pattern": r"^sup-"},
                {"type": "integer", "minimum": 1},
            ]
        }
        validate_json_schema("sup-001", schema)
        validate_json_schema(3, schema)
        with self.assertRaisesRegex(SchemaValidationError, "failed oneOf"):
            validate_json_schema("vendor-001", schema)

    def test_grounded_entity_reference_is_flagged(self):
        agent = _ConfidenceAgent()
        warnings = agent._validate_grounded_findings(
            [agent._make_finding("signal", Severity.INFO, "x", entity_id="missing")],
            {"supplier": "known"},
        )
        self.assertIn("ungrounded finding reference", warnings[0])

    def test_numeric_evidence_reference_is_flagged(self):
        agent = _ConfidenceAgent()
        warnings = agent._validate_grounded_findings(
            [agent._make_finding("signal", Severity.INFO, "x", evidence={"risk_score": 99})],
            {"risk_score": 2},
        )
        self.assertTrue(any("evidence.risk_score" in warning for warning in warnings))

    def test_numeric_text_claim_is_flagged(self):
        agent = _ConfidenceAgent()
        warnings = agent._validate_grounded_findings(
            [agent._make_finding("signal", Severity.INFO, "Risk increased to 99 percent")],
            {"risk_score": 2},
        )
        self.assertTrue(any("ungrounded numeric claim" in warning for warning in warnings))

    def test_currency_and_comma_formatted_claim_is_normalized_before_grounding_check(self):
        agent = _ConfidenceAgent()
        grounded = agent._validate_grounded_findings(
            [agent._make_finding("signal", Severity.INFO, "Savings of $1,234.56 identified")],
            {"savings_usd": 1234.56},
        )
        self.assertEqual(grounded, [])
        ungrounded = agent._validate_grounded_findings(
            [agent._make_finding("signal", Severity.INFO, "Savings of $9,999.00 identified")],
            {"savings_usd": 1234.56},
        )
        self.assertTrue(any("ungrounded numeric claim" in warning for warning in ungrounded))

    def test_secret_redaction_applies_even_without_pii_mode(self):
        register_secret("phase0-secret")
        self.assertEqual(redact_secrets("token=phase0-secret"), "token=[SECRET_REDACTED]")

    def test_llm_output_schema_is_enforced(self):
        class Strategy:
            provider_name = "test"

            def reason(self, request):
                return ReasoningResponse(
                    content='{"score": "not-a-number"}',
                    strategy_used=ReasoningStrategy.STRUCTURED,
                    parsed_output={"score": "not-a-number"},
                )

        engine = ReasoningEngine(Strategy())
        with self.assertRaisesRegex(ValueError, "expected number"):
            engine.reason(ReasoningRequest(
                system_prompt="system", user_prompt="user", observations={},
                output_schema={
                    "type": "object",
                    "required": ["score"],
                    "properties": {"score": {"type": "number"}},
                },
            ))


if __name__ == "__main__":
    unittest.main()