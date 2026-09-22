"""
platform/databricks/adapter.py
================================
Databricks platform adapter for SKEIN.

Provides:
  1. DeltaTableMemoryStore  — InstitutionalMemory backed by Delta Lake
  2. MLflowGovernanceTracker — Log governance events as MLflow runs
  3. DatabricksJobRunner    — Launch SKEIN workflows as Databricks Jobs
  4. SkeinDatabricksApp     — Entry point for Databricks notebook execution

DEPENDENCIES (install in Databricks cluster):
  %pip install databricks-sdk mlflow pyyaml

USAGE (Databricks notebook):
  from platform.databricks.adapter import SkeinDatabricksApp

  app = SkeinDatabricksApp.from_notebook_context()
  result = app.run_workflow("supplier-risk-review", {
      "transaction_data": spark.table("procurement.supplier_transactions").toPandas().to_dict("records")
  })

DESIGN
======
DeltaTableMemoryStore wraps InstitutionalMemory interface.
All agents receive this store via dependency injection — no code changes required.
The SKEIN framework is platform-agnostic; only the entry point changes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, ClassVar, Dict, List, Optional

from framework.governance.hashchain import GENESIS, compute_chained_entry, verify_chained_entries
from framework.multitenancy.context import TenantContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeltaTableMemoryStore — wraps InstitutionalMemory interface
# ---------------------------------------------------------------------------

class DeltaTableMemoryStore:
    """
    Delta Lake backed institutional memory for SKEIN.

    Reads and writes to a Delta table:
        CREATE TABLE IF NOT EXISTS skein.institutional_memory (
            key STRING NOT NULL,
            value_json STRING NOT NULL,
            stored_by STRING,
            stored_at TIMESTAMP,
            updated_at TIMESTAMP
        ) USING DELTA TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

    Implements the same interface as InstitutionalMemory so agents
    receive it via dependency injection without any code changes.

    Thread-safe: Delta Lake provides ACID guarantees on concurrent writes.
    """

    def __init__(
        self,
        table_name:  str = "skein.institutional_memory",
        catalog:     str = "main",
        spark=       None,  # SparkSession — injected when running in Databricks
        tenant:      Optional[TenantContext] = None,
    ) -> None:
        # Physical multi-tenant isolation: when a TenantContext is given,
        # route to that tenant's dedicated catalog instead of the shared
        # table_name/catalog defaults (which remain for single-tenant use).
        self._tenant  = tenant
        self._table   = tenant.qualified_table(tenant.memory_table) if tenant else table_name
        self._catalog = tenant.catalog if tenant else catalog
        self._spark   = spark
        self._cache:  Dict[str, Any] = {}  # local read cache
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Return True if Delta Lake is accessible."""
        try:
            if self._spark is None:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.getActiveSession()
            return self._spark is not None
        except ImportError:
            log.warning("[databricks] PySpark not available — falling back to in-memory store")
            return False

    def set(self, key: str, value: Any, session_id=None, agent_id=None, ttl_seconds=None) -> None:
        if not self._available:
            self._cache[key] = value
            return

        value_json = json.dumps(value, default=str)
        stored_by  = str(agent_id) if agent_id else "skein"

        try:
            self._spark.sql(
                f"""
                MERGE INTO {self._table} AS target
                USING (SELECT :key AS key, :value_json AS value_json,
                               :stored_by AS stored_by,
                               current_timestamp() AS updated_at) AS source
                ON target.key = source.key
                WHEN MATCHED THEN UPDATE SET
                    value_json = source.value_json,
                    stored_by  = source.stored_by,
                    updated_at = source.updated_at
                WHEN NOT MATCHED THEN INSERT (key, value_json, stored_by, stored_at, updated_at)
                VALUES (source.key, source.value_json, source.stored_by,
                        current_timestamp(), current_timestamp())
                """,
                args={"key": key, "value_json": value_json, "stored_by": stored_by},
            )
            self._cache[key] = value
        except Exception as exc:
            log.error("[databricks] DeltaMemory.set failed for key=%s: %s", key, exc)
            self._cache[key] = value  # fallback to cache

    def get(self, key: str, session_id=None) -> Optional[Any]:
        if not self._available:
            return self._cache.get(key)

        # Check local cache first
        if key in self._cache:
            return self._cache[key]

        try:
            rows = self._spark.sql(
                f"SELECT value_json FROM {self._table} WHERE key = :key LIMIT 1",
                args={"key": key},
            ).collect()
            if rows:
                value = json.loads(rows[0]["value_json"])
                self._cache[key] = value
                return value
        except Exception as exc:
            log.error("[databricks] DeltaMemory.get failed for key=%s: %s", key, exc)
        return None

    def delete(self, key: str, session_id=None) -> None:
        self._cache.pop(key, None)
        if self._available:
            try:
                self._spark.sql(
                    f"DELETE FROM {self._table} WHERE key = :key", args={"key": key}
                )
            except Exception as exc:
                log.error("[databricks] DeltaMemory.delete failed: %s", exc)

    def keys(self, session_id=None) -> List[str]:
        if not self._available:
            return list(self._cache.keys())
        try:
            rows = self._spark.sql(
                f"SELECT key FROM {self._table}"
            ).collect()
            return [r["key"] for r in rows]
        except Exception:
            return list(self._cache.keys())

    def ensure_table_exists(self) -> None:
        """Create the Delta table if it does not exist."""
        if not self._available:
            return
        self._spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                key        STRING NOT NULL,
                value_json STRING NOT NULL,
                stored_by  STRING,
                stored_at  TIMESTAMP,
                updated_at TIMESTAMP
            ) USING DELTA
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
        """)
        log.info("[databricks] Ensured Delta table: %s", self._table)


# ---------------------------------------------------------------------------
# DeltaGovernanceStore — tenant-partitioned, hash-chained governance log
# ---------------------------------------------------------------------------

class DeltaGovernanceStore:
    """
    Tenant-partitioned, hash-chained governance log backed by a Delta table
    instead of local JSONL files.

    Required once replicas are horizontally scaled: framework.governance.
    logger.GovernanceLogger writes to a local file per pod, so N replicas
    produce N disconnected hash chains with no fleet-wide or cross-restart
    audit trail. This store persists to one Delta table per tenant
    (physical isolation boundary, per TenantContext) instead, reusing the
    exact same chain-construction algorithm (framework.governance.
    hashchain) that framework.governance.logger.HashChainedWriter uses, so
    the concurrency/restart-safety fixes made there apply here too.

    All four record types (execution/decision/escalation/audit) are stored
    as rows in ONE per-tenant table (distinguished by an event_type column)
    rather than four separate per-tenant tables, so there is a single
    chronological, hash-chained sequence per tenant to verify.

    Schema (create once per tenant via ensure_table_exists()):
        CREATE TABLE IF NOT EXISTS <catalog>.<schema>.governance_events (
            event_type  STRING NOT NULL,
            record_json STRING NOT NULL,
            prev_hash   STRING NOT NULL,
            hash        STRING NOT NULL,
            inserted_at TIMESTAMP
        ) USING DELTA;

    CAVEAT: not exercised against a live Databricks/Unity Catalog cluster —
    validated only against a fake Spark double in
    tests/unit/test_databricks_adapter.py. Resuming the chain on restart
    via `ORDER BY inserted_at DESC LIMIT 1` assumes no two rows share a
    timestamp; genuinely concurrent multi-pod writers against the SAME
    tenant need validation against a live cluster (e.g. an IDENTITY/
    monotonic sequence column) before production use.
    """

    _table_locks:     ClassVar[Dict[str, threading.Lock]] = {}
    _table_prev_hash: ClassVar[Dict[str, str]] = {}
    _registry_lock:   ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, tenant: TenantContext, spark=None) -> None:
        self._tenant = tenant
        self._table  = tenant.qualified_table(tenant.governance_table)
        self._spark  = spark
        self._available = self._check_available()
        self._fallback_events: List[Dict[str, Any]] = []  # used when Spark unavailable
        with DeltaGovernanceStore._registry_lock:
            if self._table not in DeltaGovernanceStore._table_locks:
                DeltaGovernanceStore._table_locks[self._table] = threading.Lock()
            if self._table not in DeltaGovernanceStore._table_prev_hash:
                DeltaGovernanceStore._table_prev_hash[self._table] = self._read_last_hash()
        self._lock = DeltaGovernanceStore._table_locks[self._table]

    def _check_available(self) -> bool:
        try:
            if self._spark is None:
                from pyspark.sql import SparkSession
                self._spark = SparkSession.getActiveSession()
            return self._spark is not None
        except ImportError:
            log.warning("[databricks] PySpark not available — governance events buffered in-memory only")
            return False

    def _read_last_hash(self) -> str:
        if not self._available:
            return GENESIS
        try:
            rows = self._spark.sql(
                f"SELECT hash FROM {self._table} ORDER BY inserted_at DESC LIMIT 1"
            ).collect()
            return rows[0]["hash"] if rows else GENESIS
        except Exception:
            return GENESIS

    def _append(self, event_type: str, record: Dict[str, Any]) -> None:
        # event_type must be folded into the record BEFORE hashing, not
        # added afterwards — otherwise the fallback branch's stored entry
        # (which includes event_type) would hash-mismatch against
        # verify_chained_entries(), which recomputes the hash over whatever
        # dict it is actually given.
        full_record = {"event_type": event_type, **record}
        with self._lock:
            prev_hash = DeltaGovernanceStore._table_prev_hash[self._table]
            entry = compute_chained_entry(full_record, prev_hash)
            if not self._available:
                self._fallback_events.append(entry)
                DeltaGovernanceStore._table_prev_hash[self._table] = entry["hash"]
                return
            try:
                self._spark.sql(
                    f"INSERT INTO {self._table} "
                    f"(event_type, record_json, prev_hash, hash, inserted_at) "
                    f"VALUES (:event_type, :record_json, :prev_hash, :hash, current_timestamp())",
                    args={
                        "event_type": event_type,
                        "record_json": json.dumps(full_record, default=str),
                        "prev_hash": prev_hash,
                        "hash": entry["hash"],
                    },
                )
                DeltaGovernanceStore._table_prev_hash[self._table] = entry["hash"]
            except Exception as exc:
                log.error("[databricks] Governance append failed for tenant=%s: %s",
                          self._tenant.tenant_id, exc)
                self._fallback_events.append(entry)

    def record_execution(self, agent_id, agent_type, task, result) -> None:
        self._append("execution", {
            "agent_id": str(agent_id), "agent_type": agent_type,
            "task_id": str(task.task_id), "session_id": str(task.session_id),
            "succeeded": result.succeeded, "duration_ms": result.duration_ms,
            "findings_count": len(result.findings),
        })

    def record_decision(self, agent_id, task, result, decision_record) -> None:
        self._append("decision", {
            "agent_id": str(agent_id), "task_id": str(task.task_id),
            "session_id": str(task.session_id), **decision_record,
        })

    def record_escalation(self, agent_id, agent_type, task, result, reason) -> None:
        self._append("escalation", {
            "agent_id": str(agent_id), "agent_type": agent_type,
            "task_id": str(task.task_id), "session_id": str(task.session_id),
            "reason": reason,
        })

    def audit(self, event_type: str, data: Dict[str, Any]) -> None:
        self._append(f"audit:{event_type}", dict(data))

    def verify_chain(self, log_file: str = "") -> bool:
        """`log_file` is accepted for interface parity with GovernanceLogger
        but ignored — verification always targets this store's own table."""
        if not self._available:
            return verify_chained_entries(self._fallback_events)
        try:
            rows = self._spark.sql(
                f"SELECT record_json, prev_hash, hash FROM {self._table} ORDER BY inserted_at ASC"
            ).collect()
        except Exception as exc:
            log.error("[databricks] Governance verify_chain read failed: %s", exc)
            return False
        entries = []
        for r in rows:
            record = json.loads(r["record_json"])
            entries.append({**record, "prev_hash": r["prev_hash"], "hash": r["hash"]})
        return verify_chained_entries(entries)

    def ensure_table_exists(self) -> None:
        if not self._available:
            return
        self._spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                event_type  STRING NOT NULL,
                record_json STRING NOT NULL,
                prev_hash   STRING NOT NULL,
                hash        STRING NOT NULL,
                inserted_at TIMESTAMP
            ) USING DELTA
        """)
        log.info("[databricks] Ensured governance Delta table: %s", self._table)


# ---------------------------------------------------------------------------
# MLflowGovernanceTracker
# ---------------------------------------------------------------------------

class MLflowGovernanceTracker:
    """
    Logs SKEIN governance events as MLflow experiments.

    Each workflow run becomes an MLflow run with:
      - Parameters: workflow_id, agent_types, task_count
      - Metrics: duration_ms, succeeded_tasks, failed_tasks, finding_counts
      - Tags: trace_id, session_id, environment

    Requires: pip install mlflow
    """

    def __init__(
        self,
        experiment_name: str = "SKEIN Procurement Intelligence",
        tracking_uri:    Optional[str] = None,
    ) -> None:
        self._experiment = experiment_name
        self._tracking_uri = tracking_uri
        self._available = self._setup_mlflow()

    def _setup_mlflow(self) -> bool:
        try:
            import mlflow
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment)
            return True
        except ImportError:
            log.warning("[databricks] MLflow not installed — governance tracking disabled")
            return False

    def log_workflow_result(self, workflow_result) -> Optional[str]:
        """Log a WorkflowResult to MLflow. Returns run_id or None."""
        if not self._available:
            return None
        try:
            import mlflow
            with mlflow.start_run(
                run_name=f"{workflow_result.workflow_name}_{int(time.time())}",
                tags={
                    "trace_id":   workflow_result.context.trace_id,
                    "session_id": str(workflow_result.session_id),
                    "workflow":   workflow_result.workflow_name,
                },
            ) as run:
                mlflow.log_params({
                    "workflow_id":   workflow_result.workflow_id,
                    "workflow_name": workflow_result.workflow_name,
                    "task_count":    len(workflow_result.task_results),
                })
                mlflow.log_metrics({
                    "duration_ms":      workflow_result.duration_ms or 0,
                    "succeeded":        1 if workflow_result.succeeded else 0,
                    "failed_tasks":     len(workflow_result.failed_tasks),
                    "cancelled_tasks":  len(workflow_result.cancelled_tasks),
                    "total_findings":   len(workflow_result.all_findings),
                    "critical_findings": sum(
                        1 for f in workflow_result.all_findings
                        if hasattr(f.severity, "value") and f.severity.value == "critical"
                    ),
                })
                return run.info.run_id
        except Exception as exc:
            log.error("[databricks] MLflow logging failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# SkeinDatabricksApp — notebook entry point
# ---------------------------------------------------------------------------

class SkeinDatabricksApp:
    """
    Entry point for running SKEIN workflows in a Databricks notebook or Job.

    Usage in a Databricks notebook:
        from platform.databricks.adapter import SkeinDatabricksApp
        app = SkeinDatabricksApp.from_notebook_context()
        result = app.run_supplier_risk_review(dbutils.widgets.get("table"))

    Usage as a Databricks Job:
        python -m platform.databricks.adapter --workflow supplier-risk-review
    """

    def __init__(
        self,
        spark=         None,
        mlflow_tracker: Optional[MLflowGovernanceTracker] = None,
        governance_dir: str = "/dbfs/tmp/skein/governance",
        config_path:    str = "config/config.yaml",
    ) -> None:
        self._spark          = spark
        self._mlflow         = mlflow_tracker
        self._governance_dir = governance_dir
        self._config_path    = config_path
        self._app            = None

    @classmethod
    def from_notebook_context(cls) -> "SkeinDatabricksApp":
        """Auto-detect Spark session from Databricks notebook environment."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None
        tracker = MLflowGovernanceTracker()
        return cls(spark=spark, mlflow_tracker=tracker)

    def _build_app(self):
        """Lazy-initialise the SKEIN framework components."""
        if self._app is not None:
            return self._app

        import sys
        import os
        # Ensure skein-rebuild is on the path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from framework.core.registry import get_registry, reset_registry
        from framework.orchestration.orchestrator import TaskOrchestrator
        from framework.governance.logger import GovernanceLogger
        from framework.memory.store import WorkingMemory

        reset_registry()
        reg = get_registry()

        # Register all 15 agents
        from agents.supply_risk.supplier_stress import SupplierStressAgent
        from agents.decision_audit.agent import DecisionAuditAgent
        from agents.cost_intelligence.should_cost import ShouldCostAgent
        from agents.cost_intelligence.total_cost import TotalCostIntelligenceAgent
        from agents.contract_analysis.value_realisation import ValueRealisationAgent
        from agents.bias_detection.bias_detector import ProcurementBiasDetectorAgent
        from agents.compliance.compliance_verification import ComplianceVerificationAgent
        from agents.market_intelligence.agents import (
            InstitutionalMemoryAgent, NegotiationIntelligenceAgent,
            SpecificationInflationAgent, WorkingCapitalOptimiserAgent,
            DemandIntelligenceAgent, SupplierInnovationAgent,
            DecisionCopilotAgent, TradeScenarioAgent,
        )
        for cls in [
            SupplierStressAgent, DecisionAuditAgent, ShouldCostAgent,
            TotalCostIntelligenceAgent, ValueRealisationAgent,
            ProcurementBiasDetectorAgent, ComplianceVerificationAgent,
            InstitutionalMemoryAgent, NegotiationIntelligenceAgent,
            SpecificationInflationAgent, WorkingCapitalOptimiserAgent,
            DemandIntelligenceAgent, SupplierInnovationAgent,
            DecisionCopilotAgent, TradeScenarioAgent,
        ]:
            reg.register_class(cls)

        gov  = GovernanceLogger(self._governance_dir)
        mem  = WorkingMemory(max_entries=50_000)
        inst = DeltaTableMemoryStore(spark=self._spark)

        orig = reg.create_instance
        def factory(at, config, **kwargs):
            inst_agent = orig(at, config, **kwargs)
            inst_agent.memory    = mem
            inst_agent.governance = gov
            return inst_agent
        reg.create_instance = factory

        orch = TaskOrchestrator(reg, config=None)
        self._app = (reg, orch, gov)
        return self._app

    def run_supplier_risk_review(self, transaction_data: list) -> dict:
        """Run supplier stress + decision audit + bias detection workflow."""
        from framework.orchestration.orchestrator import WorkflowBuilder
        from framework.core.types import SessionId, CorrelationContext

        reg, orch, gov = self._build_app()
        sid = SessionId.generate()
        ctx = CorrelationContext.new(workflow="supplier_risk_review")

        wf = (WorkflowBuilder("supplier-risk-review")
              .session(sid).trace(ctx)
              .step("SupplierStressAgent", {"transaction_data": transaction_data})
              .build())
        result = orch.run_workflow(wf)

        if self._mlflow:
            self._mlflow.log_workflow_result(result)

        return result.to_dict() if hasattr(result, "to_dict") else {
            "succeeded": result.succeeded,
            "findings":  [f.to_dict() for f in result.all_findings],
            "duration_ms": result.duration_ms,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SKEIN Databricks runner")
    parser.add_argument("--workflow", default="supplier-risk-review")
    args = parser.parse_args()

    app = SkeinDatabricksApp.from_notebook_context()
    print(f"Running workflow: {args.workflow}")
    log.info("SKEIN Databricks runner started for workflow: %s", args.workflow)
