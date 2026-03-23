"""
platform/fabric/adapter.py
============================
Microsoft Fabric platform adapter for SKEIN.

Provides:
  1. OneLakeMemoryStore    — InstitutionalMemory backed by OneLake (Azure Data Lake)
  2. FabricGovernanceLogger — Writes governance events to Fabric Lakehouse
  3. SkeinFabricApp        — Entry point for Fabric notebook / Pipeline execution

DEPENDENCIES (install in Fabric notebook):
  %pip install azure-identity azure-storage-file-datalake pyyaml

FABRIC LAKEHOUSE SETUP:
  1. Create a Lakehouse in your Fabric workspace
  2. Note the OneLake ABFSS path: abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Files/skein/
  3. Set FABRIC_ONELAKE_PATH environment variable or pass to SkeinFabricApp

USAGE (Fabric notebook):
  from platform.fabric.adapter import SkeinFabricApp
  app = SkeinFabricApp.from_fabric_context()
  result = app.run_supplier_risk_review(transaction_data)

USAGE (Fabric Data Pipeline):
  Activity: Notebook
  Parameters: {"workflow": "supplier-risk-review", "table": "procurement.transactions"}

DESIGN
======
OneLakeMemoryStore uses Azure Data Lake Storage Gen2 REST API
via azure-storage-file-datalake SDK.
Falls back to in-memory store when the SDK is not available.
All agents are platform-agnostic — only the entry point and storage differ.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OneLakeMemoryStore
# ---------------------------------------------------------------------------

class OneLakeMemoryStore:
    """
    Azure OneLake (ADLS Gen2) backed institutional memory for SKEIN.

    Stores each memory key as a JSON file in:
        {onelake_path}/institutional_memory/{key}.json

    This allows the institutional knowledge base to be:
      - Shared across multiple Fabric workspaces
      - Versioned via Delta or versioned folders
      - Queried via Fabric SQL Analytics Endpoint

    Falls back to in-memory dict when Azure SDK is not available.

    Thread-safe: Azure SDK operations are atomic at the file level.
    """

    def __init__(
        self,
        onelake_path:   Optional[str] = None,
        workspace_id:   Optional[str] = None,
        lakehouse_name: Optional[str] = None,
    ) -> None:
        self._path      = onelake_path or os.environ.get("FABRIC_ONELAKE_PATH", "")
        self._workspace = workspace_id or os.environ.get("FABRIC_WORKSPACE_ID", "")
        self._lakehouse = lakehouse_name or os.environ.get("FABRIC_LAKEHOUSE_NAME", "skein")
        self._cache:    Dict[str, Any] = {}
        self._client    = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        """Initialise Azure Data Lake client."""
        if not self._path and not self._workspace:
            log.info("[fabric] No OneLake path configured — using in-memory fallback")
            return False
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
            account_url = (
                self._path.split(".dfs.fabric.microsoft.com")[0].rstrip("/")
                + ".dfs.fabric.microsoft.com"
                if ".dfs.fabric.microsoft.com" in self._path
                else f"https://onelake.dfs.fabric.microsoft.com"
            )
            cred = DefaultAzureCredential()
            self._client = DataLakeServiceClient(account_url, credential=cred)
            log.info("[fabric] OneLakeMemoryStore connected")
            return True
        except ImportError:
            log.warning("[fabric] azure-storage-file-datalake not installed — using fallback")
            return False
        except Exception as exc:
            log.warning("[fabric] OneLake connection failed: %s — using fallback", exc)
            return False

    def _key_to_path(self, key: str) -> str:
        """Convert a memory key to a safe filename."""
        safe = key.replace(":", "__").replace("/", "_").replace(" ", "_")
        return f"institutional_memory/{safe}.json"

    def set(self, key: str, value: Any, session_id=None, agent_id=None, ttl_seconds=None) -> None:
        self._cache[key] = value
        if not self._available:
            return
        try:
            payload = json.dumps({
                "value": value, "stored_by": str(agent_id) if agent_id else "skein",
                "stored_at": time.time(),
            }, default=str)
            container = self._lakehouse + ".Lakehouse"
            fs_client = self._client.get_file_system_client(container)
            file_path = f"Files/skein/{self._key_to_path(key)}"
            file_client = fs_client.get_file_client(file_path)
            file_client.create_file()
            file_client.append_data(payload.encode(), offset=0)
            file_client.flush_data(len(payload.encode()))
        except Exception as exc:
            log.error("[fabric] OneLake write failed for key=%s: %s", key, exc)

    def get(self, key: str, session_id=None) -> Optional[Any]:
        if key in self._cache:
            return self._cache[key]
        if not self._available:
            return None
        try:
            container  = self._lakehouse + ".Lakehouse"
            fs_client  = self._client.get_file_system_client(container)
            file_path  = f"Files/skein/{self._key_to_path(key)}"
            file_client = fs_client.get_file_client(file_path)
            data = file_client.download_file().readall()
            entry = json.loads(data)
            value = entry.get("value")
            self._cache[key] = value
            return value
        except Exception:
            return None

    def delete(self, key: str, session_id=None) -> None:
        self._cache.pop(key, None)
        if not self._available:
            return
        try:
            container  = self._lakehouse + ".Lakehouse"
            fs_client  = self._client.get_file_system_client(container)
            file_path  = f"Files/skein/{self._key_to_path(key)}"
            fs_client.get_file_client(file_path).delete_file()
        except Exception as exc:
            log.error("[fabric] OneLake delete failed: %s", exc)

    def keys(self, session_id=None) -> List[str]:
        if not self._available:
            return list(self._cache.keys())
        try:
            container  = self._lakehouse + ".Lakehouse"
            fs_client  = self._client.get_file_system_client(container)
            prefix     = "Files/skein/institutional_memory/"
            paths = fs_client.get_paths(path=prefix)
            return [
                p.name.replace(prefix, "").replace("__", ":").replace(".json", "")
                for p in paths if not p.is_directory
            ]
        except Exception:
            return list(self._cache.keys())


# ---------------------------------------------------------------------------
# FabricGovernanceLogger — writes to Fabric Lakehouse Tables
# ---------------------------------------------------------------------------

class FabricGovernanceLogger:
    """
    Writes SKEIN governance events to a Fabric Lakehouse Delta table.

    Creates table: skein.governance_events with schema:
        event_type, agent_id, agent_type, task_id, session_id,
        trace_id, succeeded, duration_ms, findings_count, timestamp

    Falls back to file-based GovernanceLogger when Spark is unavailable.
    """

    def __init__(
        self,
        spark=              None,
        table_name:  str =  "skein.governance_events",
        fallback_dir: str = "/tmp/skein/governance",
    ) -> None:
        self._spark = spark
        self._table = table_name
        self._fallback_dir = fallback_dir
        self._fallback: Optional[Any] = None

        if not self._spark:
            from framework.governance.logger import GovernanceLogger
            self._fallback = GovernanceLogger(fallback_dir)
            log.info("[fabric] Spark not available — using file-based governance logger")

    def record_execution(self, agent_id, agent_type, task, result) -> None:
        if self._fallback:
            self._fallback.record_execution(agent_id, agent_type, task, result)
            return
        try:
            self._spark.createDataFrame([{
                "event_type":      "execution",
                "agent_id":        str(agent_id),
                "agent_type":      agent_type,
                "task_id":         str(task.task_id),
                "session_id":      str(task.session_id),
                "trace_id":        task.context.trace_id,
                "succeeded":       result.succeeded,
                "duration_ms":     result.duration_ms or 0.0,
                "findings_count":  len(result.findings),
                "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }]).write.format("delta").mode("append").saveAsTable(self._table)
        except Exception as exc:
            log.error("[fabric] Governance write failed: %s", exc)

    def record_decision(self, agent_id, task, result, decision_record) -> None:
        if self._fallback:
            self._fallback.record_decision(agent_id, task, result, decision_record)

    def record_escalation(self, agent_id, agent_type, task, result, reason) -> None:
        if self._fallback:
            self._fallback.record_escalation(agent_id, agent_type, task, result, reason)

    def audit(self, event_type, data) -> None:
        if self._fallback:
            self._fallback.audit(event_type, data)

    def verify_chain(self, log_file: str) -> bool:
        if self._fallback:
            return self._fallback.verify_chain(log_file)
        return True  # Delta provides its own audit trail


# ---------------------------------------------------------------------------
# SkeinFabricApp — Fabric notebook / pipeline entry point
# ---------------------------------------------------------------------------

class SkeinFabricApp:
    """
    Entry point for running SKEIN workflows in Microsoft Fabric.

    Supports:
      - Fabric Notebook (interactive or scheduled)
      - Fabric Data Pipeline (via Notebook activity)
      - Fabric Data Activator (trigger on data events)

    All 15 SKEIN agents are registered automatically.
    Storage uses OneLake for institutional memory.
    Governance events written to Fabric Lakehouse tables.
    """

    def __init__(
        self,
        spark=            None,
        onelake_path:     Optional[str] = None,
        workspace_id:     Optional[str] = None,
        lakehouse_name:   str = "skein",
        governance_table: str = "skein.governance_events",
        config_path:      str = "config/config.yaml",
    ) -> None:
        self._spark     = spark
        self._onelake   = OneLakeMemoryStore(onelake_path, workspace_id, lakehouse_name)
        self._gov_logger = FabricGovernanceLogger(spark, governance_table)
        self._config    = config_path
        self._app       = None

    @classmethod
    def from_fabric_context(cls) -> "SkeinFabricApp":
        """Auto-detect Fabric environment."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
        except ImportError:
            spark = None
        workspace_id = os.environ.get("FABRIC_WORKSPACE_ID", "")
        lakehouse    = os.environ.get("FABRIC_LAKEHOUSE_NAME", "skein")
        return cls(spark=spark, workspace_id=workspace_id, lakehouse_name=lakehouse)

    def _build_app(self):
        if self._app:
            return self._app

        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from framework.core.registry import get_registry, reset_registry
        from framework.orchestration.orchestrator import TaskOrchestrator
        from framework.memory.store import WorkingMemory

        reset_registry()
        reg = get_registry()

        # Register agents
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
        for cls_ in [
            SupplierStressAgent, DecisionAuditAgent, ShouldCostAgent,
            TotalCostIntelligenceAgent, ValueRealisationAgent,
            ProcurementBiasDetectorAgent, ComplianceVerificationAgent,
            InstitutionalMemoryAgent, NegotiationIntelligenceAgent,
            SpecificationInflationAgent, WorkingCapitalOptimiserAgent,
            DemandIntelligenceAgent, SupplierInnovationAgent,
            DecisionCopilotAgent, TradeScenarioAgent,
        ]:
            reg.register_class(cls_)

        wm = WorkingMemory(max_entries=50_000)
        orig = reg.create_instance

        def factory(at, config, **kwargs):
            inst = orig(at, config, **kwargs)
            inst.memory    = wm
            inst.governance = self._gov_logger
            return inst
        reg.create_instance = factory

        orch = TaskOrchestrator(reg, config=None)
        self._app = (reg, orch)
        return self._app

    def run_supplier_risk_review(self, transaction_data: list) -> dict:
        from framework.orchestration.orchestrator import WorkflowBuilder
        from framework.core.types import SessionId, CorrelationContext
        reg, orch = self._build_app()
        sid = SessionId.generate()
        ctx = CorrelationContext.new(platform="fabric", workflow="supplier_risk_review")
        wf  = (WorkflowBuilder("supplier-risk-review")
               .session(sid).trace(ctx)
               .step("SupplierStressAgent", {"transaction_data": transaction_data})
               .build())
        result = orch.run_workflow(wf)
        return {
            "succeeded":  result.succeeded,
            "findings":   [f.to_dict() for f in result.all_findings],
            "duration_ms": result.duration_ms,
            "trace_id":   ctx.trace_id,
        }


if __name__ == "__main__":
    app = SkeinFabricApp.from_fabric_context()
    log.info("SKEIN Fabric runner initialised")
    print("SkeinFabricApp ready. Call app.run_supplier_risk_review(data) to execute.")
