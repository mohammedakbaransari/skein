# SKEIN on Databricks

Run SKEIN procurement intelligence workflows on Databricks with Delta Lake memory and MLflow governance tracking.

> **Validation status (2026-09-16):** adapter behavior is covered by fake-Spark unit/integration tests only. Unity Catalog permissions, concurrent writers, IDENTITY behavior, MLflow wiring, and restart recovery have not been validated against a live workspace. Complete roadmap R23 before production traffic.

## Prerequisites

- Databricks Runtime 14.0+ (Python 3.11)
- MLflow experiment tracking (included in Databricks)
- Delta Lake (included in Databricks)
- Cluster with internet access (for LLM provider calls)

## Installation

In your Databricks notebook or cluster init script:

```python
%pip install pyyaml requests anthropic
# Or: pip install pyyaml requests openai
```

For the full framework, build an archive from the current checkout and upload it to DBFS:
```bash
git archive --format=zip --output=skein.zip HEAD
dbfs cp skein.zip dbfs:/FileStore/skein/
```

Then in your notebook:
```python
import zipfile, os
with zipfile.ZipFile('/dbfs/FileStore/skein/skein.zip') as z:
    z.extractall('/tmp/skein')
import sys
sys.path.insert(0, '/tmp/skein/skein')
```

## Quick Start

```python
from platform.databricks.adapter import SkeinDatabricksApp

# Auto-detects SparkSession and MLflow from Databricks environment
app = SkeinDatabricksApp.from_notebook_context()

# Run supplier risk review
transaction_data = (
    spark.table("procurement.supplier_transactions")
    .toPandas()
    .to_dict("records")
)
result = app.run_supplier_risk_review(transaction_data)

print(f"Succeeded: {result['succeeded']}")
print(f"Findings:  {len(result['findings'])}")
for finding in result["findings"]:
    print(f"  [{finding['severity'].upper()}] {finding['summary']}")
```

## Delta Table Memory

SKEIN stores institutional knowledge in a Delta table for persistence across runs:

```sql
-- Create the memory table (run once)
CREATE TABLE IF NOT EXISTS skein.institutional_memory (
    key        STRING NOT NULL,
    value_json STRING NOT NULL,
    stored_by  STRING,
    stored_at  TIMESTAMP,
    updated_at TIMESTAMP
) USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
```

```python
from platform.databricks.adapter import DeltaTableMemoryStore

memory = DeltaTableMemoryStore(
    table_name="skein.institutional_memory",
    spark=spark,
)
memory.ensure_table_exists()
```

For multi-tenant deployments, use a validated `TenantContext` per tenant and provision dedicated catalogs only when the physical-isolation profile is selected. Logical isolation remains the product default; catalog/container provisioning is outside this repository.

## MLflow Governance Tracking

Every workflow run is logged as an MLflow experiment:

```python
from platform.databricks.adapter import MLflowGovernanceTracker

tracker = MLflowGovernanceTracker(
    experiment_name="SKEIN Procurement Intelligence"
)
app = SkeinDatabricksApp.from_notebook_context(mlflow_tracker=tracker)
```

View results in the MLflow UI under **Experiments → SKEIN Procurement Intelligence**.

Each run logs:
- Parameters: `workflow_id`, `workflow_name`, `task_count`
- Metrics: `duration_ms`, `succeeded`, `failed_tasks`, `total_findings`, `critical_findings`
- Tags: `trace_id`, `session_id`, `workflow`

## Databricks Jobs

Schedule SKEIN as a Databricks Job:

1. Create a new Job
2. Task type: Notebook
3. Source: your notebook with `SkeinDatabricksApp` code
4. Cluster: choose a standard cluster with the SKEIN packages installed
5. Parameters: pass `table` as a widget parameter

```python
# In the notebook
dbutils.widgets.text("table", "procurement.supplier_transactions")
table_name = dbutils.widgets.get("table")

data = spark.table(table_name).toPandas().to_dict("records")
app  = SkeinDatabricksApp.from_notebook_context()
result = app.run_supplier_risk_review(data)
```

## Governance Log Storage

Governance JSONL logs are written to DBFS by default:

```python
app = SkeinDatabricksApp(
    governance_dir="/dbfs/tmp/skein/governance"
)
```

To verify chain integrity after a run:
```python
from framework.governance.logger import GovernanceLogger
gov = GovernanceLogger("/dbfs/tmp/skein/governance")
ok  = gov.verify_chain("/dbfs/tmp/skein/governance/executions.jsonl")
print(f"Chain integrity: {'OK' if ok else 'COMPROMISED'}")
```

## Secrets and audit integration

Choose a `SecretsProvider` through deployment configuration; do not hard-code tokens in notebooks. Environment, file, and Vault KV v2 adapters exist. Databricks Secrets integration is not implemented in this repository. Governance can additionally emit the vendor-neutral `AuditEvent` contract through an `AuditSink`; no live SIEM endpoint has been certified.
