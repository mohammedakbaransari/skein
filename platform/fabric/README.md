# SKEIN on Microsoft Fabric

Run SKEIN procurement intelligence workflows on Microsoft Fabric with OneLake memory storage and Fabric Lakehouse governance tracking.

## Prerequisites

- Microsoft Fabric workspace with a Lakehouse
- Fabric Notebook or Data Pipeline
- Python 3.11+ (default in Fabric notebooks)

## Installation

In your Fabric notebook:

```python
%pip install pyyaml requests anthropic
%pip install azure-identity azure-storage-file-datalake
```

Upload the SKEIN package to your Lakehouse Files:
1. In Fabric, open your Lakehouse
2. Upload `skein-framework-v2-FINAL.zip` to `Files/skein/`
3. In your notebook:

```python
import zipfile, sys
with zipfile.ZipFile('/lakehouse/default/Files/skein/skein-framework-v2-FINAL.zip') as z:
    z.extractall('/tmp/skein')
sys.path.insert(0, '/tmp/skein/skein')
```

## Quick Start

```python
from platform.fabric.adapter import SkeinFabricApp

# Auto-detects Fabric environment
app = SkeinFabricApp.from_fabric_context()

# Load data from Fabric Lakehouse table
transaction_data = spark.table("procurement.supplier_transactions") \
                        .toPandas().to_dict("records")

result = app.run_supplier_risk_review(transaction_data)
print(f"Trace ID: {result['trace_id']}")
print(f"Findings: {len(result['findings'])}")
```

## OneLake Memory Setup

SKEIN uses OneLake (Azure Data Lake Gen2) for persistent institutional memory:

```python
import os
os.environ["FABRIC_WORKSPACE_ID"]   = "your-workspace-id"
os.environ["FABRIC_LAKEHOUSE_NAME"] = "skein"

from platform.fabric.adapter import OneLakeMemoryStore
memory = OneLakeMemoryStore(workspace_id=os.environ["FABRIC_WORKSPACE_ID"])
```

Memory keys are stored as JSON files at:
`Files/skein/institutional_memory/{key}.json`

## Governance Logging to Lakehouse Tables

```python
from platform.fabric.adapter import FabricGovernanceLogger

# Write governance events directly to a Delta table
gov = FabricGovernanceLogger(
    spark=spark,
    table_name="skein.governance_events"
)

# Create the table (run once)
spark.sql("""
CREATE TABLE IF NOT EXISTS skein.governance_events (
    event_type     STRING,
    agent_id       STRING,
    agent_type     STRING,
    task_id        STRING,
    session_id     STRING,
    trace_id       STRING,
    succeeded      BOOLEAN,
    duration_ms    DOUBLE,
    findings_count INT,
    timestamp      TIMESTAMP
) USING DELTA
""")
```

Query governance events from the Fabric SQL Analytics Endpoint:
```sql
SELECT agent_type,
       COUNT(*)                          AS total_runs,
       AVG(duration_ms)                  AS avg_duration_ms,
       SUM(CASE WHEN NOT succeeded THEN 1 ELSE 0 END) AS failures
FROM   skein.governance_events
WHERE  event_type = 'execution'
GROUP  BY agent_type
ORDER  BY total_runs DESC;
```

## Fabric Data Pipeline

Use SKEIN in a Fabric Pipeline for scheduled execution:

1. Create a new Data Pipeline
2. Add a **Notebook** activity
3. Set the notebook to your SKEIN notebook
4. Add parameters: `table_name`, `workflow_type`
5. Schedule via the Pipeline trigger

```python
# In your Fabric notebook (parameterised)
table_name    = mssparkutils.runtime.context.get("table_name",
                "procurement.supplier_transactions")
workflow_type = mssparkutils.runtime.context.get("workflow_type",
                "supplier_risk_review")

data = spark.table(table_name).toPandas().to_dict("records")
app  = SkeinFabricApp.from_fabric_context()

if workflow_type == "supplier_risk_review":
    result = app.run_supplier_risk_review(data)
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FABRIC_WORKSPACE_ID` | Your Fabric workspace GUID | `abc123-...` |
| `FABRIC_LAKEHOUSE_NAME` | Lakehouse name | `skein` |
| `FABRIC_ONELAKE_PATH` | Full ABFSS path | `abfss://ws@onelake.dfs.fabric.microsoft.com/skein.Lakehouse/Files/skein/` |
| `LLM_PROVIDER` | LLM provider | `anthropic` |
| `LLM_API_KEY` | API key | `sk-ant-...` |
