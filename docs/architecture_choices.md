# Architecture Choices

## Why These Tools?

### DLT (Data Load Tool)

**Problem**: Fetching data from APIs requires handling pagination, rate limits, incremental state, and schema evolution.

**Why DLT**:
- Built-in incremental loading with cursor state management
- Automatic schema inference and evolution
- Handles API pagination and retries out of the box
- State stored alongside data (Azure Blob) - no external DB needed
- Swappable destinations (local → Azure → BigQuery) without code changes

**Alternative considered**: Custom scripts with manual state tracking. Rejected because maintaining cursor state across runs is error-prone.

---

### Weights & Biases Model Registry

**Problem**: Need versioned model storage with metadata, and a way to promote models without code changes.

**Why W&B**:
- Model versioning with aliases (`staging`, `production`) - no file renaming
- Metrics attached to each version (MAE, R², training params)
- API to fetch models by alias - container pulls latest `production` at startup
- Free tier sufficient for this use case
- Experiment tracking included (training runs, hyperparameters)

**Alternative considered**: MLflow. Viable but requires hosting the tracking server. W&B is managed.

---

### GitHub Actions

**Problem**: Need automated pipeline that runs ingest → features → train → promote in sequence.

**Why GitHub Actions**:
- `workflow_run` triggers chain jobs after upstream completion
- Secrets management built-in
- No infrastructure to maintain
- Cron scheduling for hourly ingestion
- Logs and artifacts retained for debugging

**Alternative considered**: Azure Data Factory, Airflow. Both require additional infrastructure. GitHub Actions is already where the code lives.

---

### FastAPI Dashboard

**Problem**: Need quick visualization of predictions and data quality without a separate frontend deployment.

**Why embedded Plotly**:
- Single container serves API + dashboard
- No JavaScript build step or separate hosting
- Plotly generates interactive HTML server-side
- Good enough for monitoring - not a customer-facing product

**Alternative considered**: Streamlit, Grafana. Streamlit requires separate deployment. Grafana would be an improvement given more time.

---

### Azure Container Apps

**Problem**: Need serverless container hosting with auto-scaling and managed infrastructure.

**Why Container Apps**:
- Scale to zero when idle (cost control)
- Built-in HTTPS with automatic certs
- Terraform-native resource
- Pulls from ACR with managed identity (no credentials in container)
- Log Analytics integration for debugging

**Alternative considered**: Azure Kubernetes Service (AKS). Overkill for a single-container workload. Container Apps is simpler and cheaper.

---

## Shortcomings & Improvements

### Current Limitations

| Issue | Impact | Severity |
|-------|--------|----------|
| **No feature store** | Features recomputed from raw data each training run | Medium |
| **Single model type** | Only GradientBoosting - no model comparison | Low |
| **No data validation** | Bad data from Fingrid silently propagates | High |
| **Batch predictions only** | `/predict` fetches live data - adds latency | Medium |
| **No rollback mechanism** | If promoted model is bad, manual intervention needed | High |
| **Hourly granularity** | DLT runs hourly - could miss intra-hour spikes | Low |

### Recommended Improvements

**1. Add data validation (Priority: High)**
```
Ingest → [Great Expectations / Pandera] → Features → Train
```
Validate row counts, null percentages, value ranges before training. Fail pipeline early if data quality degrades.

**2. Implement model rollback (Priority: High)**
- Store last 3 production model versions
- Add `/model/rollback` endpoint that swaps alias back
- Or: automatic rollback if prediction latency/errors spike post-promotion

**3. Add feature store (Priority: Medium)**
- Use Feast or just versioned parquet files in Azure Blob
- Decouple feature computation from training
- Enable feature reuse across models

**4. Shadow mode for challengers (Priority: Medium)**
- Run staging model in parallel with production
- Log both predictions without serving staging
- Compare real-world performance before promotion

**5. Real-time feature pipeline (Priority: Low)**
- Replace batch DLT with streaming (Kafka, Event Hubs)
- Pre-compute features and cache
- Reduce `/predict` latency from ~2s to ~100ms

**6. Multi-model experiments (Priority: Low)**
- Add LightGBM, XGBoost to training
- Promote best performer automatically
- Would require restructuring train.py

---

## Architecture Evolution Path

```
Current State (MVP)
    │
    ├── Add data validation (Great Expectations)
    │
    ├── Add rollback mechanism
    │
    └── Add feature store (Feast)
            │
            ├── Shadow mode evaluation
            │
            └── Real-time streaming (if latency matters)
```

The current architecture is appropriate for a proof-of-concept. The primary gaps are around reliability (validation, rollback) rather than features.
