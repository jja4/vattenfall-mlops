# Copilot Instructions: Vattenfall MLOps

## Project Overview
Real-time electricity imbalance price prediction service for Finland, integrating Fingrid API data with machine learning models served via FastAPI. The system uses a modern MLOps architecture with:
- **DLT (dlthub)** for batch data ingestion to Azure Blob Storage
- **W&B Model Registry** for model versioning with `production`/`staging` aliases
- **Azure Container Apps** for serverless deployment with model hot-reload

## Architecture & Data Flow

```
Fingrid API → DLT Pipeline → Azure Blob Storage → Feature Engineering → Model Training → W&B Registry
                                      ↓                                                       ↓
                              pipeline/features.py ←──────────────── FastAPI ← Model Loader ←┘
```

### Key Layers
- **Batch Ingestion** (`pipeline/ingest.py`): DLT pipeline fetches Fingrid data (Wind #181, mFRR #342, Price #319) → Azure Blob
- **Feature Engineering** (`pipeline/features.py`): Loads from Blob, applies transformations via `ingestion/processor.py`
- **Training** (`pipeline/train.py`): Trains GradientBoostingRegressor, logs to W&B, registers model with alias
- **Serving** (`app/`): FastAPI loads model from W&B Registry on startup, serves `/predict` endpoint

## Critical Conventions

### Package Management: uv-first
This project uses `uv` (not pip/poetry). All dependency operations:
```bash
uv sync              # Install/update deps from uv.lock
uv add <package>     # Add new dependency
uv run <command>     # Execute in uv-managed venv
```
Dockerfile leverages `uv sync --frozen --no-dev` for reproducible builds.

### Data Storage Pattern
Data flows through Azure Blob Storage via DLT:
- **Raw data**: `fingrid_data/{table_name}/` in Azure Blob container
- **Local fallback**: `data/raw/{table_name}.parquet` for development
- **Feature loading**: `pipeline.features.read_dlt_table(name)` handles both Azure and local

Use `pipeline.features` module for all data access - it abstracts the storage layer.

### Model Registry Pattern
Models are versioned in W&B Model Registry:
- **Entity**: `vattenfall-team` 
- **Registry**: `model-registry/imbalance-price-model`
- **Aliases**: `production` (serving), `staging` (testing), `latest` (new)

Production app loads from `production` alias via `USE_WANDB_REGISTRY=true` environment variable.

### Fingrid API Integration
`FingridClient` expects `FINGRID_API_KEY` from environment. Dataset IDs:
- Wind: 181
- mFRR: 342  
- Price: 319 (target variable)

API key stored in `.env` (gitignored), also in Azure Key Vault for production.

### FastAPI Lifespan Pattern
Model loading happens in lifespan context manager, NOT in global scope:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model from W&B or local
    yield
    # Cleanup
```

### Feature Engineering (processor.py)
Pure transformation functions - no I/O:
- `resample_to_15min(df, col, method)`: Time-series resampling
- `merge_datasets(wind, mfrr, price)`: Inner join on timestamp
- `create_lag_features(df)`: Historical lags with proper `.shift()`
- `create_rolling_features(df)`: Rolling statistics
- `create_temporal_features(df)`: Hour, day-of-week cyclical encoding

## Development Workflows

### Local Development
```bash
uv sync                                    # Setup environment
uv run python -m pipeline.ingest           # Fetch data to local/Azure
uv run python -m pipeline.train            # Train and register model
uv run uvicorn app.main:app --reload       # Run API (port 8000)
uv run pytest                              # Run tests
```

### Docker Build & Test
```bash
docker build -t vattenfall-ml .
docker run -p 8080:8080 --env-file .env vattenfall-ml
```
Container loads model from W&B at runtime (no bundled model.pkl).

### Deployment (Azure)
Terraform in `infra/` manages:
- Container Registry (ACR)
- Container App with auto-scaling
- Storage Account for DLT data
- Key Vault for secrets

GitHub Actions workflow `.github/workflows/deploy.yml` handles CI/CD.

## Key Integration Points

### Weights & Biases
- **Experiment tracking**: `wandb.init()`, `wandb.log()` during training
- **Model registry**: `wandb.link_artifact()` for versioning
- **API key**: `WANDB_API_KEY` env var

### Azure Integration
- **Storage**: `AZURE_STORAGE_CONNECTION_STRING` for DLT destination
- **Container Apps**: Deployed via Terraform with environment variables from Key Vault

### Time Zone Handling
Fingrid API returns UTC timestamps. All processing maintains UTC consistency.

## Common Pitfalls

1. **Parquet Dependencies**: Requires `pyarrow` (already in pyproject.toml)
2. **Azure Auth**: Ensure connection string includes account key or use managed identity
3. **Datetime Alignment**: Merge on exact timestamps using `merge(..., on='timestamp', how='inner')`
4. **Docker Layer Caching**: `uv.lock` and `pyproject.toml` copied before app code
5. **Model Hot-Reload**: FastAPI loads model at startup - restart required for new model version

## File Structure
```
├── app/                    # FastAPI serving
│   ├── main.py            # Endpoints, lifespan
│   └── schemas.py         # Pydantic models
├── pipeline/              # MLOps pipeline modules
│   ├── ingest.py          # DLT data ingestion
│   ├── features.py        # Feature loading/engineering
│   └── train.py           # Model training + W&B loading
├── ingestion/             # Data utilities
│   ├── client.py          # Fingrid API client
│   └── processor.py       # Pure transformation functions
├── infra/                 # Terraform configs
└── tests/                 # pytest test suite
```

## File References
- Architecture decisions: [docs/project_plan.md](docs/project_plan.md)
- API entry point: [app/main.py](app/main.py)
- Pipeline modules: [pipeline/](pipeline/)
- Terraform: [infra/](infra/)
