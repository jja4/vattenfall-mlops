# Copilot Instructions: Vattenfall MLOps

## Project Overview
Real-time electricity imbalance price prediction service for Finland, integrating Fingrid API data with machine learning models served via FastAPI. The system follows a classic MLOps pattern: ingestion → processing → training → serving.

## Architecture & Data Flow
- **Ingestion Layer** (`ingestion/`): Fetches external data from Fingrid API (Wind Power #181, mFRR Activation #342, Imbalance Price #319), caches locally as parquet in `data/raw/`
- **Processing Layer** (`ingestion/processor.py`): Cleans, merges time-series datasets, engineers lag features and rolling means
- **Training Layer** (`models/train.py`): Trains scikit-learn models with Weights & Biases logging, serializes to `models/*.pkl`
- **Serving Layer** (`app/`): FastAPI service loads model on startup via lifespan handler, exposes `/predict` endpoint that fetches latest Fingrid data and returns predictions

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
All raw data flows through `ingestion/storage.py`:
- **Save**: `save_parquet(df, name)` → `data/raw/{name}.parquet` (auto-creates dirs)
- **Load**: `load_parquet(name)` → returns None if missing (cache-friendly pattern)
  
Never write directly to `data/` - always use storage helpers.

### Fingrid API Integration
`FingridClient` expects `FINGRID_API_KEY` from environment. Dataset IDs are hardcoded constants:
- Wind: 181
- mFRR: 342  
- Price: 319 (target variable)

API key stored in `.env` (gitignored), loaded via `python-dotenv`.

### FastAPI Lifespan Pattern
Model loading happens in lifespan context manager (see [project_plan.md](../docs/project_plan.md#L47)), NOT in global scope:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model.pkl here
    yield
    # Cleanup
```

### Feature Engineering Requirements
When implementing `processor.py`, maintain temporal consistency:
- Lag features must match training window (e.g., `price_lag_1h`, `wind_lag_2h`)
- Rolling means should use `.rolling(window=...).mean()` with proper `.shift(1)` to avoid data leakage
- Hour-of-day extraction via `.dt.hour` for capturing diurnal patterns

## Development Workflows

### Local Development
```bash
uv sync                           # Setup environment
uv run uvicorn app.main:app --reload  # Run API locally (port 8000)
uv run python models/train.py     # Train model
uv run pytest                     # Run tests
```

### Docker Build & Test
```bash
docker build -t vattenfall-ml .
docker run -p 8080:8080 --env-file .env vattenfall-ml
```
Container expects model.pkl to exist before build (add explicit copy in Dockerfile).

### Deployment Target
GCP Cloud Run (see [project_plan.md](../docs/project_plan.md#L67-L69)). Port 8080 is hardcoded in Dockerfile CMD.

## Key Integration Points

### Weights & Biases
Training script uses `wandb.init()` and `wandb.log()` for experiment tracking. API key expected via `WANDB_API_KEY` env var.

### Time Zone Handling
Fingrid API returns UTC timestamps. Ensure consistent timezone handling in feature engineering and prediction pipeline.

### Model Serialization
Standard pickle format (`model.pkl`) in `models/` directory. When updating training script, ensure prediction endpoint loads compatible format.

## Common Pitfalls

1. **Parquet Dependencies**: Requires both `pyarrow` AND `fastparquet` (already in pyproject.toml)
2. **Cache Invalidation**: `load_parquet()` returns `None` for missing data - pipeline orchestrator should detect and trigger fetch
3. **Datetime Alignment**: When merging wind/mfrr/price datasets, align on exact timestamps using `merge(..., on='timestamp', how='inner')`
4. **Docker Layer Caching**: `uv.lock` and `pyproject.toml` copied before app code to optimize rebuild times

## File References
- Architecture decisions: [docs/project_plan.md](docs/project_plan.md)
- API entry point: [app/main.py](app/main.py)
- Data pipeline: [ingestion/client.py](ingestion/client.py), [ingestion/storage.py](ingestion/storage.py), [ingestion/processor.py](ingestion/processor.py)
- Model training: [models/train.py](models/train.py)
