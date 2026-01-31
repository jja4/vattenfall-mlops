# Vattenfall Project Plan: Real-Time Imbalance Price Prediction

## 1. Project Initialization
- [x] Initialize project folder structure with skeleton files.
- [x] Configure `uv` project (`pyproject.toml`) and install dependencies (`fastapi`, `pandas`, `scikit-learn`, `wandb`, `requests`, `uvicorn`, `python-dotenv`).

## 2. Ingestion Layer
- [x] Implement `ingestion/client.py`: Fingrid API wrapper.
    - [x] `Class FingridClient`: Helper to manage API key and base URL.
    - [x] Method `get_data(dataset_id, start_time, end_time)`: Generic fetcher.
    - [x] Convenience methods for Wind (181), mFRR (342), Price (319).
    - [x] Monthly chunking with `relativedelta` for large date ranges.
    - [x] Rate limiting with exponential backoff for 429 errors.
- [x] Implement `ingestion/storage.py`:
    - [x] `save_parquet(df, name)`: Save to `data/raw/{name}.parquet`.
    - [x] `load_parquet(name)`: Load from `data/raw/{name}.parquet`.
    - [x] `save_processed(df, name)`: Cache processed data to `data/processed/`.
    - [x] `load_processed(name)`: Load cached processed data.
    - [x] `PROJECT_ROOT` constant for absolute path resolution.
- [x] Implement `ingestion/pipeline.py`: Orchestrator to check cache -> fetch if missing -> save.

## 3. Data Processing
- [x] Implement `ingestion/processor.py`:
    - [x] `resample_to_15min(df, value_col, method)`: Resample to 15-min intervals.
    - [x] `merge_datasets(wind, mfrr, price)`: Join on timestamp with inner join.
    - [x] `create_lag_features(df)`: Add lag features (1h, 2h, 3h) for price/wind/mfrr.
    - [x] `create_rolling_features(df)`: Rolling means (1h, 3h) with shift to avoid leakage.
    - [x] `create_temporal_features(df)`: hour_of_day, day_of_week, is_weekend.
    - [x] `validate_dataframe(df, required_columns, name)`: Input validation.
    - [x] `process_features(df, use_cache)`: Main pipeline with processed data caching.
    - [x] Constants: `PERIODS_PER_HOUR`, `LAG_1H/2H/3H`, `ROLLING_1H/3H`.
    - [x] Required column sets: `REQUIRED_RAW_COLUMNS`, `REQUIRED_MERGED_COLUMNS`.

## 4. Modeling Layer
- [x] Create `models/train.py`:
    - [x] CLI args for training parameters (n-estimators, max-depth, test-size, etc.).
    - [x] Fetch/Load data using `ingestion` module with caching.
    - [x] Temporal Train/Test split (chronological, not random).
    - [x] Train Model (`RandomForestRegressor` with regularization).
    - [x] Evaluate (MAE, RMSE, R²) on held-out test set.
    - [x] Integration with Weights & Biases (`wandb.init`, `wandb.log`).
    - [x] `save_model(model, feature_names, path)`: Save model with metadata.
    - [x] `load_model(path)`: Load model and feature names (handles legacy format).
- [x] Create `run_pipeline.py`: Single entry point for ingestion → processing → training.
    - [x] CLI flags: `--skip-ingestion`, `--skip-processing`, `--skip-training`, `--no-wandb`.
    - [x] Date range args: `--start`, `--end`.

## 4.1 Testing
- [x] Create `tests/` directory with pytest tests.
    - [x] `test_processor.py`: Tests for resample, lag, rolling, temporal features, validation.
    - [x] `test_train.py`: Tests for prepare_data, train_model, evaluate_model, save/load.
- [x] Move diagnostic scripts to `scripts/` folder.
    - [x] `scripts/analyze_intervals.py`: Temporal pattern analysis.
    - [x] `scripts/verify_data.py`: Data verification report.

## 5. Serving Layer (FastAPI)
- [ ] Implement `app/schemas.py`: Pydantic models for Input/Output.
- [ ] Implement `app/main.py`:
    - [ ] `lifespan` handler: Load `model.pkl` on startup.
    - [ ] `GET /health`: Health check.
    - [ ] `GET /predict`: 
        - [ ] Fetch *latest* available data from Fingrid (past 1-2 hours).
        - [ ] Process features matching training format.
        - [ ] Return prediction.

## 6. Dockerization
- [ ] Create `Dockerfile`:
    - [ ] Base python image (slim).
    - [ ] Install `uv`.
    - [ ] Copy code and model.
    - [ ] Entrypoint: `uvicorn app.main:app`.
- [ ] Create `.dockerignore`.
- [ ] Test build and run locally.

## 7. Cloud Deployment (GCP Cloud Run)
- [ ] Setup GCP Project & Artifact Registry.
- [ ] Build and Push Image:
- [ ] Deploy to Cloud Run:
