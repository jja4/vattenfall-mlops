# Vattenfall Project Plan: Real-Time Imbalance Price Prediction

## 1. Project Initialization
- [x] Initialize project folder structure with skeleton files.
- [ ] Configure `uv` project (`pyproject.toml`) and install dependencies (`fastapi`, `pandas`, `scikit-learn`, `wandb`, `requests`, `uvicorn`, `python-dotenv`).

## 2. Ingestion Layer
- [ ] Implement `ingestion/client.py`: Fingrid API wrapper.
    - [ ] `Class FingridClient`: Helper to manage API key and base URL.
    - [ ] Method `get_data(dataset_id, start_time, end_time)`: Generic fetcher.
    - [ ] Convenience methods for Wind (181), mFRR (342), Price (319).
- [ ] Implement `ingestion/storage.py`:
    - [ ] `save_raw(df, name)`: Save to `data/raw/{name}.parquet`.
    - [ ] `load_raw(name)`: Load from `data/raw/{name}.parquet`.
- [ ] Implement `ingestion/pipeline.py`: Orchestrator to check cache -> fetch if missing -> save.

## 3. Data Processing
- [ ] Implement `ingestion/processor.py`:
    - [ ] `clean_data(df)`: Handle missing values (ffill/bfill).
    - [ ] `merge_datasets(wind, mfrr, price)`: Join on timestamp.
    - [ ] `create_features(df)`: Add lag features (e.g., `price_lag_1h`), rolling means, hour-of-day.

## 4. Modeling Layer
- [ ] Create `models/train.py`:
    - [ ] CLI args for date range.
    - [ ] Fetch/Load data using `ingestion` module.
    - [ ] Split Train/Test.
    - [ ] Train Model (e.g., `RandomForestRegressor`).
    - [ ] Evaluate (MAE, RMSE).
    - [ ] Integration with Weights & Biases (`wandb.init`, `wandb.log`).
    - [ ] Serialization: Save `model.pkl`.

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
