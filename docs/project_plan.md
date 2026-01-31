# Vattenfall Project Plan: Real-Time Imbalance Price Prediction

## 1. Project Initialization
- [x] Initialize project folder structure with skeleton files.
- [x] Configure `uv` project (`pyproject.toml`) and install dependencies (`fastapi`, `pandas`, `scikit-learn`, `wandb`, `requests`, `uvicorn`, `python-dotenv`).

## 2. Ingestion Layer
- [x] Implement `ingestion/client.py`: Fingrid API wrapper.
    - [x] `Class FingridClient`: Helper to manage API key and base URL.
    - [x] Method `get_data(dataset_id, start_time, end_time)`: Generic fetcher.
    - [x] Convenience methods for Wind (181), mFRR (342), Price (319).
- [x] Implement `ingestion/storage.py`:
    - [x] `save_raw(df, name)`: Save to `data/raw/{name}.parquet`.
    - [x] `load_raw(name)`: Load from `data/raw/{name}.parquet`.
- [x] Implement `ingestion/pipeline.py`: Orchestrator to check cache -> fetch if missing -> save.

## 3. Data Processing
- [x] Implement `ingestion/processor.py`:
    - [x] `clean_data(df)`: Handle missing values (ffill/bfill).
    - [x] `merge_datasets(wind, mfrr, price)`: Join on timestamp.
    - [x] `create_features(df)`: Add lag features (e.g., `price_lag_1h`), rolling means, hour-of-day.

## 4. Modeling Layer
- [x] Create `models/train.py`:
    - [x] CLI args for training parameters.
    - [x] Fetch/Load data using `ingestion` module.
    - [x] Split Train/Test.
    - [x] Train Model (e.g., `RandomForestRegressor`).
    - [x] Evaluate (MAE, RMSE).
    - [ ] Integration with Weights & Biases (`wandb.init`, `wandb.log`).
    - [x] Serialization: Save `model.pkl`.

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
