"""
FastAPI application for serving imbalance price predictions.

The API fetches the latest data from Fingrid, processes features matching
the training format, and returns predictions using the trained model.

Usage:
    uvicorn app.main:app --reload --port 8000
"""
import os
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.schemas import HealthResponse, PredictionResponse, ErrorResponse
from models.train import load_model
from ingestion.client import FingridClient
from ingestion.processor import (
    resample_to_15min,
    merge_datasets,
    create_lag_features,
    create_rolling_features,
    create_temporal_features,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for model and scaler
_model = None
_feature_names = None
_scaler = None
_model_created_at = None

# Cache for Fingrid data (avoid hammering the API)
_data_cache = {
    "data": None,
    "timestamp": None,
    "ttl_seconds": 300  # 5 minute cache
}


def get_model_path() -> str:
    """Get model path from environment or default."""
    return os.getenv("MODEL_PATH", "models/model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    Loads model on startup, cleanup on shutdown.
    """
    global _model, _feature_names, _scaler, _model_created_at
    
    model_path = get_model_path()
    logger.info(f"Loading model from {model_path}...")
    
    try:
        _model, _feature_names, _scaler = load_model(model_path)
        
        # Try to get model creation time from artifact
        import pickle
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
            if isinstance(artifact, dict):
                _model_created_at = artifact.get("created_at")
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Features: {len(_feature_names) if _feature_names else 'unknown'}")
        logger.info(f"  Scaler: {'yes' if _scaler else 'no'}")
        logger.info(f"  Created: {_model_created_at or 'unknown'}")
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train a model first: python -m models.train")
        raise RuntimeError(f"Model not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up...")
    _model = None
    _feature_names = None
    _scaler = None


# Create FastAPI app with lifespan
app = FastAPI(
    title="Vattenfall Imbalance Price Predictor",
    description="Real-time electricity imbalance price prediction for Finland using Fingrid data",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "Vattenfall ML Service", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for k8s/Cloud Run.
    Returns model status and basic info.
    """
    return HealthResponse(
        status="healthy" if _model is not None else "unhealthy",
        model_loaded=_model is not None,
        model_features=len(_feature_names) if _feature_names else 0,
        timestamp=datetime.now(timezone.utc)
    )


def _is_cache_valid() -> bool:
    """Check if cached data is still valid."""
    if _data_cache["data"] is None or _data_cache["timestamp"] is None:
        return False
    age = (datetime.now(timezone.utc) - _data_cache["timestamp"]).total_seconds()
    return age < _data_cache["ttl_seconds"]


def _fetch_latest_data() -> pd.DataFrame:
    """
    Fetch latest data from Fingrid API.
    Uses cache to avoid hammering the API.
    
    Returns:
        DataFrame with merged wind, mfrr, and price data
    """
    global _data_cache
    
    # Return cached data if valid
    if _is_cache_valid():
        logger.info("Using cached Fingrid data")
        return _data_cache["data"]
    
    logger.info("Fetching fresh data from Fingrid API...")
    
    # Need at least 24h of data for lag features
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=30)  # Extra buffer for safety
    
    try:
        client = FingridClient()
        
        # Fetch all three datasets (pass datetime objects, not strings)
        wind_df = client.get_wind_power(start_time, end_time)
        mfrr_df = client.get_mfrr_activation(start_time, end_time)
        price_df = client.get_imbalance_price(start_time, end_time)
        
        if wind_df.empty or mfrr_df.empty or price_df.empty:
            raise ValueError("One or more datasets returned empty")
        
        logger.info(f"  Wind: {len(wind_df)} rows")
        logger.info(f"  mFRR: {len(mfrr_df)} rows")
        logger.info(f"  Price: {len(price_df)} rows")
        
        # Resample to 15-min intervals
        wind_df = resample_to_15min(wind_df, "wind_power_mw", method="mean")
        mfrr_df = resample_to_15min(mfrr_df, "mfrr_price", method="ffill")
        price_df = resample_to_15min(price_df, "imbalance_price", method="ffill")
        
        # Merge datasets
        merged = merge_datasets(wind_df, mfrr_df, price_df)
        
        # Update cache
        _data_cache["data"] = merged
        _data_cache["timestamp"] = datetime.now(timezone.utc)
        
        logger.info(f"  Merged: {len(merged)} rows")
        return merged
        
    except Exception as e:
        logger.error(f"Failed to fetch Fingrid data: {e}")
        
        # Return stale cache if available (better than nothing)
        if _data_cache["data"] is not None:
            logger.warning("Returning stale cached data")
            return _data_cache["data"]
        
        raise


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, datetime, datetime]:
    """
    Prepare features for prediction from raw merged data.
    
    Returns:
        Tuple of (features array, prediction_for timestamp, data_timestamp)
    """
    # Apply feature engineering (same as training)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_temporal_features(df)
    
    # Drop NaN rows from lag/rolling operations
    df = df.dropna().reset_index(drop=True)
    
    if df.empty:
        raise ValueError("No valid data after feature engineering")
    
    # Get the latest row for prediction
    latest = df.iloc[[-1]]  # Keep as DataFrame for column selection
    
    # Get timestamps
    data_timestamp = latest["timestamp"].iloc[0]
    prediction_for = data_timestamp + timedelta(minutes=15)  # Predicting next interval
    
    # Select features in the same order as training
    if _feature_names is None:
        raise ValueError("Feature names not loaded from model")
    
    # Check for missing features
    missing_features = set(_feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    X = latest[_feature_names].values
    
    # Apply scaler if available
    if _scaler is not None:
        X = _scaler.transform(X)
    
    return X, prediction_for, data_timestamp


@app.get(
    "/predict",
    response_model=PredictionResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    },
    tags=["Prediction"]
)
async def predict():
    """
    Get price prediction for the next 15-minute interval.
    
    This endpoint:
    1. Fetches the latest data from Fingrid API (cached for 5 min)
    2. Processes features matching the training format
    3. Returns the predicted imbalance price
    
    The prediction is for the next 15-minute interval after the latest available data.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch latest data
        merged_df = _fetch_latest_data()
        
        # Prepare features
        X, prediction_for, data_timestamp = _prepare_features(merged_df)
        
        # Make prediction
        prediction = _model.predict(X)[0]
        
        logger.info(f"Prediction: {prediction:.2f} EUR/MWh for {prediction_for}")
        
        return PredictionResponse(
            predicted_price=round(float(prediction), 2),
            unit="EUR/MWh",
            prediction_for=prediction_for,
            data_timestamp=data_timestamp,
            model_version=_model_created_at
        )
        
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/invalidate-cache", tags=["Admin"])
async def invalidate_cache():
    """
    Invalidate the Fingrid data cache.
    Forces fresh data fetch on next prediction.
    """
    global _data_cache
    _data_cache["data"] = None
    _data_cache["timestamp"] = None
    return {"message": "Cache invalidated"}


@app.get("/model/info", tags=["Admin"])
async def model_info():
    """Get information about the loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(_model).__name__,
        "n_features": len(_feature_names) if _feature_names else 0,
        "feature_names": _feature_names,
        "has_scaler": _scaler is not None,
        "created_at": _model_created_at,
        "model_path": get_model_path()
    }
