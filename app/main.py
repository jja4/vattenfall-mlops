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
from fastapi.responses import JSONResponse, HTMLResponse
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


@app.get("/dashboard", response_class=HTMLResponse, tags=["Visualization"])
async def dashboard():
    """
    Interactive dashboard showing recent data and predictions.
    Returns an HTML page with Plotly charts.
    
    Note: Predictions are made for T+15min using features at time T.
    We align predictions with actual future values for proper comparison.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    try:
        # Fetch latest data
        raw_data = _fetch_latest_data()
        
        if raw_data.empty:
            return HTMLResponse("<h1>No data available</h1>", status_code=503)
        
        # Apply feature engineering to get features for ALL rows (not just latest)
        df = raw_data.copy()
        df = create_lag_features(df)
        df = create_rolling_features(df)
        df = create_temporal_features(df)
        
        # Drop NaN rows from lag/rolling operations
        df = df.dropna().reset_index(drop=True)
        
        if df.empty or len(df) < 10:
            return HTMLResponse("<h1>Insufficient data for visualization</h1>", status_code=503)
        
        # Get predictions for all available rows
        # Prediction at row i is for timestamp[i] + 15min = timestamp[i+1]
        X = df[_feature_names].values
        if _scaler:
            X = _scaler.transform(X)
        predictions = _model.predict(X)
        
        # ALIGNMENT FIX: Predictions made at T are for T+15min
        # So prediction[i] should compare with actual[i+1]
        # We shift predictions back by 1 to align with actuals
        # Or equivalently: for time T, we show prediction made at T-15min
        
        # Use timestamps as the common x-axis (excluding first point for alignment)
        # Convert to Python lists to avoid Plotly binary encoding issues
        aligned_timestamps = df['timestamp'].values[1:].tolist()  # T+15min timestamps
        aligned_actuals = df['imbalance_price'].values[1:].tolist()  # Actual at T+15min
        aligned_predictions = predictions[:-1].tolist()  # Prediction made at T for T+15min
        
        # Wind and mFRR also aligned to same timestamps
        aligned_wind = df['wind_power_mw'].values[1:].tolist()
        aligned_mfrr = df['mfrr_price'].values[1:].tolist()
        
        # Create subplots with shared x-axis for alignment
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '<b>Imbalance Price: Actual vs Model Prediction</b>',
                '<b>Wind Power Generation</b>',
                '<b>mFRR Activation Price</b>'
            ),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True  # Share x-axis for time alignment
        )
        
        # Plot 1: Actual vs Predicted (properly aligned)
        fig.add_trace(
            go.Scatter(
                x=aligned_timestamps, 
                y=aligned_actuals, 
                name='Actual Price',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='Actual: %{y:.2f} EUR/MWh<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=aligned_timestamps, 
                y=aligned_predictions, 
                name='Predicted Price (15min ahead)',
                line=dict(color='#E94F37', width=2, dash='dash'),
                hovertemplate='Predicted: %{y:.2f} EUR/MWh<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Wind Power
        fig.add_trace(
            go.Scatter(
                x=aligned_timestamps, 
                y=aligned_wind, 
                name='Wind Power (MW)',
                line=dict(color='#44AF69', width=1.5),
                fill='tozeroy', 
                fillcolor='rgba(68, 175, 105, 0.2)',
                hovertemplate='Wind: %{y:.0f} MW<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 3: mFRR Price
        fig.add_trace(
            go.Scatter(
                x=aligned_timestamps, 
                y=aligned_mfrr, 
                name='mFRR Price (EUR/MWh)',
                line=dict(color='#F18F01', width=1.5),
                hovertemplate='mFRR: %{y:.2f} EUR/MWh<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Calculate metrics on aligned data (convert back to numpy for calculations)
        actuals_arr = np.array(aligned_actuals)
        preds_arr = np.array(aligned_predictions)
        mae = np.mean(np.abs(actuals_arr - preds_arr))
        rmse = np.sqrt(np.mean((actuals_arr - preds_arr) ** 2))
        
        # Correlation between prediction and actual
        corr = np.corrcoef(actuals_arr, preds_arr)[0, 1]
        
        # Update layout with better legend and spacing
        fig.update_layout(
            title=dict(
                text=f'<b>Finland Electricity Imbalance Price Dashboard</b><br>'
                     f'<span style="font-size:14px">MAE: {mae:.2f} EUR/MWh | '
                     f'RMSE: {rmse:.2f} EUR/MWh | '
                     f'Correlation: {corr:.3f} | '
                     f'Data Points: {len(aligned_predictions)}</span>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=900,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.25,  # Move legend further down to avoid overlap with x-axis title
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                font=dict(size=12),
                itemsizing='constant',
                itemwidth=40
            ),
            template='plotly_white',
            hovermode='x unified',
            margin=dict(b=120, l=80, r=50, t=120)  # More margins for labels
        )
        
        # Update axes labels - show time on all charts
        fig.update_yaxes(title_text='Price (EUR/MWh)', row=1, col=1)
        fig.update_xaxes(title_text='Time (UTC)', showticklabels=True, row=1, col=1)
        
        fig.update_yaxes(title_text='Power (MW)', row=2, col=1)
        fig.update_xaxes(title_text='Time (UTC)', showticklabels=True, row=2, col=1)
        
        fig.update_yaxes(title_text='Price (EUR/MWh)', row=3, col=1)
        fig.update_xaxes(title_text='Time (UTC)', row=3, col=1)
        
        # Calculate data lag
        latest_data_time = df['timestamp'].max()
        data_lag_minutes = (datetime.now(timezone.utc) - latest_data_time.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 60
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vattenfall MLOps Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                }}
                .header {{ 
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 1px solid #eee;
                }}
                .refresh-btn {{
                    background: #2E86AB;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                .refresh-btn:hover {{ background: #1d5d7a; }}
                .timestamp {{ color: #666; font-size: 14px; }}
                .info {{ 
                    background: #f8f9fa; 
                    padding: 10px 15px; 
                    border-radius: 5px; 
                    margin-bottom: 15px;
                    font-size: 13px;
                    color: #555;
                }}
                .warning {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    color: #856404;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <h1 style="margin: 0;">🇫🇮 Finland Imbalance Price Prediction</h1>
                        <p class="timestamp">Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    </div>
                    <button class="refresh-btn" onclick="location.reload()">↻ Refresh</button>
                </div>
                <div class="info">
                    ℹ️ Predictions are made 15 minutes ahead. The model uses current wind power, mFRR prices, 
                    and historical lags to predict the next interval's imbalance price.
                </div>
                <div class="info warning">
                    ⏱️ <strong>Data Lag:</strong> Latest data point is from {latest_data_time.strftime('%Y-%m-%d %H:%M UTC')} 
                    (~{int(data_lag_minutes)} min / {data_lag_minutes/60:.1f} hours behind current time). 
                    This is normal — Fingrid API publishes data with ~1-2 hour delay.
                </div>
                <div id="chart"></div>
            </div>
            <script>
                var plotlyData = {fig.to_json()};
                Plotly.newPlot('chart', plotlyData.data, plotlyData.layout, {{responsive: true}});
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)
