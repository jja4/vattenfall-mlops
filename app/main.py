"""
FastAPI application for serving imbalance price predictions.

The API fetches the latest data from Fingrid, processes features matching
the training format, and returns predictions using the trained model.

Features:
- Real-time predictions using latest Fingrid data
- Model hot-reload from W&B Model Registry
- Background model version checking
- Zero-downtime model updates

Usage:
    uvicorn app.main:app --reload --port 8000
"""
import os
import asyncio
import threading
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from app.schemas import HealthResponse, PredictionResponse, ErrorResponse, ModelReloadResponse
from pipeline.train import load_model, load_model_from_wandb, get_production_model_version
from ingestion.client import FingridClient
from ingestion.processor import (
    resample_to_15min,
    merge_datasets,
    merge_datasets_realtime,
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
_model_version = None  # W&B version (e.g., 'v3')

# Thread lock for safe model swapping
_model_lock = threading.Lock()

# Background reload configuration
MODEL_CHECK_INTERVAL_SECONDS = 300  # Check W&B every 5 minutes
USE_WANDB_REGISTRY = os.getenv("USE_WANDB_REGISTRY", "false").lower() == "true"

# Cache for Fingrid data (avoid hammering the API)
_data_cache = {
    "data": None,
    "timestamp": None,
    "ttl_seconds": 300  # 5 minute cache
}


def get_model_path() -> str:
    """Get model path from environment or default."""
    return os.getenv("MODEL_PATH", "models/model.pkl")


def _load_model_from_source():
    """
    Load model from configured source (W&B Registry or local file).
    
    Returns:
        Tuple of (model, feature_names, scaler, version, created_at)
    """
    if USE_WANDB_REGISTRY:
        logger.info("Loading model from W&B Model Registry...")
        model, feature_names, scaler, version, created_at = load_model_from_wandb("production")
        return model, feature_names, scaler, version, created_at
    else:
        logger.info("Loading model from local file...")
        model_path = get_model_path()
        model, feature_names, scaler = load_model(model_path)
        
        # Try to get metadata from local file
        import pickle
        version = None
        created_at = None
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
            if isinstance(artifact, dict):
                created_at = artifact.get("created_at")
                version = artifact.get("version", "local")
        
        return model, feature_names, scaler, version, created_at


async def _background_model_checker():
    """
    Background task that periodically checks for model updates in W&B.
    
    Runs every MODEL_CHECK_INTERVAL_SECONDS and reloads model if
    production version has changed.
    """
    global _model, _feature_names, _scaler, _model_version, _model_created_at
    
    if not USE_WANDB_REGISTRY:
        logger.info("W&B registry disabled, background checker not starting")
        return
    
    logger.info(f"Starting background model checker (interval: {MODEL_CHECK_INTERVAL_SECONDS}s)")
    
    while True:
        await asyncio.sleep(MODEL_CHECK_INTERVAL_SECONDS)
        
        try:
            # Check current production version
            current_version = get_production_model_version()
            
            if current_version and current_version != _model_version:
                logger.info(f"New model version detected: {current_version} (was: {_model_version})")
                
                # Load new model
                new_model, new_features, new_scaler, version, created_at = load_model_from_wandb("production")
                
                # Atomic swap with lock
                with _model_lock:
                    _model = new_model
                    _feature_names = new_features
                    _scaler = new_scaler
                    _model_version = version
                    _model_created_at = created_at
                
                logger.info(f"Model hot-reloaded to version {version}")
            else:
                logger.debug(f"Model version unchanged: {_model_version}")
                
        except Exception as e:
            logger.error(f"Background model check failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    Loads model on startup, starts background checker, cleanup on shutdown.
    """
    global _model, _feature_names, _scaler, _model_created_at, _model_version
    
    logger.info("Starting Vattenfall ML Service...")
    
    try:
        # Load initial model
        _model, _feature_names, _scaler, _model_version, _model_created_at = _load_model_from_source()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Version: {_model_version or 'unknown'}")
        logger.info(f"  Features: {len(_feature_names) if _feature_names else 'unknown'}")
        logger.info(f"  Scaler: {'yes' if _scaler else 'no'}")
        logger.info(f"  Created: {_model_created_at or 'unknown'}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Please train a model first: python -m pipeline.train")
        raise RuntimeError(f"Model not found: {e}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Start background model checker
    background_task = asyncio.create_task(_background_model_checker())
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up...")
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    
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
        model_version=_model_version,
        timestamp=datetime.now(timezone.utc)
    )


@app.post("/model/reload", response_model=ModelReloadResponse, tags=["Admin"])
async def reload_model():
    """
    Manually trigger model reload from W&B Model Registry.
    
    This endpoint forces an immediate reload of the production model
    from W&B, useful for testing or forcing updates.
    """
    global _model, _feature_names, _scaler, _model_version, _model_created_at
    
    if not USE_WANDB_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail="W&B registry not enabled. Set USE_WANDB_REGISTRY=true"
        )
    
    old_version = _model_version
    
    try:
        new_model, new_features, new_scaler, version, created_at = load_model_from_wandb("production")
        
        with _model_lock:
            _model = new_model
            _feature_names = new_features
            _scaler = new_scaler
            _model_version = version
            _model_created_at = created_at
        
        logger.info(f"Model manually reloaded: {old_version} -> {version}")
        
        return ModelReloadResponse(
            success=True,
            message="Model reloaded successfully",
            old_version=old_version,
            new_version=version,
        )
        
    except Exception as e:
        logger.error(f"Manual model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _is_cache_valid() -> bool:
    """Check if cached data is still valid."""
    if _data_cache["data"] is None or _data_cache["timestamp"] is None:
        return False
    age = (datetime.now(timezone.utc) - _data_cache["timestamp"]).total_seconds()
    return age < _data_cache["ttl_seconds"]


def _fetch_latest_data(realtime: bool = False) -> pd.DataFrame:
    """
    Fetch latest data from Fingrid API.
    Uses cache to avoid hammering the API.
    
    Args:
        realtime: If True, use left-join to include timestamps without price data
                  (for real-time prediction). If False, only return rows where
                  all data is available (for metrics calculation).
    
    Returns:
        DataFrame with merged wind, mfrr, and price data
    """
    global _data_cache
    
    cache_key = "data_realtime" if realtime else "data"
    
    # Return cached data if valid
    if _is_cache_valid():
        if cache_key in _data_cache and _data_cache[cache_key] is not None:
            logger.info(f"Using cached Fingrid data (realtime={realtime})")
            return _data_cache[cache_key]
    
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
        
        if wind_df.empty or mfrr_df.empty:
            raise ValueError("Wind or mFRR datasets returned empty")
        
        logger.info(f"  Wind: {len(wind_df)} rows, latest: {wind_df['timestamp'].max()}")
        logger.info(f"  mFRR: {len(mfrr_df)} rows, latest: {mfrr_df['timestamp'].max()}")
        logger.info(f"  Price: {len(price_df)} rows, latest: {price_df['timestamp'].max() if not price_df.empty else 'N/A'}")
        
        # Resample to 15-min intervals
        wind_df = resample_to_15min(wind_df, "wind_power_mw", method="mean")
        mfrr_df = resample_to_15min(mfrr_df, "mfrr_price", method="ffill")
        price_df = resample_to_15min(price_df, "imbalance_price", method="ffill") if not price_df.empty else pd.DataFrame(columns=["timestamp", "imbalance_price"])
        
        # Merge datasets - use inner join for historical (with actuals), left join for realtime
        merged = merge_datasets(wind_df, mfrr_df, price_df)
        merged_realtime = merge_datasets_realtime(wind_df, mfrr_df, price_df)
        
        # Update cache with both versions
        _data_cache["data"] = merged
        _data_cache["data_realtime"] = merged_realtime
        _data_cache["timestamp"] = datetime.now(timezone.utc)
        
        logger.info(f"  Merged (inner): {len(merged)} rows")
        logger.info(f"  Merged (realtime): {len(merged_realtime)} rows")
        
        return merged_realtime if realtime else merged
        
    except Exception as e:
        logger.error(f"Failed to fetch Fingrid data: {e}")
        
        # Return stale cache if available (better than nothing)
        if cache_key in _data_cache and _data_cache[cache_key] is not None:
            logger.warning("Returning stale cached data")
            return _data_cache[cache_key]
        
        raise


def _prepare_features(df: pd.DataFrame, for_realtime: bool = False) -> tuple[np.ndarray, datetime, datetime]:
    """
    Prepare features for prediction from raw merged data.
    
    Args:
        df: Raw merged data
        for_realtime: If True, use forward-filled prices for lag features
                      (allows prediction even when price is not yet known)
    
    Returns:
        Tuple of (features array, prediction_for timestamp, data_timestamp)
    """
    # Apply feature engineering (same as training, but with realtime option)
    df = create_lag_features(df, for_realtime=for_realtime)
    df = create_rolling_features(df, for_realtime=for_realtime)
    df = create_temporal_features(df)
    
    # Drop NaN rows from lag/rolling operations
    df = df.dropna(subset=_feature_names).reset_index(drop=True)
    
    if df.empty:
        raise ValueError("No valid data after feature engineering")
    
    # Get the latest row for prediction
    latest = df.iloc[[-1]]  # Keep as DataFrame for column selection
    
    # Get timestamps
    data_timestamp = latest["timestamp"].iloc[0]
    # For real-time: we're predicting the price AT this timestamp (nowcast)
    prediction_for = data_timestamp
    
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
    Get real-time price prediction for the current/latest interval.
    
    This endpoint:
    1. Fetches the latest wind and mFRR data from Fingrid API (cached for 5 min)
    2. Uses the most recent available price for lag features
    3. Returns the predicted imbalance price for the latest timestamp
    
    The prediction is made in real-time using the most recent input data,
    even if the actual price for that timestamp hasn't been published yet.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch latest data (realtime=True to include timestamps without price)
        merged_df = _fetch_latest_data(realtime=True)
        
        # Prepare features (for_realtime=True uses forward-filled prices for lags)
        X, prediction_for, data_timestamp = _prepare_features(merged_df, for_realtime=True)
        
        # Make prediction
        prediction = _model.predict(X)[0]
        
        logger.info(f"Real-time prediction: {prediction:.2f} EUR/MWh for {prediction_for}")
        
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


@app.get("/model/info", tags=["Admin"])
async def model_info():
    """Get information about the loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(_model).__name__,
        "model_version": _model_version,  # W&B version (e.g., 'v1') or 'local'
        "model_source": "wandb_registry" if USE_WANDB_REGISTRY else "local_file",
        "n_features": len(_feature_names) if _feature_names else 0,
        "feature_names": _feature_names,
        "has_scaler": _scaler is not None,
        "created_at": _model_created_at,
    }


@app.get("/dashboard", response_class=HTMLResponse, tags=["Visualization"])
async def dashboard():
    """
    Interactive dashboard showing recent data and real-time predictions.
    Returns an HTML page with Plotly charts.
    
    Shows:
    - Historical predictions vs actual prices (where actuals are available)
    - Real-time predictions for timestamps where price is not yet published
    - Current prediction highlighted
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    try:
        # Fetch data with realtime=True to get timestamps without price
        raw_data = _fetch_latest_data(realtime=True)
        
        if raw_data.empty:
            return HTMLResponse("<h1>No data available</h1>", status_code=503)
        
        # Apply feature engineering with realtime=True (forward-fill prices for lags)
        df = raw_data.copy()
        df = create_lag_features(df, for_realtime=True)
        df = create_rolling_features(df, for_realtime=True)
        df = create_temporal_features(df)
        
        # Drop rows where we can't compute features (early rows without enough lag history)
        df = df.dropna(subset=_feature_names).reset_index(drop=True)
        
        if df.empty or len(df) < 5:
            return HTMLResponse("<h1>Insufficient data for visualization</h1>", status_code=503)
        
        # Get predictions for ALL rows (including those without actual price)
        X = df[_feature_names].values
        if _scaler:
            X = _scaler.transform(X)
        predictions = _model.predict(X)
        
        # Separate data into historical (with actuals) and forecast (without actuals)
        has_actual = ~df['imbalance_price'].isna()
        
        # Historical data (where we have actual prices)
        hist_timestamps = df.loc[has_actual, 'timestamp'].tolist()
        hist_actuals = df.loc[has_actual, 'imbalance_price'].tolist()
        hist_preds = predictions[has_actual].tolist()
        
        # Forecast data (where we don't have actual prices yet)
        forecast_timestamps = df.loc[~has_actual, 'timestamp'].tolist()
        forecast_preds = predictions[~has_actual].tolist()
        
        # All timestamps for wind/mFRR (always available)
        all_timestamps = df['timestamp'].tolist()
        wind = df['wind_power_mw'].tolist()
        mfrr = df['mfrr_price'].tolist()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '<b>Imbalance Price: Actual vs Real-Time Prediction</b>',
                '<b>Wind Power Generation</b>',
                '<b>mFRR Activation Price</b>'
            ),
            vertical_spacing=0.16,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True
        )
        
        # Plot 1a: Actual prices (historical)
        if hist_actuals:
            fig.add_trace(
                go.Scatter(
                    x=hist_timestamps, 
                    y=hist_actuals, 
                    name='Actual Price',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='Actual: %{y:.2f} EUR/MWh<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 1b: Historical predictions (where we have actuals)
        if hist_preds:
            fig.add_trace(
                go.Scatter(
                    x=hist_timestamps, 
                    y=hist_preds, 
                    name='Model Prediction (Historical)',
                    line=dict(color='#E94F37', width=2, dash='dash'),
                    hovertemplate='Predicted: %{y:.2f} EUR/MWh<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 1c: Real-time predictions (where we don't have actuals yet)
        if forecast_preds:
            fig.add_trace(
                go.Scatter(
                    x=forecast_timestamps, 
                    y=forecast_preds, 
                    name='Real-Time Prediction',
                    mode='lines+markers',
                    line=dict(color='#9B59B6', width=3),
                    marker=dict(size=10, symbol='diamond'),
                    hovertemplate='<b>LIVE</b> Predicted: %{y:.2f} EUR/MWh<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Wind Power
        fig.add_trace(
            go.Scatter(
                x=all_timestamps, 
                y=wind, 
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
                x=all_timestamps, 
                y=mfrr, 
                name='mFRR Price (EUR/MWh)',
                line=dict(color='#F18F01', width=1.5),
                hovertemplate='mFRR: %{y:.2f} EUR/MWh<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Calculate metrics only on historical data (where we have actuals)
        if len(hist_actuals) > 1:
            actuals_arr = np.array(hist_actuals)
            preds_arr = np.array(hist_preds)
            mae = np.mean(np.abs(actuals_arr - preds_arr))
            rmse = np.sqrt(np.mean((actuals_arr - preds_arr) ** 2))
            corr = np.corrcoef(actuals_arr, preds_arr)[0, 1]
            metrics_text = (f'MAE: {mae:.2f} EUR/MWh | RMSE: {rmse:.2f} EUR/MWh | '
                          f'Correlation: {corr:.3f} | Historical Points: {len(hist_actuals)}')
        else:
            metrics_text = "Insufficient historical data for metrics"
        
        # Latest prediction info
        latest_pred = predictions[-1]
        latest_ts = df['timestamp'].iloc[-1]
        has_latest_actual = not pd.isna(df['imbalance_price'].iloc[-1])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Finland Electricity Imbalance Price Dashboard</b><br>'
                     f'<span style="font-size:14px">{metrics_text}</span><br>'
                     f'<span style="font-size:16px; color:#9B59B6"><b>Latest Prediction: {latest_pred:.2f} EUR/MWh for {latest_ts.strftime("%H:%M UTC")}'
                     f'{" (actual available)" if has_latest_actual else " (LIVE)"}</b></span>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=900,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.25,
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
            margin=dict(b=120, l=80, r=50, t=140)
        )
        
        # Update axes labels
        fig.update_yaxes(title_text='Price (EUR/MWh)', row=1, col=1)
        fig.update_xaxes(title_text='Time (UTC)', showticklabels=True, row=1, col=1)
        
        fig.update_yaxes(title_text='Power (MW)', row=2, col=1)
        fig.update_xaxes(title_text='Time (UTC)', showticklabels=True, row=2, col=1)
        
        fig.update_yaxes(title_text='Price (EUR/MWh)', row=3, col=1)
        fig.update_xaxes(title_text='Time (UTC)', row=3, col=1)
        
        # Calculate data lag
        latest_data_time = df['timestamp'].max()
        data_lag_minutes = (datetime.now(timezone.utc) - latest_data_time.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 60
        
        # Count realtime predictions
        n_realtime = len(forecast_preds)
        
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
                    ℹ️ The model uses current wind power, mFRR prices, 
                    and historical lags to predict the imbalance price in real-time.
                </div>
                <div class="info {'warning' if n_realtime > 0 else ''}">
                    ⏱️ <strong>{'🔴 LIVE PREDICTIONS' if n_realtime > 0 else 'Data Status'}:</strong> 
                    {'<b>' + str(n_realtime) + ' real-time prediction(s)</b> shown for intervals where actual price is not yet available. ' if n_realtime > 0 else ''}
                    Latest data: {latest_data_time.strftime('%Y-%m-%d %H:%M UTC')} 
                    (~{int(data_lag_minutes)} min behind).
                </div>
                <div id="chart"></div>
            </div>
            <script>
                var plotlyData = {fig.to_json()};
                Plotly.newPlot('chart', plotlyData.data, plotlyData.layout, {{responsive: true}});
                // Auto-refresh every 2 minutes
                setTimeout(function() {{ location.reload(); }}, 120000);
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
