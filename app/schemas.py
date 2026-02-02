"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool
    model_features: int
    model_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    """Response for price prediction endpoint."""
    predicted_price: float = Field(..., description="Predicted imbalance price in EUR/MWh")
    unit: str = "EUR/MWh"
    prediction_for: datetime = Field(..., description="Timestamp the prediction is for (next 15-min interval)")
    data_timestamp: datetime = Field(..., description="Timestamp of latest input data used")
    model_version: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 45.67,
                "unit": "EUR/MWh",
                "prediction_for": "2025-01-15T14:00:00Z",
                "data_timestamp": "2025-01-15T13:45:00Z",
                "model_version": "2025-01-10T12:00:00"
            }
        }


class FeatureInput(BaseModel):
    """Input features for custom prediction (optional endpoint)."""
    wind_power_mw: float = Field(..., description="Current wind power generation in MW")
    mfrr_price: float = Field(..., description="Current mFRR activation price")
    price_lag_1h: float = Field(..., description="Price 1 hour ago")
    price_lag_24h: float = Field(..., description="Price 24 hours ago")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    # Additional features can be added as needed


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions with custom features."""
    features: List[FeatureInput]


class ModelReloadResponse(BaseModel):
    """Response for model reload endpoint."""
    success: bool
    message: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
