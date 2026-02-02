"""
Feature engineering transformations for imbalance price prediction.

This module contains PURE TRANSFORMATION FUNCTIONS only - no I/O.
All data loading/saving is handled by:
- pipeline/features.py (batch processing from Azure Blob)
- app/main.py (real-time processing from Fingrid API)

Functions:
- resample_to_15min(): Resample time-series to 15-minute intervals
- merge_datasets(): Merge wind, mFRR, and price data
- merge_datasets_realtime(): Merge with lookback data for real-time predictions
- create_lag_features(): Create lagged features
- create_rolling_features(): Create rolling statistics
- create_temporal_features(): Create time-based features
"""
import pandas as pd
import numpy as np

# =============================================================================
# Constants for time intervals (15-minute resolution)
# =============================================================================
PERIODS_PER_HOUR = 4  # 60 min / 15 min = 4 periods per hour

# Lag feature periods
LAG_1H = PERIODS_PER_HOUR * 1   # 4 periods
LAG_2H = PERIODS_PER_HOUR * 2   # 8 periods
LAG_3H = PERIODS_PER_HOUR * 3   # 12 periods
LAG_6H = PERIODS_PER_HOUR * 6   # 24 periods
LAG_12H = PERIODS_PER_HOUR * 12 # 48 periods
LAG_24H = PERIODS_PER_HOUR * 24 # 96 periods (same time yesterday)

# Rolling window periods
ROLLING_1H = PERIODS_PER_HOUR * 1   # 4 periods
ROLLING_3H = PERIODS_PER_HOUR * 3   # 12 periods
ROLLING_6H = PERIODS_PER_HOUR * 6   # 24 periods
ROLLING_24H = PERIODS_PER_HOUR * 24 # 96 periods

# Required columns for validation
REQUIRED_RAW_COLUMNS = {
    "wind": {"timestamp", "wind_power_mw"},
    "mfrr": {"timestamp", "mfrr_price"},
    "price": {"timestamp", "imbalance_price"},
}
REQUIRED_MERGED_COLUMNS = {"timestamp", "wind_power_mw", "mfrr_price", "imbalance_price"}


def validate_dataframe(df: pd.DataFrame, required_columns: set, name: str = "DataFrame"):
    """Validate that a DataFrame contains required columns."""
    if df is None:
        raise ValueError(f"{name} is None - data may not be loaded")
    
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    return True


def resample_to_15min(df: pd.DataFrame, value_col: str, method: str = "mean") -> pd.DataFrame:
    """
    Resample time-series data to 15-minute intervals.
    
    Args:
        df: DataFrame with 'timestamp' and value column
        value_col: Name of the value column to resample
        method: Aggregation method - 'mean' for downsampling, 'ffill' for upsampling
    
    Returns:
        DataFrame resampled to 15-minute intervals
    """
    df = df.copy()
    df = df.set_index("timestamp")
    
    if method == "mean":
        # Downsample high-frequency data by averaging
        df_resampled = df[[value_col]].resample("15min").mean()
    elif method == "ffill":
        # Upsample low-frequency data by forward-filling
        df_resampled = df[[value_col]].resample("15min").ffill()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Remove any remaining NaN values at the start using bfill
    df_resampled = df_resampled.bfill()
    
    df_resampled = df_resampled.reset_index()
    return df_resampled


def merge_datasets(wind_df: pd.DataFrame, mfrr_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all three datasets on timestamp using inner join.
    
    Args:
        wind_df: Wind power data (resampled to 15min)
        mfrr_df: mFRR activation data (resampled to 15min)
        price_df: Imbalance price data (native 15min)
    
    Returns:
        Merged DataFrame with all features
    """
    # Merge wind and mfrr
    merged = pd.merge(wind_df, mfrr_df, on="timestamp", how="inner")
    
    # Merge with price (target variable)
    merged = pd.merge(merged, price_df, on="timestamp", how="inner")
    
    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    return merged


def merge_datasets_realtime(wind_df: pd.DataFrame, mfrr_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge datasets for real-time prediction, keeping rows where wind/mFRR exist
    even if price is not yet available.
    
    Uses left join to preserve wind/mFRR timestamps that don't have price data yet.
    This allows predicting for timestamps where price hasn't been published.
    
    Args:
        wind_df: Wind power data (resampled to 15min)
        mfrr_df: mFRR activation data (resampled to 15min)
        price_df: Imbalance price data (native 15min)
    
    Returns:
        Merged DataFrame with NaN for missing price values
    """
    # Merge wind and mfrr (inner join - both should be available for prediction)
    merged = pd.merge(wind_df, mfrr_df, on="timestamp", how="inner")
    
    # Merge with price using LEFT join - keep rows even without price
    merged = pd.merge(merged, price_df, on="timestamp", how="left")
    
    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    
    return merged


def create_lag_features(df: pd.DataFrame, for_realtime: bool = False) -> pd.DataFrame:
    """
    Create lag features for temporal patterns.
    Uses multiple lag horizons to capture short and long-term dependencies.
    
    Args:
        df: DataFrame with timestamp and value columns
        for_realtime: If True, forward-fill NaN prices before computing lags
                      (allows prediction for timestamps without known price)
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    # For real-time prediction, use the last known price for lag calculations
    # This allows predicting for timestamps where price isn't available yet
    price_col = df["imbalance_price"]
    if for_realtime:
        price_col = price_col.ffill()
    
    # Price lags - most important for prediction
    df["price_lag_1h"] = price_col.shift(LAG_1H)
    df["price_lag_2h"] = price_col.shift(LAG_2H)
    df["price_lag_3h"] = price_col.shift(LAG_3H)
    df["price_lag_6h"] = price_col.shift(LAG_6H)
    df["price_lag_12h"] = price_col.shift(LAG_12H)
    df["price_lag_24h"] = price_col.shift(LAG_24H)  # Same time yesterday
    
    # Wind lags - key driver of price
    df["wind_lag_1h"] = df["wind_power_mw"].shift(LAG_1H)
    df["wind_lag_2h"] = df["wind_power_mw"].shift(LAG_2H)
    df["wind_lag_6h"] = df["wind_power_mw"].shift(LAG_6H)
    df["wind_lag_24h"] = df["wind_power_mw"].shift(LAG_24H)
    
    # mFRR lags
    df["mfrr_lag_1h"] = df["mfrr_price"].shift(LAG_1H)
    df["mfrr_lag_3h"] = df["mfrr_price"].shift(LAG_3H)
    
    # Price change features (momentum)
    df["price_change_1h"] = price_col.shift(LAG_1H) - price_col.shift(LAG_2H)
    df["price_change_3h"] = price_col.shift(LAG_1H) - price_col.shift(LAG_1H + LAG_3H)
    
    return df


def create_rolling_features(df: pd.DataFrame, for_realtime: bool = False) -> pd.DataFrame:
    """
    Create rolling statistics to capture trends and volatility.
    Uses multiple window sizes, all shifted by 1 to avoid data leakage.
    
    Args:
        df: DataFrame with timestamp and value columns
        for_realtime: If True, forward-fill NaN prices before computing rolling stats
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    # For real-time prediction, use the last known price for rolling calculations
    price_col = df["imbalance_price"]
    if for_realtime:
        price_col = price_col.ffill()
    
    # Rolling means - capture trends
    df["price_rolling_1h"] = price_col.shift(1).rolling(window=ROLLING_1H).mean()
    df["price_rolling_3h"] = price_col.shift(1).rolling(window=ROLLING_3H).mean()
    df["price_rolling_6h"] = price_col.shift(1).rolling(window=ROLLING_6H).mean()
    df["price_rolling_24h"] = price_col.shift(1).rolling(window=ROLLING_24H).mean()
    
    df["wind_rolling_1h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_1H).mean()
    df["wind_rolling_3h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_3H).mean()
    df["wind_rolling_6h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_6H).mean()
    
    # Rolling standard deviation - capture volatility
    df["price_volatility_3h"] = price_col.shift(1).rolling(window=ROLLING_3H).std()
    df["price_volatility_6h"] = price_col.shift(1).rolling(window=ROLLING_6H).std()
    df["wind_volatility_3h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_3H).std()
    
    # Rolling min/max - capture range
    df["price_min_6h"] = price_col.shift(1).rolling(window=ROLLING_6H).min()
    df["price_max_6h"] = price_col.shift(1).rolling(window=ROLLING_6H).max()
    df["price_range_6h"] = df["price_max_6h"] - df["price_min_6h"]
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features to capture diurnal and seasonal patterns.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Hour of day (0-23) - captures daily price patterns
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    # Cyclical encoding of hour (better for models)
    import numpy as np
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Cyclical encoding of day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Is weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Month (captures seasonal patterns)
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Peak hours (typically 8-20 in Finland)
    df["is_peak_hour"] = ((df["hour_of_day"] >= 8) & (df["hour_of_day"] <= 20)).astype(int)
    
    # Morning ramp (6-9), evening peak (17-20)
    df["is_morning_ramp"] = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] <= 9)).astype(int)
    df["is_evening_peak"] = ((df["hour_of_day"] >= 17) & (df["hour_of_day"] <= 20)).astype(int)
    
    return df
