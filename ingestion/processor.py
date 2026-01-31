import pandas as pd
from .storage import load_parquet, save_processed, load_processed

# =============================================================================
# Constants for time intervals (15-minute resolution)
# =============================================================================
PERIODS_PER_HOUR = 4  # 60 min / 15 min = 4 periods per hour

# Lag feature periods
LAG_1H = PERIODS_PER_HOUR * 1   # 4 periods
LAG_2H = PERIODS_PER_HOUR * 2   # 8 periods
LAG_3H = PERIODS_PER_HOUR * 3   # 12 periods

# Rolling window periods
ROLLING_1H = PERIODS_PER_HOUR * 1   # 4 periods
ROLLING_3H = PERIODS_PER_HOUR * 3   # 12 periods

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


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for temporal patterns.
    Uses 1h, 2h, and 3h lags (corresponding to LAG_1H, LAG_2H, LAG_3H periods).
    
    Args:
        df: DataFrame with timestamp and value columns
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    # 1-hour lag
    df["price_lag_1h"] = df["imbalance_price"].shift(LAG_1H)
    df["wind_lag_1h"] = df["wind_power_mw"].shift(LAG_1H)
    df["mfrr_lag_1h"] = df["mfrr_price"].shift(LAG_1H)
    
    # 2-hour lag
    df["price_lag_2h"] = df["imbalance_price"].shift(LAG_2H)
    df["wind_lag_2h"] = df["wind_power_mw"].shift(LAG_2H)
    
    # 3-hour lag
    df["price_lag_3h"] = df["imbalance_price"].shift(LAG_3H)
    
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling mean features to capture trends.
    Uses 1-hour and 3-hour windows, shifted to avoid data leakage.
    
    Args:
        df: DataFrame with timestamp and value columns
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    # 1-hour rolling mean, shifted by 1 to avoid leakage
    df["price_rolling_1h"] = df["imbalance_price"].shift(1).rolling(window=ROLLING_1H).mean()
    df["wind_rolling_1h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_1H).mean()
    
    # 3-hour rolling mean
    df["price_rolling_3h"] = df["imbalance_price"].shift(1).rolling(window=ROLLING_3H).mean()
    df["wind_rolling_3h"] = df["wind_power_mw"].shift(1).rolling(window=ROLLING_3H).mean()
    
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features to capture diurnal patterns.
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Hour of day (0-23)
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Is weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    return df


def process_features(df: pd.DataFrame = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    If df is None, loads raw data from storage, otherwise processes provided DataFrame.
    Uses cache by default to avoid reprocessing.
    
    Steps:
    1. Check cache (if use_cache=True)
    2. Load raw data
    3. Validate data
    4. Resample to 15-minute intervals
    5. Merge datasets
    6. Create lag features
    7. Create rolling features
    8. Create temporal features
    9. Drop rows with NaN (from lag/rolling operations)
    10. Cache result
    
    Args:
        df: Optional pre-loaded DataFrame to process
        use_cache: If True, use cached processed data if available
    
    Returns:
        DataFrame ready for model training
    """
    # Check cache first
    if df is None and use_cache:
        cached = load_processed("features")
        if cached is not None:
            print("📦 Using cached processed data")
            print(f"  Shape: {cached.shape}")
            return cached
    
    if df is None:
        print("📥 Loading raw data...")
        wind_df = load_parquet("wind")
        mfrr_df = load_parquet("mfrr")
        price_df = load_parquet("price")
        
        if wind_df is None or mfrr_df is None or price_df is None:
            raise ValueError("Missing raw data. Run ingestion pipeline first.")
        
        # Validate raw data
        validate_dataframe(wind_df, REQUIRED_RAW_COLUMNS["wind"], "Wind data")
        validate_dataframe(mfrr_df, REQUIRED_RAW_COLUMNS["mfrr"], "mFRR data")
        validate_dataframe(price_df, REQUIRED_RAW_COLUMNS["price"], "Price data")
        
        print(f"  Wind: {len(wind_df):,} rows ✓")
        print(f"  mFRR: {len(mfrr_df):,} rows ✓")
        print(f"  Price: {len(price_df):,} rows ✓")
        
        # Resample to 15-minute intervals
        print("\n⏱️  Resampling to 15-minute intervals...")
        wind_df = resample_to_15min(wind_df, "wind_power_mw", method="mean")
        mfrr_df = resample_to_15min(mfrr_df, "mfrr_price", method="ffill")
        # Price is already at 15-min, but ensure consistency
        price_df = resample_to_15min(price_df, "imbalance_price", method="ffill")
        
        print(f"  Wind: {len(wind_df):,} rows")
        print(f"  mFRR: {len(mfrr_df):,} rows")
        print(f"  Price: {len(price_df):,} rows")
        
        # Merge datasets
        print("\n🔗 Merging datasets...")
        df = merge_datasets(wind_df, mfrr_df, price_df)
        print(f"  Merged: {len(df):,} rows")
        
        # Validate merged data
        validate_dataframe(df, REQUIRED_MERGED_COLUMNS, "Merged data")
    
    # Feature engineering
    print("\n🔧 Creating features...")
    df = create_lag_features(df)
    print("  ✓ Lag features (1h, 2h, 3h)")
    
    df = create_rolling_features(df)
    print("  ✓ Rolling means (1h, 3h)")
    
    df = create_temporal_features(df)
    print("  ✓ Temporal features (hour, day, weekend)")
    
    # Drop rows with NaN from lag/rolling operations
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"\n🧹 Dropped {initial_len - len(df):,} rows with NaN (from lag/rolling)")
    print(f"  Final dataset: {len(df):,} rows with {len(df.columns)} features")
    
    # Cache the processed data
    if use_cache:
        save_processed(df, "features")
    
    return df


if __name__ == "__main__":
    # Test the processor
    print("="*60)
    print("Testing Feature Processing Pipeline")
    print("="*60)
    
    processed_df = process_features()
    
    print(f"\n📊 Processed Data Summary:")
    print(f"  Shape: {processed_df.shape}")
    print(f"\n  Columns: {list(processed_df.columns)}")
    print(f"\n  Time range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")
    print(f"\n  Sample rows:")
    print(processed_df.head())
    
    print(f"\n✅ Processing complete!")
