import pandas as pd
from .storage import load_parquet


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
    Uses 1h, 2h, and 3h lags (corresponding to 4, 8, 12 periods at 15-min intervals).
    
    Args:
        df: DataFrame with timestamp and value columns
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    # 1-hour lag (4 periods * 15min)
    df["price_lag_1h"] = df["imbalance_price"].shift(4)
    df["wind_lag_1h"] = df["wind_power_mw"].shift(4)
    df["mfrr_lag_1h"] = df["mfrr_price"].shift(4)
    
    # 2-hour lag (8 periods)
    df["price_lag_2h"] = df["imbalance_price"].shift(8)
    df["wind_lag_2h"] = df["wind_power_mw"].shift(8)
    
    # 3-hour lag (12 periods)
    df["price_lag_3h"] = df["imbalance_price"].shift(12)
    
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
    
    # 1-hour rolling mean (4 periods * 15min), shifted by 1 to avoid leakage
    df["price_rolling_1h"] = df["imbalance_price"].shift(1).rolling(window=4).mean()
    df["wind_rolling_1h"] = df["wind_power_mw"].shift(1).rolling(window=4).mean()
    
    # 3-hour rolling mean (12 periods)
    df["price_rolling_3h"] = df["imbalance_price"].shift(1).rolling(window=12).mean()
    df["wind_rolling_3h"] = df["wind_power_mw"].shift(1).rolling(window=12).mean()
    
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


def process_features(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    If df is None, loads raw data from storage, otherwise processes provided DataFrame.
    
    Steps:
    1. Load raw data
    2. Resample to 15-minute intervals
    3. Merge datasets
    4. Create lag features
    5. Create rolling features
    6. Create temporal features
    7. Drop rows with NaN (from lag/rolling operations)
    
    Returns:
        DataFrame ready for model training
    """
    if df is None:
        print("📥 Loading raw data...")
        wind_df = load_parquet("wind")
        mfrr_df = load_parquet("mfrr")
        price_df = load_parquet("price")
        
        if wind_df is None or mfrr_df is None or price_df is None:
            raise ValueError("Missing raw data. Run ingestion pipeline first.")
        
        print(f"  Wind: {len(wind_df):,} rows")
        print(f"  mFRR: {len(mfrr_df):,} rows")
        print(f"  Price: {len(price_df):,} rows")
        
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
