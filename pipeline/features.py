"""
Feature generation pipeline for ML training.

This module reads raw data from DLT output (Azure Blob or local),
applies feature engineering, and writes partitioned feature datasets.

Features:
- Reads parquet data from DLT destination
- Applies all feature engineering from processor.py
- Writes date-partitioned features for efficient training
- Idempotent execution (can re-run safely)

Usage:
    python -m pipeline.features
    
Environment Variables:
    AZURE_STORAGE_CONNECTION_STRING: Azure Blob connection string
"""
import os
import sys
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Azure Storage configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Local paths for development
LOCAL_DLT_OUTPUT = Path(__file__).parent.parent / "data" / "dlt_output"
LOCAL_FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

# Column mappings from DLT output to processor expectations
DLT_TO_PROCESSOR_COLUMNS = {
    "wind_power": {"value_col": "wind_power_mw"},
    "mfrr_activation": {"value_col": "mfrr_price"},
    "imbalance_price": {"value_col": "imbalance_price"},
}


def get_storage_options() -> Optional[dict]:
    """Get storage options for pandas Azure Blob access."""
    if AZURE_CONNECTION_STRING:
        return {"connection_string": AZURE_CONNECTION_STRING}
    return None


def read_dlt_table(table_name: str) -> pd.DataFrame:
    """
    Read a DLT output table from Azure Blob or local filesystem.
    
    Args:
        table_name: Name of the DLT resource (wind_power, mfrr_activation, imbalance_price)
        
    Returns:
        DataFrame with timestamp and value columns
    """
    if AZURE_CONNECTION_STRING:
        # Read from Azure Blob Storage using adlfs
        from adlfs import AzureBlobFileSystem
        
        fs = AzureBlobFileSystem(connection_string=AZURE_CONNECTION_STRING)
        
        # List all parquet files in the table directory
        blob_path = f"raw/fingrid/{table_name}"
        logger.info(f"Reading {table_name} from Azure Blob: az://{blob_path}/*.parquet")
        
        parquet_files = [f for f in fs.ls(blob_path) if f.endswith('.parquet')]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {blob_path}")
        
        # Read all parquet files
        dfs = []
        for pq_file in parquet_files:
            with fs.open(pq_file, 'rb') as f:
                df = pd.read_parquet(f)
                dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        # Read from local filesystem
        path = LOCAL_DLT_OUTPUT / "fingrid" / table_name
        if not path.exists():
            raise FileNotFoundError(f"DLT output not found: {path}")
        logger.info(f"Reading {table_name} from local: {path}")
        df = pd.read_parquet(path)
    
    # Deduplicate by timestamp (keeps latest value for each timestamp)
    original_len = len(df)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    elif 'start_time' in df.columns:
        df = df.sort_values('start_time').drop_duplicates(subset=['start_time'], keep='last')
    
    if len(df) < original_len:
        logger.info(f"  Deduplicated: {original_len:,} -> {len(df):,} rows")
    else:
        logger.info(f"  Loaded {len(df):,} rows")
    
    return df


def write_features(df: pd.DataFrame, partition_col: str = "date"):
    """
    Write feature DataFrame to storage, partitioned by date.
    
    Handles idempotent writes - deduplicates by timestamp before writing.
    
    Args:
        df: Feature DataFrame with timestamp column
        partition_col: Column name to use for partitioning (will be created from timestamp)
    """
    df = df.copy()
    
    # Deduplicate by timestamp before writing
    original_len = len(df)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    if len(df) < original_len:
        logger.info(f"  Deduplicated features: {original_len:,} -> {len(df):,} rows")
    
    # Create partition column
    df[partition_col] = df["timestamp"].dt.date.astype(str)
    
    if AZURE_CONNECTION_STRING:
        # Write to Azure Blob Storage
        # Write as partitioned parquet to: az://features/data/
        base_path = "az://features/data"
        logger.info(f"Writing features to Azure Blob: {base_path}")
        
        # Write partitioned parquet using pyarrow
        import pyarrow as pa
        import pyarrow.parquet as pq
        from adlfs import AzureBlobFileSystem
        
        fs = AzureBlobFileSystem(connection_string=AZURE_CONNECTION_STRING)
        
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path="features/data",
            partition_cols=[partition_col],
            filesystem=fs,
            existing_data_behavior="overwrite_or_ignore",
        )
    else:
        # Write to local filesystem
        LOCAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = LOCAL_FEATURES_DIR / "features.parquet"
        logger.info(f"Writing features to local: {output_path}")
        df.to_parquet(output_path, index=False)
    
    logger.info(f"  Wrote {len(df):,} feature rows")


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the dataset for reproducibility tracking.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        SHA256 hash of the DataFrame
    """
    # Hash based on timestamp range and row count
    # This is fast and deterministic
    content = f"{df['timestamp'].min()}|{df['timestamp'].max()}|{len(df)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_features() -> pd.DataFrame:
    """
    Load feature dataset from storage.
    
    Returns:
        Feature DataFrame ready for training
    """
    if AZURE_CONNECTION_STRING:
        # Read from Azure Blob Storage using adlfs
        from adlfs import AzureBlobFileSystem
        
        fs = AzureBlobFileSystem(connection_string=AZURE_CONNECTION_STRING)
        
        # List all parquet files in the features directory
        base_path = "features/data"
        logger.info(f"Loading features from Azure Blob: az://{base_path}/**/*.parquet")
        
        # Recursively list all parquet files
        parquet_files = []
        try:
            for item in fs.ls(base_path):
                if fs.isdir(item):
                    # Check partition subdirectories (date=YYYY-MM-DD)
                    for subitem in fs.ls(item):
                        if subitem.endswith('.parquet'):
                            parquet_files.append(subitem)
                elif item.endswith('.parquet'):
                    parquet_files.append(item)
        except FileNotFoundError:
            raise FileNotFoundError(f"No feature data found in {base_path}")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {base_path}")
        
        # Read all parquet files
        dfs = []
        for pq_file in parquet_files:
            with fs.open(pq_file, 'rb') as f:
                df = pd.read_parquet(f)
                dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    else:
        path = LOCAL_FEATURES_DIR / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Features not found: {path}. Run feature generation first.")
        logger.info(f"Loading features from local: {path}")
        df = pd.read_parquet(path)
    
    # Drop partition column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    
    # Deduplicate by timestamp (in case of overlapping feature writes)
    original_len = len(df)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    if len(df) < original_len:
        logger.info(f"Loaded {len(df):,} feature rows (deduplicated from {original_len:,})")
    else:
        logger.info(f"Loaded {len(df):,} feature rows")
    return df


def run_feature_generation() -> dict:
    """
    Run the complete feature generation pipeline.
    
    Returns:
        Dict with generation statistics including dataset hash
    """
    logger.info("=" * 60)
    logger.info("🔧 Starting Feature Generation")
    logger.info("=" * 60)
    
    # Load raw data from DLT output
    logger.info("\n📥 Loading raw data from DLT output...")
    
    wind_df = read_dlt_table("wind_power")
    mfrr_df = read_dlt_table("mfrr_activation")
    price_df = read_dlt_table("imbalance_price")
    
    # Ensure timestamp is datetime
    for df in [wind_df, mfrr_df, price_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Resample to 15-minute intervals
    logger.info("\n⏱️  Resampling to 15-minute intervals...")
    wind_df = resample_to_15min(wind_df, "wind_power_mw", method="mean")
    mfrr_df = resample_to_15min(mfrr_df, "mfrr_price", method="ffill")
    price_df = resample_to_15min(price_df, "imbalance_price", method="ffill")
    
    logger.info(f"  Wind: {len(wind_df):,} rows")
    logger.info(f"  mFRR: {len(mfrr_df):,} rows")
    logger.info(f"  Price: {len(price_df):,} rows")
    
    # Merge datasets
    logger.info("\n🔗 Merging datasets...")
    df = merge_datasets(wind_df, mfrr_df, price_df)
    logger.info(f"  Merged: {len(df):,} rows")
    
    # Feature engineering
    logger.info("\n🔧 Creating features...")
    df = create_lag_features(df)
    logger.info("  ✓ Lag features")
    
    df = create_rolling_features(df)
    logger.info("  ✓ Rolling features")
    
    df = create_temporal_features(df)
    logger.info("  ✓ Temporal features")
    
    # Drop NaN rows from lag/rolling operations
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    logger.info(f"\n🧹 Dropped {dropped:,} rows with NaN")
    
    # Compute dataset hash
    dataset_hash = compute_dataset_hash(df)
    logger.info(f"\n🔑 Dataset hash: {dataset_hash}")
    
    # Write features
    logger.info("\n💾 Writing features...")
    write_features(df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Feature Generation Complete")
    logger.info("=" * 60)
    
    results = {
        "row_count": len(df),
        "feature_count": len(df.columns),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "dataset_hash": dataset_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    logger.info(f"  Rows: {results['row_count']:,}")
    logger.info(f"  Features: {results['feature_count']}")
    logger.info(f"  Time range: {results['timestamp_min']} to {results['timestamp_max']}")
    
    return results


def main():
    """Entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate features from DLT output")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_feature_generation()
        return 0
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
