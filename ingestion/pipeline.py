"""
Ingestion pipeline orchestrator.
Checks local cache -> fetches from Fingrid API if missing -> saves locally.
"""
import os
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from .client import FingridClient
from .storage import save_parquet, load_parquet

# Load environment variables
load_dotenv()


def fetch_and_cache_data(
    dataset_name: str,
    start_time: datetime,
    end_time: datetime,
    force_refresh: bool = False
) -> None:
    """
    Orchestrates data fetching: checks cache, fetches if needed, saves locally.
    
    Args:
        dataset_name: One of 'wind', 'mfrr', 'price'
        start_time: Start of time range (UTC)
        end_time: End of time range (UTC)
        force_refresh: If True, bypass cache and fetch fresh data
    """
    # Check cache first
    if not force_refresh:
        cached_df = load_parquet(dataset_name)
        if cached_df is not None:
            print(f"✓ Using cached data for {dataset_name}: {len(cached_df)} rows")
            return
    
    # Fetch from API
    api_key = os.getenv("FINGRID_API_KEY")
    if not api_key:
        raise ValueError("FINGRID_API_KEY not found in environment variables")
    
    client = FingridClient(api_key)
    
    print(f"⬇ Fetching {dataset_name} from Fingrid API...")
    
    if dataset_name == "wind":
        df = client.get_wind_power(start_time, end_time)
    elif dataset_name == "mfrr":
        df = client.get_mfrr_activation(start_time, end_time)
    elif dataset_name == "price":
        df = client.get_imbalance_price(start_time, end_time)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wind', 'mfrr', or 'price'")
    
    if df.empty:
        print(f"⚠ No data returned for {dataset_name}")
        return
    
    # Save to cache
    save_parquet(df, dataset_name)
    print(f"✓ Saved {dataset_name}: {len(df)} rows ({df['timestamp'].min()} to {df['timestamp'].max()})")


def fetch_all_datasets(
    start_time: datetime,
    end_time: datetime,
    force_refresh: bool = False
) -> None:
    """
    Fetch all three datasets (wind, mfrr, price) for the given time range.
    
    Args:
        start_time: Start of time range (UTC)
        end_time: End of time range (UTC)
        force_refresh: If True, bypass cache and fetch fresh data
    """
    datasets = ["wind", "mfrr", "price"]
    
    for dataset in datasets:
        try:
            fetch_and_cache_data(dataset, start_time, end_time, force_refresh)
        except Exception as e:
            print(f"✗ Error fetching {dataset}: {e}")
            raise


if __name__ == "__main__":
    # Fetch previous year of data (2025)
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    end_time = datetime(2026, 1, 1, 0, 0, 0)
    
    print(f"Fetching data from {start_time} to {end_time}")
    fetch_all_datasets(start_time, end_time, force_refresh=False)
    print("\n✓ Pipeline complete!")
