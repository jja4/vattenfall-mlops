"""
DLT Source for Fingrid API data.

This module defines a DLT source that fetches electricity market data
from the Fingrid API incrementally using timestamp-based cursors.

Datasets:
- Wind Power (#181): Wind generation in MW
- mFRR Activation (#342): mFRR activation price in EUR/MWh
- Imbalance Price (#319): Imbalance price in EUR/MWh (target variable)

Usage:
    import dlt
    from ingestion.dlt_source import fingrid_source
    
    pipeline = dlt.pipeline(
        pipeline_name="fingrid",
        destination="filesystem",
        dataset_name="raw"
    )
    
    load_info = pipeline.run(fingrid_source())
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import dlt
from dlt.sources.helpers import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fingrid API configuration
FINGRID_BASE_URL = "https://data.fingrid.fi/api/datasets"
FINGRID_API_KEY = os.getenv("FINGRID_API_KEY")

# Dataset IDs
WIND_POWER_ID = 181
MFRR_ACTIVATION_ID = 342
IMBALANCE_PRICE_ID = 319

# Initial lookback for first ingestion
INITIAL_LOOKBACK_DAYS = 365

# API pagination settings
PAGE_SIZE = 20000  # Max allowed by Fingrid API


def _get_headers() -> dict:
    """Get API headers with authentication."""
    api_key = FINGRID_API_KEY
    if not api_key:
        raise ValueError("FINGRID_API_KEY environment variable not set")
    return {"x-api-key": api_key}


def _fetch_dataset_page(
    dataset_id: int,
    start_time: datetime,
    end_time: datetime,
    page: int = 1
) -> dict:
    """
    Fetch a single page of data from Fingrid API.
    
    Args:
        dataset_id: Fingrid dataset ID
        start_time: Start of time range (UTC)
        end_time: End of time range (UTC)
        page: Page number (1-indexed)
        
    Returns:
        API response as dict with 'data' and 'pagination' keys
    """
    url = f"{FINGRID_BASE_URL}/{dataset_id}/data"
    params = {
        "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json",
        "page": page,
        "pageSize": PAGE_SIZE,
        "sortOrder": "asc"
    }
    
    response = requests.get(url, headers=_get_headers(), params=params)
    response.raise_for_status()
    return response.json()


def _fetch_dataset_records(
    dataset_id: int,
    start_time: datetime,
    end_time: datetime
) -> Iterator[dict]:
    """
    Fetch all records from Fingrid API with pagination.
    
    Yields records one at a time for memory efficiency.
    
    Args:
        dataset_id: Fingrid dataset ID
        start_time: Start of time range (UTC)
        end_time: End of time range (UTC)
        
    Yields:
        Individual records as dicts
    """
    page = 1
    
    while True:
        result = _fetch_dataset_page(dataset_id, start_time, end_time, page)
        data = result.get("data", [])
        
        if not data:
            break
        
        for record in data:
            yield record
        
        # Check for more pages
        pagination = result.get("pagination", {})
        if pagination.get("nextPage") is None:
            break
        
        page += 1


@dlt.source(name="fingrid")
def fingrid_source(
    initial_lookback_days: int = INITIAL_LOOKBACK_DAYS
):
    """
    DLT source for Fingrid electricity market data.
    
    This source fetches three datasets incrementally:
    - Wind power generation
    - mFRR activation prices
    - Imbalance prices
    
    State is automatically managed by DLT using the timestamp column
    as the incremental cursor.
    
    Args:
        initial_lookback_days: Days to look back on first run (default: 365)
        
    Returns:
        DLT source with three resources
    """
    return [
        wind_power(initial_lookback_days),
        mfrr_activation(initial_lookback_days),
        imbalance_price(initial_lookback_days),
    ]


@dlt.resource(
    name="wind_power",
    write_disposition="merge",
    primary_key="timestamp",
    columns={"timestamp": {"data_type": "timestamp", "nullable": False}},
)
def wind_power(
    initial_lookback_days: int = INITIAL_LOOKBACK_DAYS,
    last_timestamp: dlt.sources.incremental[datetime] = dlt.sources.incremental(
        "timestamp",
        primary_key=(),
    ),
) -> Iterator[dict]:
    """
    Fetch wind power generation data from Fingrid.
    
    Dataset ID: 181
    Resolution: 3 minutes (aggregated to 15-min by processor)
    Unit: MW
    
    Args:
        initial_lookback_days: Days to look back on first run
        last_timestamp: DLT incremental cursor (auto-managed)
        
    Yields:
        Records with timestamp and wind_power_mw
    """
    end_time = datetime.now(timezone.utc)
    
    # Determine start time: use DLT state or initial lookback
    if last_timestamp.start_value is not None:
        start_time = last_timestamp.start_value - timedelta(minutes=15)
        print(f"🌬️  Incremental fetch wind power: {start_time} to {end_time}")
    else:
        start_time = end_time - timedelta(days=initial_lookback_days)
        print(f"🌬️  Initial fetch wind power ({initial_lookback_days} days): {start_time} to {end_time}")
    
    record_count = 0
    for record in _fetch_dataset_records(WIND_POWER_ID, start_time, end_time):
        record_count += 1
        yield {
            "timestamp": datetime.fromisoformat(record["endTime"].replace("Z", "+00:00")),
            "wind_power_mw": float(record["value"]),
        }
    
    print(f"   ✓ Fetched {record_count:,} wind power records")


@dlt.resource(
    name="mfrr_activation",
    write_disposition="merge",
    primary_key="timestamp",
    columns={"timestamp": {"data_type": "timestamp", "nullable": False}},
)
def mfrr_activation(
    initial_lookback_days: int = INITIAL_LOOKBACK_DAYS,
    last_timestamp: dlt.sources.incremental[datetime] = dlt.sources.incremental(
        "timestamp",
        primary_key=(),
    ),
) -> Iterator[dict]:
    """
    Fetch mFRR activation price data from Fingrid.
    
    Dataset ID: 342
    Resolution: 15 minutes
    Unit: EUR/MWh
    
    Args:
        initial_lookback_days: Days to look back on first run
        last_timestamp: DLT incremental cursor (auto-managed)
        
    Yields:
        Records with timestamp and mfrr_price
    """
    end_time = datetime.now(timezone.utc)
    
    if last_timestamp.start_value is not None:
        start_time = last_timestamp.start_value - timedelta(minutes=15)
        print(f"⚡ Incremental fetch mFRR: {start_time} to {end_time}")
    else:
        start_time = end_time - timedelta(days=initial_lookback_days)
        print(f"⚡ Initial fetch mFRR ({initial_lookback_days} days): {start_time} to {end_time}")
    
    record_count = 0
    for record in _fetch_dataset_records(MFRR_ACTIVATION_ID, start_time, end_time):
        record_count += 1
        yield {
            "timestamp": datetime.fromisoformat(record["endTime"].replace("Z", "+00:00")),
            "mfrr_price": float(record["value"]),
        }
    
    print(f"   ✓ Fetched {record_count:,} mFRR records")


@dlt.resource(
    name="imbalance_price",
    write_disposition="merge",
    primary_key="timestamp",
    columns={"timestamp": {"data_type": "timestamp", "nullable": False}},
)
def imbalance_price(
    initial_lookback_days: int = INITIAL_LOOKBACK_DAYS,
    last_timestamp: dlt.sources.incremental[datetime] = dlt.sources.incremental(
        "timestamp",
        primary_key=(),
    ),
) -> Iterator[dict]:
    """
    Fetch imbalance price data from Fingrid.
    
    This is the TARGET VARIABLE for prediction.
    
    Dataset ID: 319
    Resolution: 15 minutes
    Unit: EUR/MWh
    
    Args:
        initial_lookback_days: Days to look back on first run
        last_timestamp: DLT incremental cursor (auto-managed)
        
    Yields:
        Records with timestamp and imbalance_price
    """
    end_time = datetime.now(timezone.utc)
    
    if last_timestamp.start_value is not None:
        start_time = last_timestamp.start_value - timedelta(minutes=15)
        print(f"💰 Incremental fetch imbalance price: {start_time} to {end_time}")
    else:
        start_time = end_time - timedelta(days=initial_lookback_days)
        print(f"💰 Initial fetch imbalance price ({initial_lookback_days} days): {start_time} to {end_time}")
    
    record_count = 0
    for record in _fetch_dataset_records(IMBALANCE_PRICE_ID, start_time, end_time):
        record_count += 1
        yield {
            "timestamp": datetime.fromisoformat(record["endTime"].replace("Z", "+00:00")),
            "imbalance_price": float(record["value"]),
        }
    
    print(f"   ✓ Fetched {record_count:,} imbalance price records")


if __name__ == "__main__":
    # Test the source locally
    print("Testing Fingrid DLT source...")
    
    for resource in fingrid_source(initial_lookback_days=7):
        print(f"\nResource: {resource.name}")
        records = list(resource)
        print(f"  Records: {len(records)}")
        if records:
            print(f"  First: {records[0]}")
            print(f"  Last: {records[-1]}")
