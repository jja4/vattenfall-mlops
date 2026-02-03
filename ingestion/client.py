import os
import requests
import pandas as pd
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FingridClient:
    BASE_URL = "https://data.fingrid.fi/api/datasets"
    
    # Dataset IDs
    WIND_POWER_ID = 181
    MFRR_ACTIVATION_ID = 342
    IMBALANCE_PRICE_ID = 319
    
    def __init__(self, api_key: str = None):
        """
        Initialize Fingrid API client.
        
        Args:
            api_key: Fingrid API key. If not provided, reads from FINGRID_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FINGRID_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set FINGRID_API_KEY env var or pass api_key argument.")
        self.headers = {"x-api-key": self.api_key}

    def get_dataset(self, dataset_id: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetch data from Fingrid API with automatic pagination and date chunking.
        
        API has a hard limit on total records, so for large date ranges we chunk
        by month to avoid hitting limits.
        
        Args:
            dataset_id: Fingrid dataset ID
            start_time: Start of time range (UTC)
            end_time: End of time range (UTC)
            
        Returns:
            DataFrame with columns: timestamp, value
        """
        # Calculate if we need to chunk (more than ~30 days to be safe)
        time_delta = end_time - start_time
        needs_chunking = time_delta.days > 30
        
        if needs_chunking:
            print("  📅 Large date range detected, chunking by month...")
            return self._get_dataset_chunked(dataset_id, start_time, end_time)
        else:
            return self._get_dataset_single(dataset_id, start_time, end_time)
    
    def _get_dataset_single(self, dataset_id: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch data for a single time range (no chunking)."""
        all_data = []
        page = 1
        page_size = 20000  # Max allowed by API
        
        # Format times as ISO 8601 strings
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        while True:
            url = f"{self.BASE_URL}/{dataset_id}/data"
            params = {
                "startTime": start_str,
                "endTime": end_str,
                "format": "json",
                "page": page,
                "pageSize": page_size,
                "sortOrder": "asc"
            }
            
            # Retry logic for rate limiting
            max_retries = 5
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=self.headers, params=params)
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"  ⏳ Rate limited, waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            raise  # Max retries reached
                    else:
                        raise  # Other HTTP error
            
            result = response.json()
            data = result.get("data", [])
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Check if there are more pages
            pagination = result.get("pagination", {})
            if pagination.get("nextPage") is None:
                break
                
            page += 1
            # Small delay between pages to be nice to the API
            time.sleep(0.5)
        
        # Convert to DataFrame
        if not all_data:
            return pd.DataFrame(columns=["timestamp", "value"])
        
        df = pd.DataFrame(all_data)
        # Use endTime as the timestamp (represents the completion of the 15-min interval)
        df["timestamp"] = pd.to_datetime(df["endTime"])
        df = df[["timestamp", "value"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def _get_dataset_chunked(self, dataset_id: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch data by chunking into monthly periods to avoid API limits."""
        all_chunks = []
        
        current_start = start_time
        chunk_num = 0
        
        while current_start < end_time:
            # Calculate next chunk end (exactly 1 month forward using calendar months)
            current_end = min(
                current_start + relativedelta(months=1),
                end_time
            )
            
            chunk_num += 1
            print(f"    Chunk {chunk_num}: {current_start.date()} to {current_end.date()}")
            
            chunk_df = self._get_dataset_single(dataset_id, current_start, current_end)
            
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
            
            current_start = current_end
        
        # Combine all chunks
        if not all_chunks:
            return pd.DataFrame(columns=["timestamp", "value"])
        
        combined_df = pd.concat(all_chunks, ignore_index=True)
        # Remove any duplicates that might occur at chunk boundaries
        combined_df = combined_df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        
        return combined_df
    
    def get_wind_power(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch wind power generation data (MW)."""
        df = self.get_dataset(self.WIND_POWER_ID, start_time, end_time)
        df = df.rename(columns={"value": "wind_power_mw"})
        return df
    
    def get_mfrr_activation(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch mFRR activation price data (EUR/MWh)."""
        df = self.get_dataset(self.MFRR_ACTIVATION_ID, start_time, end_time)
        df = df.rename(columns={"value": "mfrr_price"})
        return df
    
    def get_imbalance_price(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch imbalance price data (EUR/MWh) - this is our target variable."""
        df = self.get_dataset(self.IMBALANCE_PRICE_ID, start_time, end_time)
        df = df.rename(columns={"value": "imbalance_price"})
        return df
