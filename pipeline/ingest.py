"""
DLT Pipeline runner for incremental data ingestion.

This module configures and runs the DLT pipeline to fetch data from
Fingrid API and store it in Azure Blob Storage as parquet files.

Features:
- Incremental ingestion using DLT state management
- Azure Blob Storage as destination (filesystem destination)
- Automatic state persistence in blob storage
- Resume-safe execution

Usage:
    # From command line
    python -m pipeline.ingest
    
    # From GitHub Actions
    - uses: actions/checkout@v4
    - run: uv run python -m pipeline.ingest

Environment Variables:
    AZURE_STORAGE_CONNECTION_STRING: Azure Blob connection string
    FINGRID_API_KEY: Fingrid API authentication key
"""
import os
import sys
import logging
from pathlib import Path

import dlt
from dlt.destinations import filesystem
from dotenv import load_dotenv

from ingestion.dlt_source import fingrid_source

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
STORAGE_CONTAINER = "raw"  # Container name in Azure Blob

# Local fallback for development
LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "dlt_output"


def parse_connection_string(conn_str: str) -> dict:
    """Parse Azure Storage connection string into components."""
    parts = {}
    for part in conn_str.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            parts[key] = value
    return parts


def get_destination():
    """
    Get the DLT destination based on environment.
    
    If AZURE_STORAGE_CONNECTION_STRING is set, uses Azure Blob Storage.
    Otherwise, falls back to local filesystem for development.
    
    Returns:
        DLT filesystem destination
    """
    if AZURE_CONNECTION_STRING:
        logger.info("Using Azure Blob Storage destination")
        
        # Parse connection string to extract account name and key
        conn_parts = parse_connection_string(AZURE_CONNECTION_STRING)
        account_name = conn_parts.get("AccountName")
        account_key = conn_parts.get("AccountKey")
        
        if not account_name or not account_key:
            raise ValueError("Invalid AZURE_STORAGE_CONNECTION_STRING: missing AccountName or AccountKey")
        
        logger.info(f"Using Azure Storage Account: {account_name}")
        
        # Azure Blob Storage via filesystem destination
        # Format: az://<container>/<path>
        return filesystem(
            bucket_url=f"az://{STORAGE_CONTAINER}",
            credentials={
                "azure_storage_account_name": account_name,
                "azure_storage_account_key": account_key,
            },
            layout="{table_name}/{load_id}.{file_id}.parquet",
        )
    else:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING not set, using local filesystem")
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return filesystem(
            bucket_url=str(LOCAL_DATA_DIR),
            layout="{table_name}/{load_id}.{file_id}.parquet",
        )


def create_pipeline() -> dlt.Pipeline:
    """
    Create and configure the DLT pipeline.
    
    The pipeline:
    - Fetches data incrementally from Fingrid API
    - Writes parquet files to Azure Blob Storage (or local)
    - Stores state in the same destination for resume capability
    
    Returns:
        Configured DLT pipeline
    """
    destination = get_destination()
    
    pipeline = dlt.pipeline(
        pipeline_name="fingrid_ingestion",
        destination=destination,
        dataset_name="fingrid",  # Creates fingrid/ subfolder
        progress="log",  # Show progress in logs
    )
    
    return pipeline


def run_ingestion(initial_lookback_days: int = 365) -> dict:
    """
    Run the incremental ingestion pipeline.
    
    Args:
        initial_lookback_days: Days to look back on first run
        
    Returns:
        Dict with ingestion statistics
    """
    logger.info("=" * 60)
    logger.info("🔄 Starting Incremental Data Ingestion")
    logger.info("=" * 60)
    logger.info(f"Initial lookback: {initial_lookback_days} days")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Show current state
    try:
        state = pipeline.state
        sources = state.get('sources', {})
        if sources:
            logger.info(f"Pipeline state sources: {list(sources.keys())}")
        else:
            logger.info(f"🎯 First run detected - will fetch {initial_lookback_days} days of history")
    except Exception:
        logger.info(f"🎯 No existing pipeline state - will fetch {initial_lookback_days} days")
    
    # Create source
    source = fingrid_source(initial_lookback_days=initial_lookback_days)
    
    # Run pipeline
    logger.info("-" * 60)
    logger.info("Running DLT pipeline...")
    logger.info("-" * 60)
    
    # Use parquet format for efficient storage and reading
    load_info = pipeline.run(source, loader_file_format="parquet")
    
    # Extract results
    logger.info("=" * 60)
    logger.info("✅ Ingestion Complete")
    logger.info("=" * 60)
    
    # Log load info
    logger.info(f"Load IDs: {load_info.loads_ids}")
    logger.info(f"Destination: {load_info.destination_name}")
    logger.info(f"Dataset: {load_info.dataset_name}")
    
    # Count loaded rows per table
    results = {}
    for package in load_info.load_packages:
        # DLT 1.x uses .schema_update for table info
        if hasattr(package, 'schema_update'):
            for table_name, table_info in package.schema_update.items():
                if isinstance(table_info, dict):
                    row_count = table_info.get("row_count", 0)
                    if row_count > 0:
                        results[table_name] = row_count
                        logger.info(f"  {table_name}: {row_count:,} rows loaded")
    
    # Show any failed jobs
    if load_info.has_failed_jobs:
        logger.error("Some jobs failed:")
        for package in load_info.load_packages:
            for job in getattr(package, "failed_jobs", []):
                logger.error(f"  - {job}")
    
    return results


def main():
    """Entry point for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Fingrid data ingestion pipeline")
    parser.add_argument(
        "--lookback",
        type=int,
        default=365,
        help="Initial lookback days for first run (default: 365)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = run_ingestion(initial_lookback_days=args.lookback)
        
        # Exit with success
        total_rows = sum(results.values())
        logger.info(f"\nTotal rows loaded: {total_rows:,}")
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
