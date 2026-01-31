import pandas as pd
from pathlib import Path

# Use absolute path based on project root to work from any directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def save_parquet(df: pd.DataFrame, name: str):
    """Save raw data to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / f"{name}.parquet")


def load_parquet(name: str) -> pd.DataFrame:
    """Load raw data from parquet."""
    path = DATA_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_processed(df: pd.DataFrame, name: str = "features"):
    """Save processed/feature-engineered data to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet")
    print(f"💾 Cached processed data: {PROCESSED_DIR / f'{name}.parquet'}")


def load_processed(name: str = "features") -> pd.DataFrame:
    """Load processed data from cache if available."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None
