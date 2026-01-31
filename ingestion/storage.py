import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def save_parquet(df: pd.DataFrame, name: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / f"{name}.parquet")

def load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None
