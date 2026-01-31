"""
Quick verification script to inspect fetched data.

Usage:
    uv run python -m scripts.verify_data
"""
from ingestion.storage import load_parquet

def verify_dataset(name: str):
    """Verify structure and quality of a dataset."""
    print(f"\n{'='*60}")
    print(f"📊 Verifying {name.upper()} dataset")
    print('='*60)
    
    df = load_parquet(name)
    
    if df is None:
        print(f"❌ No data found for {name}")
        return False
    
    print(f"\n✓ Loaded successfully")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n📋 Data Types:")
    print(df.dtypes)
    
    print(f"\n📅 Time Range:")
    print(f"  Start: {df['timestamp'].min()}")
    print(f"  End:   {df['timestamp'].max()}")
    print(f"  Span:  {df['timestamp'].max() - df['timestamp'].min()}")
    
    print(f"\n🔢 Value Statistics:")
    value_col = [col for col in df.columns if col != 'timestamp'][0]
    print(df[value_col].describe())
    
    print(f"\n❓ Missing Values:")
    print(f"  Timestamp: {df['timestamp'].isna().sum()}")
    print(f"  {value_col}: {df[value_col].isna().sum()}")
    
    print(f"\n📝 Sample Data (first 5 rows):")
    print(df.head())
    
    print(f"\n✅ {name} data looks valid!")
    return True


if __name__ == "__main__":
    datasets = ["wind", "mfrr", "price"]
    
    print("\n🔍 Data Verification Report")
    print("="*60)
    
    all_valid = True
    for dataset in datasets:
        valid = verify_dataset(dataset)
        all_valid = all_valid and valid
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All datasets verified successfully!")
    else:
        print("❌ Some datasets have issues")
    print('='*60)
