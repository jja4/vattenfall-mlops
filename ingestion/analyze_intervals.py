"""
Analyze the temporal patterns of each dataset.
"""
import pandas as pd
from ingestion.storage import load_parquet

def analyze_intervals(df: pd.DataFrame, name: str):
    """Analyze time intervals between consecutive records."""
    print(f"\n{'='*60}")
    print(f"📊 Analyzing {name.upper()} temporal patterns")
    print('='*60)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate intervals between consecutive timestamps
    df["time_diff"] = df["timestamp"].diff()
    
    print(f"\n⏱️  Interval Statistics:")
    print(df["time_diff"].describe())
    
    print(f"\n📈 Most common intervals:")
    interval_counts = df["time_diff"].value_counts().head(10)
    for interval, count in interval_counts.items():
        if pd.notna(interval):
            print(f"  {interval} : {count:,} occurrences ({count/len(df)*100:.1f}%)")
    
    # Check for gaps
    print(f"\n🕳️  Large gaps (> 1 hour):")
    large_gaps = df[df["time_diff"] > pd.Timedelta(hours=1)]
    if len(large_gaps) > 0:
        print(f"  Found {len(large_gaps)} gaps")
        for _, row in large_gaps.head().iterrows():
            print(f"    {row['timestamp']} - Gap: {row['time_diff']}")
    else:
        print("  No large gaps found")
    
    return df["time_diff"]


def check_data_source_consistency(df: pd.DataFrame, name: str):
    """Check if there are any anomalies suggesting multiple sources."""
    print(f"\n🔍 Data source consistency check for {name}:")
    
    value_col = [col for col in df.columns if col != 'timestamp'][0]
    
    # Check for sudden jumps that might indicate source switches
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    df_sorted["value_diff"] = df_sorted[value_col].diff().abs()
    
    print(f"  Value change statistics:")
    print(f"    Mean change: {df_sorted['value_diff'].mean():.2f}")
    print(f"    Max change: {df_sorted['value_diff'].max():.2f}")
    print(f"    95th percentile: {df_sorted['value_diff'].quantile(0.95):.2f}")
    
    # Look for extreme jumps
    threshold = df_sorted['value_diff'].quantile(0.999)
    extreme_jumps = df_sorted[df_sorted['value_diff'] > threshold]
    print(f"  Extreme jumps (>99.9th percentile): {len(extreme_jumps)}")
    
    return df_sorted


if __name__ == "__main__":
    print("="*60)
    print("🔬 Temporal Pattern Analysis")
    print("="*60)
    
    # Load datasets
    wind_df = load_parquet("wind")
    mfrr_df = load_parquet("mfrr")
    price_df = load_parquet("price")
    
    # Analyze each dataset
    wind_intervals = analyze_intervals(wind_df, "wind")
    mfrr_intervals = analyze_intervals(mfrr_df, "mfrr")
    price_intervals = analyze_intervals(price_df, "price")
    
    # Check consistency
    print(f"\n{'='*60}")
    print("🔍 Data Source Consistency")
    print('='*60)
    
    wind_df_checked = check_data_source_consistency(wind_df, "wind")
    mfrr_df_checked = check_data_source_consistency(mfrr_df, "mfrr")
    price_df_checked = check_data_source_consistency(price_df, "price")
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print('='*60)
