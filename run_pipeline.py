#!/usr/bin/env python
"""
Main pipeline entry point for the Vattenfall MLOps project.

This script orchestrates the full data pipeline:
  1. Ingestion: Fetch data from Fingrid API
  2. Processing: Resample, merge, and engineer features
  3. Training: Train and evaluate the model

Usage:
    uv run python run_pipeline.py                    # Full pipeline
    uv run python run_pipeline.py --skip-ingestion   # Skip API calls, use cached data
    uv run python run_pipeline.py --no-wandb         # Disable W&B logging
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


def run_ingestion(start_date: str, end_date: str):
    """Run the ingestion pipeline to fetch Fingrid data."""
    from datetime import datetime
    from ingestion.client import FingridClient
    from ingestion.storage import save_parquet
    
    print("\n" + "="*60)
    print("📥 STEP 1: DATA INGESTION")
    print("="*60)
    
    client = FingridClient()
    
    # Convert string dates to datetime objects
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    
    # Fetch all datasets
    print(f"\n⏳ Fetching data from {start_date} to {end_date}...")
    
    print("\n🌬️ Fetching wind power data...")
    wind_df = client.get_wind_power(start_dt, end_dt)
    save_parquet(wind_df, "wind")
    print(f"   ✓ {len(wind_df):,} records saved")
    
    print("\n⚡ Fetching mFRR activation data...")
    mfrr_df = client.get_mfrr_activation(start_dt, end_dt)
    save_parquet(mfrr_df, "mfrr")
    print(f"   ✓ {len(mfrr_df):,} records saved")
    
    print("\n💰 Fetching imbalance price data...")
    price_df = client.get_imbalance_price(start_dt, end_dt)
    save_parquet(price_df, "price")
    print(f"   ✓ {len(price_df):,} records saved")
    
    print("\n✅ Ingestion complete!")
    return wind_df, mfrr_df, price_df


def run_processing(use_cache: bool = True):
    """Run the processing pipeline to create features."""
    from ingestion.processor import process_features
    
    print("\n" + "="*60)
    print("🔧 STEP 2: DATA PROCESSING")
    print("="*60)
    
    df = process_features(use_cache=use_cache)
    
    print(f"\n📊 Processed dataset:")
    print(f"   Samples: {len(df):,}")
    print(f"   Features: {df.shape[1] - 1}")  # -1 for target
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\n✅ Processing complete!")
    return df


def run_training(use_wandb: bool = True, processed_df: Optional[pd.DataFrame] = None, **hyperparams):
    """Run the training pipeline."""
    from ingestion.processor import process_features
    from models.train import prepare_data, train_model, evaluate_model, save_model
    import wandb
    
    print("\n" + "="*60)
    print("🎯 STEP 3: MODEL TRAINING")
    print("="*60)
    
    # Default hyperparameters
    config = {
        "n_estimators": hyperparams.get("n_estimators", 100),
        "max_depth": hyperparams.get("max_depth", 15),
        "min_samples_split": hyperparams.get("min_samples_split", 10),
        "min_samples_leaf": hyperparams.get("min_samples_leaf", 5),
        "test_size": hyperparams.get("test_size", 0.2)
    }
    
    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="vattenfall-imbalance-price",
            config={"model": "RandomForestRegressor", **config}
        )
        print("✓ Weights & Biases initialized")
    
    # Load processed data
    print("\n📥 Loading processed data...")
    if processed_df is not None:
        df = processed_df
        print(f"  Using provided data: {len(df):,} rows")
    else:
        df = process_features(use_cache=True)
    
    # Prepare data
    print("🔧 Preparing features and target...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df, test_size=config["test_size"])
    print(f"   Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")
    
    # Train
    print("\n🏋️ Training model...")
    model = train_model(
        X_train, y_train,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"]
    )
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "train")
    test_metrics = evaluate_model(model, X_test, y_test, "test")
    
    print(f"\n📊 Results:")
    print(f"   Train MAE: {train_metrics['train_mae']:.2f} EUR/MWh")
    print(f"   Test MAE:  {test_metrics['test_mae']:.2f} EUR/MWh")
    print(f"   Test R²:   {test_metrics['test_r2']:.4f}")
    
    if use_wandb:
        wandb.log({**train_metrics, **test_metrics})
    
    # Save model
    output_path = "models/model.pkl"
    save_model(model, feature_names, output_path)
    
    if use_wandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print("\n✅ Training complete!")
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run the Vattenfall MLOps pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run_pipeline.py                     # Full pipeline
  uv run python run_pipeline.py --skip-ingestion    # Use cached raw data
  uv run python run_pipeline.py --no-wandb          # No experiment tracking
  uv run python run_pipeline.py --start 2024-01-01  # Custom date range
"""
    )
    
    # Pipeline control
    parser.add_argument("--skip-ingestion", action="store_true", 
                        help="Skip data ingestion, use cached raw data")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Skip processing, use cached features")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    
    # Date range for ingestion
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date for data ingestion (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01",
                        help="End date for data ingestion (YYYY-MM-DD)")
    
    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 VATTENFALL MLOps PIPELINE")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Step 1: Ingestion
        if not args.skip_ingestion:
            run_ingestion(args.start, args.end)
        else:
            print("\n⏭️  Skipping ingestion (using cached raw data)")
        
        # Step 2: Processing
        processed_df = None
        if not args.skip_processing:
            # Force reprocessing if we just ingested new data
            use_cache = args.skip_ingestion
            processed_df = run_processing(use_cache=use_cache)
        else:
            print("\n⏭️  Skipping processing (using cached features)")
        
        # Step 3: Training
        if not args.skip_training:
            run_training(
                use_wandb=not args.no_wandb,
                processed_df=processed_df,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                test_size=args.test_size
            )
        else:
            print("\n⏭️  Skipping training")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
