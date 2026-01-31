"""
Model training script for imbalance price prediction.

Usage:
    python -m models.train [--no-wandb] [--n-estimators 100]
"""
import argparse
import os
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
from dotenv import load_dotenv

from ingestion.processor import process_features

# Load environment variables
load_dotenv()


def prepare_data(df: pd.DataFrame, test_size=0.2, scale_features=True, clip_outliers=True):
    """
    Prepare features and target, with proper train/test temporal split.
    
    CRITICAL: Split is done chronologically (not random) to preserve temporal order
    and prevent any data leakage. Test set is the most recent 20% of data.
    
    Args:
        df: Processed dataframe with all features
        test_size: Fraction of data for test set (most recent data)
        scale_features: Whether to apply StandardScaler to features
        clip_outliers: Whether to clip extreme target values (robust to outliers)
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler (or None)
    """
    df = df.copy()
    
    # Target variable
    target_col = "imbalance_price"
    
    # Clip extreme outliers in target using IQR method (only on training portion to avoid leakage)
    if clip_outliers:
        split_idx = int(len(df) * (1 - test_size))
        train_target = df.iloc[:split_idx][target_col]
        q1 = train_target.quantile(0.01)
        q99 = train_target.quantile(0.99)
        
        original_range = (df[target_col].min(), df[target_col].max())
        df[target_col] = df[target_col].clip(lower=q1, upper=q99)
        print(f"  ✓ Target clipped to [{q1:.2f}, {q99:.2f}] (was {original_range[0]:.2f} to {original_range[1]:.2f})")
    
    # Drop timestamp and target to get features
    feature_cols = [col for col in df.columns if col not in ["timestamp", target_col]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Temporal split: train on first 80%, test on last 20%
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Feature scaling (fit on train, transform both)
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("  ✓ Features scaled with StandardScaler")
    
    print(f"\n📊 Train/Test Split (Temporal):")
    print(f"  Train: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test:  {len(X_test):,} samples ({test_size*100:.0f}%)")
    print(f"  Train period: {df.iloc[0]['timestamp']} to {df.iloc[split_idx-1]['timestamp']}")
    print(f"  Test period:  {df.iloc[split_idx]['timestamp']} to {df.iloc[-1]['timestamp']}")
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_model(
    X_train, 
    y_train, 
    model_type="rf",
    n_estimators=100, 
    max_depth=15, 
    min_samples_split=10,
    min_samples_leaf=5,
    learning_rate=0.1,
    random_state=42
):
    """
    Train regression model with regularization to prevent overfitting.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: "rf" for RandomForest, "gb" for GradientBoosting
        n_estimators: Number of trees
        max_depth: Maximum depth of trees (lower = more regularization)
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in leaf node
        learning_rate: Learning rate for GradientBoosting
        random_state: Random seed
        
    Returns:
        Trained model
    """
    if model_type == "gb":
        print(f"\n🚀 Training GradientBoostingRegressor...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=0.8,  # Use 80% of samples per tree for regularization
            random_state=random_state,
            verbose=1
        )
    else:
        print(f"\n🌲 Training RandomForestRegressor...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X, y, split_name="test"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: True target values
        split_name: Name of the split (train/val/test)
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {
        f"{split_name}_mae": mae,
        f"{split_name}_rmse": rmse,
        f"{split_name}_r2": r2
    }
    
    print(f"\n📊 {split_name.upper()} Metrics:")
    print(f"  MAE:  {mae:.2f} EUR/MWh")
    print(f"  RMSE: {rmse:.2f} EUR/MWh")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def save_model(model, feature_names, output_path="models/model.pkl", scaler=None):
    """
    Save trained model and feature metadata to disk.
    
    Args:
        model: Trained model
        feature_names: List of feature names (order matters for prediction)
        output_path: Path to save the model
        scaler: Optional StandardScaler fitted on training data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model with metadata for safe loading later
    model_artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "scaler": scaler,
        "created_at": datetime.now().isoformat()
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_artifact, f)
    
    print(f"\n💾 Model saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   Features: {len(feature_names)}")
    print(f"   Scaler included: {scaler is not None}")


def load_model(model_path="models/model.pkl"):
    """
    Load trained model and feature metadata from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (model, feature_names, scaler)
    """
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    
    # Handle both old format (just model) and new format (dict with metadata)
    if isinstance(artifact, dict):
        return artifact["model"], artifact["feature_names"], artifact.get("scaler")
    else:
        # Old format - model only, no feature names or scaler
        return artifact, None, None


def main():
    parser = argparse.ArgumentParser(description="Train imbalance price prediction model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--model", type=str, default="gb", choices=["rf", "gb"],
                        help="Model type: rf=RandomForest, gb=GradientBoosting (default)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=8, help="Max tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (GB only)")
    parser.add_argument("--min-samples-split", type=int, default=10, help="Min samples to split")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="Min samples in leaf")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0-1)")
    parser.add_argument("--no-scaling", action="store_true", help="Disable feature scaling")
    parser.add_argument("--output", type=str, default="models/model.pkl", help="Output path for model")
    
    args = parser.parse_args()
    
    use_wandb = not args.no_wandb
    scale_features = not args.no_scaling
    
    print("="*60)
    print("🚀 Imbalance Price Prediction - Model Training")
    print("="*60)
    
    # Initialize W&B
    if use_wandb:
        wandb.init(
            project="vattenfall-imbalance-price",
            config={
                "model": "GradientBoostingRegressor" if args.model == "gb" else "RandomForestRegressor",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "test_size": args.test_size,
                "feature_scaling": scale_features
            }
        )
        print("✓ Weights & Biases initialized")
    
    # Load and process data (force refresh to get new features)
    print("\n📥 Loading and processing data...")
    df = process_features(use_cache=False)
    
    # Prepare features and target WITH PROPER TRAIN/TEST SPLIT
    print("\n🔧 Preparing features and target...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(
        df, test_size=args.test_size, scale_features=scale_features
    )
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Target range (train): [{y_train.min():.2f}, {y_train.max():.2f}] EUR/MWh")
    print(f"  Target range (test):  [{y_test.min():.2f}, {y_test.max():.2f}] EUR/MWh")
    
    # Train model on training data
    print(f"\n{'='*60}")
    print("🎯 Training Model on Training Set")
    print("   ⚠️  Test set is completely held out")
    print('='*60)
    
    model = train_model(
        X_train, y_train,
        model_type=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
    
    # Evaluate on training set
    train_metrics = evaluate_model(model, X_train, y_train, "train")
    
    # Log feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 15 Feature Importances:")
    print(feature_importance.head(15).to_string(index=False))
    
    if use_wandb:
        wandb.log(train_metrics)
        wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance)})
    
    # FINAL EVALUATION ON HELD-OUT TEST SET
    print(f"\n{'='*60}")
    print("🎯 FINAL EVALUATION ON HELD-OUT TEST SET")
    print("   ⚠️  First time this data has been seen by the model!")
    print('='*60)
    
    test_metrics = evaluate_model(model, X_test, y_test, "test")
    
    if use_wandb:
        wandb.log(test_metrics)
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY: Train vs Test Performance")
    print('='*60)
    print(f"\n  Train MAE:  {train_metrics['train_mae']:.2f} EUR/MWh")
    print(f"  Test MAE:   {test_metrics['test_mae']:.2f} EUR/MWh")
    print(f"  Gap:        {test_metrics['test_mae'] - train_metrics['train_mae']:.2f} EUR/MWh")
    print(f"\n  Train R²:   {train_metrics['train_r2']:.4f}")
    print(f"  Test R²:    {test_metrics['test_r2']:.4f}")
    
    if use_wandb:
        wandb.log({
            "train_test_gap": test_metrics['test_mae'] - train_metrics['train_mae']
        })
    
    # Save model with feature names and scaler
    save_model(model, feature_names, args.output, scaler=scaler)
    
    if use_wandb:
        # Save model to W&B
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(args.output)
        wandb.log_artifact(artifact)
        wandb.finish()
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
