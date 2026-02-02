"""
Challenger model training with W&B Model Registry integration.

This module trains challenger models and registers them in W&B Model Registry
with proper versioning and aliasing for the champion/challenger pattern.

Features:
- Loads features from Azure Blob Storage or local
- Trains models with configurable hyperparameters
- Logs metrics, artifacts, and feature importance to W&B
- Registers models in W&B Model Registry with 'staging' alias
- Tracks dataset hash for reproducibility

Usage:
    python -m pipeline.train [--model gb] [--n-estimators 200]
    
Environment Variables:
    WANDB_API_KEY: W&B authentication key
    AZURE_STORAGE_CONNECTION_STRING: Azure Blob connection string (optional)
"""
import argparse
import os
import sys
import pickle
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
from dotenv import load_dotenv

from pipeline.features import load_features, compute_dataset_hash

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# W&B configuration
WANDB_PROJECT = "vattenfall-imbalance-price"
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # Optional: your W&B username/team
MODEL_REGISTRY_NAME = "imbalance-price-model"  # Name in W&B Model Registry


def prepare_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    scale_features: bool = True, 
    clip_outliers: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, Optional[StandardScaler]]:
    """
    Prepare features and target with proper temporal train/test split.
    
    CRITICAL: Split is chronological to preserve temporal order and prevent data leakage.
    
    Args:
        df: Feature DataFrame with timestamp and imbalance_price columns
        test_size: Fraction of data for test set (most recent data)
        scale_features: Whether to apply StandardScaler
        clip_outliers: Whether to clip extreme target values
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    df = df.copy()
    target_col = "imbalance_price"
    
    # Clip outliers using IQR on training portion only
    if clip_outliers:
        split_idx = int(len(df) * (1 - test_size))
        train_target = df.iloc[:split_idx][target_col]
        q1 = train_target.quantile(0.01)
        q99 = train_target.quantile(0.99)
        
        original_range = (df[target_col].min(), df[target_col].max())
        df[target_col] = df[target_col].clip(lower=q1, upper=q99)
        logger.info(f"Target clipped to [{q1:.2f}, {q99:.2f}] (was {original_range[0]:.2f} to {original_range[1]:.2f})")
    
    # Get feature columns (exclude timestamp and target)
    feature_cols = [col for col in df.columns if col not in ["timestamp", target_col, "date"]]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Temporal split
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Feature scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "gb",
    n_estimators: int = 200,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
):
    """
    Train a regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'rf' for RandomForest, 'gb' for GradientBoosting
        Other args: Model hyperparameters
        
    Returns:
        Trained model
    """
    if model_type == "gb":
        logger.info(f"Training GradientBoostingRegressor (n={n_estimators}, depth={max_depth}, lr={learning_rate})")
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=0.8,
            random_state=random_state,
        )
    else:
        logger.info(f"Training RandomForestRegressor (n={n_estimators}, depth={max_depth})")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray, prefix: str = "test") -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: True target values
        prefix: Metric name prefix (train/test)
        
    Returns:
        Dict of metrics
    """
    y_pred = model.predict(X)
    
    metrics = {
        f"{prefix}_mae": mean_absolute_error(y, y_pred),
        f"{prefix}_rmse": np.sqrt(mean_squared_error(y, y_pred)),
        f"{prefix}_r2": r2_score(y, y_pred),
    }
    
    logger.info(f"{prefix.upper()} - MAE: {metrics[f'{prefix}_mae']:.2f}, RMSE: {metrics[f'{prefix}_rmse']:.2f}, R²: {metrics[f'{prefix}_r2']:.4f}")
    
    return metrics


def save_model_artifact(
    model,
    feature_names: list,
    scaler: Optional[StandardScaler],
    metrics: Dict[str, float],
    dataset_hash: str,
) -> str:
    """
    Save model artifact to temporary file.
    
    Returns:
        Path to saved model file
    """
    artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "scaler": scaler,
        "metrics": metrics,
        "dataset_hash": dataset_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "model.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    
    logger.info(f"Model saved to {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    return str(model_path)


def register_model_to_wandb(
    run: wandb.Run,
    model_path: str,
    metrics: Dict[str, float],
    dataset_hash: str,
    alias: str = "staging",
) -> str:
    """
    Register model artifact to W&B Model Registry.
    
    Args:
        run: Active W&B run
        model_path: Path to model pickle file
        metrics: Model metrics to log as metadata
        dataset_hash: Hash of training dataset
        alias: Alias to assign ('staging' for challenger, 'production' for champion)
        
    Returns:
        Artifact version string (e.g., 'v3')
    """
    # Create artifact
    artifact = wandb.Artifact(
        name=MODEL_REGISTRY_NAME,
        type="model",
        description=f"Imbalance price prediction model (dataset: {dataset_hash})",
        metadata={
            **metrics,
            "dataset_hash": dataset_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    
    # Add model file
    artifact.add_file(model_path, name="model.pkl")
    
    # Log artifact with alias
    logged_artifact = run.log_artifact(artifact, aliases=[alias])
    
    # Wait for artifact to be committed
    logged_artifact.wait()
    
    logger.info(f"Model registered to W&B: {MODEL_REGISTRY_NAME}:{logged_artifact.version} (alias: {alias})")
    
    return logged_artifact.version


def run_training(
    model_type: str = "gb",
    n_estimators: int = 200,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    test_size: float = 0.2,
    scale_features: bool = True,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.
    
    Args:
        Model hyperparameters and training options
        
    Returns:
        Dict with training results
    """
    logger.info("=" * 60)
    logger.info("🚀 Challenger Model Training")
    logger.info("=" * 60)
    
    # Load features
    logger.info("\n📥 Loading features...")
    df = load_features()
    
    # Compute dataset hash for reproducibility
    dataset_hash = compute_dataset_hash(df)
    logger.info(f"Dataset hash: {dataset_hash}")
    
    # Initialize W&B
    run = None
    if use_wandb:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="training",
            config={
                "model_type": model_type,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "test_size": test_size,
                "scale_features": scale_features,
                "dataset_hash": dataset_hash,
                "dataset_rows": len(df),
            },
            tags=["challenger", model_type],
        )
        logger.info(f"W&B run: {run.url}")
    
    try:
        # Prepare data
        logger.info("\n🔧 Preparing data...")
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(
            df, test_size=test_size, scale_features=scale_features
        )
        
        # Train model
        logger.info("\n🎯 Training model...")
        model = train_model(
            X_train, y_train,
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        
        # Evaluate
        logger.info("\n📊 Evaluating model...")
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        test_metrics = evaluate_model(model, X_test, y_test, "test")
        
        all_metrics = {**train_metrics, **test_metrics}
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Log to W&B
        if run:
            wandb.log(all_metrics)
            wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance)})
        
        # Save and register model
        logger.info("\n💾 Saving model...")
        model_path = save_model_artifact(model, feature_names, scaler, all_metrics, dataset_hash)
        
        model_version = None
        if run:
            model_version = register_model_to_wandb(
                run, model_path, all_metrics, dataset_hash, alias="staging"
            )
        
        # Also save locally for backwards compatibility
        local_path = Path(__file__).parent.parent / "models" / "model.pkl"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())
        logger.info(f"Model also saved to {local_path}")
        
        results = {
            "model_type": model_type,
            "metrics": all_metrics,
            "dataset_hash": dataset_hash,
            "model_version": model_version,
            "feature_count": len(feature_names),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Training Complete")
        logger.info("=" * 60)
        logger.info(f"Test MAE: {all_metrics['test_mae']:.2f} EUR/MWh")
        logger.info(f"Test R²: {all_metrics['test_r2']:.4f}")
        if model_version:
            logger.info(f"Model version: {model_version} (alias: staging)")
        
        return results
        
    finally:
        if run:
            wandb.finish()


def main():
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(description="Train challenger model")
    parser.add_argument("--model", type=str, default="gb", choices=["rf", "gb"],
                        help="Model type: rf=RandomForest, gb=GradientBoosting")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-samples-split", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        results = run_training(
            model_type=args.model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            test_size=args.test_size,
            scale_features=not args.no_scaling,
            use_wandb=not args.no_wandb,
        )
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================
# Model Loading Utilities (used by app/main.py and notebooks)
# ============================================================================

def load_model(model_path: str) -> Tuple[Any, list, Any]:
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


def load_model_from_wandb(
    alias: str = "production",
    entity: Optional[str] = None,
) -> Tuple[Any, list, Any, str, str]:
    """
    Load model from W&B Model Registry.
    
    Args:
        alias: Model alias to load ('production', 'staging')
        entity: W&B entity (defaults to WANDB_ENTITY env var)
        
    Returns:
        Tuple of (model, feature_names, scaler, version, created_at)
    """
    api = wandb.Api()
    entity = entity or os.getenv("WANDB_ENTITY") or api.default_entity
    
    artifact_path = f"{entity}/{WANDB_PROJECT}/{MODEL_REGISTRY_NAME}:{alias}"
    logger.info(f"Loading model from W&B: {artifact_path}")
    
    artifact = api.artifact(artifact_path)
    version = artifact.version
    
    # Download to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = artifact.download(root=tmpdir)
        model_path = Path(artifact_dir) / "model.pkl"
        
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    
    model = data["model"]
    feature_names = data["feature_names"]
    scaler = data.get("scaler")
    created_at = data.get("created_at", "unknown")
    
    logger.info(f"Loaded model {version} (created: {created_at})")
    
    return model, feature_names, scaler, version, created_at


def get_production_model_version(entity: Optional[str] = None) -> Optional[str]:
    """
    Get the current production model version from W&B.
    
    Args:
        entity: W&B entity
        
    Returns:
        Version string (e.g., 'v3') or None if no production model
    """
    api = wandb.Api()
    entity = entity or os.getenv("WANDB_ENTITY") or api.default_entity
    
    try:
        artifact_path = f"{entity}/{WANDB_PROJECT}/{MODEL_REGISTRY_NAME}:production"
        artifact = api.artifact(artifact_path)
        return artifact.version
    except wandb.errors.CommError:
        return None
