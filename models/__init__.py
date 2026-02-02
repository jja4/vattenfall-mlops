"""Model utilities for loading trained models."""
import pickle
from pathlib import Path


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
