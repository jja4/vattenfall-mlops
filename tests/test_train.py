"""
Tests for the model training module.
"""
import pytest
import pandas as pd
import numpy as np
import pickle
import tempfile
from pathlib import Path

from models.train import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


class TestPrepareData:
    """Tests for data preparation."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing with correct column names."""
        n_samples = 100
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="15min")
        return pd.DataFrame({
            "timestamp": timestamps,
            "imbalance_price": np.random.randn(n_samples) * 50 + 100,
            "wind_power_mw": np.random.randn(n_samples) * 100 + 500,
            "mfrr_price": np.random.randn(n_samples) * 20,
            "price_lag_1h": np.random.randn(n_samples) * 50 + 100,
            "price_lag_2h": np.random.randn(n_samples) * 50 + 100,
            "hour_of_day": np.random.randint(0, 24, n_samples),
        })
    
    def test_returns_train_test_split(self, sample_df):
        """Should return X_train, X_test, y_train, y_test, feature_names."""
        result = prepare_data(sample_df, test_size=0.2)
        
        assert len(result) == 5
        X_train, X_test, y_train, y_test, feature_names = result
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(feature_names, list)
    
    def test_split_is_temporal(self, sample_df):
        """Train/test split should be chronological, not random."""
        X_train, X_test, y_train, y_test, _ = prepare_data(sample_df, test_size=0.2)
        
        # With temporal split, train should have first 80%, test should have last 20%
        expected_train_size = int(len(sample_df) * 0.8)
        assert len(X_train) == expected_train_size
        assert len(X_test) == len(sample_df) - expected_train_size
    
    def test_timestamp_excluded_from_features(self, sample_df):
        """Timestamp should not be in feature names."""
        _, _, _, _, feature_names = prepare_data(sample_df)
        
        assert "timestamp" not in feature_names
        assert "imbalance_price" not in feature_names  # target excluded
    
    def test_price_is_target(self, sample_df):
        """Imbalance_price column should be the target variable."""
        _, _, y_train, y_test, _ = prepare_data(sample_df)
        
        # y should be the price values
        expected_train_y = sample_df["imbalance_price"].values[:int(len(sample_df) * 0.8)]
        np.testing.assert_array_equal(y_train, expected_train_y)


class TestTrainModel:
    """Tests for model training."""
    
    def test_trains_random_forest(self):
        """Should train a RandomForestRegressor."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = train_model(X, y, n_estimators=10, max_depth=5)
        
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)
    
    def test_model_can_predict(self):
        """Trained model should be able to make predictions."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = train_model(X, y, n_estimators=10)
        predictions = model.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(np.isfinite(predictions))


class TestEvaluateModel:
    """Tests for model evaluation."""
    
    def test_returns_metrics_dict(self):
        """Should return dictionary with mae, rmse, r2 metrics."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = train_model(X, y, n_estimators=10)
        
        metrics = evaluate_model(model, X, y, split_name="test")
        
        assert "test_mae" in metrics
        assert "test_rmse" in metrics
        assert "test_r2" in metrics
    
    def test_metrics_are_numeric(self):
        """All metrics should be finite numbers."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = train_model(X, y, n_estimators=10)
        
        metrics = evaluate_model(model, X, y, split_name="test")
        
        for value in metrics.values():
            assert np.isfinite(value)


class TestSaveLoadModel:
    """Tests for model serialization."""
    
    def test_save_and_load_roundtrip(self):
        """Model should be identical after save/load."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = train_model(X, y, n_estimators=10)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            save_model(model, feature_names, str(path))
            
            loaded_model, loaded_features = load_model(str(path))
        
        # Predictions should be identical
        original_pred = model.predict(X[:5])
        loaded_pred = loaded_model.predict(X[:5])
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Feature names should be preserved
        assert loaded_features == feature_names
    
    def test_saves_metadata(self):
        """Should save feature names and metadata."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = train_model(X, y, n_estimators=10)
        feature_names = ["a", "b", "c", "d", "e"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            save_model(model, feature_names, str(path))
            
            # Load raw pickle to check structure
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        
        assert "model" in artifact
        assert "feature_names" in artifact
        assert "n_features" in artifact
        assert artifact["n_features"] == 5
    
    def test_load_handles_old_format(self):
        """Should handle old format (model only, no metadata)."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        model = train_model(X, y, n_estimators=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_model.pkl"
            
            # Save in old format (just the model)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            
            # Should still load without error
            loaded_model, loaded_features = load_model(str(path))
        
        assert loaded_model is not None
        assert loaded_features is None  # No feature names in old format
