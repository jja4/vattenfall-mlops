"""
Tests for the data processor.
"""
import pytest
import pandas as pd
import numpy as np

from ingestion.processor import (
    resample_to_15min,
    create_lag_features,
    create_rolling_features,
    create_temporal_features,
    validate_dataframe,
    PERIODS_PER_HOUR,
    LAG_1H,
    LAG_2H,
    LAG_3H,
    ROLLING_1H,
    ROLLING_3H,
)


class TestResampleTo15Min:
    """Tests for resample_to_15min function."""
    
    def test_downsamples_3min_to_15min(self):
        """Wind data at 3-min intervals should be downsampled to 15-min."""
        # Create 3-minute data
        timestamps = pd.date_range("2024-01-01", periods=10, freq="3min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        })
        
        result = resample_to_15min(df, "value", "mean")
        
        # Should have fewer rows (downsampled)
        assert len(result) < len(df)
        # All intervals should be 15 minutes
        time_diffs = result["timestamp"].diff().dropna()
        assert all(time_diffs == pd.Timedelta(minutes=15))
    
    def test_upsamples_hourly_to_15min(self):
        """Hourly data should be upsampled to 15-min with forward fill."""
        timestamps = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": [100.0, 200.0, 300.0]
        })
        
        result = resample_to_15min(df, "value", "ffill")
        
        # Should have more rows (upsampled) - roughly 4x
        assert len(result) > len(df)
    
    def test_preserves_15min_data(self):
        """Data already at 15-min should remain unchanged."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="15min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "value": range(10)
        })
        
        result = resample_to_15min(df, "value", "mean")
        
        # Should have same number of rows
        assert len(result) == len(df)


class TestLagFeatures:
    """Tests for lag feature creation."""
    
    @pytest.fixture
    def merged_df(self):
        """Create a sample merged dataframe with correct column names."""
        timestamps = pd.date_range("2024-01-01", periods=20, freq="15min")
        return pd.DataFrame({
            "timestamp": timestamps,
            "imbalance_price": list(range(20)),  # 0, 1, 2, ..., 19
            "wind_power_mw": [500 + i*10 for i in range(20)],
            "mfrr_price": [10 + i for i in range(20)],
        })
    
    def test_creates_lag_columns(self, merged_df):
        """Should create lag_1h, lag_2h, lag_3h columns."""
        result = create_lag_features(merged_df)
        
        assert "price_lag_1h" in result.columns
        assert "price_lag_2h" in result.columns
        assert "price_lag_3h" in result.columns
        assert "wind_lag_1h" in result.columns
        assert "mfrr_lag_1h" in result.columns
    
    def test_lag_values_are_correct(self, merged_df):
        """Lag values should be shifted by correct number of periods."""
        result = create_lag_features(merged_df)
        
        # LAG_1H = 4 periods (1 hour at 15-min intervals)
        # At index 4, lag_1h should equal value at index 0
        assert result.iloc[LAG_1H]["price_lag_1h"] == result.iloc[0]["imbalance_price"]
        
        # LAG_2H = 8 periods
        assert result.iloc[LAG_2H]["price_lag_2h"] == result.iloc[0]["imbalance_price"]
    
    def test_lag_introduces_nans(self, merged_df):
        """First few rows should have NaN lag values."""
        result = create_lag_features(merged_df)
        
        # First LAG_1H rows should have NaN for price_lag_1h
        assert result["price_lag_1h"].iloc[:LAG_1H].isna().all()
        assert not result["price_lag_1h"].iloc[LAG_1H:].isna().any()


class TestRollingFeatures:
    """Tests for rolling feature creation."""
    
    @pytest.fixture
    def merged_df(self):
        """Create a sample merged dataframe with correct column names."""
        timestamps = pd.date_range("2024-01-01", periods=50, freq="15min")
        return pd.DataFrame({
            "timestamp": timestamps,
            "imbalance_price": np.random.randn(50) * 50 + 100,
            "wind_power_mw": np.random.randn(50) * 100 + 500,
            "mfrr_price": np.random.randn(50) * 20,
        })
    
    def test_creates_rolling_columns(self, merged_df):
        """Should create rolling mean columns."""
        result = create_rolling_features(merged_df)
        
        assert "wind_rolling_1h" in result.columns
        assert "wind_rolling_3h" in result.columns
        assert "price_rolling_1h" in result.columns
    
    def test_rolling_mean_is_average(self):
        """Rolling mean should be average of window."""
        timestamps = pd.date_range("2024-01-01", periods=20, freq="15min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "imbalance_price": [10.0] * 20,  # Constant value
            "wind_power_mw": [100.0] * 20,
            "mfrr_price": [5.0] * 20,
        })
        
        result = create_rolling_features(df)
        
        # Rolling mean of constant should equal that constant (after initial NaNs)
        non_nan_values = result["wind_rolling_1h"].dropna()
        if len(non_nan_values) > 0:
            assert non_nan_values.iloc[0] == 100.0


class TestTemporalFeatures:
    """Tests for temporal feature creation."""
    
    def test_creates_temporal_columns(self):
        """Should create hour_of_day, day_of_week, is_weekend columns."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="15min")
        df = pd.DataFrame({"timestamp": timestamps})
        
        result = create_temporal_features(df)
        
        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
    
    def test_hour_extraction_is_correct(self):
        """Hour should match the timestamp hour."""
        timestamps = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:30", "2024-01-01 23:45"])
        df = pd.DataFrame({"timestamp": timestamps})
        
        result = create_temporal_features(df)
        
        assert list(result["hour_of_day"]) == [0, 12, 23]
    
    def test_day_of_week_is_correct(self):
        """Day of week should be 0-6 (Monday-Sunday)."""
        # 2024-01-01 is a Monday (0)
        timestamps = pd.date_range("2024-01-01", periods=7, freq="1D")
        df = pd.DataFrame({"timestamp": timestamps})
        
        result = create_temporal_features(df)
        
        assert list(result["day_of_week"]) == [0, 1, 2, 3, 4, 5, 6]
    
    def test_is_weekend_is_correct(self):
        """is_weekend should be 1 for Saturday/Sunday, 0 otherwise."""
        # 2024-01-01 is Monday, so 2024-01-06 is Saturday, 2024-01-07 is Sunday
        timestamps = pd.date_range("2024-01-01", periods=7, freq="1D")
        df = pd.DataFrame({"timestamp": timestamps})
        
        result = create_temporal_features(df)
        
        # Mon, Tue, Wed, Thu, Fri, Sat, Sun
        assert list(result["is_weekend"]) == [0, 0, 0, 0, 0, 1, 1]


class TestValidation:
    """Tests for data validation."""
    
    def test_valid_dataframe_passes(self):
        """Valid dataframe should not raise."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min"),
            "value": [1, 2, 3, 4, 5]
        })
        
        # Should not raise
        validate_dataframe(df, {"timestamp", "value"}, "test")
    
    def test_missing_column_raises(self):
        """Missing required column should raise ValueError."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="15min")
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, {"timestamp", "value"}, "test")
    
    def test_empty_dataframe_raises(self):
        """Empty dataframe should raise ValueError."""
        df = pd.DataFrame(columns=["timestamp", "value"])
        
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df, {"timestamp", "value"}, "test")


class TestConstants:
    """Tests for feature engineering constants."""
    
    def test_periods_per_hour(self):
        """4 periods per hour for 15-min intervals."""
        assert PERIODS_PER_HOUR == 4
    
    def test_lag_constants_are_multiples_of_periods_per_hour(self):
        """Lag constants should be multiples of periods per hour."""
        assert LAG_1H == PERIODS_PER_HOUR * 1
        assert LAG_2H == PERIODS_PER_HOUR * 2
        assert LAG_3H == PERIODS_PER_HOUR * 3
    
    def test_rolling_constants(self):
        """Rolling window constants should match expected hours."""
        assert ROLLING_1H == PERIODS_PER_HOUR * 1
        assert ROLLING_3H == PERIODS_PER_HOUR * 3
