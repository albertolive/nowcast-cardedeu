"""
Test suite: Data pipeline integrity.
Catches: API contract changes, graceful degradation failures,
missing columns, incorrect data types.

Tests that require network are marked with @pytest.mark.network.
Run offline tests:  pytest tests/ -m "not network"
Run all tests:      pytest tests/
"""
import pytest
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TestOpenMeteoContract:
    """Verify Open-Meteo API variable lists are consistent."""

    def test_hourly_vars_no_duplicates(self):
        dupes = [v for v in config.OPEN_METEO_HOURLY_VARS
                 if config.OPEN_METEO_HOURLY_VARS.count(v) > 1]
        if dupes:
            pytest.fail(f"Duplicate vars in OPEN_METEO_HOURLY_VARS: {set(dupes)}")

    def test_pressure_level_vars_no_duplicates(self):
        from src.data.open_meteo import PRESSURE_LEVEL_VARS
        dupes = [v for v in PRESSURE_LEVEL_VARS if PRESSURE_LEVEL_VARS.count(v) > 1]
        if dupes:
            pytest.fail(f"Duplicate vars in PRESSURE_LEVEL_VARS: {set(dupes)}")


@pytest.mark.network
class TestGracefulDegradation:
    """Every data module must return safe defaults on error, never crash."""

    def test_meteocardedeu_returns_empty_df_on_error(self):
        """fetch_series must return empty DataFrame, never raise."""
        from src.data.meteocardedeu import fetch_series
        # With a very short timeout, it should still not crash
        result = fetch_series(hours=0)
        assert isinstance(result, pd.DataFrame)

    def test_rainviewer_dbz_handles_edge_cases(self):
        from src.data.rainviewer import _radar_intensity_to_dbz
        # All edge cases must return a float, never None or raise
        for val in [0, -1, -100, 255, 128, 1]:
            result = _radar_intensity_to_dbz(val)
            assert isinstance(result, float)

    def test_open_meteo_pressure_levels_returns_dict_with_all_keys(self):
        """Even on failure, fetch_pressure_levels must return dict with all expected keys."""
        from src.data.open_meteo import fetch_pressure_levels
        # The function handles its own exceptions
        result = fetch_pressure_levels()
        assert isinstance(result, dict)
        required_keys = [
            "temp_925", "rh_925", "wind_925_speed", "wind_925_dir",
            "wind_850_speed", "wind_850_dir", "temp_850", "rh_850",
            "rh_700", "temp_700", "temp_500",
            "wind_300_speed", "wind_300_dir", "gph_300",
            "vt_index", "tt_index", "li_index",
        ]
        for key in required_keys:
            assert key in result, f"fetch_pressure_levels missing key: {key}"

    def test_sst_forecast_returns_dict(self):
        from src.data.open_meteo import fetch_sst_forecast
        result = fetch_sst_forecast()
        assert isinstance(result, dict)
        assert "sst_med" in result


@pytest.mark.network
class TestPressureLevelsHourly:
    """Validate fetch_pressure_levels_hourly contract."""

    def test_returns_dataframe(self):
        from src.data.open_meteo import fetch_pressure_levels_hourly
        result = fetch_pressure_levels_hourly(hours_ahead=6)
        assert isinstance(result, pd.DataFrame)

    def test_with_past_hours(self):
        from src.data.open_meteo import fetch_pressure_levels_hourly
        result = fetch_pressure_levels_hourly(hours_ahead=6, past_hours=6)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "datetime" in result.columns
            # With past_hours=6, first row should be ~6h ago
            min_dt = result["datetime"].min()
            now = pd.Timestamp.now()
            # Allow generous tolerance for timezone differences
            hours_ago = (now - min_dt).total_seconds() / 3600
            assert hours_ago > 3, f"past_hours=6 but earliest data is only {hours_ago:.1f}h ago"

    def test_has_key_columns_when_not_empty(self):
        from src.data.open_meteo import fetch_pressure_levels_hourly
        result = fetch_pressure_levels_hourly(hours_ahead=6)
        if not result.empty:
            key_cols = ["wind_850_dir", "temp_850", "rh_700", "temp_500"]
            for col in key_cols:
                assert col in result.columns, f"Missing column {col}"


@pytest.mark.network
class TestForecastContract:
    """Validate fetch_forecast returns expected structure."""

    def test_returns_dataframe(self):
        from src.data.open_meteo import fetch_forecast
        result = fetch_forecast(hours_ahead=6)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "datetime" in result.columns

    def test_has_key_weather_columns(self):
        from src.data.open_meteo import fetch_forecast
        result = fetch_forecast(hours_ahead=6)
        if not result.empty:
            # These are the most critical columns for the model
            for col in ["temperature_2m", "precipitation", "weather_code", "cape"]:
                assert col in result.columns, f"Forecast missing {col}"


@pytest.mark.network
class TestEnsembleContract:
    """Validate ensemble agreement returns expected keys."""

    def test_returns_dict(self):
        from src.data.ensemble import fetch_ensemble_agreement
        result = fetch_ensemble_agreement()
        assert isinstance(result, dict)
        assert "ensemble_rain_agreement" in result


class TestFeatureEngineeringPipeline:
    """Test the complete feature engineering pipeline with synthetic data."""

    def test_build_features_from_hourly_produces_key_features(self):
        """The pipeline must produce key features from hourly data."""
        from src.features.engineering import build_features_from_hourly

        # Create minimal hourly data
        hours = 12
        now = pd.Timestamp.now().floor("h")
        df = pd.DataFrame({
            "datetime": [now + pd.Timedelta(hours=i) for i in range(hours)],
            "temperature_2m": np.linspace(15, 22, hours),
            "relative_humidity_2m": np.linspace(80, 50, hours),
            "pressure_msl": np.linspace(1015, 1010, hours),
            "wind_speed_10m": np.linspace(5, 15, hours),
            "wind_direction_10m": np.linspace(90, 180, hours),
            "precipitation": [0.0] * hours,
            "cloud_cover": np.linspace(70, 30, hours),
            "weather_code": [3] * hours,
            "cape": np.linspace(0, 300, hours),
            "rain": [0.0] * hours,
            "wind_850_dir": np.linspace(100, 160, hours),
            "wind_850_speed": np.linspace(20, 30, hours),
            "temp_850": np.linspace(8, 10, hours),
            "rh_850": np.linspace(85, 60, hours),
            "rh_700": np.linspace(50, 30, hours),
            "temp_700": np.linspace(0, 2, hours),
            "temp_500": np.linspace(-15, -12, hours),
            "wind_300_speed": np.linspace(50, 80, hours),
            "wind_300_dir": np.linspace(250, 270, hours),
            "gph_300": np.linspace(9200, 9300, hours),
        })

        result = build_features_from_hourly(df)
        assert not result.empty
        assert len(result) == hours

        # Key features that must be produced
        key_features = [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "dew_point", "dew_point_depression",
            "wind_u", "wind_v",
            "model_predicts_precip",
            "has_pressure_levels",
            "vt_index", "tt_index",
            "wind_shear_speed",
            "moisture_flux_850",
        ]
        for feat in key_features:
            assert feat in result.columns, f"Missing key feature: {feat}"

    def test_temporal_features_range(self):
        """hour_sin/cos must be in [-1, 1]."""
        from src.features.engineering import build_features_from_hourly

        hours = 24
        now = pd.Timestamp.now().floor("h")
        df = pd.DataFrame({
            "datetime": [now + pd.Timedelta(hours=i) for i in range(hours)],
            "temperature_2m": [20.0] * hours,
            "relative_humidity_2m": [60.0] * hours,
            "pressure_msl": [1013.0] * hours,
            "wind_speed_10m": [10.0] * hours,
            "wind_direction_10m": [180.0] * hours,
            "precipitation": [0.0] * hours,
            "cloud_cover": [50.0] * hours,
            "weather_code": [3] * hours,
        })
        result = build_features_from_hourly(df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            if col in result.columns:
                vals = result[col].dropna()
                assert vals.min() >= -1.0001, f"{col} below -1"
                assert vals.max() <= 1.0001, f"{col} above 1"
