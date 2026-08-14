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

    def test_pressure_column_selection_no_duplicates(self):
        """Regression: _PRESSURE_RENAME values + _PASSTHROUGH_COLS must not overlap."""
        from src.data.open_meteo import _PRESSURE_RENAME
        renamed_values = list(_PRESSURE_RENAME.values())
        dupes = [c for c in renamed_values if renamed_values.count(c) > 1]
        if dupes:
            pytest.fail(
                f"Duplicate values in _PRESSURE_RENAME: {set(dupes)}. "
                f"Each rename target must be unique to avoid duplicate columns in parquet."
            )


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


class TestConfigNotHardcoded:
    """Regression: config constants must auto-update, not be hardcoded."""

    def test_year_is_current(self):
        from datetime import datetime
        assert config.YEAR == str(datetime.now().year), (
            f"config.YEAR={config.YEAR} is hardcoded! Must be dynamic."
        )

    def test_history_years_includes_current(self):
        from datetime import datetime
        current = datetime.now().year
        assert current in config.HISTORY_YEARS, (
            f"config.HISTORY_YEARS ends at {max(config.HISTORY_YEARS)}, "
            f"must include current year {current}"
        )


class TestForecastBiasEdgeCases:
    """Regression: forecast bias must handle 0C and missing humidity correctly."""

    def test_bias_at_zero_celsius(self):
        """0C temp must produce a valid bias, not NaN."""
        from src.data.ensemble import compute_forecast_bias
        now = pd.Timestamp.now().floor("h")
        fdf = pd.DataFrame({
            "datetime": [now],
            "temperature_2m": [5.0],
            "relative_humidity_2m": [80.0],
        })
        result = compute_forecast_bias(station_temp=0.0, station_hum=50.0, forecast_df=fdf)
        assert not np.isnan(result["forecast_temp_bias"]), (
            "forecast_temp_bias is NaN when station_temp=0.0"
        )
        assert abs(result["forecast_temp_bias"] - 5.0) < 0.01

    def test_bias_with_nan_humidity(self):
        """NaN humidity (station offline) must produce NaN bias, not crash."""
        from src.data.ensemble import compute_forecast_bias
        now = pd.Timestamp.now().floor("h")
        fdf = pd.DataFrame({
            "datetime": [now],
            "temperature_2m": [20.0],
            "relative_humidity_2m": [60.0],
        })
        result = compute_forecast_bias(station_temp=20.0, station_hum=np.nan, forecast_df=fdf)
        assert np.isnan(result["forecast_humidity_bias"]), (
            "forecast_humidity_bias should be NaN when station_hum is NaN"
        )

    def test_bias_with_none_forecast(self):
        """No forecast data must return NaN biases, not crash."""
        from src.data.ensemble import compute_forecast_bias
        result = compute_forecast_bias(station_temp=20.0, station_hum=60.0, forecast_df=None)
        assert np.isnan(result["forecast_temp_bias"])
        assert np.isnan(result["forecast_humidity_bias"])


class TestSentinelHumidityEdgeCases:
    """Regression: sentinel features must handle NaN station humidity."""

    def test_sentinel_nan_station_humidity(self):
        """NaN station_humidity must not produce wrong sentinel_humidity_diff."""
        from src.data.meteocat import compute_sentinel_features
        sentinel = {"sentinel_temp": 18.0, "sentinel_humidity": 70.0,
                    "sentinel_precip": 0.0}
        result = compute_sentinel_features(sentinel, station_temp=20.0,
                                           station_humidity=float("nan"))
        assert result["sentinel_humidity_diff"] is None, (
            "With NaN station humidity, sentinel_humidity_diff should be None"
        )

    def test_sentinel_zero_humidity_is_valid(self):
        """0 percent humidity must still compute the diff."""
        from src.data.meteocat import compute_sentinel_features
        sentinel = {"sentinel_temp": 18.0, "sentinel_humidity": 5.0,
                    "sentinel_precip": 0.0}
        result = compute_sentinel_features(sentinel, station_temp=20.0,
                                           station_humidity=0.0)
        assert result["sentinel_humidity_diff"] == 5.0


class TestAemetCrossMidnight:
    """Regression: AEMET 6h window must work after 18:00."""

    def test_late_night_covers_next_day(self):
        """At 22:00, the 6h window (22-04) must include tomorrow 0006 period."""
        current_hour = 22
        today_periods = [("0006", 80), ("0612", 30), ("1218", 40), ("1824", 60)]
        tomorrow_periods = [("0006", 90), ("0612", 20)]

        max_prob = 0
        for day_idx, periods in enumerate([today_periods, tomorrow_periods]):
            offset = 24 * day_idx
            for periodo, valor in periods:
                h_start = int(periodo[:2]) + offset
                h_end = int(periodo[2:]) + offset
                if h_start <= current_hour + 6 and h_end > current_hour:
                    max_prob = max(max_prob, valor)

        assert max_prob == 90, (
            f"At 22:00, expected tomorrow 0006 period (prob=90) but got {max_prob}. "
            f"Cross-midnight bug."
        )

    def test_afternoon_no_false_positives(self):
        """At 14:00, must not match tomorrow periods."""
        current_hour = 14
        today_periods = [("1218", 40), ("1824", 20)]
        tomorrow_periods = [("0006", 90)]

        max_prob = 0
        for day_idx, periods in enumerate([today_periods, tomorrow_periods]):
            offset = 24 * day_idx
            for periodo, valor in periods:
                h_start = int(periodo[:2]) + offset
                h_end = int(periodo[2:]) + offset
                if h_start <= current_hour + 6 and h_end > current_hour:
                    max_prob = max(max_prob, valor)

        assert max_prob == 40


class TestRainGateAemetStorm:
    """Regression: rain gate AEMET storm check must handle all edge cases."""

    def test_nan_does_not_trigger_gate(self):
        from src.model.predict import _aemet_storm_above_threshold
        assert _aemet_storm_above_threshold({"aemet_prob_storm": np.nan}) is False

    def test_none_does_not_trigger_gate(self):
        from src.model.predict import _aemet_storm_above_threshold
        assert _aemet_storm_above_threshold({"aemet_prob_storm": None}) is False
        assert _aemet_storm_above_threshold({}) is False

    def test_above_threshold_triggers(self):
        from src.model.predict import _aemet_storm_above_threshold
        assert _aemet_storm_above_threshold(
            {"aemet_prob_storm": config.RAIN_GATE_AEMET_STORM + 1}
        ) is True

    def test_below_threshold_does_not_trigger(self):
        from src.model.predict import _aemet_storm_above_threshold
        assert _aemet_storm_above_threshold({"aemet_prob_storm": 0}) is False


class TestXemaSentinelFallback:
    """Regression tests for XEMA sentinel yesterday-fallback and cache behaviour."""

    def test_empty_sentinel_has_required_keys(self):
        from src.data.meteocat import _empty_sentinel
        result = _empty_sentinel()
        assert "sentinel_temp" in result
        assert "sentinel_humidity" in result
        assert "sentinel_precip" in result

    def test_compute_sentinel_features_with_none_data(self):
        """When XEMA returns all None, features should gracefully degrade."""
        from src.data.meteocat import compute_sentinel_features
        result = compute_sentinel_features(
            {"sentinel_temp": None, "sentinel_humidity": None, "sentinel_precip": None},
            station_temp=15.0, station_humidity=60.0,
        )
        assert result["sentinel_temp_diff"] is None
        assert result["sentinel_humidity_diff"] is None
        assert result["sentinel_precip"] is None
        assert result["sentinel_raining"] == 0

    def test_compute_sentinel_features_with_real_data(self):
        """When XEMA returns data, features should be computed."""
        from src.data.meteocat import compute_sentinel_features
        result = compute_sentinel_features(
            {"sentinel_temp": 13.0, "sentinel_humidity": 72.0, "sentinel_precip": 0.2,
             "local_rain_xema": 0.5, "local_rain_xema_3h": 1.2},
            station_temp=15.0, station_humidity=60.0,
        )
        assert result["sentinel_temp_diff"] == pytest.approx(2.0)
        assert result["sentinel_humidity_diff"] == pytest.approx(12.0)
        assert result["sentinel_precip"] == pytest.approx(0.2)
        assert result["sentinel_raining"] == 1
        assert result["local_rain_xema"] == pytest.approx(0.5)
        assert result["local_rain_xema_3h"] == pytest.approx(1.2)

    def test_fetch_variable_caches_empty_responses(self, tmp_path, monkeypatch):
        """Empty API responses should be cached to avoid burning quota."""
        import time
        from src.data import meteocat_cache
        from src.data.meteocat import fetch_variable_all_stations
        from datetime import date

        # Redirect cache to tmp_path
        cache_file = str(tmp_path / "meteocat_cache.json")
        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", cache_file)
        # Fake API configured
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        # Mock the HTTP call to return empty data
        call_count = 0
        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return []

        import src.data.meteocat as meteocat_mod
        original_get = meteocat_mod.SESSION.get
        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse()
        monkeypatch.setattr(meteocat_mod.SESSION, "get", mock_get)

        test_date = date(2026, 3, 24)
        # First call: API hit
        df1 = fetch_variable_all_stations(32, test_date)
        assert df1.empty
        assert call_count == 1

        # Second call: should use cache, NOT call API again
        df2 = fetch_variable_all_stations(32, test_date)
        assert df2.empty
        assert call_count == 1, "Empty response should be cached — API called twice!"

    def test_meteocat_cache_prune(self, tmp_path, monkeypatch):
        """Cache should prune old entries when exceeding max."""
        import time
        from src.data import meteocat_cache

        cache_file = str(tmp_path / "meteocat_cache.json")
        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", cache_file)
        monkeypatch.setattr(meteocat_cache, "_MAX_ENTRIES", 5)

        # Write 7 entries
        for i in range(7):
            meteocat_cache.set_cached(f"key_{i}", {"val": i})

        # Read back — should only have 5 most recent
        cache = meteocat_cache._load_cache()
        assert len(cache) == 5
        # Oldest keys (key_0, key_1) should be pruned
        assert "key_0" not in cache
        assert "key_1" not in cache
        assert "key_6" in cache

class TestXemaRateLimitCooldown:
    """HTTP 429 must stop the next prediction from hammering XEMA."""

    def test_429_persists_cooldown_for_other_variables(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_variable_all_stations

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        calls = []

        class FakeResponse:
            status_code = 429
            def raise_for_status(self):
                import requests
                error = requests.HTTPError("rate limited")
                error.response = self
                raise error

        def mock_get(*args, **kwargs):
            calls.append(args[0])
            return FakeResponse()

        monkeypatch.setattr(meteocat_mod.SESSION, "get", mock_get)
        day = date(2026, 3, 24)
        assert fetch_variable_all_stations(32, day).empty
        assert len(calls) == 1
        # Different variable: the persistent breaker prevents another HTTP request.
        assert fetch_variable_all_stations(33, day).empty
        assert len(calls) == 1

    def test_expired_cooldown_allows_a_new_request(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_variable_all_stations

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import time
        meteocat_cache.set_cached("xema_rate_limit_cooldown", True)
        cache = meteocat_cache._load_cache()
        cache["xema_rate_limit_cooldown"]["timestamp"] = time.time() - 61 * 60
        meteocat_cache._save_cache(cache)
        calls = []

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return []

        monkeypatch.setattr(meteocat_mod.SESSION, "get", lambda *a, **k: (calls.append(a[0]) or FakeResponse()))
        assert fetch_variable_all_stations(32, date(2026, 3, 24)).empty
        assert len(calls) == 1

    def test_non_429_error_does_not_start_cooldown(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_variable_all_stations

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        class FakeResponse:
            status_code = 500
            def raise_for_status(self):
                import requests
                error = requests.HTTPError("server error")
                error.response = self
                raise error

        monkeypatch.setattr(meteocat_mod.SESSION, "get", lambda *a, **k: FakeResponse())
        assert fetch_variable_all_stations(32, date(2026, 3, 24)).empty
        assert meteocat_cache.get_cached(
            "xema_rate_limit_cooldown", config.METEOCAT_XEMA_429_COOLDOWN_MIN
        ) is None

    def test_cooldown_reuses_bounded_stale_data(self, tmp_path, monkeypatch):
        from datetime import date
        import time
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_variable_all_stations

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        rows = [{
            "station_code": "YM", "datetime": "2026-03-24T12:00:00",
            "value": 18.5, "estat": "",
        }]
        meteocat_cache.set_cached("xema_32_2026-03-24", rows)
        meteocat_cache.set_cached("xema_rate_limit_cooldown", True)
        cache = meteocat_cache._load_cache()
        cache["xema_32_2026-03-24"]["timestamp"] = time.time() - 90 * 60
        meteocat_cache._save_cache(cache)
        monkeypatch.setattr(meteocat_mod.SESSION, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP must be skipped")))

        result = fetch_variable_all_stations(32, date(2026, 3, 24))
        assert len(result) == 1
        assert result.iloc[0]["value"] == 18.5
        assert isinstance(result.iloc[0]["datetime"], pd.Timestamp)
        assert result.attrs["xema_stale"] is True


class TestXemaStationEndpoint:
    """SMC-recommended per-station endpoint: 1 call = all variables of 1 station."""

    def test_parses_all_variables_for_one_station(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_station_all_variables

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        payload = [{
            "codi": "YM",
            "variables": [
                {"codi": 32, "lectures": [{"data": "2026-03-24T12:00Z", "valor": 18.5, "estat": "V"}]},
                {"codi": 33, "lectures": [{"data": "2026-03-24T12:00Z", "valor": 62.0, "estat": "V"}]},
                {"codi": 35, "lectures": [{"data": "2026-03-24T12:00Z", "valor": 0.0, "estat": "V"}]},
            ],
        }]

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return payload

        monkeypatch.setattr(meteocat_mod.SESSION, "get", lambda *a, **k: FakeResponse())
        df = fetch_station_all_variables("YM", date(2026, 3, 24))
        assert len(df) == 3
        assert set(df["variable_code"]) == {32, 33, 35}
        assert set(df["station_code"]) == {"YM"}

    def test_filters_unavailable_readings_before_cache(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_station_all_variables

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        payload = [{
            "codi": "YM",
            "variables": [
                {"codi": 32, "lectures": [
                    {"data": "2026-03-24T12:00Z", "valor": 18.5, "estat": "V"},
                    {"data": "2026-03-24T12:30Z", "valor": None, "estat": "T"},
                ]},
            ],
        }]

        calls = []
        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return payload
        monkeypatch.setattr(meteocat_mod.SESSION, "get", lambda *a, **k: (calls.append(1) or FakeResponse()))

        df = fetch_station_all_variables("YM", date(2026, 3, 24))
        assert len(df) == 1
        assert df.iloc[0]["value"] == 18.5

        # Cache hit: no HTTP, and the "T" row was never cached.
        df2 = fetch_station_all_variables("YM", date(2026, 3, 24))
        assert len(df2) == 1
        assert len(calls) == 1

    def test_uses_per_station_endpoint(self, tmp_path, monkeypatch):
        from datetime import date
        from src.data import meteocat_cache
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_station_all_variables

        monkeypatch.setattr(meteocat_cache, "CACHE_FILE", str(tmp_path / "cache.json"))
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return [{"codi": "YM", "variables": []}]

        seen_urls = []
        monkeypatch.setattr(meteocat_mod.SESSION, "get",
                            lambda url, *a, **k: (seen_urls.append(url) or FakeResponse()))
        fetch_station_all_variables("YM", date(2026, 3, 24))
        assert any("estacions/mesurades/YM/2026/03/24" in u for u in seen_urls)


class TestXemaStationSentinelFlow:
    """fetch_sentinel_latest should use 2 station calls and still build features."""

    def test_sentinel_latest_uses_two_station_calls(self, monkeypatch):
        import src.data.meteocat as meteocat_mod
        from src.data.meteocat import fetch_sentinel_latest

        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")

        def fake_station(station_code, target_date):
            if station_code == config.SENTINEL_STATION_CODE:
                return pd.DataFrame({
                    "station_code": [station_code] * 3,
                    "variable_code": [32, 33, 35],
                    "datetime": [pd.to_datetime("2026-03-24T12:00:00")] * 3,
                    "value": [18.0, 65.0, 0.4],
                    "estat": ["", "", ""],
                })
            return pd.DataFrame({
                "station_code": [station_code],
                "variable_code": [35],
                "datetime": [pd.to_datetime("2026-03-24T12:00:00")],
                "value": [0.4],
                "estat": [""],
            })

        calls = []
        def recording(station_code, target_date):
            calls.append(station_code)
            return fake_station(station_code, target_date)
        monkeypatch.setattr(meteocat_mod, "fetch_station_all_variables", recording)

        result = fetch_sentinel_latest()
        assert result["sentinel_temp"] == 18.0
        assert result["sentinel_humidity"] == 65.0
        assert result["sentinel_precip"] == 0.4
        assert result["local_rain_xema"] == 0.4
        assert len(calls) == 2
        assert sorted(calls) == [config.LOCAL_RAIN_STATION_CODE, config.SENTINEL_STATION_CODE]
