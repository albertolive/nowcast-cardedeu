"""
Test suite: Feature engineering health checks.
Catches: null features from scalar broadcast, missing extra_cols, broken diffs,
station unavailability crashes, feature count mismatches.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Helpers to create synthetic data ──

def _make_forecast_df(hours: int = 24) -> pd.DataFrame:
    """Creates a realistic forecast DataFrame with varying values (future-only)."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=i) for i in range(hours)]
    return _build_forecast_rows(times)


def _make_forecast_df_with_past(past_hours: int = 12, future_hours: int = 12) -> pd.DataFrame:
    """Creates a forecast DataFrame covering past+future hours (simulates past_hours param)."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=i) for i in range(-past_hours, future_hours)]
    return _build_forecast_rows(times)


def _build_forecast_rows(times) -> pd.DataFrame:
    """Builds forecast rows from a list of datetime objects."""
    hours = len(times)
    df = pd.DataFrame({
        "datetime": times,
        "temperature_2m": np.linspace(15, 25, hours),
        "relative_humidity_2m": np.linspace(80, 50, hours),
        "dew_point_2m": np.linspace(12, 10, hours),
        "pressure_msl": np.linspace(1013, 1010, hours),
        "surface_pressure": np.linspace(990, 988, hours),
        "precipitation": [0.0] * hours,
        "rain": [0.0] * hours,
        "cloud_cover": np.linspace(70, 30, hours),
        "cloud_cover_low": np.linspace(40, 10, hours),
        "cloud_cover_mid": np.linspace(20, 5, hours),
        "cloud_cover_high": np.linspace(10, 15, hours),
        "wind_speed_10m": np.linspace(10, 15, hours),
        "wind_direction_10m": np.linspace(90, 180, hours),
        "wind_gusts_10m": np.linspace(20, 30, hours),
        "cape": np.linspace(0, 500, hours),
        "shortwave_radiation": np.linspace(100, 300, hours),
        "direct_radiation": np.linspace(80, 250, hours),
        "diffuse_radiation": np.linspace(20, 50, hours),
        "weather_code": [3] * hours,  # overcast
        "vapour_pressure_deficit": np.linspace(0.2, 0.8, hours),
        "convective_inhibition": np.linspace(-10, -5, hours),
        "wet_bulb_temperature_2m": np.linspace(13, 15, hours),
        "showers": [0.0] * hours,
        "et0_fao_evapotranspiration": np.linspace(0.1, 0.3, hours),
        "soil_temperature_0_to_7cm": [np.nan] * hours,  # archive-only
        "sunshine_duration": np.linspace(2000, 3600, hours),
        "wind_speed_100m": np.linspace(15, 25, hours),
        "wind_direction_100m": np.linspace(90, 180, hours),
        "snowfall": [0.0] * hours,
        "total_column_integrated_water_vapour": np.linspace(20, 25, hours),
        "boundary_layer_height": np.linspace(500, 1500, hours),
        "terrestrial_radiation": np.linspace(300, 350, hours),
        "soil_moisture_0_to_7cm": [np.nan] * hours,  # archive-only
        "soil_moisture_7_to_28cm": [np.nan] * hours,  # archive-only
        "soil_moisture_28_to_100cm": [np.nan] * hours,  # archive-only
    })
    return df


def _make_pressure_hourly_df(hours: int = 36, start_offset: int = -12) -> pd.DataFrame:
    """Creates a realistic pressure levels DataFrame (past+future)."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=start_offset + i) for i in range(hours)]
    return pd.DataFrame({
        "datetime": times,
        "temp_925": np.linspace(12, 14, hours),
        "rh_925": np.linspace(85, 70, hours),
        "wind_925_speed": np.linspace(15, 20, hours),
        "wind_925_dir": np.linspace(100, 150, hours),
        "wind_850_speed": np.linspace(20, 30, hours),
        "wind_850_dir": np.linspace(110, 160, hours),
        "temp_850": np.linspace(8, 10, hours),
        "rh_850": np.linspace(90, 60, hours),
        "rh_700": np.linspace(50, 30, hours),
        "temp_700": np.linspace(0, 2, hours),
        "temp_500": np.linspace(-15, -12, hours),
        "wind_300_speed": np.linspace(50, 80, hours),
        "wind_300_dir": np.linspace(250, 270, hours),
        "gph_300": np.linspace(9200, 9300, hours),
        "gph_850": np.linspace(1500, 1520, hours),
        "rh_500": np.linspace(25, 15, hours),
        "wind_700_speed": np.linspace(20, 30, hours),
        "wind_700_dir": np.linspace(200, 230, hours),
        "nwp_lifted_index": np.linspace(5, -2, hours),
    })


def _make_station_df(hours: int = 24) -> pd.DataFrame:
    """Creates a station DataFrame matching MeteoCardedeu format."""
    now = datetime.now()
    # Minute-by-minute data
    n = hours * 60
    times = [now - timedelta(minutes=n - i) for i in range(n)]
    return pd.DataFrame({
        "datetime": times,
        "TEMP": np.random.uniform(15, 25, n),
        "HUM": np.random.uniform(40, 90, n),
        "BAR": np.random.uniform(1010, 1015, n),
        "VEL": np.random.uniform(0, 20, n),
        "DIR": ["NE"] * n,
        "DIR_DEG": np.random.uniform(30, 60, n),
        "PREC": [0.0] * n,
        "SUN": np.random.uniform(0, 500, n),
        "UVI": np.random.uniform(0, 5, n),
    })


# ── Test Classes ──


class TestFeatureColumnConsistency:
    """Validate FEATURE_COLUMNS is complete and consistent."""

    def test_feature_count(self):
        """FEATURE_COLUMNS should have 209 entries."""
        from src.features.engineering import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 209, (
            f"Expected 209 features, got {len(FEATURE_COLUMNS)}. "
            f"If you added/removed features, update this test."
        )

    def test_no_duplicates(self):
        from src.features.engineering import FEATURE_COLUMNS
        dupes = [f for f in FEATURE_COLUMNS if FEATURE_COLUMNS.count(f) > 1]
        assert len(dupes) == 0, f"Duplicate features: {set(dupes)}"

    def test_model_feature_names_subset(self):
        """All features the model expects must be in FEATURE_COLUMNS."""
        import json
        from src.features.engineering import FEATURE_COLUMNS
        with open(config.FEATURE_NAMES_PATH) as f:
            model_features = json.load(f)
        missing = set(model_features) - set(FEATURE_COLUMNS)
        assert len(missing) == 0, f"Model expects features not in FEATURE_COLUMNS: {missing}"


class TestBuildFeaturesFromRealtime:
    """Integration test: build_features_from_realtime produces valid features."""

    def test_with_station_and_forecast(self):
        """Normal case: both station and forecast available."""
        from src.features.engineering import build_features_from_realtime, FEATURE_COLUMNS
        station_df = _make_station_df(hours=6)
        forecast_df = _make_forecast_df(hours=12)
        result = build_features_from_realtime(station_df, forecast_df)

        assert not result.empty, "Should produce features"
        assert "datetime" in result.columns

    def test_diff_features_with_past_forecast(self):
        """REGRESSION: diff(3) features must NOT be NaN when forecast includes past hours.
        Bug: fetch_forecast was future-only, so merge_asof left past station rows
        with NaN for forecast columns → .diff(3) on last row = NaN.
        Fix: fetch_forecast now includes past_hours=12."""
        from src.features.engineering import build_features_from_realtime, FEATURE_COLUMNS

        station_df = _make_station_df(hours=24)
        # Simulate forecast with past_hours=12: covers -12h to +12h
        forecast_df = _make_forecast_df_with_past(past_hours=12, future_hours=12)
        # Add pressure levels covering past+future
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])
        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        if pl_cols:
            forecast_df = pd.merge_asof(
                forecast_df.sort_values("datetime"),
                pl_df[["datetime"] + pl_cols].sort_values("datetime"),
                on="datetime", direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )

        result = build_features_from_realtime(station_df, forecast_df)
        assert not result.empty

        latest = result.iloc[-1]
        # These diff(3) features MUST be non-NaN with past forecast data
        must_be_populated = [
            "vpd_change_3h", "cloud_change_3h", "weather_code_change_3h",
            "nwp_rain_trend_3h", "tcwv_change_3h", "blh_change_3h",
            "rh_700_change_3h", "temp_850_change_3h", "gph_850_change_3h",
            "cape_change_3h",
        ]
        missing = [f for f in must_be_populated if f in latest.index and pd.isna(latest[f])]
        assert len(missing) == 0, (
            f"diff(3) features still NaN with past forecast data: {missing}. "
            f"This means forecast past hours are not reaching the station merge."
        )

    def test_diff_features_nan_without_past_forecast(self):
        """Confirms the bug: future-only forecast → diff(3) features are NaN."""
        from src.features.engineering import build_features_from_realtime

        station_df = _make_station_df(hours=24)
        # Future-only forecast (the old buggy behavior)
        forecast_df = _make_forecast_df(hours=12)

        result = build_features_from_realtime(station_df, forecast_df)
        if result.empty:
            pytest.skip("Could not build features")

        latest = result.iloc[-1]
        # With future-only forecast, these SHOULD be NaN (the bug we're documenting)
        forecast_diff_features = ["vpd_change_3h", "cloud_change_3h"]
        nan_count = sum(1 for f in forecast_diff_features
                        if f in latest.index and pd.isna(latest[f]))
        assert nan_count > 0, (
            "Expected diff(3) features to be NaN with future-only forecast — "
            "test setup may be wrong (forecast overlapping with station hours?)"
        )

    def test_empty_station_does_not_crash(self):
        """When MeteoCardedeu is down, must NOT crash — use forecast only."""
        from src.features.engineering import build_features_from_realtime
        station_df = pd.DataFrame()  # empty — station down
        forecast_df = _make_forecast_df(hours=12)
        result = build_features_from_realtime(station_df, forecast_df)

        assert not result.empty, "Must produce features from forecast alone"

    def test_station_without_datetime_does_not_crash(self):
        """Station returned data but missing datetime column."""
        from src.features.engineering import build_features_from_realtime
        station_df = pd.DataFrame({"TEMP": [20]})  # no datetime
        forecast_df = _make_forecast_df(hours=12)
        result = build_features_from_realtime(station_df, forecast_df)
        assert not result.empty

    def test_both_empty_returns_empty(self):
        """Both station and forecast empty → empty result (caught by predict_now)."""
        from src.features.engineering import build_features_from_realtime
        result = build_features_from_realtime(pd.DataFrame(), pd.DataFrame())
        assert result.empty


class TestPressureLevelMerge:
    """Test the hourly pressure level merge into forecast_df.
    Catches: scalar broadcast bug where .diff(3) produced all NaN."""

    def test_merge_produces_varying_pressure_columns(self):
        """After merge, pressure level columns must have varying values (not constant)."""
        forecast_df = _make_forecast_df(hours=24)
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)

        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])

        # Merge the same way predict.py does
        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        merged = pd.merge_asof(
            forecast_df.sort_values("datetime"),
            pl_df[["datetime"] + pl_cols].sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

        # Key check: pressure columns must NOT be constant (scalar broadcast = all same)
        for col in ["rh_700", "temp_850", "wind_850_dir", "gph_850", "rh_500"]:
            if col in merged.columns:
                values = merged[col].dropna()
                if len(values) > 3:
                    assert values.nunique() > 1, (
                        f"{col} has constant value {values.iloc[0]} — "
                        f"likely scalar broadcast bug"
                    )

    def test_diff_features_not_all_nan(self):
        """diff(3) on varying data must produce non-NaN values beyond row 3."""
        forecast_df = _make_forecast_df(hours=24)
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)

        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])

        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        merged = pd.merge_asof(
            forecast_df.sort_values("datetime"),
            pl_df[["datetime"] + pl_cols].sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

        # Simulate what build_features_from_hourly does
        if "rh_700" in merged.columns:
            rh700 = pd.to_numeric(merged["rh_700"], errors="coerce")
            rh_700_change = rh700.diff(3)
            # After 3 rows, at least some values must be non-NaN
            non_null = rh_700_change.iloc[3:].dropna()
            assert len(non_null) > 0, (
                "rh_700_change_3h is all NaN after diff(3) — indicates constant values"
            )

    def test_scalar_fallback_columns_are_constant(self):
        """Verify: if we DO scalar-inject (fallback), columns ARE constant →
        diff(3) WILL be NaN. This is the bug scenario we must prevent."""
        forecast_df = _make_forecast_df(hours=24)

        # Scalar injection (the old buggy way)
        pressure_data = {"rh_700": 50.0, "temp_850": 8.0, "wind_850_dir": 120.0}
        for k, v in pressure_data.items():
            if k not in forecast_df.columns:
                forecast_df[k] = v  # scalar → broadcast to all rows

        # These MUST be constant — that's why this approach is buggy
        for col in pressure_data:
            if col in forecast_df.columns:
                assert forecast_df[col].nunique() == 1, (
                    f"Scalar injection should produce constant column"
                )
                # And diff(3) MUST be all-zero or NaN
                diff3 = forecast_df[col].diff(3)
                assert (diff3.dropna() == 0).all(), (
                    f"Scalar injection diff(3) should be zero (constant input)"
                )


class TestExtraColsCoverage:
    """Verify the extra_cols list in build_features_from_realtime
    includes all variables that the feature engineering pipeline needs."""

    def test_pressure_level_vars_in_extra_cols(self):
        """All PL vars that the feature pipeline references must be in extra_cols."""
        from src.features.engineering import build_features_from_realtime
        import inspect

        source = inspect.getsource(build_features_from_realtime)

        # These pressure level columns MUST be in extra_cols for the merge to work
        required_pl_cols = [
            "wind_850_dir", "wind_850_speed", "temp_850", "rh_850",
            "wind_925_dir", "wind_925_speed", "temp_925", "rh_925",
            "rh_700", "temp_700", "temp_500",
            "wind_300_speed", "wind_300_dir", "gph_300",
            "gph_850", "rh_500", "wind_700_speed", "wind_700_dir",
            "nwp_lifted_index",
        ]
        for col in required_pl_cols:
            assert f'"{col}"' in source, (
                f"{col} missing from extra_cols in build_features_from_realtime — "
                f"it won't be merged from forecast_df"
            )

    def test_key_forecast_vars_in_extra_cols(self):
        """Important forecast vars that caught bugs previously."""
        from src.features.engineering import build_features_from_realtime
        import inspect

        source = inspect.getsource(build_features_from_realtime)
        required = [
            "vapour_pressure_deficit", "convective_inhibition",
            "cape", "cloud_cover", "weather_code",
        ]
        for col in required:
            assert f'"{col}"' in source, f"{col} missing from extra_cols"


class TestFeatureNullBudget:
    """Test that the feature pipeline produces a reasonable null count."""

    def test_forecast_only_null_budget(self):
        """With only forecast data (no station, no rain gate), expected nulls are bounded."""
        from src.features.engineering import build_features_from_realtime, FEATURE_COLUMNS

        forecast_df = _make_forecast_df(hours=24)
        # Add pressure levels to forecast
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])
        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        if pl_cols:
            forecast_df = pd.merge_asof(
                forecast_df.sort_values("datetime"),
                pl_df[["datetime"] + pl_cols].sort_values("datetime"),
                on="datetime", direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )

        result = build_features_from_realtime(pd.DataFrame(), forecast_df)
        if result.empty:
            pytest.skip("Could not build features — may need more data points")

        latest = result.iloc[-1]
        null_count = sum(1 for col in FEATURE_COLUMNS if col not in latest.index or pd.isna(latest.get(col)))

        # Expected nulls: soil(~7) + radar(~22) + sentinel(~6) + ensemble(~6)
        # + aemet(~3) + lightning(~7) + smc(~3) + bias(~2) + radar_bearing(~2)
        # ≈ ~58 rain-gated/archive-only features
        # Non-gated features that use station data: pressure_msl trends ≈ ~5
        # Total expected: ~60-70 (many are rain-gated or need station)
        assert null_count < 100, (
            f"Too many null features ({null_count}/{len(FEATURE_COLUMNS)}). "
            f"Expected < 100 with forecast-only data."
        )

    def test_anti_fp_features_populated_with_pressure_data(self):
        """Key anti-FP features MUST be populated when pressure data is available."""
        from src.features.engineering import build_features_from_realtime, FEATURE_COLUMNS

        forecast_df = _make_forecast_df(hours=24)
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])
        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        if pl_cols:
            forecast_df = pd.merge_asof(
                forecast_df.sort_values("datetime"),
                pl_df[["datetime"] + pl_cols].sort_values("datetime"),
                on="datetime", direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )

        result = build_features_from_realtime(pd.DataFrame(), forecast_df)
        if result.empty:
            pytest.skip("Could not build features")

        latest = result.iloc[-1]

        # These features were ALL null before the fix and are critical for reducing FPs
        anti_fp_features = [
            "nwp_rain_trend_3h",
            "weather_code_change_3h",
            "cloud_humidity_convergence",
            "precip_trend_3h",
            "nwp_rain_persistence_6h",
            "vapour_pressure_deficit",
            "deep_layer_shear",
            "moisture_flux_850",
        ]
        missing = []
        for feat in anti_fp_features:
            if feat in latest.index and pd.notna(latest[feat]):
                continue
            missing.append(feat)

        assert len(missing) == 0, (
            f"Anti-FP features still null: {missing}. "
            f"These are critical for reducing false positives."
        )

    def test_pressure_trend_features_not_all_nan(self):
        """Features derived from .diff(3) must have non-NaN values beyond row 3."""
        from src.features.engineering import build_features_from_realtime

        forecast_df = _make_forecast_df(hours=24)
        pl_df = _make_pressure_hourly_df(hours=36, start_offset=-12)
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        pl_df["datetime"] = pd.to_datetime(pl_df["datetime"])
        pl_cols = [c for c in pl_df.columns if c != "datetime" and c not in forecast_df.columns]
        if pl_cols:
            forecast_df = pd.merge_asof(
                forecast_df.sort_values("datetime"),
                pl_df[["datetime"] + pl_cols].sort_values("datetime"),
                on="datetime", direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )

        result = build_features_from_realtime(pd.DataFrame(), forecast_df)
        if result.empty or len(result) <= 6:
            pytest.skip("Need more rows for diff tests")

        diff_features = ["rh_700_change_3h", "temp_850_change_3h", "gph_850_change_3h"]
        for feat in diff_features:
            if feat in result.columns:
                # After first 3 rows, there should be non-NaN values
                non_null = result[feat].iloc[3:].dropna()
                assert len(non_null) > 0, (
                    f"{feat} is all NaN after row 3 — "
                    f"pressure data likely broadcast as scalar (constant diff=0 or NaN)"
                )


class TestPressureRenameMap:
    """Verify the _PRESSURE_RENAME dict covers all needed variables."""

    def test_all_pressure_level_vars_mapped(self):
        """Every var in PRESSURE_LEVEL_VARS that needs renaming must be in _PRESSURE_RENAME."""
        from src.data.open_meteo import _PRESSURE_RENAME, PRESSURE_LEVEL_VARS

        # Vars that don't need renaming (already correct names)
        passthrough = {"cape", "convective_inhibition", "visibility", "freezing_level_height"}
        unmapped = []
        for var in PRESSURE_LEVEL_VARS:
            if var in passthrough or var in _PRESSURE_RENAME:
                continue
            unmapped.append(var)

        assert len(unmapped) == 0, (
            f"PRESSURE_LEVEL_VARS not in _PRESSURE_RENAME: {unmapped}. "
            f"These columns will be silently dropped."
        )

    def test_rename_values_match_feature_columns(self):
        """Renamed pressure columns must appear somewhere in FEATURE_COLUMNS
        or be used as intermediate vars."""
        from src.data.open_meteo import _PRESSURE_RENAME
        from src.features.engineering import FEATURE_COLUMNS

        # Some renamed cols are used as intermediaries (e.g., gph_500)
        intermediary_cols = {"gph_500"}
        for api_name, internal_name in _PRESSURE_RENAME.items():
            if internal_name in intermediary_cols:
                continue
            assert internal_name in FEATURE_COLUMNS, (
                f"_PRESSURE_RENAME maps {api_name}→{internal_name} "
                f"but {internal_name} not in FEATURE_COLUMNS"
            )
