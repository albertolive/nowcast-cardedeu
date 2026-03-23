"""
Test suite: Physics formulas and feature engineering functions.
Catches: dew point formula bugs, wind decomposition errors,
pressure trend calculation, solar timing, VT/TT/LI index formulas.
"""
import pytest
import math
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDewPoint:
    """Test Magnus formula for dew point calculation."""

    def setup_method(self):
        from src.features.engineering import dew_point
        self.dew_point = dew_point

    def test_saturated_air(self):
        """At 100% RH, dew point == temperature."""
        assert self.dew_point(20, 100) == pytest.approx(20, abs=0.5)

    def test_low_humidity(self):
        """At low humidity, dew point << temperature."""
        dp = self.dew_point(30, 20)
        assert dp < 10  # much lower than temp

    def test_typical_summer(self):
        """25°C, 60% RH → dew point ~16.7°C."""
        dp = self.dew_point(25, 60)
        assert 15 < dp < 18

    def test_zero_humidity(self):
        """0% humidity → returns temp (edge case guard)."""
        assert self.dew_point(20, 0) == 20

    def test_negative_humidity(self):
        """Negative humidity (sensor error) → returns temp."""
        assert self.dew_point(20, -5) == 20

    def test_cold_conditions(self):
        """Negative temperature, high humidity."""
        dp = self.dew_point(-5, 90)
        assert -7 < dp < -5  # close to temp at high RH

    def test_dew_point_always_less_than_temp(self):
        """Dew point must be ≤ temperature for any RH < 100%."""
        for temp in range(-10, 40, 5):
            for rh in range(10, 100, 10):
                dp = self.dew_point(temp, rh)
                assert dp <= temp + 0.1, f"dp({temp}°C, {rh}%)={dp} > temp"


class TestWindComponents:
    """Test U/V wind decomposition."""

    def setup_method(self):
        from src.features.engineering import wind_components
        self.wind_components = wind_components

    def test_north_wind(self):
        """0° (N) wind: u=0, v=-speed (southward)."""
        u, v = self.wind_components(10, 0)
        assert u == pytest.approx(0, abs=0.01)
        assert v == pytest.approx(-10, abs=0.01)

    def test_east_wind(self):
        """90° (E) wind: u=-speed (westward), v=0."""
        u, v = self.wind_components(10, 90)
        assert u == pytest.approx(-10, abs=0.01)
        assert v == pytest.approx(0, abs=0.01)

    def test_south_wind(self):
        """180° (S) wind: u=0, v=+speed (northward)."""
        u, v = self.wind_components(10, 180)
        assert u == pytest.approx(0, abs=0.01)
        assert v == pytest.approx(10, abs=0.01)

    def test_west_wind(self):
        """270° (W) wind: u=+speed (eastward), v=0."""
        u, v = self.wind_components(10, 270)
        assert u == pytest.approx(10, abs=0.01)
        assert v == pytest.approx(0, abs=0.01)

    def test_zero_speed(self):
        """Calm wind → u=v=0 regardless of direction."""
        u, v = self.wind_components(0, 123)
        assert u == pytest.approx(0)
        assert v == pytest.approx(0)

    def test_magnitude_preserved(self):
        """sqrt(u² + v²) must equal original speed."""
        for deg in range(0, 360, 15):
            u, v = self.wind_components(10, deg)
            magnitude = math.sqrt(u**2 + v**2)
            assert magnitude == pytest.approx(10, abs=0.01), f"At {deg}°: magnitude={magnitude}"


class TestPressureFeatures:
    """Test pressure trend calculations."""

    def test_diff_values(self):
        from src.features.engineering import _add_pressure_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=8, freq="h"),
            "pressure_msl": [1015, 1014, 1013, 1012, 1011, 1010, 1009, 1008],
        })
        result = _add_pressure_features(df)
        # diff(1) = -1 hPa/h
        assert result["pressure_change_1h"].iloc[1] == pytest.approx(-1)
        # diff(3) = -3 hPa/3h
        assert result["pressure_change_3h"].iloc[3] == pytest.approx(-3)
        # diff(6) = -6 hPa/6h
        assert result["pressure_change_6h"].iloc[7] == pytest.approx(-6)

    def test_first_rows_are_nan(self):
        from src.features.engineering import _add_pressure_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "pressure_msl": [1015, 1014, 1013, 1012],
        })
        result = _add_pressure_features(df)
        assert pd.isna(result["pressure_change_1h"].iloc[0])
        assert pd.isna(result["pressure_change_3h"].iloc[0])
        assert pd.isna(result["pressure_change_3h"].iloc[2])

    def test_missing_pressure_column(self):
        from src.features.engineering import _add_pressure_features
        df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=4, freq="h")})
        result = _add_pressure_features(df)
        assert "pressure_change_1h" not in result.columns


class TestHumidityFeatures:
    """Test humidity feature engineering."""

    def test_dew_point_depression(self):
        from src.features.engineering import _add_humidity_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temperature_2m": [25.0, 25.0, 25.0, 25.0],
            "relative_humidity_2m": [80.0, 80.0, 80.0, 80.0],
        })
        result = _add_humidity_features(df)
        # dpd should be positive (temp > dew point)
        assert all(result["dew_point_depression"] > 0)
        # At 80% RH, dpd should be ~4°C
        assert all(result["dew_point_depression"] < 8)

    def test_humidity_change(self):
        from src.features.engineering import _add_humidity_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temperature_2m": [20.0] * 4,
            "relative_humidity_2m": [60.0, 65.0, 70.0, 75.0],
        })
        result = _add_humidity_features(df)
        assert result["humidity_change_1h"].iloc[1] == pytest.approx(5)


class TestSolarTiming:
    """Test hours_since_sunrise calculation."""

    def test_midnight_is_zero(self):
        from src.features.engineering import _add_solar_timing_features
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2026-06-21 02:00")],  # well before sunrise
        })
        result = _add_solar_timing_features(df)
        assert result["hours_since_sunrise"].iloc[0] == pytest.approx(0, abs=0.01)

    def test_noon_is_positive(self):
        from src.features.engineering import _add_solar_timing_features
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2026-06-21 12:00")],
        })
        result = _add_solar_timing_features(df)
        assert result["hours_since_sunrise"].iloc[0] > 4  # well after sunrise

    def test_summer_sunrise_earlier(self):
        from src.features.engineering import _add_solar_timing_features
        summer = pd.DataFrame({"datetime": [pd.Timestamp("2026-06-21 08:00")]})
        winter = pd.DataFrame({"datetime": [pd.Timestamp("2026-12-21 08:00")]})
        s = _add_solar_timing_features(summer)["hours_since_sunrise"].iloc[0]
        w = _add_solar_timing_features(winter)["hours_since_sunrise"].iloc[0]
        # In summer, 8:00 is much later after sunrise than in winter
        assert s > w


class TestInstabilityIndices:
    """Test VT/TT/LI computed from pressure level data."""

    def test_vt_index(self):
        """VT = T850 - T500."""
        from src.features.engineering import _add_pressure_level_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temperature_2m": [20.0] * 4,
            "relative_humidity_2m": [70.0] * 4,
            "wind_speed_10m": [10.0] * 4,
            "wind_direction_10m": [180.0] * 4,
            "wind_850_dir": [180.0] * 4,
            "wind_850_speed": [20.0] * 4,
            "temp_850": [10.0] * 4,
            "temp_500": [-15.0] * 4,
            "rh_850": [80.0] * 4,
            "rh_700": [50.0] * 4,
            "temp_700": [0.0] * 4,
            "wind_300_speed": [60.0] * 4,
            "wind_300_dir": [270.0] * 4,
            "gph_300": [9300.0] * 4,
        })
        result = _add_pressure_level_features(df)
        assert result["vt_index"].iloc[0] == pytest.approx(25)  # 10 - (-15)

    def test_deep_layer_shear(self):
        """DLS = |wind_300_speed - wind_850_speed|."""
        from src.features.engineering import _add_pressure_level_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temperature_2m": [20.0] * 4,
            "relative_humidity_2m": [70.0] * 4,
            "wind_speed_10m": [10.0] * 4,
            "wind_direction_10m": [180.0] * 4,
            "wind_850_dir": [180.0] * 4,
            "wind_850_speed": [20.0] * 4,
            "temp_850": [10.0] * 4,
            "temp_500": [-15.0] * 4,
            "rh_850": [80.0] * 4,
            "wind_300_speed": [70.0] * 4,
            "wind_300_dir": [270.0] * 4,
            "gph_300": [9300.0] * 4,
        })
        result = _add_pressure_level_features(df)
        assert result["deep_layer_shear"].iloc[0] == pytest.approx(50)  # |70 - 20|

    def test_inversion_925(self):
        """Inversion = T925 - T_sfc. Positive = inversion layer."""
        from src.features.engineering import _add_pressure_level_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=4, freq="h"),
            "temperature_2m": [15.0] * 4,
            "relative_humidity_2m": [70.0] * 4,
            "wind_speed_10m": [10.0] * 4,
            "wind_direction_10m": [180.0] * 4,
            "wind_850_dir": [180.0] * 4,
            "wind_850_speed": [20.0] * 4,
            "temp_925": [12.0] * 4,  # 925 warmer than sfc
            "rh_925": [80.0] * 4,
            "wind_925_speed": [15.0] * 4,
            "wind_925_dir": [180.0] * 4,
            "temp_850": [10.0] * 4,
            "temp_500": [-15.0] * 4,
            "rh_850": [80.0] * 4,
        })
        result = _add_pressure_level_features(df)
        # T925(12) - T_sfc(15) = -3 (no inversion, normal lapse rate)
        assert result["inversion_925"].iloc[0] == pytest.approx(-3)


class TestModelFeatures:
    """Test NWP-derived features (weather code decomposition, severity)."""

    def test_weather_code_decomposition(self):
        from src.features.engineering import _add_model_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=5, freq="h"),
            "weather_code": [0, 51, 61, 95, 3],  # clear, drizzle, rain, thunderstorm, overcast
            "precipitation": [0, 0.1, 2.0, 5.0, 0],
            "rain": [0, 0.1, 2.0, 5.0, 0],
            "cloud_cover": [0, 80, 90, 100, 70],
            "relative_humidity_2m": [40, 70, 85, 90, 60],
        })
        result = _add_model_features(df)
        # WC 0 = clear
        assert result["model_predicts_precip"].iloc[0] == 0
        assert result["wc_is_rain"].iloc[0] == 0
        # WC 51 = drizzle
        assert result["wc_is_drizzle"].iloc[1] == 1
        assert result["model_predicts_precip"].iloc[1] == 1
        # WC 61 = rain
        assert result["wc_is_rain"].iloc[2] == 1
        # WC 95 = thunderstorm
        assert result["wc_is_thunderstorm"].iloc[3] == 1

    def test_nwp_precip_severity_scale(self):
        """NWP severity: drizzle=1, rain=2, showers=3, snow=4, thunder=5."""
        from src.features.engineering import _add_model_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=6, freq="h"),
            "weather_code": [0, 51, 61, 80, 71, 95],
            "precipitation": [0] * 6,
            "rain": [0] * 6,
            "cloud_cover": [0] * 6,
            "relative_humidity_2m": [50] * 6,
        })
        result = _add_model_features(df)
        assert result["nwp_precip_severity"].iloc[0] == 0  # clear
        assert result["nwp_precip_severity"].iloc[1] == 1  # drizzle
        assert result["nwp_precip_severity"].iloc[2] == 2  # rain
        assert result["nwp_precip_severity"].iloc[3] == 3  # showers
        assert result["nwp_precip_severity"].iloc[5] == 5  # thunder
