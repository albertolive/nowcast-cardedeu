"""
Test suite: Wind regime classification and direction utilities.
Catches: direction boundary wrapping at 360/0, regime overlap/gaps,
850hPa vs 10m fallback logic, interaction term formulas.
"""
import pytest
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDirInRange:
    """Test _dir_in_range — especially 360°/0° wrap-around."""

    def setup_method(self):
        from src.features.engineering import _dir_in_range
        self._dir_in_range = _dir_in_range

    def test_simple_range(self):
        """60°-150° (Llevantada) — no wrap."""
        dirs = pd.Series([90, 60, 150, 59, 151])
        result = self._dir_in_range(dirs, 60, 150)
        assert list(result) == [1, 1, 1, 0, 0]

    def test_wrap_around_tramuntana(self):
        """340°-60° (Tramuntana) — wraps around 0°."""
        dirs = pd.Series([0, 350, 60, 340, 30, 61, 339])
        result = self._dir_in_range(dirs, 340, 60)
        assert list(result) == [1, 1, 1, 1, 1, 0, 0]

    def test_exact_boundaries(self):
        """Boundary values must be inclusive."""
        dirs = pd.Series([340, 60, 150, 190, 250])
        # Tramuntana 340-60: 340=in, 60=in
        assert self._dir_in_range(dirs, 340, 60).iloc[0] == 1
        assert self._dir_in_range(dirs, 340, 60).iloc[1] == 1
        # Llevantada 60-150: 60=in, 150=in
        assert self._dir_in_range(dirs, 60, 150).iloc[1] == 1
        assert self._dir_in_range(dirs, 60, 150).iloc[2] == 1
        # Migjorn 150-190: 150=in, 190=in
        assert self._dir_in_range(dirs, 150, 190).iloc[2] == 1
        assert self._dir_in_range(dirs, 150, 190).iloc[3] == 1

    def test_full_360_coverage_no_gaps(self):
        """Every degree 0-359 must fall into exactly one regime."""
        all_dirs = pd.Series(range(360))
        tramuntana = self._dir_in_range(all_dirs, 340, 60)
        llevantada = self._dir_in_range(all_dirs, 60, 150)
        migjorn = self._dir_in_range(all_dirs, 150, 190)
        garbi = self._dir_in_range(all_dirs, 190, 250)
        ponent = self._dir_in_range(all_dirs, 250, 340)

        total = tramuntana + llevantada + migjorn + garbi + ponent

        # At boundaries (60, 150, 190, 250, 340) exactly 2 regimes overlap
        # That's expected — the boundary directions belong to both adjacent regimes
        gaps = (total == 0).sum()
        assert gaps == 0, f"{gaps} degrees with no regime — coverage gap!"

        # Interior degrees should be exactly 1
        for deg in [0, 30, 90, 120, 170, 220, 300]:
            assert total.iloc[deg] == 1, f"Degree {deg} maps to {total.iloc[deg]} regimes"


class TestAngularDiff:
    """Test angular difference across 360°/0° boundary."""

    def setup_method(self):
        from src.features.engineering import _angular_diff
        self._angular_diff = _angular_diff

    def test_simple_veering(self):
        """90° → 120° = +30° (veering/clockwise)."""
        dirs = pd.Series([90, 100, 110, 120])
        result = self._angular_diff(dirs, 1)
        assert result.iloc[1] == pytest.approx(10)
        assert result.iloc[3] == pytest.approx(10)

    def test_simple_backing(self):
        """120° → 90° = -30° (backing/counter-clockwise)."""
        dirs = pd.Series([120, 110, 100, 90])
        result = self._angular_diff(dirs, 1)
        assert result.iloc[1] == pytest.approx(-10)

    def test_wrap_around_north(self):
        """350° → 10° should be +20°, not -340°."""
        dirs = pd.Series([350, 0, 10])
        result = self._angular_diff(dirs, 1)
        assert result.iloc[1] == pytest.approx(10)  # 0 - 350 → normalized to +10
        assert result.iloc[2] == pytest.approx(10)  # 10 - 0 = +10

    def test_wrap_around_clockwise(self):
        """10° → 350° should be -20°, not +340°."""
        dirs = pd.Series([10, 0, 350])
        result = self._angular_diff(dirs, 1)
        assert result.iloc[1] == pytest.approx(-10)
        assert result.iloc[2] == pytest.approx(-10)

    def test_180_degree_change(self):
        """Exactly 180° is ambiguous but must not crash."""
        dirs = pd.Series([0, 180])
        result = self._angular_diff(dirs, 1)
        assert abs(result.iloc[1]) == pytest.approx(180)

    def test_first_period_is_nan(self):
        """First `periods` values should be NaN."""
        dirs = pd.Series([90, 120, 150, 180])
        result = self._angular_diff(dirs, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == pytest.approx(90)


class TestWindRegimeClassification:
    """Test complete wind regime feature generation."""

    def _make_df(self, wind_850_dir, wind_850_speed=20, humidity=70, n=6):
        now = pd.Timestamp.now().floor("h")
        return pd.DataFrame({
            "datetime": [now + pd.Timedelta(hours=i) for i in range(n)],
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
            "relative_humidity_2m": [humidity] * n,
            "wind_850_dir": [wind_850_dir] * n,
            "wind_850_speed": [wind_850_speed] * n,
        })

    def test_llevantada_classification(self):
        from src.features.engineering import _add_wind_regime_features
        df = self._make_df(wind_850_dir=90)
        result = _add_wind_regime_features(df)
        assert result["is_llevantada"].iloc[0] == 1
        assert result["is_tramuntana"].iloc[0] == 0

    def test_tramuntana_wraps_around_north(self):
        from src.features.engineering import _add_wind_regime_features
        for deg in [350, 0, 10, 50]:
            df = self._make_df(wind_850_dir=deg)
            result = _add_wind_regime_features(df)
            assert result["is_tramuntana"].iloc[0] == 1, f"{deg}° should be Tramuntana"

    def test_garbi_classification(self):
        from src.features.engineering import _add_wind_regime_features
        df = self._make_df(wind_850_dir=220)
        result = _add_wind_regime_features(df)
        assert result["is_garbi"].iloc[0] == 1

    def test_interaction_terms(self):
        from src.features.engineering import _add_wind_regime_features
        df = self._make_df(wind_850_dir=90, wind_850_speed=25, humidity=80)
        result = _add_wind_regime_features(df)
        assert result["llevantada_strength"].iloc[0] == pytest.approx(25)
        assert result["llevantada_moisture"].iloc[0] == pytest.approx(0.80)

    def test_non_active_regime_has_zero_interaction(self):
        from src.features.engineering import _add_wind_regime_features
        df = self._make_df(wind_850_dir=220)  # Garbí, not Llevantada
        result = _add_wind_regime_features(df)
        assert result["llevantada_strength"].iloc[0] == 0
        assert result["llevantada_moisture"].iloc[0] == 0

    def test_850hpa_preferred_over_10m(self):
        """When 850hPa is available, 10m wind should NOT be used for regimes."""
        from src.features.engineering import _add_wind_regime_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=6, freq="h"),
            "wind_speed_10m": [10.0] * 6,
            "wind_direction_10m": [90.0] * 6,   # 10m says Llevantada
            "relative_humidity_2m": [70.0] * 6,
            "wind_850_dir": [220.0] * 6,         # 850hPa says Garbí
            "wind_850_speed": [25.0] * 6,
        })
        result = _add_wind_regime_features(df)
        # 850hPa wins: Garbí, not Llevantada
        assert result["is_garbi"].iloc[0] == 1
        assert result["is_llevantada"].iloc[0] == 0

    def test_fallback_to_10m_when_850_nan(self):
        """When 850hPa is all NaN, fall back to 10m surface wind."""
        from src.features.engineering import _add_wind_regime_features
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=6, freq="h"),
            "wind_speed_10m": [10.0] * 6,
            "wind_direction_10m": [90.0] * 6,   # 10m says Llevantada
            "relative_humidity_2m": [70.0] * 6,
            "wind_850_dir": [np.nan] * 6,        # All NaN
            "wind_850_speed": [np.nan] * 6,
        })
        result = _add_wind_regime_features(df)
        assert result["is_llevantada"].iloc[0] == 1


class TestWindRegimeConsistency:
    """
    Verifica coherència entre wind_850_dir i els règims derivats.
    Reprodueix el bug del 22/03: 850hPa Garbí (248°) + superfície ESE (112°)
    → feature engineering ha d'usar 850hPa, no superfície.
    """

    def test_garbi_850_with_ese_surface(self):
        """850hPa a 248° (Garbí) + superfície ESE (112°) → usa 850hPa."""
        from src.features.engineering import _add_wind_regime_features
        df = pd.DataFrame({
            "wind_850_dir": [248.0],
            "wind_850_speed": [8.4],
            "wind_speed_10m": [4.8],
            "wind_direction_10m": [112.0],
            "relative_humidity_2m": [69.0],
        })
        result = _add_wind_regime_features(df)
        assert result["garbi_strength"].values[0] == pytest.approx(8.4)
        assert result["llevantada_strength"].values[0] == 0.0
        assert result["is_garbi"].values[0] == 1
        assert result["is_llevantada"].values[0] == 0

    def test_partial_nan_850_no_surface_fallback(self):
        """
        wind_850_dir NaN al row actual però disponible en altres rows →
        usa 850hPa (NaN) per row actual, NO cau a superfície.
        Resultat: regimes = 0 per row actual (NaN no és cap direcció).
        """
        from src.features.engineering import _add_wind_regime_features
        df = pd.DataFrame({
            "wind_850_dir": [np.nan, 248.0],
            "wind_850_speed": [np.nan, 8.4],
            "wind_speed_10m": [4.8, 5.0],
            "wind_direction_10m": [112.0, 200.0],
            "relative_humidity_2m": [69.0, 50.0],
        })
        result = _add_wind_regime_features(df)
        # Row 0: NaN → tots els règims = 0
        assert result["llevantada_strength"].values[0] == 0.0
        assert result["garbi_strength"].values[0] == 0.0
        # Row 1: 248° → Garbí
        assert result["garbi_strength"].values[1] == pytest.approx(8.4)
        assert result["llevantada_strength"].values[1] == 0.0
