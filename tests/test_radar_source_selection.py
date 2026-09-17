"""Tests per a la selecció de font de radar (src/model/predict.py).

El 2026-09-16 RainViewer va servir frames congelats mentre el radar de Meteocat
(local, 6 min, dBZ exacte) veia la tempesta correctament. Aquests tests fixen
la degradació: Meteocat fresca → RainViewer → Meteocat amb dades velles →
RainViewer (encara que estigui congelat, amb les regles físiques desactivades).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.data.aemet_radar import AEMET_FRAME_STALE_STREAK
from src.data.meteocat_radar import to_radar_data
from src.model.predict import _select_radar_source, _aemet_to_radar_data


def _meteocat(**overrides):
    base = {
        "meteocat_radar_available": True,
        "meteocat_radar_dbz": 0.0,
        "meteocat_radar_has_echo": False,
        "meteocat_radar_nearest_echo_km": 2.1,
        "meteocat_radar_nearest_echo_bearing": 157.0,
        "meteocat_radar_nearest_echo_compass": "SSE",
        "meteocat_radar_max_dbz_20km": 52.5,
        "meteocat_radar_coverage_20km": 0.519,
        "meteocat_radar_upwind_nearest_echo_km": 2.1,
        "meteocat_radar_upwind_max_dbz": 52.5,
        "meteocat_radar_storm_drift_kmh": 20.3,
        "meteocat_radar_storm_bearing": 84.0,
        "meteocat_radar_storm_eta_min": None,
        "meteocat_radar_storm_approaching": False,
        "meteocat_radar_frame_time": "2026-09-16T16:48:00+00:00",
        "meteocat_radar_frame_age_min": 17.6,
        "meteocat_radar_same_frame_streak": 0,
    }
    base.update(overrides)
    return base


def _rainviewer(**overrides):
    base = {
        "radar_dbz": 0.0, "radar_has_echo": False, "radar_rain_rate": 0.0,
        "radar_nearest_echo_km": 7.2, "radar_nearest_echo_compass": "SSE",
        "radar_max_dbz_20km": 56.5, "radar_coverage_20km": 0.01,
        "radar_upwind_nearest_echo_km": 7.3, "radar_upwind_max_dbz": 56.5,
        "radar_storm_approaching": 0, "radar_approaching": 0,
        "radar_storm_eta_min": None, "radar_storm_velocity_kmh": 0.0,
        "radar_frames_frozen": False,
    }
    base.update(overrides)
    return base


class TestRadarSourceSelection:
    def test_fresh_meteocat_is_primary(self):
        radar, source = _select_radar_source(_meteocat(), _rainviewer())
        assert source == "meteocat"
        assert radar["radar_source"] == "meteocat"
        assert radar["radar_max_dbz_20km"] == pytest.approx(52.5)
        assert radar["radar_nearest_echo_km"] == pytest.approx(2.1)
        assert radar["radar_frames_frozen"] is False

    def test_rainviewer_used_when_meteocat_unavailable(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(mc, _rainviewer())
        assert source == "rainviewer"
        assert radar["radar_source"] == "rainviewer"
        assert radar["radar_max_dbz_20km"] == pytest.approx(56.5)

    def test_frozen_meteocat_falls_back_to_rainviewer(self):
        mc = _meteocat(meteocat_radar_same_frame_streak=config.METEO_RADAR_FRAME_STALE_STREAK)
        radar, source = _select_radar_source(mc, _rainviewer())
        assert source == "rainviewer"
        assert radar["radar_frames_frozen"] is False

    def test_both_bad_prefers_stale_meteocat_over_frozen_rainviewer(self):
        # Meteocat congelada però amb dades reals > RainViewer congelat
        mc = _meteocat(meteocat_radar_available=False,
                       meteocat_radar_same_frame_streak=5)
        radar, source = _select_radar_source(mc, _rainviewer(radar_frames_frozen=True))
        assert source == "meteocat_stale"
        assert radar["radar_frames_frozen"] is True     # les regles es desactivaran

    def test_nothing_usable_returns_frozen_rainviewer(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(mc, _rainviewer(radar_frames_frozen=True))
        assert source == "rainviewer_frozen"
        assert radar["radar_frames_frozen"] is True


class TestMeteocatToRadarData:
    def test_adapter_exposes_the_features_the_model_consumes(self):
        from src.features.engineering import FEATURE_COLUMNS
        radar = to_radar_data(_meteocat())
        for key in ("radar_dbz", "radar_has_echo", "radar_nearest_echo_km",
                    "radar_max_dbz_20km", "radar_coverage_20km",
                    "radar_upwind_nearest_echo_km", "radar_upwind_max_dbz",
                    "radar_approaching", "radar_storm_approaching",
                    "radar_rain_rate"):
            assert key in radar, f"falta {key}"
            assert key in FEATURE_COLUMNS, f"{key} no és una feature del model"

    def test_rain_rate_derived_from_dbz(self):
        radar = to_radar_data(_meteocat(meteocat_radar_dbz=30.0))
        # Marshall-Palmer: 30 dBZ ≈ 2.7 mm/h
        assert 2.0 < radar["radar_rain_rate"] < 3.5

    def test_velocity_components_from_drift(self):
        radar = to_radar_data(_meteocat())
        # 20.3 km/h cap a 84° → gairebé tot component EW
        assert radar["radar_storm_velocity_ew"] == pytest.approx(20.2, abs=0.5)
        assert radar["radar_storm_velocity_ns"] == pytest.approx(2.1, abs=0.5)

    def test_missing_drift_yields_nan_not_zero(self):
        radar = to_radar_data(_meteocat(meteocat_radar_storm_drift_kmh=None,
                                        meteocat_radar_storm_bearing=None))
        assert radar["radar_storm_velocity_kmh"] != radar["radar_storm_velocity_kmh"]  # NaN
        assert radar["radar_storm_velocity_ew"] != radar["radar_storm_velocity_ew"]

    def test_streak_sets_frozen_flag(self):
        radar = to_radar_data(_meteocat(
            meteocat_radar_same_frame_streak=config.METEO_RADAR_FRAME_STALE_STREAK))
        assert radar["radar_frames_frozen"] is True


def _aemet(**overrides):
    base = {
        "aemet_radar_available": True,
        "aemet_radar_dbz": 38.0,
        "aemet_radar_has_echo": True,
        "aemet_radar_nearest_echo_km": 6.0,
        "aemet_radar_nearest_echo_compass": "SO",
        "aemet_radar_max_dbz_20km": 40.0,
        "aemet_radar_coverage_20km": 0.12,
        "aemet_radar_echoes_found": True,
        "aemet_radar_same_frame_streak": 0,
    }
    base.update(overrides)
    return base


class TestAemetFallback:
    """El mode de fallada d'AEMET (429 sorollós) és millor fallback que el
    congelat silenciós de RainViewer: la cadena és Meteocat → AEMET → RV."""

    def test_aemet_is_second_in_chain(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(mc, _rainviewer(), _aemet())
        assert source == "aemet"
        assert radar["radar_source"] == "aemet"
        assert radar["radar_max_dbz_20km"] == pytest.approx(40.0)
        assert radar["radar_has_echo"] is True
        assert radar["radar_frames_frozen"] is False

    def test_rainviewer_only_after_aemet(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(
            mc, _rainviewer(),
            _aemet(aemet_radar_available=False, aemet_radar_same_frame_streak=None))
        assert source == "rainviewer"
        assert radar["radar_source"] == "rainviewer"

    def test_frozen_aemet_skips_to_rainviewer(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(
            mc, _rainviewer(),
            _aemet(aemet_radar_same_frame_streak=AEMET_FRAME_STALE_STREAK))
        assert source == "rainviewer"
        assert radar["radar_frames_frozen"] is False

    def test_meteocat_stale_beats_frozen_aemet_and_rainviewer(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_same_frame_streak=5)
        radar, source = _select_radar_source(
            mc, _rainviewer(radar_frames_frozen=True),
            _aemet(aemet_radar_same_frame_streak=AEMET_FRAME_STALE_STREAK))
        assert source == "meteocat_stale"

    def test_nothing_usable_still_returns_frozen_rainviewer(self):
        mc = _meteocat(meteocat_radar_available=False, meteocat_radar_frame_time=None)
        radar, source = _select_radar_source(
            mc, _rainviewer(radar_frames_frozen=True),
            _aemet(aemet_radar_same_frame_streak=AEMET_FRAME_STALE_STREAK))
        assert source == "rainviewer_frozen"
        assert radar["radar_frames_frozen"] is True


class TestAemetToRadarData:
    def test_adapter_maps_schema(self):
        from src.features.engineering import FEATURE_COLUMNS
        from src.model.predict import _aemet_to_radar_data
        radar = _aemet_to_radar_data(_aemet())
        for key in ("radar_dbz", "radar_has_echo", "radar_nearest_echo_km",
                    "radar_max_dbz_20km", "radar_coverage_20km",
                    "radar_rain_rate", "radar_approaching",
                    "radar_storm_approaching", "radar_frames_frozen"):
            assert key in radar, f"falta {key}"
        # Les claus que alimenten el vector de features han de ser features
        for key in ("radar_dbz", "radar_has_echo", "radar_nearest_echo_km",
                    "radar_max_dbz_20km", "radar_coverage_20km",
                    "radar_rain_rate", "radar_approaching",
                    "radar_storm_approaching"):
            assert key in FEATURE_COLUMNS, f"{key} no és una feature del model"
        assert radar["radar_source"] == "aemet"
        assert radar["radar_dbz"] == pytest.approx(38.0)

    def test_adapter_is_honest_about_no_motion_tracking(self):
        # AEMET no fa tracking: cap camp inventat, tot None/NaN/False
        from src.model.predict import _aemet_to_radar_data
        radar = _aemet_to_radar_data(_aemet())
        assert radar["radar_approaching"] is False
        assert radar["radar_storm_eta_min"] is None
        assert radar["radar_upwind_nearest_echo_km"] is None
        assert radar["radar_storm_velocity_kmh"] != radar["radar_storm_velocity_kmh"]  # NaN

    def test_frozen_frame_maps_to_frozen_flag(self):
        from src.model.predict import _aemet_to_radar_data
        radar = _aemet_to_radar_data(_aemet(
            aemet_radar_same_frame_streak=AEMET_FRAME_STALE_STREAK))
        assert radar["radar_frames_frozen"] is True

    def test_2b_rule_fires_with_aemet_as_primary(self):
        # Quan AEMET és la font primària, les regles 1/2/2b del radar han de
        # disparar amb les seves dades (no només la regla 5). Píxel sec + eco
        # fort a 7.2km: regla 2b (0.45) i regla 5 (0.55) → max 0.55.
        from src.model.predict import _apply_physical_constraints
        aemet = _aemet(aemet_radar_dbz=0.0, aemet_radar_has_echo=False,
                       aemet_radar_nearest_echo_km=7.2, aemet_radar_max_dbz_20km=56.5)
        radar_data = _aemet_to_radar_data(aemet)
        prob, adj = _apply_physical_constraints(
            0.103, radar_data, {}, aemet_radar_data=aemet)
        assert prob == pytest.approx(0.55)
        assert any("Eco fort" in a for a in adj)
        assert any("Radar AEMET" in a for a in adj)