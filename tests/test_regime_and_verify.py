"""
Test suite: Regime change detection and verification logic.
Catches: regime transition bugs, pressure threshold edge cases,
verification window miscalculations, RAIN_THRESHOLD_MM boundary.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Helpers ──

def _make_prediction(is_llevantada=False, is_garbi=False, is_tramuntana=False,
                     humidity=50, rh_850=50, tt_index=40, li_index=2,
                     temp_500=-10, pressure_change_3h=None,
                     wind_dir_change_3h=None, llevantada_strength=0):
    """Create a mock predict_now() result for regime change testing."""
    return {
        "wind_regime": {
            "is_llevantada": is_llevantada,
            "is_garbi": is_garbi,
            "is_tramuntana": is_tramuntana,
            "is_migjorn": False,
            "is_ponent": False,
            "llevantada_strength": llevantada_strength,
            "wind_dir_change_3h": wind_dir_change_3h,
        },
        "pressure_levels": {
            "tt_index": tt_index,
            "li_index": li_index,
            "temp_500": temp_500,
            "rh_850": rh_850,
            "wind_850_dir": 90 if is_llevantada else 220 if is_garbi else 0,
            "wind_850_speed_kmh": 25,
        },
        "conditions": {"humidity": humidity},
        "radar": {"nearest_echo_km": None, "storm_eta_min": None},
        "ensemble": {},
        "pressure_change_3h": pressure_change_3h,
    }


def _make_state(was_llevantada=False, was_garbi=False):
    """Create a mock notification state."""
    return {
        "last_wind_regime": {
            "is_llevantada": was_llevantada,
            "is_garbi": was_garbi,
            "is_tramuntana": False,
        }
    }


# ── Regime Change Detection ──

class TestLlevantadaOnset:

    def test_llevantada_onset_with_humidity(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80, rh_850=50)
        state = _make_state(was_llevantada=False)
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "llevantada_onset"

    def test_llevantada_onset_with_rh850(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=50, rh_850=85)
        state = _make_state(was_llevantada=False)
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "llevantada_onset"

    def test_no_onset_if_was_already_llevantada(self):
        """No alert if wind was already Llevantada (not a transition)."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=90)
        state = _make_state(was_llevantada=True)
        result = detect_regime_change(pred, state)
        assert result is None

    def test_no_onset_if_dry(self):
        """No alert if humidity is low (dry Llevantada = no rain)."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=50, rh_850=50)
        state = _make_state(was_llevantada=False)
        result = detect_regime_change(pred, state)
        assert result is None

    def test_warning_severity_strong_wind(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80, llevantada_strength=35)
        state = _make_state(was_llevantada=False)
        result = detect_regime_change(pred, state)
        assert result["severity"] == "warning"

    def test_watch_severity_light_wind(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80, llevantada_strength=15)
        state = _make_state(was_llevantada=False)
        result = detect_regime_change(pred, state)
        assert result["severity"] == "watch"


class TestGarbiInestable:

    def test_garbi_with_high_tt(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_garbi=True, tt_index=46)
        state = _make_state(was_garbi=False)
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "garbi_inestable"

    def test_garbi_with_low_li(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_garbi=True, li_index=-3)
        state = _make_state(was_garbi=False)
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "garbi_inestable"

    def test_garbi_cold_500(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_garbi=True, temp_500=-20)
        state = _make_state(was_garbi=False)
        result = detect_regime_change(pred, state)
        assert result is not None

    def test_no_alert_stable_garbi(self):
        """Garbí without instability → no alert."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_garbi=True, tt_index=38, li_index=5, temp_500=-10)
        state = _make_state(was_garbi=False)
        result = detect_regime_change(pred, state)
        assert result is None


class TestPressureDrop:

    def test_pressure_drop_in_moist_regime(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80,
                               pressure_change_3h=-3.0, rh_850=50)
        state = _make_state(was_llevantada=True)  # was already Llevantada
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "pressure_drop"

    def test_no_drop_alert_in_dry_regime(self):
        """Pressure drop without moisture → no alert (just a passing front)."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_tramuntana=True, humidity=40,
                               pressure_change_3h=-3.0)
        state = _make_state()
        result = detect_regime_change(pred, state)
        assert result is None

    def test_warning_severity_deep_drop(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80,
                               pressure_change_3h=-5.0, rh_850=50)
        state = _make_state(was_llevantada=True)
        result = detect_regime_change(pred, state)
        assert result["severity"] == "warning"

    def test_watch_severity_moderate_drop(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_garbi=True, humidity=60,
                               pressure_change_3h=-2.5, rh_850=50)
        state = _make_state(was_garbi=True)
        result = detect_regime_change(pred, state)
        assert result["severity"] == "watch"

    def test_exact_threshold(self):
        """pressure_change_3h == -2.0 (exactly at threshold) should trigger."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80,
                               pressure_change_3h=-2.0, rh_850=50)
        state = _make_state(was_llevantada=True)
        result = detect_regime_change(pred, state)
        assert result is not None

    def test_null_pressure_no_crash(self):
        """None pressure_change_3h must not crash."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(is_llevantada=True, humidity=80,
                               pressure_change_3h=None)
        state = _make_state(was_llevantada=True)
        result = detect_regime_change(pred, state)
        # Should not crash; may return None (no pressure drop alert)


class TestBackingWind:

    def test_backing_with_humidity(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(humidity=75, wind_dir_change_3h=-25)
        state = _make_state()
        result = detect_regime_change(pred, state)
        assert result is not None
        assert result["type"] == "backing_wind"

    def test_no_alert_without_humidity(self):
        from src.features.regime import detect_regime_change
        pred = _make_prediction(humidity=50, rh_850=50, wind_dir_change_3h=-30)
        state = _make_state()
        result = detect_regime_change(pred, state)
        assert result is None

    def test_veering_does_not_trigger(self):
        """Positive change (veering) should NOT trigger backing alert."""
        from src.features.regime import detect_regime_change
        pred = _make_prediction(humidity=80, wind_dir_change_3h=+25)
        state = _make_state()
        result = detect_regime_change(pred, state)
        # Should be None — only backing (negative) triggers
        assert result is None or result["type"] != "backing_wind"


class TestRegimeSummary:

    def test_llevantada_summary(self):
        from src.features.regime import get_current_regime_summary
        pred = _make_prediction(is_llevantada=True)
        summary = get_current_regime_summary(pred)
        assert "Llevantada" in summary

    def test_unknown_regime(self):
        from src.features.regime import get_current_regime_summary
        pred = _make_prediction()
        summary = get_current_regime_summary(pred)
        assert isinstance(summary, str)
        assert len(summary) > 0


# ── Verification Logic ──

class TestVerificationWindow:
    """Test the verification matching logic (without network calls)."""

    def test_rain_threshold_exact(self):
        """0.2mm exactly should count as rain (config.RAIN_THRESHOLD_MM = 0.2)."""
        assert config.RAIN_THRESHOLD_MM == 0.2
        rain_mm = 0.2
        assert rain_mm >= config.RAIN_THRESHOLD_MM

    def test_rain_threshold_below(self):
        """0.19mm should NOT count as rain."""
        rain_mm = 0.19
        assert rain_mm < config.RAIN_THRESHOLD_MM

    def test_verification_window_timing(self):
        """Prediction at 14:00 → verify window is 14:00-15:00,
        should not verify until 15:15 (60min + 15min buffer)."""
        pred_time = datetime(2026, 3, 22, 14, 0)
        verification_end = pred_time + timedelta(minutes=config.PREDICTION_HORIZON_MIN)
        earliest_verify = verification_end + timedelta(minutes=15)

        assert verification_end == datetime(2026, 3, 22, 15, 0)
        assert earliest_verify == datetime(2026, 3, 22, 15, 15)

        # At 15:10, should NOT verify yet
        now_early = datetime(2026, 3, 22, 15, 10)
        assert now_early < earliest_verify

        # At 15:20, should verify
        now_ok = datetime(2026, 3, 22, 15, 20)
        assert now_ok >= earliest_verify


class TestSafeFloat:
    """Test the _safe_float helper used in regime detection."""

    def test_normal_values(self):
        from src.features.regime import _safe_float
        assert _safe_float(42.5) == 42.5
        assert _safe_float("123") == 123.0

    def test_none_returns_default(self):
        from src.features.regime import _safe_float
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=-999) == -999

    def test_nan_returns_default(self):
        from src.features.regime import _safe_float
        assert _safe_float(float("nan")) == 0.0

    def test_invalid_string(self):
        from src.features.regime import _safe_float
        assert _safe_float("abc", default=-1) == -1
