"""Regression tests for dry/missing radar and repeated AEMET frames."""
import pytest

import config
from src.model.predict import _apply_physical_constraints


def test_frozen_aemet_coverage_cannot_raise_probability():
    aemet = {
        "aemet_radar_same_frame_streak": 100,
        "aemet_radar_nearest_echo_km": 10,
        "aemet_radar_max_dbz_20km": 40,
        "aemet_radar_coverage_20km": 0.5,
    }
    probability, adjustments = _apply_physical_constraints(
        0.02, {}, {}, aemet_radar_data=aemet)
    assert probability == pytest.approx(0.02)
    assert adjustments == []


@pytest.mark.parametrize("distance", [None, float("nan"), float("inf"), -1, "bad"])
def test_invalid_radar_distance_does_not_open_gate(distance):
    from src.model.predict import _radar_opens_rain_gate
    assert not _radar_opens_rain_gate({"radar_nearest_echo_km": distance})


def test_frozen_pixel_echo_cannot_open_gate():
    from src.model.predict import _radar_opens_rain_gate
    assert not _radar_opens_rain_gate({
        "radar_frames_frozen": True, "radar_has_echo": True,
        "radar_nearest_echo_km": 0,
    })


@pytest.mark.parametrize("distance", [0, 1, config.RAIN_GATE_RADAR_NEARBY_KM - 0.1])
def test_fresh_nearby_echo_opens_gate(distance):
    from src.model.predict import _radar_opens_rain_gate
    assert _radar_opens_rain_gate({"radar_nearest_echo_km": distance})


def test_fresh_pixel_echo_opens_gate_without_distance():
    from src.model.predict import _radar_opens_rain_gate
    assert _radar_opens_rain_gate({"radar_has_echo": True})


def test_dry_radar_does_not_open_gate():
    from src.model.predict import _radar_opens_rain_gate
    assert not _radar_opens_rain_gate({
        "radar_has_echo": False, "radar_nearest_echo_km": None,
    })
