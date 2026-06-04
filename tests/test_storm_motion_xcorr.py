"""
Tests del vector de moviment per correlació de fase.

Motivació (2026-06-04): el tracking per centroide va donar 0.0 km/h tota la
tarda amb una cèl·lula quasi-estacionària que creixia in situ, i la regla
d'ETA no va disparar mai. La correlació de fase troba el desplaçament que
millor superposa els camps d'eco entre frames, robust a cel·les que
neixen/moren i a múltiples cel·les.
"""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.rainviewer import (
    _echo_field,
    _estimate_motion_xcorr,
    _estimate_storm_tracking,
    _xcorr_displacement,
)

PIXEL_KM = 0.457


def _blob_field(cx, cy, size=256, radius=8, intensity=146.0):
    """Camp 256x256 amb un blob circular d'eco a (cx, cy)."""
    yy, xx = np.mgrid[0:size, 0:size]
    field = np.zeros((size, size))
    field[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = intensity
    return field


def _blob_png(cx, cy, size=256, radius=8, r_value=146):
    """Tile PNG RGBA amb un blob d'eco, com el que serveix RainViewer."""
    from PIL import Image
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    arr[..., 3] = 255  # cobertura radar a tot arreu
    arr[mask, 0] = r_value
    arr[mask, 1] = r_value
    arr[mask, 2] = r_value
    img = Image.fromarray(arr, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestXcorrDisplacement:
    def test_eastward_shift_detected(self):
        prev = _blob_field(100, 128)
        curr = _blob_field(110, 128)  # +10px en x (est)
        dy, dx = _xcorr_displacement(prev, curr)
        assert dx == pytest.approx(10, abs=1)
        assert dy == pytest.approx(0, abs=1)

    def test_southward_shift_detected(self):
        prev = _blob_field(128, 100)
        curr = _blob_field(128, 112)  # +12px en y (sud)
        dy, dx = _xcorr_displacement(prev, curr)
        assert dy == pytest.approx(12, abs=1)
        assert dx == pytest.approx(0, abs=1)

    def test_stationary_blob_gives_zero(self):
        prev = _blob_field(100, 100)
        curr = _blob_field(100, 100)
        dy, dx = _xcorr_displacement(prev, curr)
        assert dy == pytest.approx(0, abs=1)
        assert dx == pytest.approx(0, abs=1)

    def test_growing_stationary_blob_gives_zero(self):
        # Cas 2026-06-04: cèl·lula que creix in situ — el moviment és 0,
        # no un artefacte del creixement
        prev = _blob_field(100, 100, radius=5)
        curr = _blob_field(100, 100, radius=12)
        dy, dx = _xcorr_displacement(prev, curr)
        assert abs(dy) <= 1 and abs(dx) <= 1

    def test_impossible_displacement_rejected(self):
        # >45px/frame (~120 km/h) és soroll, no una tempesta
        prev = _blob_field(30, 128)
        curr = _blob_field(130, 128)  # 100px de salt
        assert _xcorr_displacement(prev, curr) is None


class TestEchoField:
    def test_valid_tile_returns_field(self):
        png = _blob_png(100, 100)
        field = _echo_field(png)
        assert field is not None
        assert field.max() == 146.0

    def test_too_few_echo_pixels_returns_none(self):
        png = _blob_png(100, 100, radius=1)  # ~5px, per sota del mínim de 10
        assert _echo_field(png, None) is None

    def test_clutter_mask_excludes_pixels(self):
        png = _blob_png(100, 100, radius=8)
        clutter = np.zeros((256, 256), dtype=bool)
        clutter[80:120, 80:120] = True  # tapa tot el blob
        assert _echo_field(png, clutter) is None


class TestMotionToVelocity:
    def test_eastward_10px_per_frame_is_27kmh(self):
        # 10px * 0.457 km / 10 min * 60 = 27.4 km/h cap a l'est
        tiles = [_blob_png(90, 128), _blob_png(100, 128), _blob_png(110, 128)]
        velocity = _estimate_motion_xcorr(tiles, None, PIXEL_KM)
        assert velocity is not None
        v_ew, v_ns = velocity
        assert v_ew == pytest.approx(27.4, abs=3)
        assert v_ns == pytest.approx(0, abs=3)

    def test_none_tiles_skipped(self):
        tiles = [None, _blob_png(100, 128), _blob_png(110, 128)]
        velocity = _estimate_motion_xcorr(tiles, None, PIXEL_KM)
        assert velocity is not None

    def test_all_invalid_returns_none(self):
        assert _estimate_motion_xcorr([None, None], None, PIXEL_KM) is None


class TestRadialEta:
    def test_echo_west_moving_east_approaches(self):
        # Eco a l'oest (bearing 270) a 20km, camp movent-se cap a l'est a 30km/h
        scans = [{
            "echoes_found": True,
            "nearest_echo_bearing": 270.0,
            "nearest_echo_km": 20.0,
        }]
        result = _estimate_storm_tracking(scans, PIXEL_KM, xcorr_velocity=(30.0, 0.0))
        assert result["storm_approaching"] is True
        assert result["storm_eta_min"] == pytest.approx(40, abs=2)

    def test_echo_west_moving_west_recedes(self):
        scans = [{
            "echoes_found": True,
            "nearest_echo_bearing": 270.0,
            "nearest_echo_km": 20.0,
        }]
        result = _estimate_storm_tracking(scans, PIXEL_KM, xcorr_velocity=(-30.0, 0.0))
        assert result["storm_approaching"] is False
        assert result["storm_eta_min"] is None

    def test_echo_south_moving_north_approaches(self):
        # Eco al sud (bearing 180) a 10km, movent-se cap al nord (v_ns < 0)
        scans = [{
            "echoes_found": True,
            "nearest_echo_bearing": 180.0,
            "nearest_echo_km": 10.0,
        }]
        result = _estimate_storm_tracking(scans, PIXEL_KM, xcorr_velocity=(0.0, -20.0))
        assert result["storm_approaching"] is True
        assert result["storm_eta_min"] == pytest.approx(30, abs=2)

    def test_stationary_field_no_eta(self):
        scans = [{
            "echoes_found": True,
            "nearest_echo_bearing": 180.0,
            "nearest_echo_km": 7.2,
        }]
        result = _estimate_storm_tracking(scans, PIXEL_KM, xcorr_velocity=(0.0, 0.0))
        assert result["storm_approaching"] is False
        assert result["storm_velocity_kmh"] == 0.0

    def test_no_echoes_no_eta(self):
        scans = [{"echoes_found": False, "nearest_echo_bearing": None,
                  "nearest_echo_km": 60.0}]
        result = _estimate_storm_tracking(scans, PIXEL_KM, xcorr_velocity=(30.0, 0.0))
        assert result["storm_eta_min"] is None

    def test_centroid_fallback_still_works(self):
        # Sense xcorr_velocity, el mètode antic (centroide) segueix actiu
        scans = [
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 10.0, "_centroid_dy": 0.0},
        ]
        result = _estimate_storm_tracking(scans, PIXEL_KM, frame_interval_min=10)
        assert result["storm_velocity_ew"] > 0
