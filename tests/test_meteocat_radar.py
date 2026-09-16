"""
Tests per al client del radar de Meteocat (src/data/meteocat_radar.py).

Cobreix: geometria de tiles (x, y en TMS, píxel de Cardedeu), descodificació
EXACTA de la paleta (llegenda oficial en passos de 3 dBZ), escaneig espacial
(eco més proper, dBZ màxim, cobertura, quadrants, sobrevent), deriva de la
cel·la entre frames, descobriment del darrer frame (latència 6-12 min) i els
guards d'edat/congelació.

Els tests del grup TestLive necesiten xarxa: `pytest -m network`.
"""
import io
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.data import meteocat_radar as mr


# ── Utilitats: tiles sintètiques ──

def _tile_bytes(cells=(), size=256):
    """Tile RGBA amb la paleta oficial. cells = [((px, py), color), ...]."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    for (px, py), color in cells:
        arr[py, px] = (*color, 255)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


class TestGeometry:
    def test_cardedeu_tile_is_64_80_tms(self):
        # Cas real verificat: el visor demana 07/.../064/000/000/080.png
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        assert (x, y_tms) == (64, 80)

    def test_tms_y_is_mirrored_from_xyz(self):
        # y_tms = (2^z - 1) - y_xyz → a z7, 127 - 47 = 80
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        assert y_tms == 127 - 47

    def test_pixel_of_cardedeu_inside_tile(self):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        px, py = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        assert 200 < px < 230      # prop de la vora est
        assert 165 < py < 190
        assert (round(px), round(py)) == (215, 177)

    def test_km_per_pixel_at_z7(self):
        # z7 a 41.6N → ~0.91 km/px (comparable a la config de RainViewer)
        assert 0.85 < mr._km_per_pixel(41.63) < 0.95

    def test_tile_url_format(self):
        dt = datetime(2026, 9, 16, 16, 36, tzinfo=timezone.utc)
        assert mr._tile_url(dt, 64, 80) == (
            "https://static-m.meteo.cat/tiles/radar/2026/09/16/16/36/07"
            "/000/000/064/000/000/080.png")

    def test_floor_to_6min_grid(self):
        base = datetime(2026, 9, 16, 16, 0, tzinfo=timezone.utc)
        for minute, expected in ((0, 0), (5, 0), (6, 6), (41, 36), (54, 54), (59, 54)):
            got = mr._floor_to_grid(base.replace(minute=minute))
            assert got.minute == expected, f"minut {minute}"


class TestLegendDecoding:
    def test_every_legend_color_decodes_to_its_class(self):
        # Cada color de la llegenda = una classe de 3 dBZ (centre = lower + 1.5)
        cells = [((i % 16, i // 16), color) for i, color in enumerate(mr.LEGEND_DBZ)]
        arr = np.array(Image.open(io.BytesIO(_tile_bytes(cells))).convert("RGBA"))
        has_echo, dbz = mr._decode_dbz(arr)
        for i, (color, lo) in enumerate(mr.LEGEND_DBZ.items()):
            px, py = i % 16, i // 16
            assert has_echo[py, px], f"color {color} no detectat"
            assert dbz[py, px] == lo + 1.5, f"color {color}"

    def test_off_palette_color_is_not_echo(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = (1, 2, 3, 255)      # color antic de paleta de RV
        has_echo, dbz = mr._decode_dbz(arr)
        assert not has_echo.any()
        assert dbz.max() == 0.0

    def test_transparent_pixel_is_not_echo(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[0, 0] = (0, 255, 0, 0)      # verd amb alpha 0
        has_echo, _ = mr._decode_dbz(arr)
        assert not has_echo.any()

    def test_live_palette_samples(self):
        # Valors reals del tile del 2026-09-16 16:36
        assert mr.LEGEND_DBZ[(128, 0, 255)] == 9     # feble
        assert mr.LEGEND_DBZ[(0, 255, 0)] == 24      # moderada
        assert mr.LEGEND_DBZ[(255, 255, 0)] == 36    # forta
        assert mr.LEGEND_DBZ[(255, 0, 0)] == 51      # molt forta
        assert mr.LEGEND_DBZ[(255, 0, 255)] == 63    # calamarsa


# ── Escaneig espacial ──

class TestSpatialScan:
    @staticmethod
    def _mosaic_with_cell(offset_px, color=(0, 255, 0), size=512):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        px, py = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        arr = np.zeros((256, size, 4), dtype=np.uint8)
        arr[int(py) + offset_px[1], int(px) + offset_px[0]] = (*color, 255)
        return arr, x, y_tms

    def test_empty_mosaic_has_no_echo(self):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE,
                               np.zeros((256, 512, 4), dtype=np.uint8), x, y_tms)
        assert out["nearest_echo_km"] is None
        assert out["max_dbz_20km"] == 0.0

    def test_nearest_echo_distance_and_bearing(self):
        # Eco ~10 px a l'est de Cardedeu → ~9 km, rumb a l'entorn de 90° (E)
        mosaic, x, y_tms = self._mosaic_with_cell((10, 0))
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms)
        km_px = mr._km_per_pixel(config.LATITUDE)
        assert out["nearest_echo_km"] == pytest.approx(10 * km_px, abs=1.0)
        assert out["nearest_echo_bearing"] == pytest.approx(90, abs=10)

    def test_max_dbz_20km_and_quadrant(self):
        # Cel·la forta (45-48 dBZ) a 5 px al nord → quadrant N
        mosaic, x, y_tms = self._mosaic_with_cell((0, -5), color=(255, 87, 0))
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms)
        assert out["max_dbz_20km"] == pytest.approx(46.5)
        assert out["quadrant_max_dbz"]["N"] == pytest.approx(46.5)
        assert out["quadrant_max_dbz"]["S"] == 0.0

    def test_coverage_fraction(self):
        mosaic, x, y_tms = self._mosaic_with_cell((0, 0))
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms)
        # Un sol píxel dins 20km: cobertura mínima però > 0
        assert 0 < out["coverage_20km"] < 0.01

    def test_upwind_sector_uses_wind_direction(self):
        # Eco a l'oest (270°) i vent de l'oest (270) → és al sector de sobrevent
        mosaic, x, y_tms = self._mosaic_with_cell((-10, 0))
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms,
                               wind_from_dir=270)
        assert out["upwind_nearest_echo_km"] is not None
        assert out["upwind_max_dbz"] > 0

    def test_upwind_sector_empty_when_echo_downwind(self):
        mosaic, x, y_tms = self._mosaic_with_cell((10, 0))
        out = mr._scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms,
                               wind_from_dir=270)
        assert out["upwind_nearest_echo_km"] is None


class TestStormDrift:
    def test_approaching_cell_gets_eta(self):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        cx, cy = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        km_px = mr._km_per_pixel(config.LATITUDE)
        prev = np.zeros((256, 512), dtype=float)
        curr = np.zeros((256, 512), dtype=float)
        h_prev = np.zeros((256, 512), dtype=bool)
        h_curr = np.zeros((256, 512), dtype=bool)
        # Cel·la 20 px a l'oest que avança 4 px cap a Cardedeu en 6 min
        ix, iy = int(round(cx)), int(round(cy))
        prev[iy, ix - 20] = 40.0
        curr[iy, ix - 16] = 40.0
        h_prev[iy, ix - 20] = True
        h_curr[iy, ix - 16] = True
        d = mr._estimate_drift(prev, curr, h_prev, h_curr, km_px, 6.0, cx, cy)
        assert d["approaching"] is True
        assert d["speed_kmh"] == pytest.approx(4 * km_px * 10, rel=0.15)  # km/6min → km/h
        assert d["bearing"] == pytest.approx(90, abs=15)   # movent-se cap a l'est
        assert d["eta_min"] is not None and d["eta_min"] > 0

    def test_receding_cell_has_no_eta(self):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        cx, cy = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        prev = np.zeros((256, 512), dtype=float)
        curr = np.zeros((256, 512), dtype=float)
        h_prev = np.zeros((256, 512), dtype=bool)
        h_curr = np.zeros((256, 512), dtype=bool)
        ix, iy = int(round(cx)), int(round(cy))
        prev[iy, ix - 16] = 40.0
        curr[iy, ix - 20] = 40.0
        h_prev[iy, ix - 16] = True
        h_curr[iy, ix - 20] = True
        d = mr._estimate_drift(prev, curr, h_prev, h_curr,
                               mr._km_per_pixel(config.LATITUDE), 6.0, cx, cy)
        assert d["approaching"] is False
        assert d["eta_min"] is None

    def test_no_echo_in_one_frame_gives_no_drift(self):
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        cx, cy = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        empty_h = np.zeros((256, 512), dtype=bool)
        d = mr._estimate_drift(np.zeros((256, 512)), np.zeros((256, 512)),
                               empty_h, empty_h, mr._km_per_pixel(config.LATITUDE),
                               6.0, cx, cy)
        assert d["speed_kmh"] is None and d["approaching"] is False

    def test_implausible_jump_is_discarded(self):
        # Salt de 30 px en 6 min ≈ 274 km/h: creixement/decadència, no moviment.
        # Sense el guard, aquest valor anava directe a les regles físiques
        # (2026-09-16: 586 km/h amb el mosaic desalineat).
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        cx, cy = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        ix, iy = int(round(cx)), int(round(cy))
        prev = np.zeros((256, 512), dtype=float)
        curr = np.zeros((256, 512), dtype=float)
        h_prev = np.zeros((256, 512), dtype=bool)
        h_curr = np.zeros((256, 512), dtype=bool)
        prev[iy, ix - 40] = 40.0
        curr[iy, ix - 10] = 40.0
        h_prev[iy, ix - 40] = True
        h_curr[iy, ix - 10] = True
        d = mr._estimate_drift(prev, curr, h_prev, h_curr,
                               mr._km_per_pixel(config.LATITUDE), 6.0, cx, cy)
        assert d["speed_kmh"] is None
        assert d["eta_min"] is None and d["approaching"] is False


# ── Descobriment de frame, guards i API pública ──

@pytest.fixture()
def state_file(tmp_path, monkeypatch):
    """Aïlla l'estat entre runs en un fitxer temporal."""
    p = tmp_path / "meteocat_radar_state.json"
    monkeypatch.setattr(config, "METEO_RADAR_STATE_FILE", str(p))
    return p


class TestFrameDiscovery:
    def test_picks_freshest_available_slot(self, monkeypatch):
        # La font publica amb latència: el slot actual i el següent són 404
        valid = _tile_bytes([((10, 10), (0, 255, 0))])
        calls = []

        def fake_fetch(url):
            calls.append(url)
            return None if len(calls) <= 2 else valid

        monkeypatch.setattr(mr, "_fetch_tile", fake_fetch)
        now = datetime(2026, 9, 16, 16, 42, tzinfo=timezone.utc)
        frame_dt, raw, attempts = mr._find_latest_frame(now)
        assert frame_dt == datetime(2026, 9, 16, 16, 30, tzinfo=timezone.utc)
        assert raw == valid and attempts == 3

    def test_returns_none_when_nothing_available(self, monkeypatch):
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: None)
        frame_dt, raw, attempts = mr._find_latest_frame(
            datetime(2026, 9, 16, 16, 42, tzinfo=timezone.utc))
        assert frame_dt is None and raw is None
        assert attempts == config.METEO_RADAR_MAX_LOOKBACK_SLOTS

    def test_lookback_stays_on_6min_grid(self, monkeypatch):
        seen = []
        monkeypatch.setattr(mr, "_fetch_tile",
                            lambda url: (seen.append(url), None)[1])
        mr._find_latest_frame(datetime(2026, 9, 16, 16, 41, tzinfo=timezone.utc))
        minutes = [u.split("/")[9] for u in seen]      # .../2026/09/16/16/36/...
        assert minutes[0] == "36"                      # 16:41 → slot 16:36
        assert minutes[1] == "30"


class TestPublicApi:
    @staticmethod
    def _tile_over_cardedeu(color=(0, 255, 0)):
        """Tile amb un eco exactament al píxel de Cardedeu."""
        x, y_tms = mr._tile_coords(config.LATITUDE, config.LONGITUDE)
        px, py = mr._pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        return _tile_bytes([((int(round(px)), int(round(py))), color)])

    def test_fetch_success_populates_features(self, monkeypatch, state_file):
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: self._tile_over_cardedeu())
        out = mr.fetch_meteocat_radar()
        assert out["meteocat_radar_available"] is True
        assert out["meteocat_radar_has_echo"] is True
        assert out["meteocat_radar_dbz"] == pytest.approx(25.5)
        assert out["meteocat_radar_nearest_echo_km"] == pytest.approx(0.0, abs=1.0)
        assert out["meteocat_radar_frame_md5"]

    def test_fetch_without_frames_returns_empty(self, monkeypatch, state_file):
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: None)
        out = mr.fetch_meteocat_radar()
        assert out["meteocat_radar_available"] is False
        assert out["meteocat_radar_echoes_found"] is False
        assert out["meteocat_radar_nearest_echo_km"] is None

    def test_old_frame_is_rejected_as_stale(self, monkeypatch, state_file):
        tile = _tile_bytes([((215, 176), (0, 255, 0))])
        # Frame trobat fa 90 min → per sobre del llindar (30 min)
        old_slot = mr._floor_to_grid(
            datetime.now(timezone.utc) - timedelta(minutes=90))
        monkeypatch.setattr(mr, "_fetch_tile",
                            lambda url: tile if f"{old_slot:%H/%M}" in url else None)
        out = mr.fetch_meteocat_radar()
        assert out["meteocat_radar_available"] is False

    def test_same_frame_increments_streak(self, monkeypatch, state_file):
        tile = _tile_bytes([((215, 176), (0, 255, 0))])
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: tile)
        first = mr.fetch_meteocat_radar()
        second = mr.fetch_meteocat_radar()
        assert first["meteocat_radar_same_frame_streak"] == 0
        assert second["meteocat_radar_same_frame_streak"] == 1

    def test_mosaic_shape_mismatch_skips_drift(self, monkeypatch, state_file):
        # Si el frame anterior no té la mateixa geometria (tile de l'est
        # absent), la deriva s'ha de desactivar en lloc de comparar mosaics
        # desalineats.
        calls = []

        def fake_mosaic(frame_dt, x, y_tms):
            calls.append(frame_dt)
            return np.zeros((256, 512, 4), dtype=np.uint8) if len(calls) == 1 \
                else np.zeros((256, 256, 4), dtype=np.uint8)

        monkeypatch.setattr(mr, "_mosaic", fake_mosaic)
        monkeypatch.setattr(mr, "_fetch_tile",
                            lambda url: self._tile_over_cardedeu())
        out = mr.fetch_meteocat_radar()
        assert out["meteocat_radar_available"] is True
        assert out["meteocat_radar_storm_drift_kmh"] is None
        assert out["meteocat_radar_storm_approaching"] is False

    def test_stale_fallback_returns_last_valid_result(self, monkeypatch, state_file):
        tile = self._tile_over_cardedeu()
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: tile)
        mr.fetch_meteocat_radar()                     # omple l'estat
        monkeypatch.setattr(mr, "_fetch_tile", lambda url: None)
        out = mr.fetch_meteocat_radar()               # cau al darrer estat vàlid
        assert out["meteocat_radar_available"] is False
        assert out["meteocat_radar_dbz"] == pytest.approx(25.5)   # dades encara útils


class TestLive:
    """Tests amb xarxa real (``pytest -m network``)."""

    @pytest.mark.network
    def test_live_frame_is_fresh_and_decodes(self, state_file):
        out = mr.fetch_meteocat_radar()
        assert out["meteocat_radar_available"] is True, "cap frame fresca de Meteocat"
        assert out["meteocat_radar_frame_age_min"] <= config.METEO_RADAR_STALE_MAX_MIN
        frame_dt = datetime.fromisoformat(out["meteocat_radar_frame_time"])
        assert frame_dt.minute % config.METEO_RADAR_FRAME_MINUTES == 0
        assert out["meteocat_radar_max_dbz_20km"] >= 0.0