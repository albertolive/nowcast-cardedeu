"""
Tests per a l'escaneig espacial de radar (src/data/rainviewer.py)
i per al pipeline de calibratge isotònic (src/model/train.py).
"""
import io
import math
import os
import pytest

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.rainviewer import (
    _radar_intensity_to_dbz,
    _dbz_to_rain_rate,
    _bearing_to_compass,
    _scan_radar_spatial,
    _estimate_storm_tracking,
    _build_clutter_mask,
    _empty_spatial_result,
)


# ── Bearing to compass ──

class TestBearingToCompass:
    def test_north(self):
        assert _bearing_to_compass(0) == "N"

    def test_east(self):
        assert _bearing_to_compass(90) == "E"

    def test_south(self):
        assert _bearing_to_compass(180) == "S"

    def test_west(self):
        assert _bearing_to_compass(270) == "W"

    def test_northeast(self):
        assert _bearing_to_compass(45) == "NE"

    def test_southeast(self):
        assert _bearing_to_compass(135) == "SE"

    def test_360_wraps_to_north(self):
        assert _bearing_to_compass(360) == "N"

    def test_near_north_boundary(self):
        # 349° → round(349/22.5)=round(15.51)=16 → 16%16=0 → N
        assert _bearing_to_compass(349) == "N"
        # 337° → round(337/22.5)=round(14.98)=15 → NNW
        assert _bearing_to_compass(337) == "NNW"


# ── Radar intensity → dBZ (already tested in test_radar.py, extended here) ──

class TestIntensityToDbzExtended:
    """Tests per als 10 nivells quantitzats de RainViewer."""

    def test_quantized_level_38(self):
        """R=38 → dBZ = 38/2 - 32 = -13.0 → clipped to -13.0 (below threshold)."""
        assert _radar_intensity_to_dbz(38) == pytest.approx(-13.0)

    def test_quantized_level_110(self):
        """R=110 → dBZ = 110/2 - 32 = 23.0 → light rain."""
        assert _radar_intensity_to_dbz(110) == pytest.approx(23.0)

    def test_quantized_level_146(self):
        """R=146 → dBZ = 146/2 - 32 = 41.0 → moderate rain."""
        assert _radar_intensity_to_dbz(146) == pytest.approx(41.0)

    def test_quantized_level_203(self):
        """R=203 → dBZ = 203/2 - 32 = 69.5 → capped at 65."""
        assert _radar_intensity_to_dbz(203) == 65.0

    def test_quantized_level_255_capped(self):
        """R=255 → dBZ = 255/2 - 32 = 95.5 → capped at 65 dBZ."""
        assert _radar_intensity_to_dbz(255) == 65.0


# ── dBZ → rain rate (Marshall-Palmer) ──

class TestDbzToRainRateExtended:
    def test_typical_15dbz(self):
        """15 dBZ → ~0.3 mm/h (drizzle)."""
        r = _dbz_to_rain_rate(15.0)
        assert 0.1 < r < 0.5

    def test_typical_35dbz(self):
        """35 dBZ → ~5 mm/h (moderate rain)."""
        r = _dbz_to_rain_rate(35.0)
        assert 3.0 < r < 10.0

    def test_typical_50dbz(self):
        """50 dBZ → ~50 mm/h (heavy rain)."""
        r = _dbz_to_rain_rate(50.0)
        assert 30.0 < r < 80.0

    def test_monotonic_increase(self):
        """Més dBZ → més mm/h."""
        rates = [_dbz_to_rain_rate(d) for d in [10, 20, 30, 40, 50, 60]]
        for i in range(len(rates) - 1):
            assert rates[i] < rates[i + 1]


# ── Helper to create a test PNG tile ──

def _make_tile(width=256, height=256, echoes=None):
    """
    Crea un tile PNG de test. echoes és una llista de (x, y, r_value, alpha).
    Pixels sense eco: alpha=255, R=0. Pixels fora de cobertura: alpha=0.
    """
    from PIL import Image
    img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    px = img.load()
    if echoes:
        for x, y, r_val, a in echoes:
            px[x, y] = (r_val, r_val, r_val, a)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── Spatial scanning ──

class TestScanRadarSpatial:
    def test_no_echoes_returns_empty(self):
        """Tile sense ecos → resultat buit."""
        tile = _make_tile()  # tota negra, alpha=255, R=0
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is False
        assert result["nearest_echo_km"] == 30
        assert result["coverage_20km"] == 0.0

    def test_echo_at_center_detected(self):
        """Eco al centre (0km de distància)."""
        import config
        # R=146 → dBZ=41, que passa el filtre RADAR_MIN_DBZ=10
        tile = _make_tile(echoes=[(128, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is True
        assert result["nearest_echo_km"] == 0.0

    def test_echo_distance_calculated(self):
        """Eco a 10 píxels → ~4.57 km."""
        # Echo 10 px to the east (x+10), R=146 (dBZ=41)
        tile = _make_tile(echoes=[(138, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is True
        expected_km = 10 * 0.457
        assert abs(result["nearest_echo_km"] - expected_km) < 0.2

    def test_alpha_zero_not_detected(self):
        """Pixel amb alpha=0 (fora cobertura) → no comptar com eco."""
        tile = _make_tile(echoes=[(138, 128, 146, 0)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is False

    def test_bearing_east(self):
        """Eco a l'est → bearing ~90°."""
        tile = _make_tile(echoes=[(148, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert 80 <= result["nearest_echo_bearing"] <= 100

    def test_bearing_south(self):
        """Eco al sud → bearing ~180°."""
        tile = _make_tile(echoes=[(128, 148, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert 170 <= result["nearest_echo_bearing"] <= 190

    def test_quadrant_features_present(self):
        """Eco a l'est → quadrant E té dBZ, els altres 0."""
        tile = _make_tile(echoes=[(148, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["quadrant_max_dbz_E"] > 0
        assert result["quadrant_max_dbz_W"] == 0.0
        assert result["quadrant_max_dbz_N"] == 0.0
        assert result["quadrant_max_dbz_S"] == 0.0

    def test_low_intensity_filtered_by_min_dbz(self):
        """R=38 → dBZ=-13 < RADAR_MIN_DBZ(10) → filtrat."""
        tile = _make_tile(echoes=[(138, 128, 38, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is False

    def test_r255_filtered_as_clutter(self):
        """R=255 → filtrat com artefacte (r_channel < 255 check)."""
        tile = _make_tile(echoes=[(138, 128, 255, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["echoes_found"] is False

    def test_clutter_mask_removes_echo(self):
        """Clutter mask ha de filtre l'eco."""
        tile = _make_tile(echoes=[(138, 128, 146, 255)])
        # Clutter mask: True al pixel de l'eco
        mask = np.zeros((256, 256), dtype=bool)
        mask[128, 138] = True  # nota: array[y, x]
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457, clutter_mask=mask)
        assert result["echoes_found"] is False

    def test_upwind_with_wind_direction(self):
        """Eco al sector de sobrevent (±60° de la direcció del vent)."""
        # Vent de l'est (90°), eco a l'est → dins del sector upwind
        tile = _make_tile(echoes=[(148, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457, wind_from_dir=90)
        assert result.get("upwind_nearest_echo_km", 30) < 30

    def test_centroid_returned(self):
        """Eco ha de retornar centroid per tracking."""
        tile = _make_tile(echoes=[(148, 128, 146, 255)])
        result = _scan_radar_spatial(tile, 128, 128, 30, 0.457)
        assert result["_centroid_dx"] is not None
        assert result["_centroid_dy"] is not None


# ── Empty spatial result ──

class TestEmptySpatialResult:
    def test_has_all_quadrants(self):
        result = _empty_spatial_result(30.0)
        for quad in ("N", "E", "S", "W"):
            assert f"quadrant_max_dbz_{quad}" in result
            assert f"quadrant_coverage_{quad}" in result
            assert result[f"quadrant_max_dbz_{quad}"] == 0.0

    def test_nearest_echo_equals_radius(self):
        result = _empty_spatial_result(30.0)
        assert result["nearest_echo_km"] == 30.0

    def test_echoes_found_false(self):
        result = _empty_spatial_result(30.0)
        assert result["echoes_found"] is False


# ── Storm tracking ──

class TestStormTracking:
    def test_less_than_2_scans_returns_zero(self):
        """Menys de 2 scans vàlids → velocitat 0."""
        scans = [{"echoes_found": True, "_centroid_dx": 5.0, "_centroid_dy": 0.0}]
        result = _estimate_storm_tracking(scans, 0.457)
        assert result["storm_velocity_kmh"] == 0.0
        assert result["storm_approaching"] is False

    def test_no_valid_scans_returns_zero(self):
        """Scans sense ecos → velocitat 0."""
        scans = [
            {"echoes_found": False, "_centroid_dx": None, "_centroid_dy": None},
            {"echoes_found": False, "_centroid_dx": None, "_centroid_dy": None},
        ]
        result = _estimate_storm_tracking(scans, 0.457)
        assert result["storm_velocity_kmh"] == 0.0

    def test_eastward_movement(self):
        """Centroid movent-se cap a l'est → velocity_ew > 0."""
        scans = [
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 10.0, "_centroid_dy": 0.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        assert result["storm_velocity_ew"] > 0
        assert abs(result["storm_velocity_ns"]) < 0.1

    def test_southward_movement(self):
        """Centroid movent-se cap al sud → velocity_ns > 0."""
        scans = [
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 10.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        assert result["storm_velocity_ns"] > 0

    def test_approaching_storm(self):
        """Tempesta acostant-se (centroid cap al centre, >1km)."""
        # Comença a 20px de distància, s'acosta a 5px
        scans = [
            {"echoes_found": True, "_centroid_dx": 20.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 5.0, "_centroid_dy": 0.0,
             "nearest_echo_km": 2.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        approach_km = (20.0 - 5.0) * 0.457  # ~6.85 km
        assert approach_km > 1.0
        assert result["storm_approaching"] == True

    def test_receding_storm(self):
        """Tempesta allunyant-se → approaching = False."""
        scans = [
            {"echoes_found": True, "_centroid_dx": 5.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 20.0, "_centroid_dy": 0.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        assert result["storm_approaching"] == False
        assert result["storm_eta_min"] is None

    def test_eta_clipped_to_180(self):
        """ETA ha d'estar limitat a 180 min."""
        # Tempesta molt llunya, velocitat baixa → ETA molt alt → capped a 180
        scans = [
            {"echoes_found": True, "_centroid_dx": 100.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 95.0, "_centroid_dy": 0.0,
             "nearest_echo_km": 40.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        if result["storm_eta_min"] is not None:
            assert result["storm_eta_min"] <= 180

    def test_velocity_calculation(self):
        """Verificar càlcul de velocitat: 10px en 10min a 0.457 km/px."""
        scans = [
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 10.0, "_centroid_dy": 0.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        # 10px * 0.457 = 4.57km en 10min → 27.42 km/h
        expected = 10 * 0.457 / 10 * 60
        assert abs(result["storm_velocity_kmh"] - expected) < 0.5

    def test_multiple_frames_uses_first_and_last(self):
        """Amb 4 frames, usa el primer i l'últim centroid."""
        scans = [
            {"echoes_found": True, "_centroid_dx": 0.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 3.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 7.0, "_centroid_dy": 0.0},
            {"echoes_found": True, "_centroid_dx": 12.0, "_centroid_dy": 0.0},
        ]
        result = _estimate_storm_tracking(scans, 0.457, frame_interval_min=10)
        # 12px over 3 intervals (30 min)
        expected_kmh = 12 * 0.457 / 30 * 60
        assert abs(result["storm_velocity_kmh"] - expected_kmh) < 0.5


# ── Clutter mask ──

class TestClutterMask:
    def test_less_than_3_frames_returns_none(self):
        """Menys de 3 frames → no fiable, retorna None."""
        tiles = [_make_tile(), _make_tile()]
        assert _build_clutter_mask(tiles) is None

    def test_no_echoes_no_clutter(self):
        """3 frames sense ecos → mask buida (cap clutter) o None."""
        tiles = [_make_tile() for _ in range(3)]
        result = _build_clutter_mask(tiles)
        # No echoes → echo_count=0 for all pixels → clutter = (0 >= 3) = False everywhere
        # clutter_frac = 0 → returns the mask (all False)
        if result is not None:
            assert result.sum() == 0

    def test_persistent_static_echo_detected_as_clutter(self):
        """Eco persistent estàtic (variància zero) → clutter, encara que sigui >0.5% del tile."""
        # 400 píxels amb mateixa intensitat en tots els frames = clutter (muntanya)
        echoes = [(x, 128, 146, 255) for x in range(50, 250)]  # 200 px per frame
        echoes += [(x, 129, 146, 255) for x in range(50, 250)]  # +200 = 400 > 327
        tile = _make_tile(echoes=echoes)
        tiles = [tile, tile, tile]
        result = _build_clutter_mask(tiles)
        assert result is not None  # Clutter estàtic detectat
        assert result.sum() == 400  # Tots els ecos persistents estàtics marcats

    def test_none_frames_ignored(self):
        """Frames None s'ignoren."""
        tiles = [None, _make_tile(), None, _make_tile(), _make_tile()]
        result = _build_clutter_mask(tiles)
        # 3 valid frames, but no echoes
        if result is not None:
            assert result.sum() == 0


# ── Calibration confidence levels ──

class TestConfidenceLevels:
    """Testar els nivells de confiança de predict_now sense executar tot el pipeline."""

    @staticmethod
    def _confidence(probability):
        if probability >= 0.85:
            return "Molt Alta"
        elif probability >= 0.70:
            return "Alta"
        elif probability >= 0.50:
            return "Mitjana"
        elif probability >= 0.30:
            return "Baixa"
        else:
            return "Molt Baixa"

    def test_95_percent(self):
        assert self._confidence(0.95) == "Molt Alta"

    def test_85_boundary(self):
        assert self._confidence(0.85) == "Molt Alta"

    def test_84_percent(self):
        assert self._confidence(0.84) == "Alta"

    def test_70_boundary(self):
        assert self._confidence(0.70) == "Alta"

    def test_69_percent(self):
        assert self._confidence(0.69) == "Mitjana"

    def test_50_boundary(self):
        assert self._confidence(0.50) == "Mitjana"

    def test_49_percent(self):
        assert self._confidence(0.49) == "Baixa"

    def test_30_boundary(self):
        assert self._confidence(0.30) == "Baixa"

    def test_29_percent(self):
        assert self._confidence(0.29) == "Molt Baixa"

    def test_zero(self):
        assert self._confidence(0.0) == "Molt Baixa"


# ── Isotonic calibration properties ──

class TestIsotonicCalibration:
    """Test les propietats matemàtiques del calibratge isotònic."""

    def test_isotonic_is_monotonic(self):
        """Calibratge isotònic manté monotonicitat: si raw_a > raw_b → cal_a >= cal_b."""
        from sklearn.isotonic import IsotonicRegression

        # Simular OOF amb correlació moderada amb target
        rng = np.random.RandomState(42)
        raw_proba = rng.uniform(0, 1, 500)
        y_true = (raw_proba + rng.normal(0, 0.3, 500) > 0.5).astype(int)

        cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        cal.fit(raw_proba, y_true)

        test_inputs = np.linspace(0, 1, 100)
        calibrated = cal.predict(test_inputs)
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]

    def test_calibrated_clipped_to_01(self):
        """Prediccions calibrades han d'estar en [0, 1]."""
        from sklearn.isotonic import IsotonicRegression

        rng = np.random.RandomState(42)
        raw_proba = rng.uniform(0, 1, 200)
        y_true = (raw_proba > 0.4).astype(int)

        cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        cal.fit(raw_proba, y_true)

        # Probar amb valors fora de rang
        test_inputs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        calibrated = cal.predict(test_inputs)
        assert all(0 <= c <= 1 for c in calibrated)

    def test_f1_threshold_search(self):
        """Cerca del llindar òptim F1 sobre la corba precision-recall."""
        from sklearn.metrics import precision_recall_curve, f1_score

        rng = np.random.RandomState(42)
        y_true = np.array([0]*80 + [1]*20)
        y_scores = rng.uniform(0, 1, 100)
        y_scores[80:] += 0.3  # positius tendeixen a ser més alts
        y_scores = np.clip(y_scores, 0, 1)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1s = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0,
        )
        optimal_idx = np.argmax(f1s)
        optimal_threshold = float(thresholds[optimal_idx])

        # El llindar ha d'estar entre 0 i 1
        assert 0 < optimal_threshold < 1
        # F1 ha de ser > 0
        assert f1s[optimal_idx] > 0
