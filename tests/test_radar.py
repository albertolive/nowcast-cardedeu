"""
Test suite: Radar processing (RainViewer).
Catches: incorrect tile coordinates, dBZ formula errors, color channel
interpretation bugs, clutter mask issues.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TestRadarIntensityToDbz:
    """Validates dBZ = R/2 - 32, capped at 65."""

    def test_zero_intensity_returns_zero(self):
        from src.data.rainviewer import _radar_intensity_to_dbz
        assert _radar_intensity_to_dbz(0) == 0.0

    def test_negative_intensity_returns_zero(self):
        from src.data.rainviewer import _radar_intensity_to_dbz
        assert _radar_intensity_to_dbz(-1) == 0.0

    def test_known_intensity_values(self):
        """RainViewer 256px quantized levels."""
        from src.data.rainviewer import _radar_intensity_to_dbz
        # R=38 → 38/2-32 = -13 dBZ (very light)
        assert _radar_intensity_to_dbz(38) == pytest.approx(-13.0)
        # R=110 → 110/2-32 = 23 dBZ (light rain)
        assert _radar_intensity_to_dbz(110) == pytest.approx(23.0)
        # R=177 → 177/2-32 = 56.5 dBZ (heavy rain)
        assert _radar_intensity_to_dbz(177) == pytest.approx(56.5)

    def test_max_dbz_capped_at_65(self):
        """R=255 → 95.5 is physically impossible, must cap at 65."""
        from src.data.rainviewer import _radar_intensity_to_dbz
        assert _radar_intensity_to_dbz(255) == 65.0
        assert _radar_intensity_to_dbz(200) == 65.0  # 200/2-32=68 > 65

    def test_cap_boundary(self):
        """R=194 → 194/2-32=65.0 exactly at the cap."""
        from src.data.rainviewer import _radar_intensity_to_dbz
        assert _radar_intensity_to_dbz(194) == 65.0
        # R=193 → 193/2-32=64.5, just under cap
        assert _radar_intensity_to_dbz(193) == pytest.approx(64.5)


class TestDbzToRainRate:
    """Marshall-Palmer Z=200*R^1.6 conversion."""

    def test_zero_dbz(self):
        from src.data.rainviewer import _dbz_to_rain_rate
        assert _dbz_to_rain_rate(0) == 0.0

    def test_negative_dbz(self):
        from src.data.rainviewer import _dbz_to_rain_rate
        assert _dbz_to_rain_rate(-5) == 0.0

    def test_moderate_rain(self):
        """35 dBZ ≈ 5.6 mm/h (moderate rain)."""
        from src.data.rainviewer import _dbz_to_rain_rate
        rate = _dbz_to_rain_rate(35)
        assert 3.0 < rate < 10.0  # reasonable range

    def test_heavy_rain(self):
        """50 dBZ ≈ 48 mm/h (heavy rain)."""
        from src.data.rainviewer import _dbz_to_rain_rate
        rate = _dbz_to_rain_rate(50)
        assert 30.0 < rate < 80.0


class TestPixelIntensityExtraction:
    """Validates pixel interpretation from PNG tiles."""

    def test_r_zero_alpha_nonzero_is_no_precipitation(self):
        """R=0 with alpha>0 means radar covers the area but NO rain detected.
        This was the critical bug: R=0 was being treated as rain."""
        from src.data.rainviewer import _extract_pixel_intensity
        from PIL import Image
        import io

        # Create a 256x256 RGBA image: R=0, G=0, B=0, A=255 (covered, no rain)
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        intensity = _extract_pixel_intensity(png_bytes, 174, 97)
        assert intensity == 0, "R=0 with alpha>0 should be NO precipitation"

    def test_alpha_zero_is_no_coverage(self):
        """alpha=0: pixel outside radar coverage."""
        from src.data.rainviewer import _extract_pixel_intensity
        from PIL import Image
        import io

        img = Image.new("RGBA", (256, 256), (100, 100, 100, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        intensity = _extract_pixel_intensity(png_bytes, 174, 97)
        assert intensity == 0

    def test_precipitation_detected(self):
        """R>0 with alpha>0: rain detected."""
        from src.data.rainviewer import _extract_pixel_intensity
        from PIL import Image
        import io

        img = Image.new("RGBA", (256, 256), (110, 110, 110, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        intensity = _extract_pixel_intensity(png_bytes, 174, 97)
        assert intensity == 110


class TestRainViewerConfig:
    """Validates tile coordinates for Cardedeu."""

    def test_tile_coordinates_for_cardedeu(self):
        """Centre Cardedeu (41.639°N, 2.356°E) at zoom=8:
        tile_x=129, tile_y=95, pixel_x=172, pixel_y=96."""
        assert config.RAINVIEWER_TILE_ZOOM == 8
        assert config.RAINVIEWER_TILE_X == 129
        assert config.RAINVIEWER_TILE_Y == 95
        assert config.RAINVIEWER_PIXEL_X == 172
        assert config.RAINVIEWER_PIXEL_Y == 96

    def test_coordinates_within_catalonia(self):
        """Basic sanity: Cardedeu must be in Catalonia."""
        assert 41.0 < config.LATITUDE < 42.5
        assert 1.5 < config.LONGITUDE < 3.5


class TestClutterMask:
    """
    Validates clutter mask doesn't filter sustained real rain.

    Regression for the bug that left RainViewer effectively blind during
    the 2026-05-15 storm: the variance-based clutter detector flagged
    sustained moderate-intensity rain (R=110-180, dBZ 23-58) as clutter
    because RainViewer quantizes intensity into ~10 discrete levels, so
    rain stuck at the same intensity for 2h has variance ≈ 0.

    Fix: clutter requires BOTH variance<1.0 AND mean_R≥200. Real mountain
    clutter saturates the radar (R≥200, dBZ≥68); sustained rain rarely
    holds R≥200 without fluctuating.
    """

    def _make_tile(self, r_value: int, alpha: int = 200) -> bytes:
        """Build a minimal 256x256 PNG with uniform R value at all pixels."""
        import io as _io
        from PIL import Image
        arr = np.zeros((256, 256, 4), dtype=np.uint8)
        arr[:, :, 0] = r_value
        arr[:, :, 1] = r_value
        arr[:, :, 2] = r_value
        arr[:, :, 3] = alpha if r_value > 0 else 0
        img = Image.fromarray(arr, mode="RGBA")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_sustained_moderate_rain_not_flagged_as_clutter(self):
        """R=146 (41 dBZ) at all pixels for 13 frames = sustained heavy rain.
        Must NOT be flagged as clutter."""
        from src.data.rainviewer import _build_clutter_mask
        tiles = [self._make_tile(146) for _ in range(13)]
        mask = _build_clutter_mask(tiles)
        # Either no clutter detected at all, or none of the rain pixels masked.
        if mask is not None:
            assert not mask.any(), (
                "Sustained R=146 rain must not be flagged as clutter "
                f"({int(mask.sum())} px filtered)"
            )

    def test_saturated_persistent_signal_is_clutter(self):
        """R=240 (mountain return) at all pixels for 13 frames = real clutter.
        MUST be flagged."""
        from src.data.rainviewer import _build_clutter_mask
        tiles = [self._make_tile(240) for _ in range(13)]
        mask = _build_clutter_mask(tiles)
        assert mask is not None and mask.any(), (
            "Saturated persistent signal (R=240) must be flagged as clutter"
        )

    def test_mixed_rain_and_mountain(self):
        """Half the tile is sustained rain (R=146), half is mountain (R=240).
        Only the mountain half should be filtered."""
        import io as _io
        from PIL import Image
        from src.data.rainviewer import _build_clutter_mask

        def make_mixed():
            arr = np.zeros((256, 256, 4), dtype=np.uint8)
            # Left half: rain
            arr[:, :128, 0] = 146
            arr[:, :128, 1] = 146
            arr[:, :128, 2] = 146
            arr[:, :128, 3] = 200
            # Right half: mountain
            arr[:, 128:, 0] = 240
            arr[:, 128:, 1] = 240
            arr[:, 128:, 2] = 240
            arr[:, 128:, 3] = 200
            buf = _io.BytesIO()
            Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
            return buf.getvalue()

        tiles = [make_mixed() for _ in range(13)]
        mask = _build_clutter_mask(tiles)
        assert mask is not None
        # Rain side must not be filtered
        assert not mask[:, :128].any(), "Rain pixels misflagged as clutter"
        # Mountain side must be fully filtered
        assert mask[:, 128:].all(), "Mountain pixels not flagged as clutter"
