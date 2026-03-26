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
