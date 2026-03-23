"""
Test suite: AEMET radar map artifact filtering.
Catches: false radar echoes from map borders/coastlines/legends.
The AEMET radar image is a pre-composited GIF where geographic borders
use yellow (255,255,0) — the SAME color as 40 dBZ radar echoes.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TestPixelToDbz:
    """Validates color → dBZ mapping with map artifact filtering."""

    def test_transparent_pixel_returns_zero(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(255, 255, 0, 0) == 0.0

    def test_black_pixel_returns_zero(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(0, 0, 0, 255) == 0.0

    def test_white_pixel_returns_zero(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(240, 240, 240, 255) == 0.0

    def test_grey_land_pixel_returns_zero(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(127, 127, 127, 255) == 0.0

    def test_yellow_classified_as_40dbz(self):
        """Yellow (255,255,0) matches the 40 dBZ threshold."""
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(255, 255, 0, 255) == 40.0

    def test_blue_classified_as_20dbz(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(0, 100, 252, 255) == 20.0

    def test_green_classified_as_30dbz(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(0, 255, 0, 255) == 30.0

    def test_red_classified_as_55dbz(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(255, 50, 50, 255) == 55.0

    def test_unrecognized_color_returns_zero(self):
        from src.data.aemet_radar import _pixel_to_dbz
        assert _pixel_to_dbz(150, 100, 150, 255) == 0.0


class TestRemoveMapArtifacts:
    """Validates morphological opening + cluster filter removes map features."""

    def test_thin_horizontal_line_removed(self):
        """A 1px-wide horizontal line (border) should be removed."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 3:17] = True  # 14px horizontal line, 1px thick
        result = _remove_map_artifacts(mask)
        assert not result.any()

    def test_thin_vertical_line_removed(self):
        """A 1px-wide vertical line (border) should be removed."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:17, 10] = True  # 14px vertical line, 1px thick
        result = _remove_map_artifacts(mask)
        assert not result.any()

    def test_thin_diagonal_line_removed(self):
        """A 1px-wide diagonal line (coastline) should be removed."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((20, 20), dtype=bool)
        for i in range(15):
            mask[2 + i, 2 + i] = True
        result = _remove_map_artifacts(mask)
        assert not result.any()

    def test_small_cluster_removed(self):
        """A small cross (5px, like a border junction) should be removed."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 9:12] = True   # horizontal piece
        mask[9:12, 10] = True   # vertical piece — 5px cross
        result = _remove_map_artifacts(mask)
        assert not result.any()

    def test_large_rain_area_preserved(self):
        """A broad area of precipitation (>10px) should survive filtering."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((30, 30), dtype=bool)
        # 6x6 solid block = 36 pixels — clearly real rain
        mask[12:18, 12:18] = True
        result = _remove_map_artifacts(mask)
        assert result.any()
        # The interior should survive (edges may shrink slightly)
        assert result.sum() >= 10

    def test_large_area_with_nearby_line_separated(self):
        """A rain area near a border line: rain preserved, line removed."""
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((30, 30), dtype=bool)
        # Real rain: 6x6 block
        mask[12:18, 12:18] = True
        # Nearby border line (not touching): 1px line
        mask[8, 5:25] = True
        result = _remove_map_artifacts(mask)
        # Rain area should survive
        assert result[12:18, 12:18].any()
        # Border line should be removed (it's thin and separate)
        assert not result[8, :].any()

    def test_empty_mask_returns_empty(self):
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.zeros((20, 20), dtype=bool)
        result = _remove_map_artifacts(mask)
        assert not result.any()

    def test_tiny_image_handled(self):
        from src.data.aemet_radar import _remove_map_artifacts
        mask = np.ones((2, 2), dtype=bool)
        result = _remove_map_artifacts(mask)
        assert result.shape == (2, 2)


class TestAemetRadarBorderRegression:
    """Regression test: yellow borders must never produce false radar echoes."""

    def test_yellow_borders_with_no_rain_produce_zero_echoes(self):
        """
        Simulates the exact AEMET radar image pattern that caused the bug:
        yellow (255,255,0) border pixels near Cardedeu at ~6km, forming
        a thin coastline/border line. After artifact removal, zero echoes
        should remain since there's no actual precipitation.
        """
        from src.data.aemet_radar import _pixel_to_dbz, _remove_map_artifacts

        # Simular una imatge amb fons negre i una línia de frontera groga
        h, w = 60, 60
        echo_mask = np.zeros((h, w), dtype=bool)

        # Afegir una línia groga (frontera) a prop del centre (Cardedeu)
        # Línia en ziga-zaga de 1-2px d'ample (com una costa real)
        for x in range(5, 55):
            y = 30 + int(3 * np.sin(x / 5))
            echo_mask[y, x] = True     # 1px wide
            if x % 3 == 0:
                echo_mask[y + 1, x] = True  # 2px in some spots

        result = _remove_map_artifacts(echo_mask)

        # Cap eco hauria de sobreviure
        assert not result.any(), (
            f"Artefactes de frontera sobreviuen el filtratge: "
            f"{result.sum()} píxels"
        )

    def test_real_rain_near_border_survives(self):
        """
        When real precipitation exists near a map border,
        the rain area survives but the border is removed.
        """
        from src.data.aemet_radar import _remove_map_artifacts

        h, w = 60, 60
        echo_mask = np.zeros((h, w), dtype=bool)

        # Frontera: línia horitzontal fina
        echo_mask[20, 5:55] = True

        # Pluja real: bloc 8x8 a prop però no tocant la frontera
        echo_mask[35:43, 25:33] = True

        result = _remove_map_artifacts(echo_mask)

        # La pluja hauria de sobreviure
        assert result[35:43, 25:33].any()
        # La frontera no
        assert not result[20, :].any()


class TestAemetRadarMinClusterConfig:
    """Verifies the cluster size threshold is configured."""

    def test_config_has_min_cluster_setting(self):
        assert hasattr(config, "AEMET_RADAR_MIN_ECHO_CLUSTER_PX")
        assert config.AEMET_RADAR_MIN_ECHO_CLUSTER_PX >= 5
