"""
Client per a l'API de RainViewer.
Obté dades de radar de precipitació en temps real per a Cardedeu.
https://www.rainviewer.com/api.html
"""
import io
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NowcastCardedeu/1.0 (research)"})


def _get_radar_frames() -> dict:
    """Obté la llista de frames de radar disponibles."""
    r = SESSION.get(config.RAINVIEWER_API_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def _extract_pixel_intensity(png_bytes: bytes, px: int, py: int) -> int:
    """
    Llegeix la intensitat del radar d'un píxel concret d'un tile PNG.
    RainViewer codifica la intensitat en el canal R del PNG (color scheme 2).
    Retorna 0 (sense pluja) a 255 (pluja molt intensa).
    """
    try:
        # Usar PIL si disponible, sinó fer fallback a raw PNG parsing
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        r, g, b, a = img.getpixel((px, py))
        # Si alfa=0, no hi ha dades de radar → sense pluja
        if a == 0:
            return 0
        return r  # Canal R conté la intensitat
    except ImportError:
        # Fallback: sense PIL, interpretar com "hi ha dades" si el PNG no està buit
        return -1  # Indica que no podem extreure el valor exacte


def _radar_intensity_to_dbz(intensity: int) -> float:
    """
    Converteix la intensitat del píxel (0-255) a dBZ aproximat.
    Color scheme 2 de RainViewer: mapping lineal 0→-32dBZ, 255→+95dBZ
    """
    if intensity <= 0:
        return 0.0
    return -32.0 + (intensity / 255.0) * 127.0


def _dbz_to_rain_rate(dbz: float) -> float:
    """
    Converteix dBZ a mm/h aproximat (fórmula Marshall-Palmer).
    Z = 200 * R^1.6 → R = (Z/200)^(1/1.6)
    """
    if dbz <= 0:
        return 0.0
    z_linear = 10 ** (dbz / 10.0)
    return (z_linear / 200.0) ** (1.0 / 1.6)


def fetch_radar_at_cardedeu() -> dict:
    """
    Obté les dades de radar actuals per a la ubicació de Cardedeu.
    Retorna un diccionari amb:
    - radar_intensity: 0-255 (intensitat del píxel)
    - radar_dbz: dBZ estimat
    - radar_rain_rate: mm/h estimat
    - radar_has_echo: bool (hi ha eco de pluja?)
    - radar_frames_with_echo: quants dels últims 6 frames tenien eco
    - radar_approaching: bool (la pluja s'acosta?)
    - radar_timestamp: str ISO
    """
    try:
        data = _get_radar_frames()
    except Exception as e:
        logger.warning(f"RainViewer API no disponible: {e}")
        return _empty_radar_result()

    past_frames = data.get("radar", {}).get("past", [])
    nowcast_frames = data.get("radar", {}).get("nowcast", [])

    if not past_frames:
        return _empty_radar_result()

    # Analitzar els últims 6 frames (~1h de radar, cada 10min)
    recent_frames = past_frames[-6:]
    intensities = []

    for frame in recent_frames:
        tile_url = (
            f"{config.RAINVIEWER_TILE_BASE}{frame['path']}/256/"
            f"{config.RAINVIEWER_TILE_ZOOM}/{config.RAINVIEWER_TILE_X}/"
            f"{config.RAINVIEWER_TILE_Y}/2/1_1.png"
        )
        try:
            r = SESSION.get(tile_url, timeout=5)
            if r.status_code == 200 and len(r.content) > 100:
                intensity = _extract_pixel_intensity(
                    r.content,
                    config.RAINVIEWER_PIXEL_X,
                    config.RAINVIEWER_PIXEL_Y,
                )
                intensities.append(intensity)
            else:
                intensities.append(0)
        except Exception:
            intensities.append(0)

    # Última intensitat (moment actual)
    current_intensity = intensities[-1] if intensities else 0
    current_dbz = _radar_intensity_to_dbz(current_intensity)
    current_rain_rate = _dbz_to_rain_rate(current_dbz)

    # Quants frames tenen eco (>10 d'intensitat ≈ >5dBZ ≈ pluja lleu)
    frames_with_echo = sum(1 for i in intensities if i > 10)

    # Detectar si la pluja s'acosta (intensitat creixent)
    approaching = False
    if len(intensities) >= 3:
        recent_trend = intensities[-1] - intensities[0]
        approaching = recent_trend > 15  # Intensitat creixent

    latest_ts = past_frames[-1]["time"]
    radar_time = datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat()

    return {
        "radar_intensity": current_intensity,
        "radar_dbz": round(current_dbz, 1),
        "radar_rain_rate": round(current_rain_rate, 2),
        "radar_has_echo": current_intensity > 10,
        "radar_frames_with_echo": frames_with_echo,
        "radar_approaching": approaching,
        "radar_max_intensity_1h": max(intensities) if intensities else 0,
        "radar_timestamp": radar_time,
    }


def _empty_radar_result() -> dict:
    return {
        "radar_intensity": 0,
        "radar_dbz": 0.0,
        "radar_rain_rate": 0.0,
        "radar_has_echo": False,
        "radar_frames_with_echo": 0,
        "radar_approaching": False,
        "radar_max_intensity_1h": 0,
        "radar_timestamp": "",
    }
