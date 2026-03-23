"""
Client per al radar d'AEMET (Red de Radares).
Obté i processa imatges del radar regional de Barcelona.
El radar de Barcelona (ba) és un C-band Doppler amb cobertura
excel·lent sobre el Vallès Oriental i Cardedeu.

Endpoints:
  - /api/red/radar/regional/{radar} — Imatge cada 10 min
  - /api/red/radar/nacional — Composició nacional cada 30 min

A diferència de RainViewer (tiles globals), AEMET proporciona
imatges composites del radar professional d'Espanya.

Documentació: https://opendata.aemet.es/dist/index.html
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
from src.data._http import create_session
from src.data.aemet_cache import get_cached, set_cached, RADAR_TTL

logger = logging.getLogger(__name__)

SESSION = create_session({"api_key": config.AEMET_API_KEY})

AEMET_BASE = "https://opendata.aemet.es/opendata/api"


# ── Paràmetres del radar de Barcelona ──
# El radar de Barcelona cobreix el NE d'Espanya.
# Cardedeu es troba dins de l'àrea de cobertura.
# La imatge retornada és un PNG/GIF georeferenciable.
# El color scheme d'AEMET utilitza una escala específica per dBZ.

# Mapping de colors AEMET (valors estimats de la paleta estàndard):
# Colors → dBZ (aproximat):
#  Blau clar:    5-15 dBZ (pluja feble)
#  Blau:        15-25 dBZ (pluja moderada)
#  Verd:        25-35 dBZ (pluja moderada-forta)
#  Groc:        35-45 dBZ (pluja forta)
#  Taronja:     45-55 dBZ (pluja molt forta)
#  Vermell:     55-65 dBZ (pluja intensa)
#  Magenta:     65+  dBZ (calamarsa/tempesta severa)
AEMET_COLOR_THRESHOLDS = [
    # (R_min, R_max, G_min, G_max, B_min, B_max, dBZ_estimate)
    (0, 50, 150, 255, 200, 255, 10.0),       # Blau clar → ~10 dBZ
    (0, 50, 50, 150, 200, 255, 20.0),         # Blau → ~20 dBZ
    (0, 100, 200, 255, 0, 100, 30.0),         # Verd → ~30 dBZ
    (200, 255, 200, 255, 0, 100, 40.0),       # Groc → ~40 dBZ
    (200, 255, 100, 200, 0, 80, 47.0),        # Taronja → ~47 dBZ
    (200, 255, 0, 100, 0, 80, 55.0),          # Vermell → ~55 dBZ
    (200, 255, 0, 100, 200, 255, 65.0),       # Magenta → ~65 dBZ
]


def _aemet_fetch_url(endpoint: str) -> Optional[str]:
    """
    Primer pas d'AEMET: obtenir la URL de dades.
    AEMET utilitza un patró de 2 passos: obtenir URL → descarregar dades.
    """
    if not config.AEMET_API_KEY:
        return None

    r = SESSION.get(f"{AEMET_BASE}{endpoint}", timeout=15)
    r.raise_for_status()
    meta = r.json()

    if meta.get("estado") != 200:
        logger.warning(f"AEMET radar error: {meta.get('descripcion', 'unknown')}")
        return None

    return meta.get("datos")


def _pixel_to_dbz(r: int, g: int, b: int, a: int) -> float:
    """
    Converteix un color de la imatge de radar d'AEMET a dBZ estimat.
    La imatge conté mapa base + ecos radar en colors específics.
    Píxels transparents o de mapa base es consideren 0 dBZ.
    """
    # Píxels transparents o gairebé blancs/negres = sense eco
    if a < 50:
        return 0.0
    if r + g + b < 50:  # negre (terra/límits)
        return 0.0
    if r > 200 and g > 200 and b > 200:  # blanc/gris clar (sense eco)
        return 0.0

    # Buscar el color més similar als llindars AEMET
    for r_min, r_max, g_min, g_max, b_min, b_max, dbz in AEMET_COLOR_THRESHOLDS:
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            return dbz

    return 0.0


def _find_cardedeu_pixel(img_array: np.ndarray, img_bounds: dict) -> Optional[tuple]:
    """
    Calcula la posició del píxel corresponent a Cardedeu dins la imatge.
    Necessita els bounds geogràfics de la imatge.
    """
    lat = config.LATITUDE
    lon = config.LONGITUDE

    lat_min = img_bounds.get("lat_min")
    lat_max = img_bounds.get("lat_max")
    lon_min = img_bounds.get("lon_min")
    lon_max = img_bounds.get("lon_max")

    if None in (lat_min, lat_max, lon_min, lon_max):
        return None

    h, w = img_array.shape[:2]
    px = int((lon - lon_min) / (lon_max - lon_min) * w)
    py = int((lat_max - lat) / (lat_max - lat_min) * h)

    if 0 <= px < w and 0 <= py < h:
        return (px, py)
    return None


def fetch_aemet_radar() -> dict:
    """
    Obté la imatge del radar regional de Barcelona i extreu
    informació de precipitació al voltant de Cardedeu.

    El radar de Barcelona ('ba') actualitza cada 10 minuts i
    cobreix tot el Vallès Oriental.

    Retorna dict amb mètriques de radar o valors buits si falla.
    """
    result = _empty_aemet_radar()

    if not config.AEMET_API_KEY:
        logger.info("AEMET radar no configurat (sense AEMET_API_KEY)")
        return result

    # Check cache first (radar updates every ~10 min)
    cached = get_cached("radar", RADAR_TTL)
    if cached is not None:
        return cached

    try:
        # Obtenir la URL de la imatge del radar de Barcelona
        data_url = _aemet_fetch_url("/red/radar/regional/ba")
        if not data_url:
            logger.warning("No s'ha obtingut URL del radar AEMET Barcelona")
            return result

        # Descarregar la imatge
        r = SESSION.get(data_url, timeout=15)
        r.raise_for_status()

        if len(r.content) < 500:
            logger.warning("Imatge de radar AEMET massa petita")
            return result

        # Processar la imatge
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            arr = np.array(img)
        except ImportError:
            logger.warning("PIL no disponible per processar radar AEMET")
            return result

        h, w = arr.shape[:2]

        # Bounds geogràfics estimats per al radar regional de Barcelona
        # El radar de Barcelona cobreix aprox. Catalunya i voltants
        img_bounds = config.AEMET_RADAR_BOUNDS

        cardedeu_px = _find_cardedeu_pixel(arr, img_bounds)
        if cardedeu_px is None:
            logger.warning("No s'ha pogut localitzar Cardedeu a la imatge de radar AEMET")
            return result

        cx, cy = cardedeu_px
        logger.debug(f"  Radar AEMET: Cardedeu al pixel ({cx}, {cy}) d'imatge {w}x{h}")

        # ── Mètriques puntuals (píxel a Cardedeu) ──
        pixel_rgba = arr[cy, cx]
        dbz_cardedeu = _pixel_to_dbz(
            int(pixel_rgba[0]), int(pixel_rgba[1]),
            int(pixel_rgba[2]), int(pixel_rgba[3])
        )

        # ── Escaneig espacial: buscar ecos en un radi al voltant ──
        # Calcular mida del píxel en km (depèn de la resolució de la imatge)
        lat_range = img_bounds["lat_max"] - img_bounds["lat_min"]
        pixel_size_km = (lat_range / h) * 111.0  # ~111 km per grau de latitud

        scan_radius_px = int(config.RADAR_SCAN_RADIUS_KM / pixel_size_km) if pixel_size_km > 0 else 50

        # Definir subregió
        y_lo = max(0, cy - scan_radius_px)
        y_hi = min(h, cy + scan_radius_px + 1)
        x_lo = max(0, cx - scan_radius_px)
        x_hi = min(w, cx + scan_radius_px + 1)

        region = arr[y_lo:y_hi, x_lo:x_hi]
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dy = yy - cy
        dx = xx - cx
        dist_px = np.sqrt(dx.astype(float)**2 + dy.astype(float)**2)
        dist_km = dist_px * pixel_size_km

        in_radius = dist_km <= config.RADAR_SCAN_RADIUS_KM

        # Detectar ecos (convertir cada píxel a dBZ)
        echo_mask = np.zeros(region.shape[:2], dtype=bool)
        dbz_map = np.zeros(region.shape[:2], dtype=float)
        for iy in range(region.shape[0]):
            for ix in range(region.shape[1]):
                px_r, px_g, px_b, px_a = region[iy, ix]
                dbz = _pixel_to_dbz(int(px_r), int(px_g), int(px_b), int(px_a))
                dbz_map[iy, ix] = dbz
                if dbz >= config.RADAR_MIN_DBZ:
                    echo_mask[iy, ix] = True

        has_echo_area = echo_mask & in_radius

        if has_echo_area.any():
            echo_distances = np.where(has_echo_area, dist_km, np.inf)
            nearest_flat = echo_distances.argmin()
            nearest_idx = np.unravel_index(nearest_flat, echo_distances.shape)
            nearest_km = float(dist_km[nearest_idx])
            nearest_dx = float(dx[nearest_idx])
            nearest_dy = float(dy[nearest_idx])
            nearest_bearing = float(
                (np.degrees(np.arctan2(nearest_dx, -nearest_dy)) + 360) % 360
            )

            # Màxim dBZ dins 20km
            within_20km = has_echo_area & (dist_km <= 20)
            max_dbz_20km = float(dbz_map[within_20km].max()) if within_20km.any() else 0.0

            # Cobertura
            total_radius = int(in_radius.sum())
            eco_radius = int(has_echo_area.sum())
            coverage = eco_radius / total_radius if total_radius > 0 else 0.0

            result.update({
                "aemet_radar_dbz": round(dbz_cardedeu, 1),
                "aemet_radar_has_echo": dbz_cardedeu >= config.RADAR_MIN_DBZ,
                "aemet_radar_nearest_echo_km": round(nearest_km, 1),
                "aemet_radar_nearest_echo_compass": _bearing_to_compass(nearest_bearing),
                "aemet_radar_max_dbz_20km": round(max_dbz_20km, 1),
                "aemet_radar_coverage_20km": round(coverage, 4),
                "aemet_radar_echoes_found": True,
                "aemet_radar_available": True,
            })
        else:
            result.update({
                "aemet_radar_dbz": round(dbz_cardedeu, 1),
                "aemet_radar_has_echo": dbz_cardedeu >= config.RADAR_MIN_DBZ,
                "aemet_radar_echoes_found": False,
                "aemet_radar_available": True,
            })

        logger.info(
            f"  Radar AEMET Barcelona: dBZ={result['aemet_radar_dbz']}, "
            f"eco_proper={result.get('aemet_radar_nearest_echo_km', 'N/A')}km, "
            f"cobertura_20km={result.get('aemet_radar_coverage_20km', 0):.1%}"
        )
        set_cached("radar", result)
        return result

    except Exception as e:
        logger.warning(f"Error processant radar AEMET: {e}")
        return result


def _bearing_to_compass(bearing: float) -> str:
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(bearing / 22.5) % 16
    return directions[idx]


def _empty_aemet_radar() -> dict:
    return {
        "aemet_radar_dbz": 0.0,
        "aemet_radar_has_echo": False,
        "aemet_radar_nearest_echo_km": config.RADAR_SCAN_RADIUS_KM,
        "aemet_radar_nearest_echo_compass": None,
        "aemet_radar_max_dbz_20km": 0.0,
        "aemet_radar_coverage_20km": 0.0,
        "aemet_radar_echoes_found": False,
        "aemet_radar_available": False,
    }
