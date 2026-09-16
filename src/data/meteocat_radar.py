"""
Client del radar de Meteocat (SMC) via les tiles públiques del visor.

Producte: composició de radar de la Xarxa de Radars de Catalunya (radar de la
Vallès, a ~20km de Cardedeu), servida com a tiles Web-Mercator de 256px a z7:

  {base}/{YYYY}/{MM}/{DD}/{HH}/{MM}/{z}/000/000/{x}/000/000/{y}.png

Particularitats (verificades el 2026-09-16 amb tempesta real sobre Cardedeu):
  - Cadència de 6 min, graella ancorada al minut 00 (00/06/12/.../54).
  - Eix Y en TMS: y_url = (2^z - 1) - y_xyz (el visor el pinta a 512px).
  - Paleta discreta = llegenda oficial (dBZ en passos de 3): cada color RGB és
    exactament una classe de 3 dBZ. Sense calibratge ni endevinalles.
  - Finestra pública ~3h i latència ~6-12 min → cal buscar enrere el darrer
    frame disponible i vigilar l'edat. Marca de temps en UTC.

Per què existeix: RainViewer ha servit frames congelats en ple episodi
(2026-06-05, 2026-09-16) i el model ML és cec al radar (0% importància). Aquesta
font és local, actualitza cada 6 min i dona dBZ exacte. AEMET es manté com a
font independent (el seu radar C-band va caure per 429 el 2026-06-04).

Documentació del visor: https://www.meteo.cat/observacions/radar
"""
import hashlib
import io
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session
from src.data._geo import _bearing_to_compass

logger = logging.getLogger(__name__)

# Sense clau: és un CDN estàtic (S3/CloudFront)
SESSION = create_session()
SESSION.headers.update({"Referer": "https://www.meteo.cat/observacions/radar"})

# Llegenda oficial "intensitat de precipitació" (extreta del visor): color RGB
# del tile → límit inferior de la classe (3 dBZ d'ample). El darrer és >66 dBZ.
LEGEND_DBZ = {
    (128, 0, 255): 9, (64, 0, 255): 12, (0, 0, 255): 15, (0, 255, 255): 18,
    (0, 255, 128): 21, (0, 255, 0): 24, (63, 255, 0): 27, (127, 255, 0): 30,
    (191, 255, 0): 33, (255, 255, 0): 36, (255, 171, 0): 39, (255, 129, 0): 42,
    (255, 87, 0): 45, (255, 45, 0): 48, (255, 0, 0): 51, (255, 0, 63): 54,
    (255, 0, 127): 57, (255, 0, 191): 60, (255, 0, 255): 63, (240, 240, 240): 66,
}


# ── Geometria ──

def _tile_coords(lat: float, lon: float,
                 z: int = config.METEO_RADAR_ZOOM) -> tuple[int, int]:
    """lat/lon → (x, y) de la tile, amb y en format TMS (origen al sud), que és
    el que espera l'URL de Meteocat: y_url = (2^z - 1) - y_xyz."""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = np.radians(lat)
    y_xyz = int((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n)
    return x, (n - 1) - y_xyz


def _pixel_in_tile(lat: float, lon: float, x: int, y_tms: int,
                   z: int = config.METEO_RADAR_ZOOM) -> tuple[float, float]:
    """Posició (px, py) d'un punt dins la tile, amb py comptat des de dalt."""
    n = 2 ** z
    frac_x = (lon + 180.0) / 360.0 * n - x
    lat_rad = np.radians(lat)
    y_xyz_norm = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n
    frac_y = y_xyz_norm - ((n - 1) - y_tms)
    return frac_x * 256.0, frac_y * 256.0


def _km_per_pixel(lat: float, z: int = config.METEO_RADAR_ZOOM) -> float:
    """Resolució real del tile a aquesta latitud (~0.91 km/px a z7 i 41.6N)."""
    return 40075.0 * np.cos(np.radians(lat)) / (2 ** z * 256.0)


def _tile_url(frame_dt: datetime, x: int, y_tms: int) -> str:
    return (f"{config.METEO_RADAR_TILE_BASE}/{frame_dt:%Y/%m/%d/%H/%M}"
            f"/{config.METEO_RADAR_ZOOM:02d}/000/000/{x:03d}/000/000/{y_tms:03d}.png")


def _floor_to_grid(dt: datetime) -> datetime:
    """Baixa al slot de 6 min de la graella (minut 00, 06, 12, ... 54)."""
    step = config.METEO_RADAR_FRAME_MINUTES
    return dt.replace(minute=(dt.minute // step) * step, second=0, microsecond=0)


def _fetch_tile(url: str) -> Optional[bytes]:
    """Descarrega una tile. Retorna None si no existeix (404 = frame no publicat)."""
    try:
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200 or len(r.content) < 200:
            return None
        return r.content
    except Exception as e:
        logger.debug(f"Tile Meteocat no disponible ({url}): {e}")
        return None


# ── Descodificació de la paleta ──

def _decode_dbz(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converteix la paleta del tile en (has_echo, dbz) de manera EXACTA.

    Cada color de la llegenda és una classe de 3 dBZ; s'usa el centre de la
    classe (límit inferior + 1.5). Píxels transparents o fora de paleta = sense
    eco: en aquest producte l'alpha=0 vol dir "sense precipitació".
    """
    h, w = arr.shape[:2]
    has_echo = np.zeros((h, w), dtype=bool)
    dbz = np.zeros((h, w), dtype=float)
    if arr.ndim < 3 or arr.shape[2] < 4:
        return has_echo, dbz
    rgb = arr[:, :, :3].astype(int)
    alpha = arr[:, :, 3]
    for color, lo in LEGEND_DBZ.items():
        m = ((rgb[:, :, 0] == color[0]) & (rgb[:, :, 1] == color[1])
             & (rgb[:, :, 2] == color[2]) & (alpha > 0))
        has_echo |= m
        dbz[m] = lo + 1.5
    return has_echo, dbz


# ── Estat entre runs (forense de frames repetits) ──

def _load_state() -> dict:
    try:
        with open(config.METEO_RADAR_STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    """Escriptura atòmica (tempfile + os.replace)."""
    try:
        os.makedirs(os.path.dirname(config.METEO_RADAR_STATE_FILE), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(config.METEO_RADAR_STATE_FILE))
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(tmp, config.METEO_RADAR_STATE_FILE)
    except Exception as e:
        logger.debug(f"No s'ha pogut desar l'estat del radar Meteocat: {e}")


# ── Descobriment del darrer frame ──

def _find_latest_frame(now: Optional[datetime] = None) -> tuple[Optional[datetime], Optional[bytes], int]:
    """Busca el frame més recent disponible endarrerint-se en slots de 6 min.

    Meteocat publica amb 6-12 min de latència, així que el slot actual sovint
    encara no hi és. Retorna (frame_dt, contingut, intents).
    """
    now = now or datetime.now(timezone.utc)
    slot = _floor_to_grid(now)
    x, y_tms = _tile_coords(config.LATITUDE, config.LONGITUDE)
    for attempt in range(config.METEO_RADAR_MAX_LOOKBACK_SLOTS):
        frame_dt = slot - timedelta(minutes=config.METEO_RADAR_FRAME_MINUTES * attempt)
        raw = _fetch_tile(_tile_url(frame_dt, x, y_tms))
        if raw is not None:
            return frame_dt, raw, attempt + 1
    return None, None, config.METEO_RADAR_MAX_LOOKBACK_SLOTS


def _mosaic(frame_dt: datetime, x: int, y_tms: int) -> Optional[np.ndarray]:
    """Mosaic de tiles al voltant de Cardedeu: la pròpia i la veïna de l'est.

    Cardedeu cau prop de la vora est de la seva tile (px ~215 de 256), així que
    el radi d'escaneig de 60km necessita la tile de l'est. Si no existeix, es
    treballa amb la tile sola (cobertura limitada a ~37km cap a l'est).
    """
    from PIL import Image
    tiles = []
    for xoff in (0, 1):
        raw = _fetch_tile(_tile_url(frame_dt, x + xoff, y_tms))
        if raw is None:
            if xoff == 0:
                return None
            break
        tiles.append(np.array(Image.open(io.BytesIO(raw)).convert("RGBA")))
    return np.hstack(tiles) if len(tiles) > 1 else tiles[0]


# ── Escaneig espacial ──

def _empty_spatial() -> dict:
    return {
        "nearest_echo_km": None, "nearest_echo_bearing": None,
        "max_dbz_20km": 0.0, "coverage_20km": 0.0,
        "quadrant_max_dbz": {}, "quadrant_coverage": {},
        "upwind_nearest_echo_km": None, "upwind_max_dbz": 0.0,
    }


def _scan_spatial(lat: float, lon: float, mosaic: np.ndarray,
                  x: int, y_tms: int,
                  wind_from_dir: Optional[float] = None) -> dict:
    """Escaneig circular al voltant de Cardedeu (anàleg a _scan_radar_spatial).

    Retorna eco més proper (km i rumb), dBZ màxim a 20km, cobertura, màxims per
    quadrant i sector de sobrevent. Els càlculs són idèntics als de RainViewer
    perquè les features siguin comparables.
    """
    has_echo, dbz = _decode_dbz(mosaic)
    km_px = _km_per_pixel(lat)
    cx, cy = _pixel_in_tile(lat, lon, x, y_tms)
    radius = config.RADAR_SCAN_RADIUS_KM
    h, w = has_echo.shape
    radius_px = int(radius / km_px)

    y_lo, y_hi = max(0, int(cy) - radius_px), min(h, int(cy) + radius_px + 1)
    x_lo, x_hi = max(0, int(cx) - radius_px), min(w, int(cx) + radius_px + 1)
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dy = (yy - cy).astype(float)
    dx = (xx - cx).astype(float)
    dist_km = np.sqrt(dx ** 2 + dy ** 2) * km_px
    in_radius = dist_km <= radius

    region_echo = has_echo[y_lo:y_hi, x_lo:x_hi] & in_radius
    region_dbz = dbz[y_lo:y_hi, x_lo:x_hi]

    out = _empty_spatial()
    if not region_echo.any():
        return out

    echo_dist = np.where(region_echo, dist_km, np.inf)
    idx = np.unravel_index(echo_dist.argmin(), echo_dist.shape)
    bearing = float((np.degrees(np.arctan2(dx[idx], -dy[idx])) + 360) % 360)
    out["nearest_echo_km"] = round(float(dist_km[idx]), 1)
    out["nearest_echo_bearing"] = round(bearing, 0)

    within_20 = region_echo & (dist_km <= 20)
    out["max_dbz_20km"] = round(float(region_dbz[within_20].max()), 1) if within_20.any() else 0.0
    total_20 = int((in_radius & (dist_km <= 20)).sum())
    out["coverage_20km"] = round(int(within_20.sum()) / total_20, 4) if total_20 else 0.0

    echo_dbz = region_dbz[region_echo]
    echo_bearings = (np.degrees(np.arctan2(dx[region_echo], -dy[region_echo])) + 360) % 360
    total_scan = int(in_radius.sum())
    for name, lo, hi in (("N", 315, 45), ("E", 45, 135), ("S", 135, 225), ("W", 225, 315)):
        q = ((echo_bearings >= lo) | (echo_bearings < hi)) if lo > hi \
            else ((echo_bearings >= lo) & (echo_bearings < hi))
        out["quadrant_max_dbz"][name] = round(float(echo_dbz[q].max()), 1) if q.any() else 0.0
        out["quadrant_coverage"][name] = (
            round(float(q.sum()) / (total_scan / 4), 4) if q.any() and total_scan else 0.0)

    if wind_from_dir is not None:
        angle_diff = ((echo_bearings - wind_from_dir + 180) % 360) - 180
        upwind = np.abs(angle_diff) <= 60
        if upwind.any():
            up_d = np.sqrt(dx[region_echo][upwind] ** 2 + dy[region_echo][upwind] ** 2) * km_px
            out["upwind_nearest_echo_km"] = round(float(up_d.min()), 1)
            out["upwind_max_dbz"] = round(float(echo_dbz[upwind].max()), 1)
    return out


def _centroid_d_to_center(has_echo: np.ndarray, dbz: np.ndarray,
                          cx: float, cy: float) -> Optional[tuple[float, float]]:
    """Centroide ponderat per dBZ, en desplaçament (dx, dy) respecte Cardedeu."""
    if not has_echo.any():
        return None
    yy, xx = np.mgrid[0:has_echo.shape[0], 0:has_echo.shape[1]]
    dy = (yy - cy).astype(float)
    dx = (xx - cx).astype(float)
    w = dbz[has_echo]
    if w.sum() <= 0:
        return None
    return float((dx[has_echo] * w).sum() / w.sum()), float((dy[has_echo] * w).sum() / w.sum())


def _estimate_drift(prev_dbz: np.ndarray, curr_dbz: np.ndarray,
                    prev_has: np.ndarray, curr_has: np.ndarray,
                    km_px: float, dt_min: float, cx: float, cy: float) -> dict:
    """Moviment de la cel·la entre dos frames (centroides ponderats per dBZ).

    Retorna velocitat (km/h), rumb, ETA a Cardedeu i si s'acosta. L'ETA només
    s'estima quan la distància al centroide disminueix; si la cel·la s'allunya,
    approaching=False i ETA=None.
    """
    c_prev = _centroid_d_to_center(prev_has, prev_dbz, cx, cy)
    c_curr = _centroid_d_to_center(curr_has, curr_dbz, cx, cy)
    if c_prev is None or c_curr is None:
        return {"speed_kmh": None, "bearing": None, "eta_min": None, "approaching": False}

    vx = (c_curr[0] - c_prev[0]) * km_px
    vy = (c_curr[1] - c_prev[1]) * km_px
    hours = dt_min / 60.0
    speed = float(np.hypot(vx, vy) / hours)
    # Guard de plausibilitat: un centroide que salta >120 km/h en 6 min és
    # creixement/decadència de la cel·la o un canvi de font, no moviment real.
    # No s'ha d'alimentar les regles físiques amb això (2026-09-16: 586 km/h).
    if speed > config.METEO_RADAR_MAX_PLAUSIBLE_KMH:
        logger.debug(f"Deriva Meteocat no plausible ({speed:.0f} km/h): descartada")
        return {"speed_kmh": None, "bearing": None, "eta_min": None, "approaching": False}

    bearing = float((np.degrees(np.arctan2(vx, -vy)) + 360) % 360)

    d_now = float(np.hypot(c_curr[0], c_curr[1]) * km_px)
    d_prev = float(np.hypot(c_prev[0], c_prev[1]) * km_px)
    closing_kmh = (d_prev - d_now) / hours
    if closing_kmh > 1.0:
        eta = d_now / closing_kmh * 60.0
        return {"speed_kmh": round(speed, 1), "bearing": round(bearing, 0),
                "eta_min": round(eta, 1), "approaching": True}
    return {"speed_kmh": round(speed, 1), "bearing": round(bearing, 0),
            "eta_min": None, "approaching": False}


# ── API pública ──

def _empty_result() -> dict:
    return {
        "meteocat_radar_available": False,
        "meteocat_radar_dbz": 0.0,
        "meteocat_radar_has_echo": False,
        "meteocat_radar_nearest_echo_km": None,
        "meteocat_radar_nearest_echo_compass": None,
        "meteocat_radar_max_dbz_20km": 0.0,
        "meteocat_radar_coverage_20km": 0.0,
        "meteocat_radar_echoes_found": False,
        "meteocat_radar_upwind_nearest_echo_km": None,
        "meteocat_radar_upwind_max_dbz": 0.0,
        "meteocat_radar_storm_drift_kmh": None,
        "meteocat_radar_storm_bearing": None,
        "meteocat_radar_storm_eta_min": None,
        "meteocat_radar_storm_approaching": False,
        "meteocat_radar_frame_time": None,
        "meteocat_radar_frame_age_min": None,
        "meteocat_radar_frame_md5": None,
        "meteocat_radar_same_frame_streak": 0,
        "meteocat_radar_fetched_at": None,
    }


def fetch_meteocat_radar(wind_from_dir: Optional[float] = None) -> dict:
    """Radar Meteocat de Cardedeu amb features anàlogues a les de RainViewer.

    Mai llança: si no hi ha tile fresca retorna el darrer estat vàlid (si encara
    és dins la finestra d'edat) o un resultat buit. El consum és de 2-3 tiles de
    ~18KB per run, sobre un CDN estàtic sense quota.
    """
    try:
        now = datetime.now(timezone.utc)
        frame_dt, raw, attempts = _find_latest_frame(now)
        if frame_dt is None or raw is None:
            logger.warning("Radar Meteocat: cap frame a la finestra pública (~3h)")
            return _stale_or_empty()

        age_min = (now - frame_dt).total_seconds() / 60.0
        if age_min > config.METEO_RADAR_STALE_MAX_MIN:
            logger.warning(f"Radar Meteocat: frame massa vell ({age_min:.0f} min)")
            return _stale_or_empty()

        x, y_tms = _tile_coords(config.LATITUDE, config.LONGITUDE)
        mosaic = _mosaic(frame_dt, x, y_tms)
        if mosaic is None:
            return _stale_or_empty()

        spatial = _scan_spatial(config.LATITUDE, config.LONGITUDE, mosaic, x, y_tms,
                                wind_from_dir=wind_from_dir)

        has_echo, dbz_grid = _decode_dbz(mosaic)
        px_f, py_f = _pixel_in_tile(config.LATITUDE, config.LONGITUDE, x, y_tms)
        px, py = int(round(px_f)), int(round(py_f))
        dbz_at_pixel = 0.0
        if 0 <= py < dbz_grid.shape[0] and 0 <= px < dbz_grid.shape[1]:
            dbz_at_pixel = float(dbz_grid[py, px])

        # Forense: MD5 del frame i ratxa de frames idèntics (congelació de font)
        md5 = hashlib.md5(raw).hexdigest()
        state = _load_state()
        streak = int(state.get("same_frame_streak", 0)) + 1 if state.get("md5") == md5 else 0

        # Moviment: frame anterior (6 min abans) amb la MATEIXA geometria de
        # mosaic. Comparar una tile sola contra un mosaic desplaça el centroide
        # artificialment (veure 2026-09-16: 586 km/h impossible).
        drift = {"speed_kmh": None, "bearing": None, "eta_min": None, "approaching": False}
        prev_mosaic = _mosaic(
            frame_dt - timedelta(minutes=config.METEO_RADAR_FRAME_MINUTES), x, y_tms)
        if prev_mosaic is not None and prev_mosaic.shape == mosaic.shape:
            prev_has, prev_dbz = _decode_dbz(prev_mosaic)
            drift = _estimate_drift(prev_dbz, dbz_grid, prev_has, has_echo,
                                    _km_per_pixel(config.LATITUDE),
                                    float(config.METEO_RADAR_FRAME_MINUTES), px_f, py_f)

        result = {
            "meteocat_radar_available": True,
            "meteocat_radar_dbz": round(dbz_at_pixel, 1),
            "meteocat_radar_has_echo": dbz_at_pixel >= config.RADAR_MIN_DBZ,
            "meteocat_radar_nearest_echo_km": spatial["nearest_echo_km"],
            "meteocat_radar_nearest_echo_compass": (
                _bearing_to_compass(spatial["nearest_echo_bearing"])
                if spatial["nearest_echo_bearing"] is not None else None),
            "meteocat_radar_max_dbz_20km": spatial["max_dbz_20km"],
            "meteocat_radar_coverage_20km": spatial["coverage_20km"],
            "meteocat_radar_echoes_found": spatial["nearest_echo_km"] is not None,
            "meteocat_radar_upwind_nearest_echo_km": spatial["upwind_nearest_echo_km"],
            "meteocat_radar_upwind_max_dbz": spatial["upwind_max_dbz"],
            "meteocat_radar_storm_drift_kmh": drift["speed_kmh"],
            "meteocat_radar_storm_bearing": drift["bearing"],
            "meteocat_radar_storm_eta_min": drift["eta_min"],
            "meteocat_radar_storm_approaching": drift["approaching"],
            "meteocat_radar_frame_time": frame_dt.isoformat(),
            "meteocat_radar_frame_age_min": round(age_min, 1),
            "meteocat_radar_frame_md5": md5,
            "meteocat_radar_same_frame_streak": streak,
            "meteocat_radar_fetched_at": now.isoformat(),
        }
        for name in ("N", "E", "S", "W"):
            result[f"meteocat_radar_quadrant_max_dbz_{name}"] = \
                spatial["quadrant_max_dbz"].get(name, 0.0)

        _save_state({"frame_time": frame_dt.isoformat(), "md5": md5,
                     "same_frame_streak": streak, "last_result": result})

        logger.info(
            f"  Radar Meteocat ({frame_dt:%H:%M}Z, fa {age_min:.0f}min, {attempts} intents): "
            f"dBZ={result['meteocat_radar_dbz']}, "
            f"eco_proper={result['meteocat_radar_nearest_echo_km']}km, "
            f"max20km={result['meteocat_radar_max_dbz_20km']}dBZ, "
            f"cobertura={result['meteocat_radar_coverage_20km']:.1%}"
        )
        return result

    except Exception as e:
        logger.warning(f"Error processant radar Meteocat: {e}")
        return _stale_or_empty()


def _stale_or_empty() -> dict:
    """Fallback: darrer estat vàlid si encara és dins la finestra d'edat.

    Un frame de fa 20-30 min encara permet que les regles físiques vegin una
    tempesta que s'acosta (mateix criteri que el radar AEMET).
    """
    cached = _load_state().get("last_result")
    if isinstance(cached, dict) and cached.get("meteocat_radar_frame_time"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(
                cached["meteocat_radar_frame_time"])).total_seconds() / 60.0
            if age <= config.METEO_RADAR_STALE_MAX_MIN:
                out = dict(cached)
                out["meteocat_radar_available"] = False  # no ve d'aquest run
                return out
        except Exception:
            pass
    return _empty_result()