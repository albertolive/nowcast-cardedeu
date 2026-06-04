"""
Client per a l'API de RainViewer.
Obté dades de radar de precipitació en temps real per a Cardedeu.
Inclou escaneig espacial: detecta ecos en un radi de 30km,
rastreja el moviment de les cel·les de pluja i estima ETA.
https://www.rainviewer.com/api.html
"""
import hashlib
import io
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session
from src.data._geo import _bearing_to_compass

logger = logging.getLogger(__name__)

SESSION = create_session()


def _get_radar_frames() -> dict:
    """Obté la llista de frames de radar disponibles."""
    r = SESSION.get(config.RAINVIEWER_API_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def _extract_pixel_intensity(png_bytes: bytes, px: int, py: int) -> int:
    """
    Llegeix la intensitat del radar d'un píxel concret d'un tile PNG.
    RainViewer escala de grisos: R=G=B (0-255), alpha>0 = cobertura radar.
    alpha=0: fora de cobertura radar.
    alpha>0, R=0: radar cobreix la zona però NO detecta precipitació.
    alpha>0, R>0: precipitació detectada (R més alt = més intensa).
    Retorna 0 (sense precipitació) o el valor R (1-255).
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        r, g, b, a = img.getpixel((px, py))
        if a == 0:
            return 0
        return r  # R=0 = no precipitació, R>0 = precipitació
    except ImportError:
        return -1


def _radar_intensity_to_dbz(intensity: int) -> float:
    """
    Converteix la intensitat del píxel (R=0-255) a dBZ.
    Fórmula estàndard RainViewer: dBZ = R / 2 - 32
    (derivada de l'esquema B&W: R_bw = dBZ+32, tiles 256px escalen ×2).
    Les tiles 256px quantitzen a 10 nivells: R=0,38,75,110,146,177,203,222,242,255.
    R=255→95.5dBZ és físicament impossible; es limita a 65 dBZ (màxim
    raonable per precipitació, valors superiors són soroll de muntanya/artefactes).
    """
    if intensity <= 0:
        return 0.0
    dbz = intensity / 2.0 - 32.0
    return min(dbz, 65.0)


def _dbz_to_rain_rate(dbz: float) -> float:
    """
    Converteix dBZ a mm/h aproximat (fórmula Marshall-Palmer).
    Z = 200 * R^1.6 → R = (Z/200)^(1/1.6)
    """
    if dbz <= 0:
        return 0.0
    z_linear = 10 ** (dbz / 10.0)
    return (z_linear / 200.0) ** (1.0 / 1.6)


def _scan_radar_spatial(png_bytes: bytes, cx: int, cy: int,
                        radius_km: float, pixel_size_km: float,
                        wind_from_dir: Optional[float] = None,
                        clutter_mask: Optional[np.ndarray] = None) -> dict:
    """
    Escaneja una zona circular del tile de radar al voltant de Cardedeu.
    Detecta ecos de pluja, calcula distància, cobertura i mètriques espacials.

    Args:
        png_bytes: Bytes del tile PNG de RainViewer
        cx, cy: Coordenades del píxel central (Cardedeu)
        radius_km: Radi d'escaneig en km
        pixel_size_km: Mida de cada píxel en km
        wind_from_dir: Direcció d'on ve el vent (graus, 0=N, 90=E).
                       Si proporcionat, calcula mètriques del sector de sobrevent.
        clutter_mask: Matriu booleana (mida del tile) amb True als píxels de clutter.
                      Píxels marcats com a clutter s'exclouen de la detecció d'eco.

    Returns:
        Dict amb mètriques espacials del radar.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        arr = np.array(img)
    except Exception:
        return _empty_spatial_result(radius_km)

    h, w = arr.shape[:2]
    radius_px = int(radius_km / pixel_size_km)

    # Definir la subregió a escanejar
    y_lo = max(0, cy - radius_px)
    y_hi = min(h, cy + radius_px + 1)
    x_lo = max(0, cx - radius_px)
    x_hi = min(w, cx + radius_px + 1)

    # Coordenades relatives al centre
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dy = yy - cy
    dx = xx - cx
    dist_km = np.sqrt(dx.astype(float)**2 + dy.astype(float)**2) * pixel_size_km

    # Màscara circular
    in_radius = dist_km <= radius_km

    # Extreure intensitat: escala de grisos R=G=B (0-255)
    # alpha=0: fora de cobertura radar
    # alpha>0, R=0: cobertura radar sense precipitació
    # alpha>0, R>0: precipitació detectada
    region = arr[y_lo:y_hi, x_lo:x_hi]
    r_channel = region[:, :, 0].astype(float)
    alpha = region[:, :, 3]
    intensity = r_channel

    # Ecos: alpha > 0 (cobertura radar) AND R > 0 (precipitació detectada)
    has_echo = (alpha > 0) & (r_channel > 0) & in_radius

    # Filtre dBZ mínim: descartar soroll, propagació anòmala i clutter transient
    # R=255 (cap a 95.5 dBZ) gairebé sempre és artefacte de terra, no pluja real
    dbz_grid = r_channel / 2.0 - 32.0
    has_echo = has_echo & (dbz_grid >= config.RADAR_MIN_DBZ) & (r_channel < 255)

    # Excloure clutter (ecos permanents detectats en tots els frames)
    if clutter_mask is not None:
        clutter_region = clutter_mask[y_lo:y_hi, x_lo:x_hi]
        has_echo = has_echo & ~clutter_region

    if not has_echo.any():
        result = _empty_spatial_result(radius_km)
        # Afegir centroid buit per tracking
        result["_centroid_dx"] = None
        result["_centroid_dy"] = None
        return result

    # ── Eco més proper ──
    echo_distances = np.where(has_echo, dist_km, np.inf)
    nearest_flat = echo_distances.argmin()
    nearest_idx = np.unravel_index(nearest_flat, echo_distances.shape)
    nearest_km = float(dist_km[nearest_idx])
    nearest_dx = float(dx[nearest_idx])
    nearest_dy = float(dy[nearest_idx])
    # Rumb geogràfic: 0=N, 90=E, 180=S, 270=W
    nearest_bearing = float((np.degrees(np.arctan2(nearest_dx, -nearest_dy)) + 360) % 360)

    # ── Màxim dBZ dins de 20km ──
    within_20km = in_radius & (dist_km <= 20) & has_echo
    max_intensity_20km = float(intensity[within_20km].max()) if within_20km.any() else 0
    max_dbz_20km = _radar_intensity_to_dbz(int(max_intensity_20km))

    # ── Cobertura dins de 20km (fracció de píxels amb eco) ──
    total_20km = int((in_radius & (dist_km <= 20)).sum())
    echo_20km = int(within_20km.sum())
    coverage_20km = float(echo_20km / total_20km) if total_20km > 0 else 0.0

    # ── Centroide ponderat de tots els ecos (per tracking de moviment) ──
    echo_intensities = intensity[has_echo]
    echo_dx_arr = dx[has_echo].astype(float)
    echo_dy_arr = dy[has_echo].astype(float)
    total_int = echo_intensities.sum()
    centroid_dx = float((echo_dx_arr * echo_intensities).sum() / total_int)
    centroid_dy = float((echo_dy_arr * echo_intensities).sum() / total_int)

    result = {
        "nearest_echo_km": round(nearest_km, 1),
        "nearest_echo_bearing": round(nearest_bearing, 0),
        "nearest_echo_compass": _bearing_to_compass(nearest_bearing),
        "max_dbz_20km": round(max_dbz_20km, 1),
        "coverage_20km": round(coverage_20km, 4),
        "echoes_found": True,
        "_centroid_dx": centroid_dx,
        "_centroid_dy": centroid_dy,
    }

    # ── Quadrant features: intensitat i cobertura per N/E/S/W ──
    echo_bearings = (np.degrees(np.arctan2(echo_dx_arr, -echo_dy_arr)) + 360) % 360
    total_echo_pixels = len(echo_dx_arr)
    # Total pixels in scan area for coverage calculation
    total_scan_pixels = int(in_radius.sum())

    for quad_name, q_lo, q_hi in [("N", 315, 45), ("E", 45, 135),
                                    ("S", 135, 225), ("W", 225, 315)]:
        if q_lo > q_hi:
            q_mask = (echo_bearings >= q_lo) | (echo_bearings < q_hi)
        else:
            q_mask = (echo_bearings >= q_lo) & (echo_bearings < q_hi)

        if q_mask.any():
            result[f"quadrant_max_dbz_{quad_name}"] = round(
                _radar_intensity_to_dbz(int(echo_intensities[q_mask].max())), 1
            )
            result[f"quadrant_coverage_{quad_name}"] = round(
                float(q_mask.sum()) / (total_scan_pixels / 4) if total_scan_pixels > 0 else 0.0, 4
            )
        else:
            result[f"quadrant_max_dbz_{quad_name}"] = 0.0
            result[f"quadrant_coverage_{quad_name}"] = 0.0

    # ── Sector de sobrevent (upwind): d'on ve el vent → d'on esperem la pluja ──
    if wind_from_dir is not None:
        angle_diff = ((echo_bearings - wind_from_dir + 180) % 360) - 180
        upwind_mask = np.abs(angle_diff) <= 60  # ±60° from wind direction

        if upwind_mask.any():
            upwind_distances = np.sqrt(
                echo_dx_arr[upwind_mask]**2 + echo_dy_arr[upwind_mask]**2
            ) * pixel_size_km
            upwind_intensities = echo_intensities[upwind_mask]
            result["upwind_nearest_echo_km"] = round(float(upwind_distances.min()), 1)
            result["upwind_max_dbz"] = round(
                _radar_intensity_to_dbz(int(upwind_intensities.max())), 1
            )
        else:
            result["upwind_nearest_echo_km"] = radius_km
            result["upwind_max_dbz"] = 0.0
    else:
        result["upwind_nearest_echo_km"] = radius_km
        result["upwind_max_dbz"] = 0.0

    return result


def _build_clutter_mask(tile_bytes_list: list) -> Optional[np.ndarray]:
    """
    Construeix una màscara booleana d'ecos permanents (clutter de terra).
    Un píxel es considera clutter si mostra eco en TOTS els frames vàlids
    i la seva intensitat NO varia entre frames (variància zero = muntanya/edifici).
    Pluja real fluctua entre frames (10 min cadascun); clutter de terra és estàtic.
    Necessita mínim 3 frames vàlids per ser fiable.
    """
    from PIL import Image

    echo_count = None
    intensity_sum = None
    intensity_sq_sum = None
    valid_count = 0

    for tile_bytes in tile_bytes_list:
        if tile_bytes is None:
            continue
        try:
            img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
            arr = np.array(img)
            has_echo = (arr[:, :, 3] > 0) & (arr[:, :, 0] > 0)
            r_ch = arr[:, :, 0].astype(float)
            dbz_arr = r_ch / 2.0 - 32.0
            has_echo = has_echo & (dbz_arr >= config.RADAR_MIN_DBZ) & (r_ch < 255)
            if echo_count is None:
                echo_count = has_echo.astype(int)
                intensity_sum = np.where(has_echo, r_ch, 0.0)
                intensity_sq_sum = np.where(has_echo, r_ch ** 2, 0.0)
            else:
                echo_count += has_echo.astype(int)
                intensity_sum += np.where(has_echo, r_ch, 0.0)
                intensity_sq_sum += np.where(has_echo, r_ch ** 2, 0.0)
            valid_count += 1
        except Exception:
            continue

    if valid_count < 3 or echo_count is None:
        return None

    persistent = echo_count >= valid_count

    if not persistent.any():
        return None

    # Variància d'intensitat per píxels persistents:
    # Clutter (muntanyes, edificis) → mateixa R cada frame → variància ≈ 0
    # Pluja real → intensitat fluctua entre frames → variància > 0
    safe_count = np.maximum(echo_count, 1)
    mean_int = intensity_sum / safe_count
    var_int = intensity_sq_sum / safe_count - mean_int ** 2
    # Llindars de variància: <1.0 = estàtic, >=1.0 = variable
    # IMPORTANT: la quantització de RainViewer (10 nivells) fa que la pluja
    # SOSTINGUDA durant >2h tingui variància 0 si la intensitat és estable.
    # Per evitar filtrar pluja real, el clutter requereix també intensitat
    # SATURADA (mean R ≥ 200, dBZ ≥ 68): les muntanyes retornen senyals
    # extrems consistentment; la pluja real, fins i tot sostinguda, rarament
    # supera R=200 sense fluctuar.
    static_persistent = persistent & (var_int < 1.0) & (mean_int >= 200)
    variable_persistent = persistent & ~static_persistent

    # Ecos persistents estàtics = clutter definitiu (sempre filtrat)
    # Ecos persistents variables = possible pluja frontal
    # Només deshabilitar el filtre de variables si la cobertura és significativa
    variable_frac = variable_persistent.sum() / persistent.size
    if variable_frac > 0.005:
        # Pluja frontal persistent (variable): no filtrar, però SÍ filtrar clutter estàtic
        clutter = static_persistent
    else:
        clutter = persistent

    clutter_count = int(clutter.sum())
    if clutter_count > 0:
        static_count = int(static_persistent.sum())
        logger.debug(f"  Clutter mask: {clutter_count} px filtrats "
                     f"({static_count} estàtics, {int(variable_persistent.sum())} variables)")

    return clutter if clutter.any() else None


def _empty_spatial_result(radius_km: float) -> dict:
    result = {
        "nearest_echo_km": radius_km,
        "nearest_echo_bearing": None,
        "nearest_echo_compass": None,
        "max_dbz_20km": 0.0,
        "coverage_20km": 0.0,
        "echoes_found": False,
        "upwind_nearest_echo_km": radius_km,
        "upwind_max_dbz": 0.0,
        "_centroid_dx": None,
        "_centroid_dy": None,
    }
    for quad in ("N", "E", "S", "W"):
        result[f"quadrant_max_dbz_{quad}"] = 0.0
        result[f"quadrant_coverage_{quad}"] = 0.0
    return result


def _echo_field(tile_bytes: bytes, clutter_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Camp d'intensitat d'eco del tile sencer (float, 0 on no hi ha eco vàlid).
    Per a la correlació de fase entre frames. Retorna None si el tile falla
    o té massa pocs píxels d'eco per correlacionar de manera fiable.
    """
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        arr = np.array(img)
    except Exception:
        return None
    r_ch = arr[:, :, 0].astype(float)
    alpha = arr[:, :, 3]
    dbz = r_ch / 2.0 - 32.0
    mask = (alpha > 0) & (r_ch > 0) & (dbz >= config.RADAR_MIN_DBZ) & (r_ch < 255)
    if clutter_mask is not None:
        mask = mask & ~clutter_mask
    if mask.sum() < 10:
        return None
    return np.where(mask, r_ch, 0.0)


def _xcorr_displacement(prev_field: np.ndarray, curr_field: np.ndarray) -> Optional[tuple]:
    """
    Desplaçament (dy_px, dx_px) del camp d'ecos prev→curr per correlació
    creuada (estàndard en nowcasting tipus TREC). A diferència del centroide,
    funciona amb múltiples cel·les i amb cel·les que neixen/moren: troba el
    desplaçament que millor superposa els dos camps. Correlació plana (no de
    fase): la normalització de fase és fràgil quan la forma del camp canvia
    entre frames (cel·les creixent) i dóna desplaçaments espuris.
    """
    f_prev = np.fft.fft2(prev_field)
    f_curr = np.fft.fft2(curr_field)
    corr = np.fft.ifft2(f_curr * np.conj(f_prev)).real
    h, w = corr.shape
    # Mitjana de l'altiplà del pic (no argmax): quan una cel·la petita queda
    # dins d'una de gran, l'overlap és constant per a un rang de desplaçaments
    # i argmax triaria un punt arbitrari de l'altiplà. La mitjana, per
    # simetria, dóna el centre (0 en el cas estacionari).
    ys, xs = np.nonzero(corr >= 0.999 * corr.max())
    dys = np.where(ys <= h // 2, ys, ys - h).astype(float)
    dxs = np.where(xs <= w // 2, xs, xs - w).astype(float)
    dy = float(dys.mean())
    dx = float(dxs.mean())
    # Desplaçaments >45px/frame (~120 km/h) són soroll de correlació, no tempestes
    if abs(dy) > 45 or abs(dx) > 45:
        return None
    return float(dy), float(dx)


def _estimate_motion_xcorr(tile_bytes_list: list,
                           clutter_mask: Optional[np.ndarray],
                           pixel_size_km: float,
                           frame_interval_min: float = 10.0) -> Optional[tuple]:
    """
    Vector de moviment (v_ew_kmh, v_ns_kmh) del camp d'ecos, com a mediana
    dels desplaçaments per correlació de fase entre frames consecutius.
    Retorna None si no hi ha prou parells vàlids.
    """
    fields = [
        _echo_field(tb, clutter_mask) if tb is not None else None
        for tb in tile_bytes_list
    ]
    displacements = []
    for prev_f, curr_f in zip(fields, fields[1:]):
        if prev_f is not None and curr_f is not None:
            d = _xcorr_displacement(prev_f, curr_f)
            if d is not None:
                displacements.append(d)
    if not displacements:
        return None
    med_dy = float(np.median([d[0] for d in displacements]))
    med_dx = float(np.median([d[1] for d in displacements]))
    km_per_min = pixel_size_km / frame_interval_min
    v_ew = med_dx * km_per_min * 60  # + = cap a l'est
    v_ns = med_dy * km_per_min * 60  # + = cap al sud (Y creix avall)
    return v_ew, v_ns


def _estimate_storm_tracking(spatial_scans: list, pixel_size_km: float,
                              frame_interval_min: float = 10.0,
                              xcorr_velocity: Optional[tuple] = None) -> dict:
    """
    Estima la velocitat i ETA de les cel·les de pluja.

    Mètode preferit: vector de correlació de fase (xcorr_velocity) projectat
    radialment sobre l'eco més proper. Fallback: moviment del centroide entre
    frames (falla amb cel·les que neixen in situ — el centroide no es mou).

    Args:
        spatial_scans: Llista de resultats de _scan_radar_spatial()
        pixel_size_km: km per píxel
        frame_interval_min: Minuts entre frames (~10 min per RainViewer)
        xcorr_velocity: (v_ew_kmh, v_ns_kmh) de _estimate_motion_xcorr(), o None

    Returns:
        Dict amb velocitat de la tempesta i ETA estimat.
    """
    if xcorr_velocity is not None:
        velocity_ew, velocity_ns = xcorr_velocity
        velocity_kmh = float(np.hypot(velocity_ew, velocity_ns))
        latest = spatial_scans[-1] if spatial_scans else {}
        bearing = latest.get("nearest_echo_bearing")
        nearest_km = latest.get("nearest_echo_km")
        approaching = False
        eta_min = None
        if latest.get("echoes_found") and bearing is not None and nearest_km is not None:
            # Posició de l'eco en coords (E, S); velocitat d'aproximació = -(v·p̂)
            theta = np.radians(bearing)
            p_e = np.sin(theta)
            p_s = -np.cos(theta)
            approach_speed_kmh = -(velocity_ew * p_e + velocity_ns * p_s)
            approaching = approach_speed_kmh > 2
            if approaching and approach_speed_kmh > 0:
                eta_min = round(nearest_km / approach_speed_kmh * 60)
                eta_min = max(0, min(eta_min, 180))
        return {
            "storm_velocity_kmh": round(velocity_kmh, 1),
            "storm_velocity_ns": round(velocity_ns, 1),
            "storm_velocity_ew": round(velocity_ew, 1),
            "storm_approaching": bool(approaching),
            "storm_eta_min": eta_min,
        }

    # Filtrar scans amb ecos i centroide vàlid
    valid = [(s["_centroid_dx"], s["_centroid_dy"])
             for s in spatial_scans
             if s.get("echoes_found") and s.get("_centroid_dx") is not None]

    if len(valid) < 2:
        return {
            "storm_velocity_kmh": 0.0,
            "storm_velocity_ns": 0.0,
            "storm_velocity_ew": 0.0,
            "storm_approaching": False,
            "storm_eta_min": None,
        }

    # Desplaçament total del centroide (primer → últim)
    first_dx, first_dy = valid[0]
    last_dx, last_dy = valid[-1]
    n_intervals = len(valid) - 1
    total_time_min = n_intervals * frame_interval_min

    # Velocitat del centroide
    move_dx = last_dx - first_dx   # píxels (positiu = cap a l'est)
    move_dy = last_dy - first_dy   # píxels (positiu = cap al sud, ja que Y creix avall)
    move_km = np.sqrt(move_dx**2 + move_dy**2) * pixel_size_km
    velocity_kmh = (move_km / total_time_min * 60) if total_time_min > 0 else 0.0

    # Descomposició N-S / E-W (km/h)
    # Convenció: NS positiu = cap al sud, EW positiu = cap a l'est
    if total_time_min > 0:
        velocity_ew = (move_dx * pixel_size_km / total_time_min) * 60  # + = est
        velocity_ns = (move_dy * pixel_size_km / total_time_min) * 60  # + = sud
    else:
        velocity_ew = 0.0
        velocity_ns = 0.0

    # Distància del centroide al centre (Cardedeu) en cada instant
    dist_first = np.sqrt(first_dx**2 + first_dy**2) * pixel_size_km
    dist_last = np.sqrt(last_dx**2 + last_dy**2) * pixel_size_km
    approach_km = dist_first - dist_last  # Positiu = s'acosta

    approaching = approach_km > 1.0  # Almenys 1km d'aproximació

    # ETA: temps estimat d'arribada a Cardedeu
    eta_min = None
    if approaching and total_time_min > 0:
        approach_speed_kmh = (approach_km / total_time_min) * 60
        if approach_speed_kmh > 2:  # Mínim 2 km/h per ser significatiu
            latest_scan = spatial_scans[-1]
            nearest_km = latest_scan.get("nearest_echo_km", config.RADAR_SCAN_RADIUS_KM)
            eta_min = round((nearest_km / approach_speed_kmh) * 60)
            eta_min = max(0, min(eta_min, 180))  # Cap a 0-180 min

    return {
        "storm_velocity_kmh": round(velocity_kmh, 1),
        "storm_velocity_ns": round(velocity_ns, 1),   # + = cap al sud
        "storm_velocity_ew": round(velocity_ew, 1),   # + = cap a l'est
        "storm_approaching": approaching,
        "storm_eta_min": eta_min,
    }


def fetch_radar_at_cardedeu(wind_from_dir: Optional[float] = None) -> dict:
    """
    Obté les dades de radar actuals per a la ubicació de Cardedeu.
    Inclou escaneig espacial en un radi de 30km al voltant del poble.

    Args:
        wind_from_dir: Direcció d'on ve el vent a 850hPa (graus).
                       Permet calcular mètriques del sector de sobrevent.

    Retorna un diccionari amb:
    - Mètriques puntuals (píxel a Cardedeu): dbz, rain_rate, has_echo
    - Mètriques espacials (radi 30km): nearest_echo_km, coverage, upwind
    - Tracking de tempesta: velocity, approaching, ETA
    """
    try:
        data = _get_radar_frames()
    except Exception as e:
        logger.warning(f"RainViewer API no disponible: {e}")
        return _empty_radar_result()

    past_frames = data.get("radar", {}).get("past", [])

    if not past_frames:
        return _empty_radar_result()

    # Analitzar els últims 6 frames (~1h de radar, cada 10min)
    recent_frames = past_frames[-6:]
    intensities = []
    tile_bytes_list = []

    # Fase 1: Descarregar tots els tiles i extreure intensitat puntual
    for frame in recent_frames:
        tile_url = (
            f"{config.RAINVIEWER_TILE_BASE}{frame['path']}/256/"
            f"{config.RAINVIEWER_TILE_ZOOM}/{config.RAINVIEWER_TILE_X}/"
            f"{config.RAINVIEWER_TILE_Y}/2/1_1.png"
        )
        try:
            r = SESSION.get(tile_url, timeout=5)
            if r.status_code == 200 and len(r.content) > 100:
                tile_bytes_list.append(r.content)
                intensity = _extract_pixel_intensity(
                    r.content,
                    config.RAINVIEWER_PIXEL_X,
                    config.RAINVIEWER_PIXEL_Y,
                )
                intensities.append(intensity)
            else:
                tile_bytes_list.append(None)
                intensities.append(0)
        except Exception:
            tile_bytes_list.append(None)
            intensities.append(0)

    # Fase 2: Construir màscara de clutter (ecos permanents en tots els frames)
    clutter_mask = _build_clutter_mask(tile_bytes_list)
    if clutter_mask is not None:
        clutter_count = int(clutter_mask.sum())
        if clutter_count > 0:
            logger.debug(f"  Clutter mask: {clutter_count} píxels permanents filtrats")

    # Fase 3: Escaneig espacial amb filtre de clutter
    spatial_scans = []
    for tile_bytes in tile_bytes_list:
        if tile_bytes is not None:
            scan = _scan_radar_spatial(
                tile_bytes,
                config.RAINVIEWER_PIXEL_X,
                config.RAINVIEWER_PIXEL_Y,
                config.RADAR_SCAN_RADIUS_KM,
                config.RADAR_PIXEL_SIZE_KM,
                wind_from_dir=wind_from_dir,
                clutter_mask=clutter_mask,
            )
            spatial_scans.append(scan)
        else:
            spatial_scans.append(_empty_spatial_result(config.RADAR_SCAN_RADIUS_KM))

    # ── Mètriques puntuals (compatibilitat amb l'existent) ──
    current_intensity = intensities[-1] if intensities else 0
    current_dbz = _radar_intensity_to_dbz(current_intensity)
    current_rain_rate = _dbz_to_rain_rate(current_dbz)
    frames_with_echo = sum(1 for i in intensities if i > 0)

    approaching = False
    if len(intensities) >= 3:
        recent_trend = intensities[-1] - intensities[0]
        approaching = recent_trend > 15

    latest_ts = past_frames[-1]["time"]
    radar_time = datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat()

    # ── Mètriques espacials (últim frame) ──
    latest_spatial = spatial_scans[-1] if spatial_scans else _empty_spatial_result(
        config.RADAR_SCAN_RADIUS_KM)

    # ── Detecció de frames congelats ──
    # RainViewer pot servir el MATEIX contingut amb timestamps nous quan el
    # composite font es queda penjat (vist el 2026-06-04: 6 frames idèntics
    # durant >1h en plena tempesta). Amb frames congelats el vector de
    # moviment és 0 per construcció i la imatge pot tenir hores; cal
    # marcar-ho perquè ningú es cregui un radar "fresc" que no ho és.
    valid_tiles = [t for t in tile_bytes_list if t is not None]
    frames_frozen = (
        len(valid_tiles) >= 3
        and len({hashlib.md5(t).hexdigest() for t in valid_tiles}) == 1
    )
    if frames_frozen:
        logger.warning(
            f"  RainViewer FROZEN: {len(valid_tiles)} frames amb contingut idèntic "
            f"(timestamps diferents) — el composite font està penjat, dades potencialment velles"
        )

    # ── Tracking de tempesta (moviment entre frames) ──
    xcorr_velocity = _estimate_motion_xcorr(
        tile_bytes_list, clutter_mask, config.RADAR_PIXEL_SIZE_KM
    )
    if frames_frozen:
        xcorr_velocity = None  # 0.0 km/h seria mentida; no hi ha informació de moviment
    storm_tracking = _estimate_storm_tracking(
        spatial_scans, config.RADAR_PIXEL_SIZE_KM,
        xcorr_velocity=xcorr_velocity,
    )

    # Combinar approaching: puntual O espacial
    spatial_approaching = storm_tracking["storm_approaching"]

    # ── Safeguard: detectar clutter post-scan ──
    # Patró: cobertura alta (>20%) + dBZ baixa (<15) + velocitat zero = clutter de terra
    # La pluja real amb 20%+ cobertura tindria dBZ molt més alt (>20 típicament)
    if (latest_spatial.get("echoes_found")
            and latest_spatial.get("coverage_20km", 0) > 0.20
            and latest_spatial.get("max_dbz_20km", 0) < 15
            and storm_tracking.get("storm_velocity_kmh", 0) == 0):
        logger.warning(
            f"  Clutter post-scan detectat: cobertura={latest_spatial['coverage_20km']:.1%} "
            f"però max_dbz={latest_spatial['max_dbz_20km']} i vel=0 — resetejant a buit"
        )
        latest_spatial = _empty_spatial_result(config.RADAR_SCAN_RADIUS_KM)
        spatial_approaching = False

    result = {
        # Puntuals (compatibilitat)
        "radar_intensity": current_intensity,
        "radar_dbz": round(current_dbz, 1),
        "radar_rain_rate": round(current_rain_rate, 2),
        "radar_has_echo": current_dbz >= config.RADAR_MIN_DBZ,
        "radar_frames_with_echo": frames_with_echo,
        "radar_approaching": approaching or spatial_approaching,
        "radar_max_intensity_1h": max(intensities) if intensities else 0,
        "radar_timestamp": radar_time,
        "radar_frames_frozen": frames_frozen,
        # Espacials (noves)
        "radar_nearest_echo_km": latest_spatial["nearest_echo_km"],
        "radar_nearest_echo_bearing": latest_spatial.get("nearest_echo_bearing"),
        "radar_nearest_echo_compass": latest_spatial.get("nearest_echo_compass"),
        "radar_max_dbz_20km": latest_spatial["max_dbz_20km"],
        "radar_coverage_20km": latest_spatial["coverage_20km"],
        "radar_upwind_nearest_echo_km": latest_spatial.get("upwind_nearest_echo_km",
                                                            config.RADAR_SCAN_RADIUS_KM),
        "radar_upwind_max_dbz": latest_spatial.get("upwind_max_dbz", 0.0),
        # Tracking
        "radar_storm_velocity_kmh": storm_tracking["storm_velocity_kmh"],
        "radar_storm_velocity_ns": storm_tracking["storm_velocity_ns"],
        "radar_storm_velocity_ew": storm_tracking["storm_velocity_ew"],
        "radar_storm_approaching": spatial_approaching,
        "radar_storm_eta_min": storm_tracking["storm_eta_min"],
        # Quadrant features (N/E/S/W)
        "radar_quadrant_max_dbz_N": latest_spatial.get("quadrant_max_dbz_N", 0.0),
        "radar_quadrant_max_dbz_E": latest_spatial.get("quadrant_max_dbz_E", 0.0),
        "radar_quadrant_max_dbz_S": latest_spatial.get("quadrant_max_dbz_S", 0.0),
        "radar_quadrant_max_dbz_W": latest_spatial.get("quadrant_max_dbz_W", 0.0),
        "radar_quadrant_coverage_N": latest_spatial.get("quadrant_coverage_N", 0.0),
        "radar_quadrant_coverage_E": latest_spatial.get("quadrant_coverage_E", 0.0),
        "radar_quadrant_coverage_S": latest_spatial.get("quadrant_coverage_S", 0.0),
        "radar_quadrant_coverage_W": latest_spatial.get("quadrant_coverage_W", 0.0),
    }

    # Log detallat
    if latest_spatial["echoes_found"]:
        eta_str = f", ETA={storm_tracking['storm_eta_min']}min" if storm_tracking["storm_eta_min"] else ""
        logger.info(
            f"  Radar espacial: eco a {latest_spatial['nearest_echo_km']}km "
            f"{latest_spatial.get('nearest_echo_compass', '?')}, "
            f"cobertura 20km={latest_spatial['coverage_20km']:.1%}, "
            f"vel={storm_tracking['storm_velocity_kmh']}km/h"
            f"{eta_str}"
        )
    else:
        logger.info(f"  Radar espacial: sense ecos dins de {config.RADAR_SCAN_RADIUS_KM}km")

    return result


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
        "radar_frames_frozen": False,
        # Espacials
        "radar_nearest_echo_km": config.RADAR_SCAN_RADIUS_KM,
        "radar_nearest_echo_bearing": None,
        "radar_nearest_echo_compass": None,
        "radar_max_dbz_20km": 0.0,
        "radar_coverage_20km": 0.0,
        "radar_upwind_nearest_echo_km": config.RADAR_SCAN_RADIUS_KM,
        "radar_upwind_max_dbz": 0.0,
        # Tracking
        "radar_storm_velocity_kmh": 0.0,
        "radar_storm_velocity_ns": 0.0,
        "radar_storm_velocity_ew": 0.0,
        "radar_storm_approaching": False,
        "radar_storm_eta_min": None,
        # Quadrant features
        "radar_quadrant_max_dbz_N": 0.0,
        "radar_quadrant_max_dbz_E": 0.0,
        "radar_quadrant_max_dbz_S": 0.0,
        "radar_quadrant_max_dbz_W": 0.0,
        "radar_quadrant_coverage_N": 0.0,
        "radar_quadrant_coverage_E": 0.0,
        "radar_quadrant_coverage_S": 0.0,
        "radar_quadrant_coverage_W": 0.0,
    }
