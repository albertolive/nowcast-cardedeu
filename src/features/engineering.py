"""
Feature Engineering per al model de Nowcasting.
Crea les features (columnes) que alimenten XGBoost.
Combina dades locals (meteocardedeu) + exteriors (Open-Meteo).
"""
import logging
import math
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def dew_point(temp_c: float, humidity_pct: float) -> float:
    """Calcula el punt de rosada (Magnus formula)."""
    if humidity_pct <= 0:
        return temp_c
    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + math.log(max(humidity_pct, 1) / 100.0)
    return (b * alpha) / (a - alpha)


def wind_components(speed: float, direction_deg: float) -> tuple[float, float]:
    """Descompon el vent en components U (est-oest) i V (nord-sud)."""
    rad = math.radians(direction_deg)
    u = -speed * math.sin(rad)  # component est-oest
    v = -speed * math.cos(rad)  # component nord-sud
    return u, v


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features temporals (hora del dia, mes, estació)."""
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    # Codificació cíclica per a hora i mes (el model entén que 23h i 0h són properes)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_solar_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hours since sunrise — convective initiation timing for Cardedeu (41.633°N)."""
    df = df.copy()
    if "datetime" not in df.columns:
        return df

    dt = pd.to_datetime(df["datetime"])
    doy = dt.dt.dayofyear
    hour_frac = dt.dt.hour + dt.dt.minute / 60.0

    # Solar declination
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 81) / 365)
    lat_rad = np.radians(config.LATITUDE)

    # Half day length in hours
    cos_ha = (-np.tan(lat_rad) * np.tan(decl)).clip(-1, 1)
    half_day = np.degrees(np.arccos(cos_ha)) / 15

    # Solar noon in Europe/Madrid local time
    # Longitude config.LONGITUDE°E → solar noon UTC offset
    # CET (UTC+1) ≈ 12:51 local, CEST (UTC+2) ≈ 13:51 local
    # Approximate DST: months 4-9 = CEST (+2), rest = CET (+1)
    month = dt.dt.month
    tz_offset = np.where((month >= 4) & (month <= 9), 2.0, 1.0)
    solar_noon_local = (12.0 - config.LONGITUDE / 15) + tz_offset

    sunrise = solar_noon_local - half_day
    df["hours_since_sunrise"] = (hour_frac - sunrise).clip(lower=0)

    return df


def _add_pressure_features(df: pd.DataFrame, pressure_col: str = "pressure_msl") -> pd.DataFrame:
    """Tendència de la pressió (clau per predir inestabilitat)."""
    df = df.copy()
    if pressure_col in df.columns:
        # Canvi de pressió en les últimes 1, 3 i 6 hores
        df["pressure_change_1h"] = df[pressure_col].diff(1)
        df["pressure_change_3h"] = df[pressure_col].diff(3)
        df["pressure_change_6h"] = df[pressure_col].diff(6)
        # Acceleració de la pressió (la derivada de la tendència)
        df["pressure_accel_3h"] = df["pressure_change_1h"].diff(3)
    return df


def _add_humidity_features(df: pd.DataFrame, temp_col: str = "temperature_2m", hum_col: str = "relative_humidity_2m") -> pd.DataFrame:
    """Features d'humitat i punt de rosada."""
    df = df.copy()
    if temp_col in df.columns and hum_col in df.columns:
        df["dew_point"] = df.apply(
            lambda r: dew_point(r[temp_col], r[hum_col]) if pd.notna(r[temp_col]) and pd.notna(r[hum_col]) else np.nan,
            axis=1,
        )
        # Dew point depression (diferència temp - punt rosada): quant més baix, més a prop de saturació
        df["dew_point_depression"] = df[temp_col] - df["dew_point"]
        # Tendència d'humitat
        df["humidity_change_1h"] = df[hum_col].diff(1)
        df["humidity_change_3h"] = df[hum_col].diff(3)
    # VPD (Vapour Pressure Deficit) — mesura directa de saturació
    # VPD=0 → aire saturat (boira/pluja imminent). Caiguda ràpida = règim canviant.
    if "vapour_pressure_deficit" in df.columns:
        df["vpd_change_3h"] = df["vapour_pressure_deficit"].diff(3)
    return df


def _add_wind_features(df: pd.DataFrame, speed_col: str = "wind_speed_10m", dir_col: str = "wind_direction_10m") -> pd.DataFrame:
    """Components U/V del vent + canvis."""
    df = df.copy()
    if speed_col in df.columns and dir_col in df.columns:
        uv = df.apply(
            lambda r: wind_components(r[speed_col], r[dir_col]) if pd.notna(r[speed_col]) and pd.notna(r[dir_col]) else (np.nan, np.nan),
            axis=1,
        )
        df["wind_u"] = uv.apply(lambda x: x[0])
        df["wind_v"] = uv.apply(lambda x: x[1])
        # Canvi de velocitat del vent
        df["wind_speed_change_1h"] = df[speed_col].diff(1)
        df["wind_speed_change_3h"] = df[speed_col].diff(3)
        # Vent del mar (component est → SSE/SE/ESE predominant a Cardedeu quan entra marinada)
        # Un wind_u negatiu significa vent del est (marinada)
        df["is_sea_breeze"] = (df["wind_u"] < -2).astype(int)
    return df


def _angular_diff(directions: pd.Series, periods: int) -> pd.Series:
    """
    Calcula el canvi angular del vent sobre `periods` timesteps.
    Gestiona correctament el pas 360°→0°.
    Positiu = veering (gir horari, pas de front fred).
    Negatiu = backing (gir antihorari, aproximació de front càlid/baixa).
    """
    prev = directions.shift(periods)
    diff = directions - prev
    # Normalitzar a [-180, 180]
    return ((diff + 180) % 360) - 180


def _dir_in_range(direction: pd.Series, lo: float, hi: float) -> pd.Series:
    """Comprova si la direcció del vent cau dins un rang (gestiona wrap 360°→0°)."""
    if lo <= hi:
        return ((direction >= lo) & (direction <= hi)).astype(int)
    else:  # wrap-around (e.g. Tramuntana 330°-30°)
        return ((direction >= lo) | (direction <= hi)).astype(int)


def _add_wind_regime_features(df: pd.DataFrame,
                              speed_col: str = "wind_speed_10m",
                              dir_col: str = "wind_direction_10m",
                              hum_col: str = "relative_humidity_2m") -> pd.DataFrame:
    """
    Classifica el vent en règims meteorològics catalans (Rosa dels Vents).
    Ref: alexmeteo.com

    IMPORTANT: La classificació sinòptica (Llevantada, Garbí…) es fa amb el
    vent a 850hPa (~1500m), que reflecteix el flux sinòptic real.
    El vent a 10m (superfície) és distorsionat per orografia (Montseny),
    canalització de valls, i brises tèrmiques.

    Si hi ha dades de 850hPa (columna wind_850_dir), s'utilitzen per
    a la classificació. Si no, fallback a 10m (entrenament històric).

    Cobertura completa de la Rosa dels Vents:
    - Tramuntana (N/NE 340°-60°): vent polar fred, cel blau, supressor de pluja.
      Inclou Gregal (NE 30°-60°) que a 850hPa és variant de Tramuntana.
    - Llevantada (E/SE 60°-150°): humitat mediterrània contra la Serralada.
    - Migjorn (S 150°-190°): vent del sud, aire africà calent i humit.
    - Garbí (SW 190°-250°): "Anuncia borrasques amb fortes precipitacions."
    - Ponent/Mestral (W/NW 250°-340°): aire continental sec.
    """
    df = df.copy()

    # Use 850hPa ONLY for synoptic regime classification.
    # Surface wind (10m) at Cardedeu has only 26% agreement with 850hPa due to
    # Montseny katabatic drainage, valley channeling, and thermal breezes.
    # Mixing the two creates semantically inconsistent features that confuse the model.
    # When 850hPa is NaN (pre-2021), the regime flags are 0 — clean signal that
    # XGBoost learns to handle. The raw surface wind (wind_u, wind_v) is still
    # available as a separate feature for the model to use directly.
    if "wind_850_dir" in df.columns and df["wind_850_dir"].notna().any():
        synoptic_dir = pd.to_numeric(df["wind_850_dir"], errors="coerce")
        synoptic_speed = pd.to_numeric(df.get("wind_850_speed", pd.Series(dtype=float)), errors="coerce").fillna(0)
    elif dir_col in df.columns:
        # Fallback: real-time prediction without pressure data
        synoptic_dir = pd.to_numeric(df[dir_col], errors="coerce")
        synoptic_speed = pd.to_numeric(df.get(speed_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    else:
        return df

    # Règims eòlics catalans a partir del vent sinòptic
    # Tramuntana estès a 60° per cobrir Gregal (NE) — a 850hPa és variant polar
    df["is_tramuntana"] = _dir_in_range(synoptic_dir, 340, 60)
    df["is_llevantada"] = _dir_in_range(synoptic_dir, 60, 150)
    df["is_migjorn"] = _dir_in_range(synoptic_dir, 150, 190)
    df["is_garbi"] = _dir_in_range(synoptic_dir, 190, 250)
    df["is_ponent"] = _dir_in_range(synoptic_dir, 250, 340)

    # Interaccions règim × velocitat/humitat per tots els règims
    # (els flags binaris sols tenen importància zero — les interaccions porten el senyal)
    humidity = pd.to_numeric(df.get(hum_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    hum_frac = humidity / 100.0

    # Llevantada (18.5% rain rate — dominant rain pattern)
    df["llevantada_strength"] = df["is_llevantada"] * synoptic_speed
    df["llevantada_moisture"] = df["is_llevantada"] * hum_frac

    # Garbí ("Anuncia borrasques amb fortes precipitacions" — ref: alexmeteo)
    df["garbi_strength"] = df["is_garbi"] * synoptic_speed

    # Tramuntana (4.8% rain rate — mostly dry Montseny air)
    df["tramuntana_strength"] = df["is_tramuntana"] * synoptic_speed
    df["tramuntana_moisture"] = df["is_tramuntana"] * hum_frac

    # Canvi de direcció sinòptica en 3h (backing/veering)
    df["wind_dir_change_3h"] = _angular_diff(synoptic_dir, 3)

    return df


def _add_pressure_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de nivells de pressió (925/850/700/500/300 hPa).

    925hPa (~750m): capa límit — low-level jet, inversions, flux d'humitat baix
    850hPa (~1500m): flux sinòptic — règims, transport humitat
    700hPa (~3000m): intrusió d'aire sec — capping, inhib tempestes
    500hPa (~5500m): aire fred — gradient tèrmic, VT/TT/LI
    300hPa (~9000m): jet stream — cisalla profunda, trigger dinàmic

    Ref: alexmeteo.com — Skew-T analysis, "Ingredients per formar Tempestes"
    """
    import math

    df = df.copy()

    # Indicador de disponibilitat de dades de nivells de pressió
    # Pre-2021: NaN → 0; Post-2021: disponible → 1
    # Permet al model distingir quan els règims venen de 850hPa (fiable) vs 10m (soroll)
    if "wind_850_dir" in df.columns:
        df["has_pressure_levels"] = df["wind_850_dir"].notna().astype(int)
    else:
        df["has_pressure_levels"] = 0

    for col in ["wind_925_speed", "wind_925_dir", "temp_925", "rh_925",
                "wind_850_speed", "wind_850_dir", "temp_850", "temp_500",
                "rh_850", "rh_700", "temp_700",
                "wind_300_speed", "wind_300_dir", "gph_300",
                "vt_index", "tt_index", "li_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Compute VT/TT/LI from raw pressure columns if not present ──
    if "vt_index" not in df.columns and "temp_850" in df.columns and "temp_500" in df.columns:
        df["vt_index"] = df["temp_850"] - df["temp_500"]

    if "tt_index" not in df.columns and "vt_index" in df.columns and "rh_850" in df.columns:
        a, b = 17.27, 237.7
        alpha = (a * df["temp_850"]) / (b + df["temp_850"]) + np.log(df["rh_850"].clip(lower=1) / 100.0)
        td_850 = (b * alpha) / (a - alpha)
        df["tt_index"] = df["vt_index"] + (td_850 - df["temp_500"])

    if "li_index" not in df.columns and all(c in df.columns for c in ["temp_850", "rh_850", "temp_500"]):
        a, b = 17.27, 237.7
        alpha_li = (a * df["temp_850"]) / (b + df["temp_850"]) + np.log(df["rh_850"].clip(lower=1) / 100.0)
        td_850_li = (b * alpha_li) / (a - alpha_li)
        dew_dep = df["temp_850"] - td_850_li
        lcl_height_m = 125 * dew_dep
        t_at_lcl = df["temp_850"] - 9.8 * (lcl_height_m / 1000.0)
        remaining_m = (3500 - lcl_height_m).clip(lower=0)
        t_parcel_500 = t_at_lcl - 6.0 * (remaining_m / 1000.0)
        df["li_index"] = df["temp_500"] - t_parcel_500

    # ── Wind shear (cisalla): diferència vent superfície vs 850hPa ──
    # Clau per organització i persistència de tempestes.
    # Ref: alexmeteo "la cisalla de vent és un factor clau en el desenvolupament
    #  i la intensitat de les tempestes"
    if "wind_850_speed" in df.columns and "wind_speed_10m" in df.columns:
        w850 = pd.to_numeric(df["wind_850_speed"], errors="coerce")
        w10 = pd.to_numeric(df["wind_speed_10m"], errors="coerce")
        df["wind_shear_speed"] = (w850 - w10).abs()

    if "wind_850_dir" in df.columns and "wind_direction_10m" in df.columns:
        d850 = pd.to_numeric(df["wind_850_dir"], errors="coerce")
        d10 = pd.to_numeric(df["wind_direction_10m"], errors="coerce")
        # Directional shear: canvi de direcció entre superfície i 850hPa
        diff = d850 - d10
        df["wind_shear_dir"] = ((diff + 180) % 360) - 180

    # ── 925hPa: boundary layer features ──
    # Inversió tèrmica: quan T925 > T_sfc, aire fred atrapat a baix → boira/estrat
    if "temp_925" in df.columns and "temperature_2m" in df.columns:
        t925 = pd.to_numeric(df["temp_925"], errors="coerce")
        t_sfc = pd.to_numeric(df["temperature_2m"], errors="coerce")
        df["inversion_925"] = t925 - t_sfc  # positiu = inversió

    # Low-level moisture flux a 925hPa — el gruix del transport d'humitat
    # en events de Llevantada és sovint a 925, no a 850 (per sobre del Montseny)
    if "wind_925_speed" in df.columns and "rh_925" in df.columns and "temp_925" in df.columns:
        w925 = pd.to_numeric(df["wind_925_speed"], errors="coerce")
        rh925 = pd.to_numeric(df["rh_925"], errors="coerce").clip(lower=1)
        t925 = pd.to_numeric(df["temp_925"], errors="coerce")
        es_925 = 6.112 * np.exp(17.67 * t925 / (t925 + 243.5))
        q_925 = (rh925 / 100.0) * es_925
        df["moisture_flux_925"] = w925 * q_925

    # ── 300hPa: jet stream features ──
    # Deep-layer shear (850-300hPa): organitza supercèl·lules i MCS
    # Més important que low-level shear per persistència de tempestes
    if "wind_300_speed" in df.columns and "wind_850_speed" in df.columns:
        w300 = pd.to_numeric(df["wind_300_speed"], errors="coerce")
        w850_dl = pd.to_numeric(df["wind_850_speed"], errors="coerce")
        df["deep_layer_shear"] = (w300 - w850_dl).abs()

    # Jet stream intensity — upper divergence proxy
    # Strong jet overhead = dynamic lifting = trigger for organized rain
    if "wind_300_speed" in df.columns:
        df["jet_speed_300"] = pd.to_numeric(df["wind_300_speed"], errors="coerce")

    # ── Cold air at 500hPa: llindars de tempesta ──
    # Ref: alexmeteo — -17°C a l'estiu = "petita bomba", -22/-24°C primavera
    if "temp_500" in df.columns:
        t500 = pd.to_numeric(df["temp_500"], errors="coerce")
        df["cold_500_moderate"] = (t500 < -17).astype(int)
        df["cold_500_strong"] = (t500 < -24).astype(int)

    # ── LI thresholds ──
    if "li_index" in df.columns:
        li = pd.to_numeric(df["li_index"], errors="coerce")
        df["li_unstable"] = (li < -2).astype(int)
        df["li_very_unstable"] = (li < -6).astype(int)

    # ── Moisture flux proxy: wind_speed × specific_humidity ──
    # This is the physical driver of precipitation: how much water is being
    # transported towards the area. Better than wind or humidity alone.
    if "wind_850_speed" in df.columns and "rh_850" in df.columns and "temp_850" in df.columns:
        w850 = pd.to_numeric(df["wind_850_speed"], errors="coerce")
        rh850 = pd.to_numeric(df["rh_850"], errors="coerce").clip(lower=1)
        t850 = pd.to_numeric(df["temp_850"], errors="coerce")
        # Approximate specific humidity from T and RH at 850hPa (Clausius-Clapeyron)
        es = 6.112 * np.exp(17.67 * t850 / (t850 + 243.5))  # sat. vapor pressure (hPa)
        q_approx = (rh850 / 100.0) * es  # mixing ratio proxy
        df["moisture_flux_850"] = w850 * q_approx

    # ── Θe lapse rate proxy (convective instability) ──
    # The difference between surface equivalent potential temperature and 850hPa.
    # When surface Θe >> 850hPa Θe, the atmosphere is convectively unstable.
    if all(c in df.columns for c in ["temperature_2m", "relative_humidity_2m", "temp_850", "rh_850"]):
        t_sfc = pd.to_numeric(df["temperature_2m"], errors="coerce")
        rh_sfc = pd.to_numeric(df["relative_humidity_2m"], errors="coerce").clip(lower=1)
        es_sfc = 6.112 * np.exp(17.67 * t_sfc / (t_sfc + 243.5))
        # Simplified Θe proxy: T + L/cp * q ≈ T + 2.5 * (RH/100) * es
        theta_e_sfc = t_sfc + 2.5 * (rh_sfc / 100.0) * es_sfc
        t850_ = pd.to_numeric(df["temp_850"], errors="coerce")
        rh850_ = pd.to_numeric(df["rh_850"], errors="coerce").clip(lower=1)
        es_850 = 6.112 * np.exp(17.67 * t850_ / (t850_ + 243.5))
        theta_e_850 = t850_ + 2.5 * (rh850_ / 100.0) * es_850
        df["theta_e_deficit"] = theta_e_sfc - theta_e_850

    # ── Mid-level drying trend at 700hPa ──
    # Controls virga (rain evaporating before surface) and entrainment.
    # Rapid drying at 700hPa kills convection by entraining dry air.
    if "rh_700" in df.columns:
        rh700 = pd.to_numeric(df["rh_700"], errors="coerce")
        df["rh_700_change_3h"] = rh700.diff(3)
        df["rh_700_change_6h"] = rh700.diff(6)

    # ── 850hPa temperature trend — warm/cold advection proxy ──
    # Positive = warm advection (isentropic lift → stratiform rain)
    # Negative = cold advection (post-frontal instability)
    if "temp_850" in df.columns:
        t850_trend = pd.to_numeric(df["temp_850"], errors="coerce")
        df["temp_850_change_3h"] = t850_trend.diff(3)

    # ── K-index: moist-layer depth ──
    # K = (T850 - T500) + Td850 - (T700 - Td700)
    # Captures "how deep is the moist layer?" — TT/VT miss this.
    # K > 25 = scattered storms, K > 35 = widespread severe
    if all(c in df.columns for c in ["temp_850", "temp_500", "rh_850", "temp_700", "rh_700"]):
        t850_k = pd.to_numeric(df["temp_850"], errors="coerce")
        t500_k = pd.to_numeric(df["temp_500"], errors="coerce")
        t700_k = pd.to_numeric(df["temp_700"], errors="coerce")
        rh850_k = pd.to_numeric(df["rh_850"], errors="coerce").clip(lower=1)
        rh700_k = pd.to_numeric(df["rh_700"], errors="coerce").clip(lower=1)
        a, b = 17.27, 237.7
        alpha_850k = (a * t850_k) / (b + t850_k) + np.log(rh850_k / 100.0)
        td_850_k = (b * alpha_850k) / (a - alpha_850k)
        alpha_700k = (a * t700_k) / (b + t700_k) + np.log(rh700_k / 100.0)
        td_700_k = (b * alpha_700k) / (a - alpha_700k)
        df["k_index"] = (t850_k - t500_k) + td_850_k - (t700_k - td_700_k)

    # ── Bulk Richardson Number ──
    # BRN = CAPE / (0.5 × shear²)
    # Determines storm mode: BRN < 10 → supercell, 10-45 → possible, > 45 → multicell
    # Supercells vs multicells produce very different precip signatures over Cardedeu.
    if "cape" in df.columns and "deep_layer_shear" in df.columns:
        cape_brn = pd.to_numeric(df["cape"], errors="coerce")
        shear_brn = pd.to_numeric(df["deep_layer_shear"], errors="coerce")
        shear_sq = (0.5 * shear_brn ** 2).clip(lower=1)  # avoid /0
        df["bulk_richardson"] = cape_brn / shear_sq

    return df


def _add_physics_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features composites basades en física atmosfèrica.
    Combinen múltiples variables existents per capturar processos
    que els models NWP aproximen malament a escala local.

    Totes les features es construeixen a partir de dades ja disponibles
    (superfície + pressure levels), sense necessitat de noves APIs.
    """
    df = df.copy()

    # ── 1. Orographic Forcing Index ──
    # Quan flux humit mediterrani xoca amb Serralada Prelitoral / Montseny,
    # l'ascens forçat desencadena precipitació orographic enhancement.
    # Component del vent perpendicular a la cresta × humitat.
    # Orientació de la Serralada: ~NE-SW (azimut ~45°), normal = ~135° (SE)
    RIDGE_NORMAL_RAD = np.radians(135)  # perpendicular a la Serralada
    if "wind_850_dir" in df.columns and "wind_850_speed" in df.columns and "rh_850" in df.columns:
        w850_dir = pd.to_numeric(df["wind_850_dir"], errors="coerce")
        w850_spd = pd.to_numeric(df["wind_850_speed"], errors="coerce")
        rh850 = pd.to_numeric(df["rh_850"], errors="coerce")
        # Component del vent perpendicular a la cresta
        wind_rad = np.radians(w850_dir)
        perp_component = w850_spd * np.cos(wind_rad - RIDGE_NORMAL_RAD)
        # Positiu = flux cap a la muntanya. Multiplicat per humitat normalitzada.
        df["orographic_forcing"] = perp_component.clip(lower=0) * (rh850 / 100.0)

    # ── 2. Frontal Passage Indicator ──
    # Detecció de pas de fronts a partir de canvis ràpids de pressió + vent + temperatura.
    # Front fred: caiguda de pressió → pujada + veering (gir horari) + caiguda de temp.
    # Front càlid: caiguda sostinguda de pressió + backing + pujada de temp + alta humitat.
    if all(c in df.columns for c in ["pressure_change_3h", "wind_dir_change_3h", "temperature_2m"]):
        p_change = pd.to_numeric(df["pressure_change_3h"], errors="coerce")
        wind_shift = pd.to_numeric(df["wind_dir_change_3h"], errors="coerce").abs()
        temp_change = pd.to_numeric(df["temperature_2m"], errors="coerce").diff(3)
        # Frontal signal: strong when pressure drops + wind shifts + temp changes
        # Normalized components to ~[0,1] range, then combined
        p_signal = (-p_change / 5.0).clip(0, 1)  # 5hPa/3h drop = max signal
        w_signal = (wind_shift / 60.0).clip(0, 1)  # 60° shift = max signal
        t_signal = (temp_change.abs() / 5.0).clip(0, 1)  # 5°C change = max signal
        df["frontal_passage"] = p_signal * w_signal + t_signal * p_signal

    # ── 3. Convective Composite Index ──
    # Combina inestabilitat (TT/LI) + humitat + trigger mecànic (shear).
    # Tots els ingredients han de ser presents → producte.
    # Ref: alexmeteo "Ingredients per formar Tempestes": inestabilitat + humitat + trigger
    composites_available = []
    if "tt_index" in df.columns:
        tt = pd.to_numeric(df["tt_index"], errors="coerce")
        # TT > 44 = convecció moderada, > 50 = severa. Normalize to ~[0,1]
        tt_signal = ((tt - 40) / 15.0).clip(0, 1)
        composites_available.append(tt_signal)
    if "moisture_flux_850" in df.columns:
        mf = pd.to_numeric(df["moisture_flux_850"], errors="coerce")
        mf_signal = (mf / 200.0).clip(0, 1)  # High moisture flux
        composites_available.append(mf_signal)
    if "deep_layer_shear" in df.columns:
        dls = pd.to_numeric(df["deep_layer_shear"], errors="coerce")
        shear_signal = (dls / 30.0).clip(0, 1)  # 30 m/s shear = max
        composites_available.append(shear_signal)
    if len(composites_available) >= 2:
        composite = composites_available[0]
        for c in composites_available[1:]:
            composite = composite * c
        df["convective_composite"] = composite

    # ── 4. Thermal Buildup Index ──
    # Calentament diürn → termals → convecció tarda.
    # Alt a l'estiu quan el terra calent genera termals.
    # Combinació: amplitud tèrmica recent × hora del dia (pic 12-18h) × radiació.
    if "temperature_2m" in df.columns:
        temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
        # Amplitud tèrmica en les últimes 12h (rolling max - min)
        temp_range_12h = temp.rolling(12, min_periods=3).max() - temp.rolling(12, min_periods=3).min()
        # Fracció del cicle diürn (pic a 15h local)
        if "hour" in df.columns:
            hour = pd.to_numeric(df["hour"], errors="coerce")
        elif "datetime" in df.columns:
            hour = pd.to_datetime(df["datetime"]).dt.hour
        else:
            hour = pd.Series(12, index=df.index)
        # Gaussian-like peak at 15h for solar heating
        diurnal_factor = np.exp(-0.5 * ((hour - 15) / 3.0) ** 2)
        df["thermal_buildup"] = temp_range_12h * diurnal_factor / 15.0  # Normalize by 15°C range

    # ── 5. Low-Level Convergence Proxy ──
    # Quan masses d'aire convergeixen → ascens → precipitació.
    # Aproximat per canvis ràpids de vent + humitat + caiguda de pressió simultanis.
    if all(c in df.columns for c in ["wind_speed_change_3h", "humidity_change_3h", "pressure_change_3h"]):
        wind_decel = pd.to_numeric(df["wind_speed_change_3h"], errors="coerce")
        hum_rise = pd.to_numeric(df["humidity_change_3h"], errors="coerce")
        p_drop = pd.to_numeric(df["pressure_change_3h"], errors="coerce")
        # Convergence: wind slowing + humidity rising + pressure dropping
        wind_conv = (-wind_decel / 10.0).clip(0, 1)  # Decelerating wind
        hum_conv = (hum_rise / 20.0).clip(0, 1)  # Rising humidity
        p_conv = (-p_drop / 3.0).clip(0, 1)  # Falling pressure
        df["low_level_convergence"] = wind_conv + hum_conv + p_conv

    # ── 6. Dry Air Intrusion (capping) ──
    # Aire sec a 700hPa sobre aire humit a 850hPa → capping inversion.
    # Pot suprimir convecció feble però energitzar tempestes fortes (loaded gun).
    if "rh_700" in df.columns and "rh_850" in df.columns:
        rh700 = pd.to_numeric(df["rh_700"], errors="coerce")
        rh850 = pd.to_numeric(df["rh_850"], errors="coerce")
        # Gran diferència = cape atrapada → tempestes potents si es trenca la capa
        df["dry_intrusion_700"] = (rh850 - rh700).clip(lower=0) / 100.0

    # ── 7. Moisture Flux Convergence ──
    # Canvi en el flux d'humitat: augmentant = front que s'apropa, disminuint = front que passa
    if "moisture_flux_850" in df.columns:
        mf = pd.to_numeric(df["moisture_flux_850"], errors="coerce")
        df["moisture_flux_change_3h"] = mf.diff(3)

    return df


def _add_soil_moisture_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features d'humitat del sòl — present en dades d'arxiu (ERA5) des de 2015."""
    df = df.copy()
    for col in ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Canvi d'humitat del sòl (indica infiltració recent per pluja)
    if "soil_moisture_0_to_7cm" in df.columns:
        df["soil_moisture_change_24h"] = df["soil_moisture_0_to_7cm"].diff(24)
    return df


def _add_atmospheric_column_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de columna atmosfèrica — TCWV, BLH, radiació terrestre, sòl profund."""
    df = df.copy()

    # TCWV (Total Column Integrated Water Vapour) — precipitable water (kg/m²)
    # The single strongest ERA5 predictor for precipitation
    if "total_column_integrated_water_vapour" in df.columns:
        tcwv = pd.to_numeric(df["total_column_integrated_water_vapour"], errors="coerce")
        df["tcwv"] = tcwv
        # TCWV change 3h — moisture loading/advection (front approaching)
        df["tcwv_change_3h"] = tcwv.diff(3)
        # TCWV change 6h — synoptic-scale moisture trend
        df["tcwv_change_6h"] = tcwv.diff(6)

        # TCWV monthly anomaly — 30mm TCWV in January is exceptional, normal in August.
        # Ratio to ERA5 climatological mean for Cardedeu (41.633°N, 2015-2025).
        _TCWV_CLIM = {
            1: 10.3, 2: 10.9, 3: 12.8, 4: 15.0, 5: 18.9, 6: 24.9,
            7: 28.4, 8: 29.0, 9: 25.1, 10: 20.8, 11: 14.6, 12: 12.1,
        }
        if "month" in df.columns:
            clim = df["month"].map(_TCWV_CLIM).clip(lower=1)
            df["tcwv_monthly_anomaly"] = tcwv / clim

    # Boundary Layer Height (m) — convective mixing depth
    # High BLH = deep convective mixing, low BLH = stable/suppressed
    if "boundary_layer_height" in df.columns:
        blh = pd.to_numeric(df["boundary_layer_height"], errors="coerce")
        df["boundary_layer_height"] = blh
        # BLH change 3h — deepening = convective development
        df["blh_change_3h"] = blh.diff(3)

    # TCWV/BLH ratio — moisture per unit mixing depth = convective efficiency
    # High TCWV + shallow BLH = trapped moisture → rain likely
    if "tcwv" in df.columns and "boundary_layer_height" in df.columns:
        blh_safe = df["boundary_layer_height"].clip(lower=50)  # avoid /0
        df["tcwv_blh_ratio"] = df["tcwv"] / blh_safe * 100  # scale to ~0-10

    # Terrestrial radiation (longwave IR, W/m²) — nighttime cloud detection
    # High = clear sky radiative cooling, low = clouds trapping heat
    if "terrestrial_radiation" in df.columns:
        df["terrestrial_radiation"] = pd.to_numeric(df["terrestrial_radiation"], errors="coerce")

    # Deep soil moisture (28-100cm) — antecedent saturation
    if "soil_moisture_28_to_100cm" in df.columns:
        df["soil_moisture_28_to_100cm"] = pd.to_numeric(df["soil_moisture_28_to_100cm"], errors="coerce")
        # Saturation ratio: shallow vs deep — high = recent rain fully infiltrated
        if "soil_moisture_0_to_7cm" in df.columns:
            shallow = pd.to_numeric(df["soil_moisture_0_to_7cm"], errors="coerce")
            deep = df["soil_moisture_28_to_100cm"].clip(lower=0.01)
            df["soil_saturation_ratio"] = shallow / deep

    return df


def _add_cloud_layer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de capes de núvols — diferenciar núvols baixos (pluja) d'alts (cirrus)."""
    df = df.copy()
    for col in ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Proporció de núvols baixos vs total — alt = pluja probable
    if "cloud_cover_low" in df.columns and "cloud_cover" in df.columns:
        total = df["cloud_cover"].clip(lower=1)  # evitar divisió per zero
        df["cloud_low_fraction"] = df["cloud_cover_low"] / total
    return df


def _add_wet_bulb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temperatura de bulb humit — indica saturació i tipus de precipitació."""
    df = df.copy()
    if "wet_bulb_temperature_2m" in df.columns:
        df["wet_bulb_temperature_2m"] = pd.to_numeric(df["wet_bulb_temperature_2m"], errors="coerce")
        # Depressió de bulb humit: gran = aire sec, petit = gairebé saturat
        if "temperature_2m" in df.columns:
            df["wet_bulb_depression"] = df["temperature_2m"] - df["wet_bulb_temperature_2m"]
    return df


def _add_radiation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de radiació — ratio difusa/total indica gruix de núvols."""
    df = df.copy()
    for col in ["direct_radiation", "diffuse_radiation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ratio difusa = difusa / (directa + difusa). Proper a 1 = cel cobert/pluja
    if "direct_radiation" in df.columns and "diffuse_radiation" in df.columns:
        total = (df["direct_radiation"] + df["diffuse_radiation"]).clip(lower=1)
        df["diffuse_fraction"] = df["diffuse_radiation"] / total
    return df


def _add_visibility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Visibilitat — baixa = pluja/boira activa. Historical Forecast API (2021-04+)."""
    df = df.copy()
    if "visibility" in df.columns:
        df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce")
    return df


def _add_freezing_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nivell de congelació (alçada isoterma 0°C) — clau per tipus/intensitat de precipitació."""
    df = df.copy()
    if "freezing_level_height" in df.columns:
        df["freezing_level_height"] = pd.to_numeric(df["freezing_level_height"], errors="coerce")
    return df


def _add_wind_gust_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ràfega de vent — fronts de ratxa convectiusindiquen tempestes."""
    df = df.copy()
    if "wind_gusts_10m" in df.columns:
        df["wind_gusts_10m"] = pd.to_numeric(df["wind_gusts_10m"], errors="coerce")
        # Gust factor: ràfega / vent mitjà. Alt = turbulència convectiva
        if "wind_speed_10m" in df.columns:
            mean_wind = df["wind_speed_10m"].clip(lower=0.5)
            df["gust_factor"] = df["wind_gusts_10m"] / mean_wind
    return df


def _add_rain_context(df: pd.DataFrame, precip_col: str = "precipitation") -> pd.DataFrame:
    """Context de pluja recent (últimes hores) i diària."""
    df = df.copy()
    if precip_col in df.columns:
        # Pluja acumulada en les últimes 3 i 6 hores
        df["rain_accum_3h"] = df[precip_col].rolling(3, min_periods=1).sum()
        df["rain_accum_6h"] = df[precip_col].rolling(6, min_periods=1).sum()
        # Ha plogut recentment?
        df["rained_last_3h"] = (df["rain_accum_3h"] > 0.2).astype(int)
        # Pluja acumulada en les últimes 24 hores — persistència sinòptica
        df["rain_accum_24h"] = df[precip_col].rolling(24, min_periods=1).sum()

    # Pressió mínima en les últimes 24h — detecta borrasques actives
    if "pressure_msl" in df.columns:
        pressure = pd.to_numeric(df["pressure_msl"], errors="coerce")
        df["pressure_min_24h"] = pressure.rolling(24, min_periods=1).min()

    return df


def _add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derivades de les previsions dels models.
    Aquestes columnes venen d'Open-Meteo.
    """
    df = df.copy()

    # CAPE (Convective Available Potential Energy) - quan és alt, risc de tempesta
    if "cape" in df.columns:
        df["cape_high"] = (df["cape"] > 500).astype(int)
        df["cape_very_high"] = (df["cape"] > 1000).astype(int)

    # Cloud cover
    if "cloud_cover" in df.columns:
        df["cloud_change_1h"] = df["cloud_cover"].diff(1)
        df["cloud_change_3h"] = df["cloud_cover"].diff(3)
        df["is_overcast"] = (df["cloud_cover"] > 80).astype(int)

    # Weather code simplificat (WMO codes)
    if "weather_code" in df.columns:
        wc = pd.to_numeric(df["weather_code"], errors="coerce")
        # Codes >= 50 indiquen precipitació
        df["model_predicts_precip"] = (wc >= 50).astype(int)
        # Codes >= 80 indiquen xàfecs
        df["model_predicts_showers"] = (wc >= 80).astype(int)
        # Weather code decomposition: thunderstorm vs stratiform vs drizzle
        # have very different predictability and physical drivers
        df["wc_is_thunderstorm"] = ((wc >= 95) | ((wc >= 17) & (wc <= 19))).astype(int)
        df["wc_is_rain"] = (((wc >= 60) & (wc <= 69)) | ((wc >= 80) & (wc <= 82))).astype(int)
        df["wc_is_drizzle"] = ((wc >= 50) & (wc <= 59)).astype(int)

        # NWP precipitation severity: continuous mapping of WMO code intensity.
        # Drizzle (51-55) has 48.8% FP rate vs rain (61-65) at 13.3%.
        # This continuous feature lets XGBoost learn the reliability gradient
        # instead of relying on the binary model_predicts_precip (54% gain).
        # 0=no precip, 1=drizzle, 2=rain, 3=showers, 4=snow, 5=thunderstorm
        import numpy as _np
        df["nwp_precip_severity"] = _np.select(
            [wc < 50, (wc >= 50) & (wc <= 59), (wc >= 60) & (wc <= 69),
             (wc >= 80) & (wc <= 84), (wc >= 70) & (wc <= 79), wc >= 95],
            [0, 1, 2, 3, 4, 5], default=0)

    # NWP error detection: when model says rain but surface conditions are dry,
    # this is likely a false positive (43% of NWP rain predictions are FP).
    # When model says no rain but humidity is very high, likely a false negative.
    if "model_predicts_precip" in df.columns and "relative_humidity_2m" in df.columns:
        rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
        mpp = df["model_predicts_precip"]
        df["nwp_dry_conflict"] = (mpp.eq(1) & rh.lt(65)).astype(int)
        df["nwp_wet_conflict"] = (mpp.eq(0) & rh.gt(90)).astype(int)

    # CAPE change rate: rapid CAPE build-up = convective environment developing
    # faster than NWP resolution can capture
    if "cape" in df.columns:
        cape = pd.to_numeric(df["cape"], errors="coerce")
        # Raw CAPE continu (44.5% cobertura) — més informatiu que cape_high binari
        df["cape"] = cape
        df["cape_change_3h"] = cape.diff(3)

        # CAPE × hora del dia: convecció solar requereix CAPE + escalfament diürn
        # CAPE matinal (pre-10h) rarament produeix tempestes; CAPE vespertí (12-18h) sí
        hour = pd.to_datetime(df.get("datetime", pd.NaT)).dt.hour
        diurnal_weight = ((hour >= 10) & (hour <= 19)).astype(float)
        df["cape_diurnal_weighted"] = cape * diurnal_weight

    # ── FP-killer features ──
    # Deep analysis shows 9,478 FP vs 8,940 TP. FP signature:
    # NWP says "maybe rain" but humidity is dropping and air is drier.
    # These interactions directly encode the error patterns.

    # 1. NWP rain amount (continuous) — much more informative than binary
    #    NWP predicting 5mm = frontal event (likely real), 0.1mm = marginal (likely FP)
    if "rain" in df.columns:
        df["nwp_rain_amount"] = pd.to_numeric(df["rain"], errors="coerce")

    # 2. NWP rain + drying humidity = classic FP pattern
    #    High value = NWP says rain but humidity is dropping → FP likely
    mpp_col = "model_predicts_precip" if "model_predicts_precip" in df.columns else None
    if mpp_col and "relative_humidity_2m" in df.columns:
        rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
        rh_change = rh.diff(3)
        df["nwp_rain_drying"] = df[mpp_col] * (-rh_change).clip(lower=0)

    # 3. NWP rain confidence: model says rain AND it's already raining = high confidence
    if mpp_col and "rain_accum_3h" in df.columns:
        df["nwp_rain_confirmed"] = df[mpp_col] * df["rain_accum_3h"]

    # 4. Afternoon convective uncertainty: NWP rain in afternoon with
    #    no low clouds = likely overestimating convection (biggest FP source)
    if mpp_col and "cloud_cover_low" in df.columns:
        hour = pd.to_datetime(df.get("datetime", pd.NaT)).dt.hour
        is_afternoon = ((hour >= 11) & (hour <= 17)).astype(float)
        low_clouds = pd.to_numeric(df["cloud_cover_low"], errors="coerce") / 100
        df["afternoon_fp_risk"] = is_afternoon * df[mpp_col] * (1 - low_clouds)

    # 5. Dew point gap when NWP says rain — large gap = air too dry for rain to reach surface
    if mpp_col and "dew_point_depression" in df.columns:
        dpd = pd.to_numeric(df["dew_point_depression"], errors="coerce")
        df["nwp_rain_dry_air"] = df[mpp_col] * dpd

    # ── NWP temporal consistency — captures WHEN the NWP is wrong ──
    # The NWP changes its mind constantly. A persistent rain prediction
    # that materializes is real; one that keeps misfiring is a stuck FP.
    # These features are the best way to beat the NWP using only NWP data.

    # 1. Persistence: how many of last 6h had NWP rain? (0–6)
    #    High = synoptic rain event (frontal, Llevantada) — high reliability
    #    Low (1) = isolated signal — often FP (convective false alarm)
    if "model_predicts_precip" in df.columns:
        df["nwp_rain_persistence_6h"] = df["model_predicts_precip"].rolling(6, min_periods=1).sum()

    # 2. NWP rain amount trend: is the NWP ramping UP or backing OFF?
    #    Increasing = front approaching → likely real
    #    Decreasing = drying event, NWP residual → FP
    if "nwp_rain_amount" in df.columns:
        df["nwp_rain_trend_3h"] = df["nwp_rain_amount"].diff(3)

    # 3. Weather code transition: captures the NWP "deciding" it will rain
    #    A sudden jump from dry→rain code is less reliable than a gradual buildup
    if "weather_code" in df.columns:
        wc = pd.to_numeric(df["weather_code"], errors="coerce")
        df["weather_code_change_3h"] = wc.diff(3)

    # 4. Cloud-humidity convergence: both increasing together = rain developing
    #    independently from what NWP says. Divergence = clearing.
    if "cloud_change_3h" in df.columns and "humidity_change_3h" in df.columns:
        cloud_ch = pd.to_numeric(df["cloud_change_3h"], errors="coerce")
        humid_ch = pd.to_numeric(df["humidity_change_3h"], errors="coerce")
        df["cloud_humidity_convergence"] = cloud_ch * humid_ch

    # 5. Precipitation trend: rain intensifying vs weakening
    if "precipitation" in df.columns:
        df["precip_trend_3h"] = pd.to_numeric(df["precipitation"], errors="coerce").diff(3)

    # ── Tier 1: New ERA5 features (100% coverage 2015+) ──

    # Showers (convective rain) — separate from stratiform rain
    if "showers" in df.columns:
        df["showers"] = pd.to_numeric(df["showers"], errors="coerce")
        # Convective fraction: how much of NWP rain is convective showers
        if "rain" in df.columns:
            total_rain = pd.to_numeric(df["rain"], errors="coerce") + df["showers"]
            df["nwp_showers_fraction"] = df["showers"] / total_rain.clip(lower=0.01)
            df["nwp_showers_fraction"] = df["nwp_showers_fraction"].where(total_rain > 0.05, 0)

    # Evapotranspiration — surface moisture flux (FAO reference ET)
    if "et0_fao_evapotranspiration" in df.columns:
        df["et0_fao_evapotranspiration"] = pd.to_numeric(df["et0_fao_evapotranspiration"], errors="coerce")

    # Soil temperature — warm soil + moist soil = thunderstorm fuel
    if "soil_temperature_0_to_7cm" in df.columns:
        df["soil_temperature_0_to_7cm"] = pd.to_numeric(df["soil_temperature_0_to_7cm"], errors="coerce")
        # Soil-air temperature gap: warm soil vs cool air = enhanced evaporation
        if "temperature_2m" in df.columns:
            df["soil_air_temp_diff"] = df["soil_temperature_0_to_7cm"] - df["temperature_2m"]

    # Sunshine duration — hours of direct sun
    if "sunshine_duration" in df.columns:
        # API returns seconds of sunshine per hour (0-3600)
        df["sunshine_duration"] = pd.to_numeric(df["sunshine_duration"], errors="coerce")
        # Cumulative sun in last 3h — boundary layer heating proxy
        df["sunshine_accum_3h"] = df["sunshine_duration"].rolling(3, min_periods=1).sum()

    # 100m wind — low-level jet detection
    if "wind_speed_100m" in df.columns:
        df["wind_speed_100m"] = pd.to_numeric(df["wind_speed_100m"], errors="coerce")
        # Wind shear 10m-100m: large = boundary layer instability
        if "wind_speed_10m" in df.columns:
            df["boundary_layer_shear"] = df["wind_speed_100m"] - df["wind_speed_10m"]

    if "wind_direction_100m" in df.columns:
        df["wind_direction_100m"] = pd.to_numeric(df["wind_direction_100m"], errors="coerce")
        # Directional shear: wind backing/veering with height = frontal signal
        if "wind_direction_10m" in df.columns:
            dir10 = pd.to_numeric(df["wind_direction_10m"], errors="coerce")
            dir100 = df["wind_direction_100m"]
            diff = dir100 - dir10
            # Normalize to [-180, 180]
            df["wind_dir_shear_100m"] = ((diff + 180) % 360) - 180

    # Snowfall — precipitation type discriminator
    if "snowfall" in df.columns:
        df["snowfall"] = pd.to_numeric(df["snowfall"], errors="coerce")

    # ── Tier 2: Historical Forecast API features (from April 2021, ~44%) ──

    # Lifted index from NWP (more accurate than our derived version)
    if "lifted_index" in df.columns:
        df["nwp_lifted_index"] = pd.to_numeric(df["lifted_index"], errors="coerce")

    # Geopotential height at 850hPa — baric topography
    if "gph_850" in df.columns:
        gph850 = pd.to_numeric(df["gph_850"], errors="coerce")
        df["gph_850"] = gph850
        # Falling gph = approaching trough = frontogenesis
        df["gph_850_change_3h"] = gph850.diff(3)

    # Humidity at 500hPa — mid-tropospheric moisture
    if "rh_500" in df.columns:
        df["rh_500"] = pd.to_numeric(df["rh_500"], errors="coerce")
        # Dry intrusion at 500hPa (better than 700hPa — deeper layer)
        if "rh_850" in df.columns:
            rh850 = pd.to_numeric(df["rh_850"], errors="coerce")
            df["dry_intrusion_500"] = (rh850 - df["rh_500"]).clip(lower=0)

    # 700hPa wind — steering level for convective cells
    if "wind_700_speed" in df.columns:
        df["wind_700_speed"] = pd.to_numeric(df["wind_700_speed"], errors="coerce")
    if "wind_700_dir" in df.columns:
        df["wind_700_dir"] = pd.to_numeric(df["wind_700_dir"], errors="coerce")
        # SE steering flow pushes Mediterranean moisture against Montseny
        dir700 = pd.to_numeric(df["wind_700_dir"], errors="coerce")
        speed700 = pd.to_numeric(df.get("wind_700_speed", pd.Series(dtype=float)), errors="coerce")
        # SE sector = 90-180° — onshore Mediterranean flow at steering level
        is_se = ((dir700 >= 90) & (dir700 <= 180)).astype(float)
        df["steering_onshore_700"] = is_se * speed700

    # ── Tier 3: Derived interaction features (no new data) ──

    # Rain ending signal: rained recently but conditions are drying
    # 70% of FP occur at rain endings — this directly targets that
    if "rained_last_3h" in df.columns and "humidity_change_3h" in df.columns:
        rained = df["rained_last_3h"]
        hum_drop = (-pd.to_numeric(df["humidity_change_3h"], errors="coerce")).clip(lower=0)
        cloud_drop = 0
        if "cloud_change_3h" in df.columns:
            cloud_drop = (-pd.to_numeric(df["cloud_change_3h"], errors="coerce")).clip(lower=0) / 100
        df["rain_ending_signal"] = rained * (hum_drop + cloud_drop * 10)

    # Cloud thickness proxy: thick low+mid = nimbostratus (real rain), high only = cirrus
    if "cloud_cover_low" in df.columns and "cloud_cover_mid" in df.columns and "cloud_cover_high" in df.columns:
        low = pd.to_numeric(df["cloud_cover_low"], errors="coerce")
        mid = pd.to_numeric(df["cloud_cover_mid"], errors="coerce")
        high = pd.to_numeric(df["cloud_cover_high"], errors="coerce")
        df["cloud_thickness_proxy"] = ((low + mid) / 2 - high) / 100

    # Radiation-rain conflict: sunshine + NWP rain = contradiction (FP signal)
    if mpp_col and "shortwave_radiation" in df.columns:
        rad = pd.to_numeric(df["shortwave_radiation"], errors="coerce")
        df["radiation_rain_conflict"] = df[mpp_col] * rad / 500  # normalize ~0-1

    return df


def _add_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivades del radar (RainViewer) — puntuals, espacials i per quadrant."""
    df = df.copy()
    # Puntuals
    if "radar_dbz" in df.columns:
        df["radar_dbz"] = pd.to_numeric(df["radar_dbz"], errors="coerce").fillna(0)
        df["radar_has_echo"] = (df["radar_dbz"] > 5).astype(int)
    if "radar_rain_rate" in df.columns:
        df["radar_rain_rate"] = pd.to_numeric(df["radar_rain_rate"], errors="coerce").fillna(0)
    if "radar_approaching" in df.columns:
        df["radar_approaching"] = df["radar_approaching"].astype(int)
    if "radar_frames_with_echo" in df.columns:
        df["radar_frames_with_echo"] = pd.to_numeric(df["radar_frames_with_echo"], errors="coerce").fillna(0)
    # Espacials
    for col in ["radar_nearest_echo_km", "radar_max_dbz_20km", "radar_coverage_20km",
                "radar_upwind_nearest_echo_km", "radar_upwind_max_dbz",
                "radar_storm_velocity_kmh", "radar_storm_velocity_ns",
                "radar_storm_velocity_ew"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "radar_storm_approaching" in df.columns:
        df["radar_storm_approaching"] = df["radar_storm_approaching"].astype(int)
    # Quadrant features (N/E/S/W)
    for quad in ("N", "E", "S", "W"):
        for prefix in ("radar_quadrant_max_dbz_", "radar_quadrant_coverage_"):
            col = f"{prefix}{quad}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Cyclic encoding of nearest echo bearing (sin/cos)
    if "radar_nearest_echo_bearing" in df.columns:
        bearing_rad = np.deg2rad(pd.to_numeric(df["radar_nearest_echo_bearing"], errors="coerce").fillna(0))
        df["radar_echo_bearing_sin"] = np.sin(bearing_rad)
        df["radar_echo_bearing_cos"] = np.cos(bearing_rad)
    return df


def _add_sentinel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de l'estació sentinella (Granollers)."""
    df = df.copy()
    for col in ["sentinel_temp_diff", "sentinel_humidity_diff", "sentinel_precip",
                "sentinel_raining", "local_rain_xema", "local_rain_xema_3h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_lightning_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de descàrregues elèctriques (XDDE Meteocat)."""
    df = df.copy()
    for col in ["lightning_count_30km", "lightning_count_15km",
                "lightning_nearest_km", "lightning_cloud_ground",
                "lightning_max_current_ka"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lightning_approaching" in df.columns:
        df["lightning_approaching"] = df["lightning_approaching"].astype(int)
    if "lightning_has_activity" in df.columns:
        df["lightning_has_activity"] = df["lightning_has_activity"].astype(int)
    return df


def _add_aemet_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features del radar AEMET Barcelona."""
    df = df.copy()
    for col in ["aemet_radar_dbz", "aemet_radar_nearest_echo_km",
                "aemet_radar_max_dbz_20km", "aemet_radar_coverage_20km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "aemet_radar_has_echo" in df.columns:
        df["aemet_radar_has_echo"] = df["aemet_radar_has_echo"].astype(int)
    if "aemet_radar_echoes_found" in df.columns:
        df["aemet_radar_echoes_found"] = df["aemet_radar_echoes_found"].astype(int)
    return df


def _add_smc_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de la predicció municipal del Meteocat (SMC)."""
    df = df.copy()
    for col in ["smc_prob_precip_1h", "smc_prob_precip_6h",
                "smc_precip_intensity", "smc_temp_forecast"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features d'acord entre models d'ensemble i bias del forecast."""
    df = df.copy()
    for col in ["ensemble_rain_agreement", "ensemble_precip_spread", "ensemble_temp_spread",
                "ensemble_max_precip", "ensemble_min_precip", "ensemble_models_rain",
                "forecast_temp_bias", "forecast_humidity_bias",
                "aemet_prob_precip", "aemet_prob_storm", "aemet_precip_today"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interaccions NWP-Ensemble: capturar desacord entre el forecast puntual i l'ensemble.
    # L'ensemble té 0.8% d'importància perquè és redundant amb el NWP puntual quan coincideixen.
    # Aquestes features aïllen els casos de DESACORD — on l'ensemble afegeix informació nova.
    if "ensemble_models_rain" in df.columns and "model_predicts_precip" in df.columns:
        # Ensemble veu pluja però el NWP puntual no (possible pluja que el punt no veu)
        df["ensemble_surprise_rain"] = (
            df["ensemble_models_rain"].fillna(0) * (1 - df["model_predicts_precip"].fillna(0))
        )
    if "nwp_precip_severity" in df.columns and "ensemble_rain_agreement" in df.columns:
        # NWP diu pluja forta però l'ensemble no hi està d'acord (possible falsa alarma)
        df["nwp_isolated_rain"] = (
            df["nwp_precip_severity"].fillna(0) * (1 - df["ensemble_rain_agreement"].fillna(0))
        )

    return df


def build_target_column(df: pd.DataFrame, precip_col: str = "precipitation", horizon: int = 1) -> pd.DataFrame:
    """
    Crea la columna target: 'will_rain' = 1 si plourà en les properes `horizon` hores.
    Per a classificació binària.
    """
    df = df.copy()
    # Mirar endavant: plourà en les properes `horizon` hores?
    future_rain = df[precip_col].rolling(horizon, min_periods=1).sum().shift(-horizon)
    df["will_rain"] = (future_rain >= config.RAIN_THRESHOLD_MM).astype(int)
    return df


def build_features_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering per a dades horàries (Open-Meteo).
    Entrada: DataFrame amb columnes d'Open-Meteo + datetime.
    Sortida: DataFrame amb totes les features calculades.
    """
    df = _add_temporal_features(df)
    df = _add_solar_timing_features(df)
    df = _add_pressure_features(df, "pressure_msl")
    df = _add_humidity_features(df, "temperature_2m", "relative_humidity_2m")
    df = _add_wind_features(df, "wind_speed_10m", "wind_direction_10m")
    df = _add_wind_regime_features(df, "wind_speed_10m", "wind_direction_10m", "relative_humidity_2m")
    df = _add_rain_context(df, "precipitation")
    df = _add_model_features(df)
    df = _add_radar_features(df)
    df = _add_sentinel_features(df)
    df = _add_lightning_features(df)
    df = _add_aemet_radar_features(df)
    df = _add_smc_forecast_features(df)
    df = _add_ensemble_features(df)
    df = _add_pressure_level_features(df)
    df = _add_physics_composites(df)
    df = _add_soil_moisture_features(df)
    df = _add_atmospheric_column_features(df)
    df = _add_cloud_layer_features(df)
    df = _add_wet_bulb_features(df)
    df = _add_radiation_features(df)
    df = _add_visibility_features(df)
    df = _add_freezing_level_features(df)
    df = _add_wind_gust_features(df)
    return df


def build_features_from_realtime(station_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina dades en temps real de l'estació amb la previsió d'Open-Meteo.
    Retorna un DataFrame amb una fila per cada moment amb totes les features.
    Si l'estació no està disponible, usa només el forecast com a base.
    """
    if station_df.empty or "datetime" not in station_df.columns:
        # Fallback: usar forecast com a base horària quan l'estació no respon
        logger.warning("Estació no disponible — usant forecast com a base horària")
        if forecast_df.empty:
            return pd.DataFrame()
        hourly = forecast_df.copy()
        hourly["datetime"] = pd.to_datetime(hourly["datetime"])
        # Aplicar feature engineering i retornar
        hourly = build_features_from_hourly(hourly)
        return hourly

    # Remapejar columnes de l'estació al format d'Open-Meteo
    station = station_df.rename(columns={
        "TEMP": "temperature_2m",
        "HUM": "relative_humidity_2m",
        "BAR": "pressure_msl",
        "VEL": "wind_speed_10m",
        "DIR_DEG": "wind_direction_10m",
        "PREC": "precipitation",
        "SUN": "shortwave_radiation",
        "UVI": "uvi_station",
    }).copy()

    # Resamplar dades minut-a-minut a horàries (mitjana)
    station["datetime"] = pd.to_datetime(station["datetime"])
    station = station.set_index("datetime")
    hourly = station.resample("1h").agg({
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "pressure_msl": "mean",
        "wind_speed_10m": "mean",
        "wind_direction_10m": lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan,
        "precipitation": "sum",
        "shortwave_radiation": "mean",
    }).reset_index()

    # Unir amb forecast d'Open-Meteo per l'hora més propera
    if not forecast_df.empty:
        forecast_df = forecast_df.copy()
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])

        # Afegir columnes del forecast que no tenim a l'estació
        extra_cols = [
            "cape", "cloud_cover", "weather_code", "wind_gusts_10m", "dew_point_2m", "rain",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "direct_radiation", "diffuse_radiation", "wet_bulb_temperature_2m",
            "visibility", "freezing_level_height",
            "vapour_pressure_deficit", "convective_inhibition",
            "showers", "et0_fao_evapotranspiration", "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
            "sunshine_duration", "wind_speed_100m", "wind_direction_100m", "snowfall",
            "total_column_integrated_water_vapour", "boundary_layer_height",
            "terrestrial_radiation", "soil_moisture_28_to_100cm",
            # Nivells de pressió (injectats per predict.py si disponibles)
            "wind_850_dir", "wind_850_speed", "temp_850", "rh_850",
            "wind_925_dir", "wind_925_speed", "temp_925", "rh_925",
            "rh_700", "temp_700", "temp_500",
            "wind_300_speed", "wind_300_dir", "gph_300",
            # Tier 2 — pressure levels extra
            "gph_850", "rh_500", "wind_700_speed", "wind_700_dir",
            "nwp_lifted_index",
        ]
        available_extra = [c for c in extra_cols if c in forecast_df.columns]

        if available_extra:
            forecast_subset = forecast_df[["datetime"] + available_extra].copy()
            hourly = pd.merge_asof(
                hourly.sort_values("datetime"),
                forecast_subset.sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("2h"),
            )

        # Afegir "model_precipitation" com a feature (què diu el model que passarà)
        if "rain" in forecast_df.columns:
            forecast_rain = forecast_df[["datetime", "rain"]].rename(columns={"rain": "model_rain_forecast"})
            hourly = pd.merge_asof(
                hourly.sort_values("datetime"),
                forecast_rain.sort_values("datetime"),
                on="datetime",
                direction="forward",  # Forecast mira endavant
                tolerance=pd.Timedelta("2h"),
            )

    # Aplicar feature engineering
    hourly = build_features_from_hourly(hourly)
    return hourly


# ── Llista de features que utilitza el model ──
FEATURE_COLUMNS = [
    # Temporals
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "hours_since_sunrise",  # Convective initiation timing (2-4h lag after sunrise)
    # Pressió
    "pressure_msl", "pressure_change_1h", "pressure_change_3h",
    "pressure_change_6h", "pressure_accel_3h",
    # Temperatura i humitat
    "temperature_2m", "relative_humidity_2m",
    "dew_point", "dew_point_depression",
    "humidity_change_1h", "humidity_change_3h",
    "vapour_pressure_deficit", "vpd_change_3h",
    # Vent
    "wind_speed_10m", "wind_u", "wind_v",
    "wind_speed_change_1h", "wind_speed_change_3h",
    "is_sea_breeze",
    # Règims eòlics catalans (Rosa dels Vents completa)
    # Classificació basada en 850hPa (sinòptic) amb fallback a 10m (superfície)
    # Binary flags (is_*) s'usen internament per derivar interaccions, NO al model
    # (importància zero demostrada). Només les interaccions règim×magnitud entren.
    "llevantada_strength", "llevantada_moisture",
    "garbi_strength",
    "tramuntana_strength", "tramuntana_moisture",
    "wind_dir_change_3h",
    # Nivells de pressió (925/850/700/500/300 hPa) — perfil vertical complet
    "has_pressure_levels",  # Indicador disponibilitat dades nivells de pressió
    "temp_925", "rh_925", "wind_925_speed", "wind_925_dir",
    "wind_850_speed", "wind_850_dir",
    "temp_850", "temp_500", "rh_850",
    "rh_700", "temp_700",
    "rh_700_change_3h", "rh_700_change_6h",  # Mid-level drying (virga/entrainment)
    "temp_850_change_3h",   # Warm/cold advection proxy
    "wind_300_speed", "wind_300_dir", "gph_300",
    # Índexs d'inestabilitat (Skew-T + Lifted Index)
    "vt_index", "tt_index", "li_index",
    "li_unstable",  # li_very_unstable: zero importance (fires too rarely)
    "k_index",              # K-index: moist layer depth (TT/VT miss this)
    # Physics: moisture flux (wind × humidity = water transport)
    "moisture_flux_850",
    "moisture_flux_925",  # low-level: bulk of Llevantada moisture
    # Physics: Θe lapse rate proxy (convective instability)
    "theta_e_deficit",
    # 925hPa: boundary layer physics
    "inversion_925",  # T925 - T_sfc: positiu = inversió (fog/stratus suppressor)
    # 300hPa: jet stream physics
    "deep_layer_shear",  # 850-300hPa speed diff: tempestes organitzades
    "bulk_richardson",      # BRN = CAPE/shear²: storm mode (supercell vs multicell)
    "jet_speed_300",  # jet stream intensity: dynamic lifting trigger
    # Cisalla de vent (wind shear) — clau per tempestes organitzades
    "wind_shear_speed", "wind_shear_dir",
    # Llindars d'aire fred a 500hPa
    # cold_500_moderate, cold_500_strong: zero importance (extreme thresholds fire too rarely)
    # Pluja recent
    "precipitation", "rain_accum_3h", "rain_accum_6h", "rained_last_3h",
    "rain_accum_24h",          # Pluja acumulada 24h — persistència sinòptica
    "pressure_min_24h",        # Pressió mínima 24h — borrasca activa
    # Model / satèl·lit
    "cloud_cover", "cloud_change_1h", "cloud_change_3h", "is_overcast",
    # cape, cape_high, cape_very_high: zero importance
    "weather_code", "model_predicts_precip",
    # Weather code decomposition (different physics for each precipitation type)
    "wc_is_thunderstorm", "wc_is_rain", "wc_is_drizzle",
    # NWP precipitation severity (continuous 0-5 scale: drizzle=1, rain=2, showers=3, snow=4, thunder=5)
    # WMO 51 (drizzle) = 83% of all FP. This continuous feature encodes reliability gradient.
    "nwp_precip_severity",
    # NWP error detection (when model disagrees with surface conditions)
    "nwp_dry_conflict", "nwp_wet_conflict",
    # NWP rain amount (continuous — more informative than binary model_predicts_precip)
    "nwp_rain_amount",
    # FP-killer interactions (target the 9K false positives)
    "nwp_rain_drying",      # NWP rain + humidity dropping → FP signal
    "nwp_rain_confirmed",   # NWP rain + already raining → high confidence TP
    "afternoon_fp_risk",    # afternoon + NWP rain + no low clouds → convective FP
    "nwp_rain_dry_air",     # NWP rain + large dew point gap → virga/evaporation
    # NWP temporal consistency — captures when the NWP is wrong
    "nwp_rain_persistence_6h", # Quantes de les últimes 6h el NWP ha predit pluja (0-6)
    "nwp_rain_trend_3h",       # Tendència quantitat NWP: pujant=front, baixant=residu FP
    "weather_code_change_3h",  # Transició WMO: salt sobtat = menys fiable
    "cloud_humidity_convergence", # Núvols + humitat pujant junts = pluja real
    "precip_trend_3h",         # Precipitació intensificant-se o debilitant-se
    # Tier 1 — ERA5 new vars (100% coverage 2015+)
    "showers",              # Pluja convectiva (xàfecs) separada de l'estratiforme
    "nwp_showers_fraction", # Fracció convectiva del total NWP rain
    "et0_fao_evapotranspiration",   # Evapotranspiració de referència FAO (mm)
    "soil_temperature_0_to_7cm",  # Temperatura del sòl
    "soil_air_temp_diff",   # Sòl calent vs aire fred = evaporació forçada
    "sunshine_duration",    # Durada de sol (s per hora)
    "sunshine_accum_3h",    # Sol acumulat 3h — proxy calentament capa límit
    "wind_speed_100m",      # Vent a 100m — low-level jet
    "boundary_layer_shear", # Cisalla 10m-100m = inestabilitat capa límit
    "wind_dir_shear_100m",  # Cisalla direccional 10m-100m = front
    "snowfall",             # Neu — discriminador tipus precipitació
    # Tier 2 — Historical Forecast API (des d'abril 2021, ~44%)
    "nwp_lifted_index",     # LI directe del NWP (millor que el derivat)
    "gph_850",              # Alçada geopotencial 850hPa
    "gph_850_change_3h",    # Canvi gph 850 — depressió/anticicló que s'apropa
    "rh_500",               # Humitat a 500hPa
    "dry_intrusion_500",    # Intrussió seca 850-500hPa (loaded gun profund)
    "wind_700_speed",       # Vent de guia (steering level)
    "wind_700_dir",         # Direcció del vent de guia
    "steering_onshore_700", # Flux marítim SE a 700hPa → humitat Mediterrània
    # Tier 3 — Derived interaction features
    "rain_ending_signal",   # Pluja acabant-se + condicions assequen-se → FP
    "cloud_thickness_proxy",# Gruix de núvols baixos+mitjos vs alts
    "radiation_rain_conflict", # Sol + NWP pluja = contradicció → FP
    "moisture_flux_change_3h", # Canvi flux d'humitat — front que s'apropa/passa
    # CAPE change rate (rapid destabilization)
    "cape",                     # CAPE continu brut (44.5% cobertura) — més informatiu que binari
    "cape_change_3h",
    "cape_diurnal_weighted",   # CAPE × hora diürna — convecció solar
    # Tier 4 — ERA5 surface expansion (100% coverage 2015+)
    "tcwv",                     # Total Column Water Vapour (kg/m²) — precipitable water
    "tcwv_change_3h",           # Càrrega/advecció d'humitat (front que arriba)
    "tcwv_change_6h",           # Tendència d'humitat sinòptica
    "boundary_layer_height",    # Alçada capa límit (m) — fondària convecció
    "blh_change_3h",            # Canvi BLH — aprofundiment = desenvolupament convectiu
    "tcwv_blh_ratio",           # Humitat / fondària mescla = eficiència convectiva
    "terrestrial_radiation",    # Radiació terrestre (W/m²) — detecció núvols nocturna
    "soil_moisture_28_to_100cm", # Humitat sòl profund — saturació antecedent
    "soil_saturation_ratio",    # Superficial / profund — infiltració recent
    "tcwv_monthly_anomaly",     # TCWV relatiu a climatologia mensual
    # Radiació solar
    "shortwave_radiation",
    # Radar (RainViewer) — puntual + espacial
    "radar_dbz", "radar_rain_rate", "radar_has_echo",
    "radar_frames_with_echo", "radar_approaching", "radar_max_intensity_1h",
    # Radar espacial (30km scan + tracking)
    "radar_nearest_echo_km", "radar_max_dbz_20km", "radar_coverage_20km",
    "radar_upwind_nearest_echo_km", "radar_upwind_max_dbz",
    "radar_storm_velocity_kmh", "radar_storm_velocity_ns", "radar_storm_velocity_ew",
    "radar_storm_approaching",
    # Radar quadrant features (N/E/S/W directional awareness)
    "radar_quadrant_max_dbz_N", "radar_quadrant_max_dbz_E",
    "radar_quadrant_max_dbz_S", "radar_quadrant_max_dbz_W",
    "radar_quadrant_coverage_N", "radar_quadrant_coverage_E",
    "radar_quadrant_coverage_S", "radar_quadrant_coverage_W",
    # Cyclic encoding of nearest echo bearing
    "radar_echo_bearing_sin", "radar_echo_bearing_cos",
    # Estació sentinella (Granollers) i pluviòmetre XEMA local
    "sentinel_temp_diff", "sentinel_humidity_diff",
    "sentinel_precip", "sentinel_raining",
    "local_rain_xema", "local_rain_xema_3h",
    # Acord entre models (Ensemble: ECMWF vs GFS vs ICON)
    "ensemble_rain_agreement", "ensemble_precip_spread",
    "ensemble_temp_spread", "ensemble_max_precip",
    "ensemble_min_precip", "ensemble_models_rain",
    # Interaccions NWP-Ensemble (desacord entre forecast puntual i ensemble)
    "ensemble_surprise_rain",  # ensemble veu pluja però NWP puntual no
    "nwp_isolated_rain",       # NWP puntual diu pluja però ensemble discrepa
    # Bias del forecast vs observació real
    "forecast_temp_bias", "forecast_humidity_bias",
    # AEMET probabilitats de precipitació i tempesta
    "aemet_prob_precip", "aemet_prob_storm", "aemet_precip_today",
    # Descàrregues elèctriques (XDDE Meteocat)
    "lightning_count_30km", "lightning_count_15km",
    "lightning_nearest_km", "lightning_cloud_ground",
    "lightning_max_current_ka", "lightning_approaching",
    "lightning_has_activity",
    # Radar AEMET Barcelona (complement a RainViewer)
    "aemet_radar_dbz", "aemet_radar_has_echo",
    "aemet_radar_nearest_echo_km", "aemet_radar_max_dbz_20km",
    "aemet_radar_coverage_20km", "aemet_radar_echoes_found",
    # Predicció municipal SMC (Meteocat)
    "smc_prob_precip_1h", "smc_prob_precip_6h",
    "smc_precip_intensity",
    # Physics composites (features derivades de variables existents)
    "orographic_forcing",     # flux humit perpendicular a la Serralada × humitat
    "frontal_passage",        # detecció de pas de front (pressió + vent + temp)
    "convective_composite",   # inestabilitat × humitat × cisalla (tots alhora)
    "thermal_buildup",        # calentament diürn → termals → convecció tarda
    "low_level_convergence",  # convergència: vent frenant + humitat pujant + pressió baixant
    "dry_intrusion_700",      # aire sec a 700hPa sobre humit a 850hPa (loaded gun)
    # Soil moisture (ERA5 archive 2015+; predicció: últim valor conegut)
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_change_24h",
    # Convective inhibition (CIN) — supressió de convecció
    "convective_inhibition",
    # SST Marine (forecast only — s'acumula via feedback loop)
    "sst_med",
    # Cloud layers (ERA5 archive 2015+)
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "cloud_low_fraction",
    # Wet bulb temperature (ERA5 archive 2015+)
    "wet_bulb_temperature_2m", "wet_bulb_depression",
    # Radiation breakdown (ERA5 archive 2015+)
    "direct_radiation", "diffuse_radiation", "diffuse_fraction",
    # Wind gusts (ERA5 archive 2015+)
    "wind_gusts_10m", "gust_factor",
    # Visibility (Historical Forecast API 2021-04+)
    "visibility",
    # Freezing level (Historical Forecast API 2021-04+)
    "freezing_level_height",
]


def build_features_from_forecast(
    forecast_df: pd.DataFrame,
    pressure_df: pd.DataFrame = None,
    smc_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Construeix vectors de features per a hores futures usant only forecast data.
    Les features de radar/sentinella/bias queden com NaN
    (XGBoost les gestiona nativament).

    Args:
        forecast_df: DataFrame d'Open-Meteo hourly forecast (temperature_2m, etc.)
        pressure_df: DataFrame de pressure levels hourly (wind_850_speed, etc.)
        smc_df: DataFrame de SMC municipal forecast (smc_prob_precip_1h, etc.)

    Returns:
        DataFrame amb una fila per hora futura i totes les FEATURE_COLUMNS.
    """
    df = forecast_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Merge pressure levels si disponible
    if pressure_df is not None and not pressure_df.empty:
        pressure_df = pressure_df.copy()
        pressure_df["datetime"] = pd.to_datetime(pressure_df["datetime"])
        df = pd.merge_asof(
            df.sort_values("datetime"),
            pressure_df.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )

    # Merge SMC municipal forecast si disponible
    if smc_df is not None and not smc_df.empty:
        smc_df = smc_df.copy()
        smc_df["datetime"] = pd.to_datetime(smc_df["datetime"])
        df = pd.merge_asof(
            df.sort_values("datetime"),
            smc_df.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )

    # Aplicar feature engineering (temporal, pressió, humitat, vent, etc.)
    df = build_features_from_hourly(df)

    return df
