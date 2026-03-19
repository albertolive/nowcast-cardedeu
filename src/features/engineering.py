"""
Feature Engineering per al model de Nowcasting.
Crea les features (columnes) que alimenten XGBoost.
Combina dades locals (meteocardedeu) + exteriors (Open-Meteo).
"""
import math
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


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
    - Tramuntana (N 340°-30°): vent polar fred, cel blau, supressor de pluja.
    - Llevantada (E/SE 60°-150°): humitat mediterrània contra la Serralada.
    - Migjorn (S 150°-190°): vent del sud, aire africà calent i sec.
    - Garbí (SW 190°-250°): "Anuncia borrasques amb fortes precipitacions."
    - Ponent/Mestral (W/NW 250°-340°): aire continental sec.
    """
    df = df.copy()

    # Preferir 850hPa per a la classificació sinòptica, fallback a 10m
    if "wind_850_dir" in df.columns and df["wind_850_dir"].notna().any():
        synoptic_dir = pd.to_numeric(df["wind_850_dir"], errors="coerce")
        synoptic_speed = pd.to_numeric(df.get("wind_850_speed", pd.Series(dtype=float)), errors="coerce").fillna(0)
    elif dir_col in df.columns:
        synoptic_dir = pd.to_numeric(df[dir_col], errors="coerce")
        synoptic_speed = pd.to_numeric(df.get(speed_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    else:
        return df

    # Règims eòlics catalans a partir del vent sinòptic
    df["is_tramuntana"] = _dir_in_range(synoptic_dir, 340, 30)
    df["is_llevantada"] = _dir_in_range(synoptic_dir, 60, 150)
    df["is_migjorn"] = _dir_in_range(synoptic_dir, 150, 190)
    df["is_garbi"] = _dir_in_range(synoptic_dir, 190, 250)
    df["is_ponent"] = _dir_in_range(synoptic_dir, 250, 340)

    # Interaccions: Llevantada × velocitat/humitat
    humidity = pd.to_numeric(df.get(hum_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["llevantada_strength"] = df["is_llevantada"] * synoptic_speed
    df["llevantada_moisture"] = df["is_llevantada"] * (humidity / 100.0)

    # Garbí × velocitat: "Anuncia borrasques amb fortes precipitacions" (ref: alexmeteo)
    df["garbi_strength"] = df["is_garbi"] * synoptic_speed

    # Canvi de direcció sinòptica en 3h (backing/veering)
    df["wind_dir_change_3h"] = _angular_diff(synoptic_dir, 3)

    return df


def _add_pressure_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de nivells de pressió (850hPa, 700hPa, 500hPa).
    Inclou índexs d'inestabilitat del Skew-T:
    - VT (Vertical Totals) = T850 - T500: gradient tèrmic vertical
    - TT (Total Totals) = VT + (Td850 - T500): combina gradient + humitat
    - LI (Lifted Index): inestabilitat a 500hPa (negatiu = inestable)

    Ref: alexmeteo.com — Skew-T analysis, "Ingredients per formar Tempestes"
    """
    import math

    df = df.copy()
    for col in ["wind_850_speed", "wind_850_dir", "temp_850", "temp_500",
                "rh_850", "rh_700", "temp_700", "vt_index", "tt_index", "li_index"]:
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

    return df


def _add_rain_context(df: pd.DataFrame, precip_col: str = "precipitation") -> pd.DataFrame:
    """Context de pluja recent (últimes hores)."""
    df = df.copy()
    if precip_col in df.columns:
        # Pluja acumulada en les últimes 3 i 6 hores
        df["rain_accum_3h"] = df[precip_col].rolling(3, min_periods=1).sum()
        df["rain_accum_6h"] = df[precip_col].rolling(6, min_periods=1).sum()
        # Ha plogut recentment?
        df["rained_last_3h"] = (df["rain_accum_3h"] > 0.2).astype(int)
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
        # Codes >= 50 indiquen precipitació
        df["model_predicts_precip"] = (df["weather_code"] >= 50).astype(int)
        # Codes >= 80 indiquen xàfecs
        df["model_predicts_showers"] = (df["weather_code"] >= 80).astype(int)

    return df


def _add_radar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivades del radar (RainViewer) — puntuals i espacials."""
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
                "radar_storm_velocity_kmh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "radar_storm_approaching" in df.columns:
        df["radar_storm_approaching"] = df["radar_storm_approaching"].astype(int)
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
    return df


def build_features_from_realtime(station_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina dades en temps real de l'estació amb la previsió d'Open-Meteo.
    Retorna un DataFrame amb una fila per cada moment amb totes les features.
    """
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
        extra_cols = ["cape", "cloud_cover", "weather_code", "wind_gusts_10m", "dew_point_2m", "rain"]
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
    # is_tramuntana, is_llevantada: zero importance (kept in engineering for derived features)
    "is_migjorn", "is_garbi", "is_ponent",
    "llevantada_strength", "llevantada_moisture", "garbi_strength",
    "wind_dir_change_3h",
    # Nivells de pressió (850hPa, 700hPa, 500hPa) — flux sinòptic real
    "wind_850_speed", "wind_850_dir",
    "temp_850", "temp_500", "rh_850",
    "rh_700", "temp_700",
    # Índexs d'inestabilitat (Skew-T + Lifted Index)
    "vt_index", "tt_index", "li_index",
    "li_unstable",  # li_very_unstable: zero importance (fires too rarely)
    # Cisalla de vent (wind shear) — clau per tempestes organitzades
    "wind_shear_speed", "wind_shear_dir",
    # Llindars d'aire fred a 500hPa
    # cold_500_moderate, cold_500_strong: zero importance (extreme thresholds fire too rarely)
    # Pluja recent
    "precipitation", "rain_accum_3h", "rain_accum_6h", "rained_last_3h",
    # Model / satèl·lit
    "cloud_cover", "cloud_change_1h", "cloud_change_3h", "is_overcast",
    # cape, cape_high, cape_very_high: zero importance
    "weather_code", "model_predicts_precip",
    # model_predicts_showers: zero importance (redundant with weather_code)
    # Radiació solar
    "shortwave_radiation",
    # Radar (RainViewer) — puntual + espacial
    "radar_dbz", "radar_rain_rate", "radar_has_echo",
    "radar_frames_with_echo", "radar_approaching", "radar_max_intensity_1h",
    # Radar espacial (30km scan + tracking)
    "radar_nearest_echo_km", "radar_max_dbz_20km", "radar_coverage_20km",
    "radar_upwind_nearest_echo_km", "radar_upwind_max_dbz",
    "radar_storm_velocity_kmh", "radar_storm_approaching",
    # Estació sentinella (Granollers) i pluviòmetre XEMA local
    "sentinel_temp_diff", "sentinel_humidity_diff",
    "sentinel_precip", "sentinel_raining",
    "local_rain_xema", "local_rain_xema_3h",
    # Acord entre models (Ensemble: ECMWF vs GFS vs ICON)
    "ensemble_rain_agreement", "ensemble_precip_spread",
    "ensemble_temp_spread", "ensemble_max_precip",
    "ensemble_min_precip", "ensemble_models_rain",
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
]


def build_features_from_forecast(
    forecast_df: pd.DataFrame,
    pressure_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Construeix vectors de features per a hores futures usant only forecast data.
    Les features de radar/sentinella/bias/AEMET queden com NaN
    (XGBoost les gestiona nativament).

    Args:
        forecast_df: DataFrame d'Open-Meteo hourly forecast (temperature_2m, etc.)
        pressure_df: DataFrame de pressure levels hourly (wind_850_speed, etc.)

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

    # Aplicar feature engineering (temporal, pressió, humitat, vent, etc.)
    df = build_features_from_hourly(df)

    return df
