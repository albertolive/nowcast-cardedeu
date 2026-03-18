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
    Classifica el vent en règims meteorològics catalans, clau per a la pluja:

    - Llevantada (E/SE 60°-150°): humitat mediterrània contra la Serralada Prelitoral.
      El patró de pluja #1 per a Cardedeu i tot el Vallès Oriental.
    - Garbí/Xaloc (SW 190°-250°): aire càlid i inestable, desencadenant de tempestes.
    - Ponent/Mestral (W/NW 260°-340°): aire continental sec, supressor de pluja.
    - wind_dir_change_3h: gir del vent en 3h (backing → front càlid s'acosta).
    - Interaccions: llevantada × velocitat, llevantada × humitat.
    """
    df = df.copy()
    if dir_col not in df.columns:
        return df

    direction = pd.to_numeric(df[dir_col], errors="coerce")

    # Règims eòlics catalans
    df["is_llevantada"] = _dir_in_range(direction, 60, 150)
    df["is_garbi"] = _dir_in_range(direction, 190, 250)
    df["is_ponent"] = _dir_in_range(direction, 260, 340)

    # Interaccions: quan hi ha Llevantada + velocitat/humitat alta → pluja quasi segura
    speed = pd.to_numeric(df.get(speed_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    humidity = pd.to_numeric(df.get(hum_col, pd.Series(dtype=float)), errors="coerce").fillna(0)

    df["llevantada_strength"] = df["is_llevantada"] * speed
    df["llevantada_moisture"] = df["is_llevantada"] * (humidity / 100.0)

    # Canvi de direcció del vent en 3h (backing/veering)
    df["wind_dir_change_3h"] = _angular_diff(direction, 3)

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
    """Features derivades del radar (RainViewer). Es passen com a columnes al df."""
    df = df.copy()
    if "radar_dbz" in df.columns:
        df["radar_dbz"] = pd.to_numeric(df["radar_dbz"], errors="coerce").fillna(0)
        df["radar_has_echo"] = (df["radar_dbz"] > 5).astype(int)
    if "radar_rain_rate" in df.columns:
        df["radar_rain_rate"] = pd.to_numeric(df["radar_rain_rate"], errors="coerce").fillna(0)
    if "radar_approaching" in df.columns:
        df["radar_approaching"] = df["radar_approaching"].astype(int)
    if "radar_frames_with_echo" in df.columns:
        df["radar_frames_with_echo"] = pd.to_numeric(df["radar_frames_with_echo"], errors="coerce").fillna(0)
    return df


def _add_sentinel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de l'estació sentinella (Granollers)."""
    df = df.copy()
    for col in ["sentinel_temp_diff", "sentinel_humidity_diff", "sentinel_precip",
                "sentinel_raining", "local_rain_xema", "local_rain_xema_3h"]:
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
    df = _add_ensemble_features(df)
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
    # Vent
    "wind_speed_10m", "wind_u", "wind_v",
    "wind_speed_change_1h", "wind_speed_change_3h",
    "is_sea_breeze",
    # Règims eòlics catalans (Llevantada, Garbí, Ponent)
    "is_llevantada", "is_garbi", "is_ponent",
    "llevantada_strength", "llevantada_moisture",
    "wind_dir_change_3h",
    # Pluja recent
    "precipitation", "rain_accum_3h", "rain_accum_6h", "rained_last_3h",
    # Model / satèl·lit
    "cloud_cover", "cloud_change_1h", "cloud_change_3h", "is_overcast",
    "cape", "cape_high", "cape_very_high",
    "weather_code", "model_predicts_precip", "model_predicts_showers",
    # Radiació solar
    "shortwave_radiation",
    # Radar (RainViewer)
    "radar_dbz", "radar_rain_rate", "radar_has_echo",
    "radar_frames_with_echo", "radar_approaching", "radar_max_intensity_1h",
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
]
