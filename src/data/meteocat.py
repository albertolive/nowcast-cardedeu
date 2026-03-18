"""
Client per a l'API del Meteocat (XEMA).
Obté dades de l'estació sentinella de Granollers i del pluviòmetre ETAP Cardedeu.
Documentació: https://apidocs.meteocat.gencat.cat/
"""
import logging
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def fetch_variable_all_stations(var_code: int, target_date: date) -> pd.DataFrame:
    """
    Obté les dades d'una variable per a TOTES les estacions en un dia.
    Endpoint: /xema/v1/variables/mesurades/{var_code}/{YYYY}/{MM}/{DD}
    Retorna DataFrame amb columnes: station_code, datetime, value
    """
    if not _is_configured():
        logger.warning("Meteocat API key no configurada")
        return pd.DataFrame()

    url = (
        f"{config.METEOCAT_BASE_URL}/xema/v1/variables/mesurades/"
        f"{var_code}/{target_date.year}/{target_date.month:02d}/{target_date.day:02d}"
    )
    try:
        r = SESSION.get(url, headers=_headers(), timeout=20)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"Meteocat API error ({var_code}, {target_date}): {e}")
        return pd.DataFrame()

    data = r.json()
    rows = []
    for station_data in data:
        station_code = station_data.get("codi", "")
        for var_info in station_data.get("variables", []):
            for lecture in var_info.get("lectures", []):
                rows.append({
                    "station_code": station_code,
                    "datetime": pd.to_datetime(lecture["data"]),
                    "value": lecture.get("valor"),
                    "estat": lecture.get("estat", "").strip(),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Filtrar lectures invàlides
        df = df[df["estat"] != "T"]  # T = valor no disponible
    return df


def fetch_sentinel_latest() -> dict:
    """
    Obté les últimes lectures de l'estació sentinella (Granollers).
    Retorna un dict amb les dades més recents de temperatura, humitat i precipitació.
    """
    if not _is_configured():
        return _empty_sentinel()

    today = date.today()
    result = {}

    # Temperatura (32), Humitat (33), Precipitació (35)
    var_map = {
        config.XEMA_VAR_TEMP: "sentinel_temp",
        config.XEMA_VAR_HUMIDITY: "sentinel_humidity",
        config.XEMA_VAR_PRECIP: "sentinel_precip",
    }

    for var_code, key in var_map.items():
        df = fetch_variable_all_stations(var_code, today)
        if df.empty:
            result[key] = None
            continue

        # Filtrar per l'estació sentinella
        sentinel = df[df["station_code"] == config.SENTINEL_STATION_CODE]
        if sentinel.empty:
            result[key] = None
            continue

        # Agafar l'última lectura
        sentinel = sentinel.sort_values("datetime")
        result[key] = float(sentinel.iloc[-1]["value"])
        result[f"{key}_time"] = sentinel.iloc[-1]["datetime"].isoformat()

    # També obtenir precipitació del pluviòmetre local (ETAP Cardedeu KX)
    df_precip = fetch_variable_all_stations(config.XEMA_VAR_PRECIP, today)
    if not df_precip.empty:
        local = df_precip[df_precip["station_code"] == config.LOCAL_RAIN_STATION_CODE]
        if not local.empty:
            local = local.sort_values("datetime")
            result["local_rain_xema"] = float(local.iloc[-1]["value"])
            # Pluja acumulada en les últimes 3h del pluviòmetre XEMA
            cutoff_3h = local.iloc[-1]["datetime"] - pd.Timedelta("3h")
            recent = local[local["datetime"] >= cutoff_3h]
            result["local_rain_xema_3h"] = float(recent["value"].sum())

    return result


def fetch_sentinel_historical(target_date: date) -> dict:
    """
    Obté les dades completes de l'estació sentinella per un dia concret.
    Retorna un dict amb arrays de lectures horàries.
    Útil per construir el dataset d'entrenament.
    """
    if not _is_configured():
        return {}

    result = {}
    var_map = {
        config.XEMA_VAR_TEMP: "sentinel_temp",
        config.XEMA_VAR_HUMIDITY: "sentinel_humidity",
        config.XEMA_VAR_PRECIP: "sentinel_precip",
    }

    for var_code, key in var_map.items():
        df = fetch_variable_all_stations(var_code, target_date)
        if df.empty:
            continue
        sentinel = df[df["station_code"] == config.SENTINEL_STATION_CODE]
        if not sentinel.empty:
            sentinel = sentinel.sort_values("datetime")
            result[key] = sentinel[["datetime", "value"]].rename(
                columns={"value": key}
            )

        # Pluviòmetre local
        if var_code == config.XEMA_VAR_PRECIP:
            local = df[df["station_code"] == config.LOCAL_RAIN_STATION_CODE]
            if not local.empty:
                local = local.sort_values("datetime")
                result["local_rain_xema"] = local[["datetime", "value"]].rename(
                    columns={"value": "local_rain_xema"}
                )

    return result


def compute_sentinel_features(sentinel_data: dict, station_temp: float, station_humidity: float) -> dict:
    """
    Calcula features derivades de les dades sentinella vs. locals.
    - Diferencial de temperatura Granollers→Cardedeu (si baixa a Granollers primer = front s'acosta)
    - Diferencial d'humitat
    """
    features = {
        "sentinel_temp_diff": None,
        "sentinel_humidity_diff": None,
        "sentinel_precip": None,
    }

    s_temp = sentinel_data.get("sentinel_temp")
    s_hum = sentinel_data.get("sentinel_humidity")
    s_precip = sentinel_data.get("sentinel_precip")

    if s_temp is not None and station_temp is not None:
        # Si Granollers és més freda que Cardedeu → possible front fred entrant
        features["sentinel_temp_diff"] = station_temp - s_temp

    if s_hum is not None and station_humidity is not None:
        # Si Granollers té més humitat → aire humit s'acosta
        features["sentinel_humidity_diff"] = s_hum - station_humidity

    if s_precip is not None:
        features["sentinel_precip"] = s_precip

    features["sentinel_raining"] = int(s_precip is not None and s_precip > 0)
    features["local_rain_xema"] = sentinel_data.get("local_rain_xema")
    features["local_rain_xema_3h"] = sentinel_data.get("local_rain_xema_3h")

    return features


def _empty_sentinel() -> dict:
    return {
        "sentinel_temp": None,
        "sentinel_humidity": None,
        "sentinel_precip": None,
    }
