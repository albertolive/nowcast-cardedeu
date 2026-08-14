"""
Client per a l'API del Meteocat (XEMA).
Obté dades de l'estació sentinella de Granollers i del pluviòmetre ETAP Cardedeu.
Documentació: https://apidocs.meteocat.gencat.cat/
"""
import logging
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session

logger = logging.getLogger(__name__)

SESSION = create_session(retry_429=False)


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def _rows_to_dataframe(rows, *, stale: bool = False) -> pd.DataFrame:
    """Restore dataframe types after JSON cache serialization."""
    df = pd.DataFrame(rows)
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    df.attrs["xema_stale"] = stale
    return df


def _parse_xema_response(data) -> list[dict]:
    """Aplana la resposta JSON de XEMA en files (estació, variable, lectura).

    Tant /variables/mesurades/{var}/{y}/{m}/{d} (totes les estacions, una
    variable) com /estacions/mesurades/{estacio}/{y}/{m}/{d} (una estació,
    totes les variables) retornen una llista d'objectes `codi` + `variables`,
    on cada variable té una llista `lectures`. Aquest parser les unifica.
    """
    rows = []
    for station_data in data:
        station_code = station_data.get("codi", "")
        for var_info in station_data.get("variables", []):
            variable_code = var_info.get("codi")
            for lecture in var_info.get("lectures", []):
                rows.append({
                    "station_code": station_code,
                    "variable_code": variable_code,
                    "datetime": pd.to_datetime(lecture["data"]).isoformat(),
                    "value": lecture.get("valor"),
                    "estat": lecture.get("estat", "").strip(),
                })
    return rows


def _fetch_xema(cache_key: str, url: str) -> pd.DataFrame:
    """Comparteix memòria cau TTL + tallafoc 429 + HTTP + parseig per a XEMA."""
    from src.data.meteocat_cache import (
        get_cached, set_cached, get_stale_cached, is_meteocat_rate_limited,
        mark_meteocat_rate_limited,
    )
    # Un 429 és un senyal de servei, no una observació que falti. Es persisteix un
    # tallafoc curt perquè la propera predicció (10 min després) no repeteixi la
    # mateixa petició rebutjada per a les altres variables/estacions.
    if is_meteocat_rate_limited("xema"):
        stale = get_stale_cached(cache_key, config.METEOCAT_XEMA_STALE_MAX_MIN)
        if stale is not None:
            logger.warning(
                "XEMA cooldown active — reusing %s-minute-old data for %s",
                config.METEOCAT_XEMA_STALE_MAX_MIN, cache_key,
            )
            return _rows_to_dataframe(stale, stale=True)
        logger.warning("XEMA cooldown active after HTTP 429 — skipping %s", cache_key)
        return pd.DataFrame()

    # Respostes buides i amb dades es reutilitzen 60 min; les prediccions segueixen
    # executant-se cada 10 min.
    cached = get_cached(cache_key, config.METEOCAT_CACHE_TTL_XEMA_EMPTY)
    if cached is not None:
        if not cached:
            logger.debug("XEMA cache hit (empty): %s", cache_key)
            return pd.DataFrame()
        cached_fresh = get_cached(cache_key, config.METEOCAT_CACHE_TTL_XEMA)
        if cached_fresh is not None:
            logger.debug("XEMA cache hit: %s", cache_key)
            return _rows_to_dataframe(cached_fresh)
        # Caducat al TTL normal però dins del TTL "empty" — tornar a demanar.

    r = None
    try:
        r = SESSION.get(url, headers=_headers(), timeout=20)
        r.raise_for_status()
    except Exception as e:
        response = getattr(e, "response", None)
        status_code = getattr(response, "status_code", None) or getattr(r, "status_code", None)
        if status_code == 429:
            mark_meteocat_rate_limited("xema")
            stale = get_stale_cached(cache_key, config.METEOCAT_XEMA_STALE_MAX_MIN)
            logger.warning(
                "Meteocat XEMA HTTP 429 (%s) — cooldown for %s minutes",
                cache_key, config.METEOCAT_XEMA_429_COOLDOWN_MIN,
            )
            return _rows_to_dataframe(stale, stale=True) if stale is not None else pd.DataFrame()
        logger.warning("Meteocat API error (%s): %s", cache_key, e)
        return pd.DataFrame()

    rows = _parse_xema_response(r.json())
    # Filtrar lectures invàlides ABANS de desar a la memòria cau, perquè un hit de
    # cau mai no torni a servir un valor "T" (no disponible).
    rows = [row for row in rows if row["estat"] != "T"]
    set_cached(cache_key, rows)
    df = pd.DataFrame(rows)
    return _rows_to_dataframe(df.to_dict("records"))


def fetch_variable_all_stations(var_code: int, target_date: date) -> pd.DataFrame:
    """
    Obté les dades d'una variable per a TOTES les estacions en un dia.
    Endpoint: /xema/v1/variables/mesurades/{var_code}/{YYYY}/{MM}/{DD}
    Retorna DataFrame amb columnes: station_code, variable_code, datetime, value.

    Preferiu fetch_station_all_variables(): descarrega totes les variables d'una
    estació en una sola crida, que és el que realment necessita el nowcast.
    """
    if not _is_configured():
        logger.warning("Meteocat API key no configurada")
        return pd.DataFrame()
    cache_key = f"xema_{var_code}_{target_date}"
    url = (
        f"{config.METEOCAT_BASE_URL}/xema/v1/variables/mesurades/"
        f"{var_code}/{target_date.year}/{target_date.month:02d}/{target_date.day:02d}"
    )
    return _fetch_xema(cache_key, url)


def fetch_station_all_variables(station_code: str, target_date: date) -> pd.DataFrame:
    """
    Obté TOTES les variables d'UNA estació per a un dia concret.
    Endpoint: /xema/v1/estacions/mesurades/{station_code}/{YYYY}/{MM}/{DD}
    Retorna DataFrame amb columnes: station_code, variable_code, datetime, value.

    Recomanat per l'SMC (agost 2026): en lloc de 3 crides (una per variable sobre
    ~200 estacions) fem 2 crides (una per estació, totes les variables).
    """
    if not _is_configured():
        logger.warning("Meteocat API key no configurada")
        return pd.DataFrame()
    cache_key = f"xema_station_{station_code}_{target_date}"
    url = (
        f"{config.METEOCAT_BASE_URL}/xema/v1/estacions/mesurades/"
        f"{station_code}/{target_date.year}/{target_date.month:02d}/{target_date.day:02d}"
    )
    return _fetch_xema(cache_key, url)


def _empty_sentinel() -> dict:
    return {"sentinel_temp": None, "sentinel_humidity": None, "sentinel_precip": None}


def fetch_sentinel_latest() -> dict:
    """
    Obté les últimes lectures de l'estació sentinella (Granollers).
    Retorna un dict amb les dades més recents de temperatura, humitat i precipitació.
    """
    if not _is_configured():
        return _empty_sentinel()
    try:
        return _fetch_sentinel_latest_inner()
    except Exception as e:
        logger.warning(f"Error obtenint dades sentinella: {e}")
        return _empty_sentinel()


def _variable_rows(df: pd.DataFrame, var_code: int) -> pd.DataFrame:
    """Filtra les files d'una variable dins un DataFrame d'estació-dia."""
    if df is None or df.empty or "variable_code" not in df.columns:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    return df[df["variable_code"] == var_code]


def _last_reading(df: pd.DataFrame):
    """Retorna (valor, timestamp) de l'última lectura d'un DataFrame."""
    if df is None or df.empty:
        return None, None
    row = df.sort_values("datetime").iloc[-1]
    return float(row["value"]), row["datetime"]


def _fetch_sentinel_latest_inner() -> dict:

    today = date.today()
    yesterday = today - timedelta(days=1)
    result = {}

    # Temperatura (32), Humitat (33), Precipitació (35)
    var_map = {
        config.XEMA_VAR_TEMP: "sentinel_temp",
        config.XEMA_VAR_HUMIDITY: "sentinel_humidity",
        config.XEMA_VAR_PRECIP: "sentinel_precip",
    }

    sentinel_today = fetch_station_all_variables(config.SENTINEL_STATION_CODE, today)
    sentinel_yesterday = None  # lazy: només es demana si avui no té cap variable
    for var_code, key in var_map.items():
        df = sentinel_today
        # Stale rows are useful as an explicit fallback for callers, but must not
        # drive a "plou ara" or front-arrival signal in the live predictor.
        if df.attrs.get("xema_stale"):
            logger.warning("XEMA %s data stale — ignoring it for live sentinel features", key)
            df = pd.DataFrame()
        value, timestamp = _last_reading(_variable_rows(df, var_code))
        # Fallback a ahir si avui no té dades (retard de publicació XEMA)
        if value is None:
            logger.info(f"XEMA {key}: sense dades avui ({today}), provant ahir ({yesterday})")
            if sentinel_yesterday is None:
                sentinel_yesterday = fetch_station_all_variables(config.SENTINEL_STATION_CODE, yesterday)
            if not sentinel_yesterday.attrs.get("xema_stale"):
                value, timestamp = _last_reading(_variable_rows(sentinel_yesterday, var_code))

        if value is None:
            logger.warning(f"XEMA {key}: sense dades ni avui ni ahir")
            result[key] = None
            continue

        result[key] = value
        result[f"{key}_time"] = timestamp.isoformat()

    # Pluviòmetre local (ETAP Cardedeu KX): mateixa estratègia, només precipitació.
    kx_today = fetch_station_all_variables(config.LOCAL_RAIN_STATION_CODE, today)
    if kx_today.attrs.get("xema_stale"):
        logger.warning("XEMA local precipitation is stale — ignoring it for live features")
        kx_today = pd.DataFrame()
    local = _variable_rows(kx_today, config.XEMA_VAR_PRECIP)
    if local.empty:
        kx_yesterday = fetch_station_all_variables(config.LOCAL_RAIN_STATION_CODE, yesterday)
        if not kx_yesterday.attrs.get("xema_stale"):
            local = _variable_rows(kx_yesterday, config.XEMA_VAR_PRECIP)
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
    Usa 2 crides (una per estació) en lloc de 3 (una per variable).
    """
    if not _is_configured():
        return {}

    result = {}
    var_map = {
        config.XEMA_VAR_TEMP: "sentinel_temp",
        config.XEMA_VAR_HUMIDITY: "sentinel_humidity",
        config.XEMA_VAR_PRECIP: "sentinel_precip",
    }

    sentinel = fetch_station_all_variables(config.SENTINEL_STATION_CODE, target_date)
    if sentinel.empty or sentinel.attrs.get("xema_stale"):
        logger.warning("XEMA historical data unavailable or stale for %s", target_date)
    else:
        for var_code, key in var_map.items():
            var_df = _variable_rows(sentinel, var_code)
            if not var_df.empty:
                var_df = var_df.sort_values("datetime")
                result[key] = var_df[["datetime", "value"]].rename(columns={"value": key})

    # Pluviòmetre local (ETAP Cardedeu KX) — només precipitació
    local_df = fetch_station_all_variables(config.LOCAL_RAIN_STATION_CODE, target_date)
    if not (local_df.empty or local_df.attrs.get("xema_stale")):
        local = _variable_rows(local_df, config.XEMA_VAR_PRECIP)
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
    try:
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

        if s_hum is not None and station_humidity is not None and not np.isnan(station_humidity):
            # Si Granollers té més humitat → aire humit s'acosta
            features["sentinel_humidity_diff"] = s_hum - station_humidity

        if s_precip is not None:
            features["sentinel_precip"] = s_precip

        features["sentinel_raining"] = int(s_precip is not None and s_precip > 0)
        features["local_rain_xema"] = sentinel_data.get("local_rain_xema")
        features["local_rain_xema_3h"] = sentinel_data.get("local_rain_xema_3h")

        return features
    except Exception as e:
        logger.warning(f"Error calculant features sentinella: {e}")
        return {
            "sentinel_temp_diff": None,
            "sentinel_humidity_diff": None,
            "sentinel_precip": None,
            "sentinel_raining": 0,
            "local_rain_xema": None,
            "local_rain_xema_3h": None,
        }


def fetch_kx_precipitation_series(hours: int = 3) -> pd.DataFrame:
    """
    Obté les dades de precipitació del pluviòmetre XEMA KX (La Roca - ETAP Cardedeu)
    per les últimes `hours` hores. Fallback per a verificació quan MeteoCardedeu.net no respon.

    Retorna DataFrame amb columnes: datetime, PREC
    Compatible amb el format que usa verify.py.
    Consumeix 1 crida XEMA (estació KX, totes les variables, dia actual).
    """
    if not _is_configured():
        logger.warning("Meteocat API key no configurada — no es pot usar KX com a fallback")
        return pd.DataFrame()

    try:
        today = date.today()
        df = fetch_station_all_variables(config.LOCAL_RAIN_STATION_CODE, today)
        if df.empty or df.attrs.get("xema_stale"):
            logger.warning("KX fallback: sense dades fresques de precipitació XEMA avui")
            return pd.DataFrame()

        # Filtrar per la variable de precipitació (35)
        kx = _variable_rows(df, config.XEMA_VAR_PRECIP).copy()
        if kx.empty:
            logger.warning("KX fallback: estació KX no trobada a les dades XEMA")
            return pd.DataFrame()

        kx = kx.sort_values("datetime")
        kx["datetime"] = pd.to_datetime(kx["datetime"]).dt.tz_localize(None)

        # Filtrar a les últimes N hores
        cutoff = datetime.now() - timedelta(hours=hours)
        kx = kx[kx["datetime"] >= cutoff]

        # Retornar en format compatible amb verify.py (columnes: datetime, PREC)
        result = kx[["datetime", "value"]].rename(columns={"value": "PREC"})
        result = result.reset_index(drop=True)
        logger.info(f"KX fallback: {len(result)} lectures de precipitació (últimes {hours}h)")
        return result

    except Exception as e:
        logger.warning(f"Error obtenint precipitació KX per fallback: {e}")
        return pd.DataFrame()
