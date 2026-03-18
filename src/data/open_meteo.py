"""
Client per a l'API d'Open-Meteo.
Proporciona dades horàries històriques i previsions actuals.
Cap API key necessària.
"""
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NowcastCardedeu/1.0 (research)"})


def fetch_historical_hourly(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Descarrega dades horàries històriques d'Open-Meteo per a Cardedeu.
    Retorna DataFrame amb columnes: datetime + totes les variables meteorològiques.
    L'API permet fins a ~2 anys per crida; per períodes llargs fem múltiples crides.
    """
    all_dfs = []
    chunk_start = start_date

    # Open-Meteo permet com a màxim ~366 dies per crida a l'arxiu
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=365), end_date)

        params = {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(config.OPEN_METEO_HOURLY_VARS),
            "timezone": "Europe/Madrid",
        }

        logger.info(f"Open-Meteo històric: {chunk_start} → {chunk_end}")
        r = SESSION.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        if "hourly" not in data:
            logger.warning(f"No hourly data for {chunk_start}-{chunk_end}")
            chunk_start = chunk_end + timedelta(days=1)
            continue

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        all_dfs.append(df)

        chunk_start = chunk_end + timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("datetime").reset_index(drop=True)
    return result


def fetch_forecast(hours_ahead: int = 48) -> pd.DataFrame:
    """
    Obté la previsió actual d'Open-Meteo per a les properes `hours_ahead` hores.
    Utilitza el model "best_match" que tria el millor model disponible automàticament.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": ",".join(config.OPEN_METEO_HOURLY_VARS),
        "timezone": "Europe/Madrid",
        "forecast_hours": hours_ahead,
        "models": ",".join(config.OPEN_METEO_FORECAST_MODELS),
    }

    r = SESSION.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data:
        raise ValueError(f"No forecast data: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def fetch_current_conditions() -> dict:
    """
    Obté les condicions actuals d'Open-Meteo (última hora + properes 2 hores).
    Retorna un diccionari amb les dades.
    """
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "surface_pressure",
            "precipitation",
            "rain",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "weather_code",
        ]),
        "timezone": "Europe/Madrid",
    }

    r = SESSION.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    return data.get("current", {})
