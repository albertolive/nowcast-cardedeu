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


# ── Dades a nivells de pressió (850hPa, 500hPa) ──

PRESSURE_LEVEL_VARS = [
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "temperature_850hPa",
    "temperature_500hPa",
    "geopotential_height_500hPa",
    "relative_humidity_850hPa",
]


def fetch_pressure_levels() -> dict:
    """
    Obté dades a 850hPa i 500hPa del forecast d'Open-Meteo.
    850hPa (~1500m) és el nivell estàndard per classificar règims sinòptics
    (Llevantada, Garbí, etc.). 500hPa per calcular Vertical Totals (VT).

    Retorna dict amb:
      - wind_850_speed: velocitat del vent a 850hPa (km/h)
      - wind_850_dir: direcció del vent a 850hPa (graus)
      - temp_850: temperatura a 850hPa (°C)
      - temp_500: temperatura a 500hPa (°C)
      - rh_850: humitat relativa a 850hPa (%)
      - vt_index: Vertical Totals = T850 - T500 (índex d'inestabilitat)
      - tt_index: Total Totals = VT + (Td850 - T500)
    """
    import numpy as np
    import math

    try:
        params = {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(PRESSURE_LEVEL_VARS),
            "timezone": "Europe/Madrid",
            "forecast_hours": 6,
        }

        r = SESSION.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})

        # Agafar primera hora (moment actual)
        def _first(key):
            vals = hourly.get(key, [])
            return vals[0] if vals and vals[0] is not None else None

        wind_850_speed = _first("wind_speed_850hPa")
        wind_850_dir = _first("wind_direction_850hPa")
        temp_850 = _first("temperature_850hPa")
        temp_500 = _first("temperature_500hPa")
        rh_850 = _first("relative_humidity_850hPa")

        # Vertical Totals (VT) — gradient tèrmic vertical
        # VT > 26: inestabilitat feble, > 30: inestabilitat clara, > 34: forta
        vt_index = (temp_850 - temp_500) if (temp_850 is not None and temp_500 is not None) else None

        # Total Totals (TT) = VT + Cross Totals (CT = Td850 - T500)
        # Necessitem punt de rosada a 850hPa: estimat a partir de T850 i RH850
        tt_index = None
        if vt_index is not None and rh_850 is not None and temp_850 is not None and temp_500 is not None:
            # Magnus formula per dew point a 850hPa
            a, b = 17.27, 237.7
            alpha = (a * temp_850) / (b + temp_850) + math.log(max(rh_850, 1) / 100.0)
            td_850 = (b * alpha) / (a - alpha)
            ct = td_850 - temp_500
            tt_index = vt_index + ct

        result = {
            "wind_850_speed": wind_850_speed,
            "wind_850_dir": wind_850_dir,
            "temp_850": temp_850,
            "temp_500": temp_500,
            "rh_850": rh_850,
            "vt_index": vt_index,
            "tt_index": tt_index,
        }

        logger.info(
            f"850hPa: vent {wind_850_dir}° @ {wind_850_speed} km/h, "
            f"T850={temp_850}°C, T500={temp_500}°C, "
            f"VT={vt_index:.1f}, TT={tt_index:.1f}" if vt_index and tt_index else
            f"850hPa: vent {wind_850_dir}° @ {wind_850_speed} km/h"
        )
        return result

    except Exception as e:
        logger.warning(f"Error obtenint dades 850hPa: {e}")
        return {
            "wind_850_speed": None,
            "wind_850_dir": None,
            "temp_850": None,
            "temp_500": None,
            "rh_850": None,
            "vt_index": None,
            "tt_index": None,
        }
