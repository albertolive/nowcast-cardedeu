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
    "relative_humidity_700hPa",
    "temperature_700hPa",
]

# Mapejat: noms de l'API → noms interns per al model
_PRESSURE_RENAME = {
    "wind_speed_850hPa": "wind_850_speed",
    "wind_direction_850hPa": "wind_850_dir",
    "temperature_850hPa": "temp_850",
    "temperature_500hPa": "temp_500",
    "relative_humidity_850hPa": "rh_850",
    "relative_humidity_700hPa": "rh_700",
    "temperature_700hPa": "temp_700",
}


def fetch_historical_pressure_levels(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Descarrega dades horàries de nivells de pressió (850/700/500hPa)
    des de l'Historical Forecast API d'Open-Meteo.

    Disponible des d'abril 2021. Per dates anteriors retorna DataFrame buit.
    Les columnes es renomenen als noms interns del model:
    wind_850_speed, wind_850_dir, temp_850, temp_500, rh_850, rh_700, temp_700.
    """
    # L'API té dades de pressure levels des d'abril 2021
    DATA_START = date(2021, 4, 1)
    effective_start = max(start_date, DATA_START)

    if effective_start >= end_date:
        logger.info(f"Pressure levels: no data before {DATA_START}, skipping")
        return pd.DataFrame()

    all_dfs = []
    chunk_start = effective_start

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=90), end_date)

        params = {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(PRESSURE_LEVEL_VARS),
            "timezone": "Europe/Madrid",
        }

        logger.info(f"Pressure levels històric: {chunk_start} → {chunk_end}")
        r = SESSION.get(
            config.OPEN_METEO_HISTORICAL_FORECAST_URL,
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        if not hourly:
            chunk_start = chunk_end + timedelta(days=1)
            continue

        df = pd.DataFrame(hourly)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])

        # Renomenar columnes API → noms interns
        df = df.rename(columns=_PRESSURE_RENAME)
        # Eliminar columnes no mapejades (ex: geopotential_height_500hPa)
        df = df[["datetime"] + [c for c in _PRESSURE_RENAME.values() if c in df.columns]]

        all_dfs.append(df)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("datetime").reset_index(drop=True)
    logger.info(f"Pressure levels: {len(result)} registres ({result['datetime'].min()} → {result['datetime'].max()})")
    return result


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
      - rh_700: humitat relativa a 700hPa (%) — clau per tempestes (ref: alexmeteo)
      - temp_700: temperatura a 700hPa (°C)
      - vt_index: Vertical Totals = T850 - T500 (índex d'inestabilitat)
      - tt_index: Total Totals = VT + (Td850 - T500)
      - li_index: Lifted Index (inestabilitat a 500hPa, negatiu = inestable)
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
        rh_700 = _first("relative_humidity_700hPa")
        temp_700 = _first("temperature_700hPa")

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

        # Lifted Index (LI) — mesura d'inestabilitat a 500hPa
        # Estima la temperatura d'una parcel·la d'aire superficial pujada a 500hPa
        # i la compara amb la temperatura ambient a 500hPa.
        # LI negatiu = inestable. LI < -3: tempestes, < -6: severes.
        # Ref: alexmeteo.com "Ingredients per formar Tempestes"
        li_index = None
        if temp_850 is not None and rh_850 is not None and temp_500 is not None:
            # Parcel·la des de 850hPa (aprox. superfície elevada a Cardedeu)
            # 1) Calcular temperatura de rosada a 850hPa
            a, b = 17.27, 237.7
            alpha_li = (a * temp_850) / (b + temp_850) + math.log(max(rh_850, 1) / 100.0)
            td_850_li = (b * alpha_li) / (a - alpha_li)
            # 2) LCL: nivell on la parcel·la satura (~125m per grau de depressió)
            #    Després, la parcel·la puja adiabàticament humida (~6°C/km)
            dew_dep = temp_850 - td_850_li
            lcl_height_m = 125 * dew_dep  # metres sobre 850hPa
            # 3) Adiabàtica seca fins LCL: ~9.8°C/km
            t_at_lcl = temp_850 - 9.8 * (lcl_height_m / 1000.0)
            # 4) Adiabàtica humida des de LCL fins 500hPa (~3500m sobre 850hPa)
            remaining_m = max(3500 - lcl_height_m, 0)
            t_parcel_500 = t_at_lcl - 6.0 * (remaining_m / 1000.0)
            # 5) LI = T_ambient(500) - T_parcel(500)
            li_index = temp_500 - t_parcel_500

        result = {
            "wind_850_speed": wind_850_speed,
            "wind_850_dir": wind_850_dir,
            "temp_850": temp_850,
            "temp_500": temp_500,
            "rh_850": rh_850,
            "rh_700": rh_700,
            "temp_700": temp_700,
            "vt_index": vt_index,
            "tt_index": tt_index,
            "li_index": li_index,
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
            "rh_700": None,
            "temp_700": None,
            "vt_index": None,
            "tt_index": None,
            "li_index": None,
        }


def fetch_pressure_levels_hourly(hours_ahead: int = 48) -> pd.DataFrame:
    """
    Obté previsió horària de nivells de pressió (850/700/500hPa).
    Retorna DataFrame amb columnes: datetime, wind_850_speed, wind_850_dir,
    temp_850, temp_500, rh_850, rh_700, temp_700.
    """
    try:
        params = {
            "latitude": config.LATITUDE,
            "longitude": config.LONGITUDE,
            "hourly": ",".join(PRESSURE_LEVEL_VARS),
            "timezone": "Europe/Madrid",
            "forecast_hours": hours_ahead,
        }

        r = SESSION.get(config.OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        if not hourly:
            return pd.DataFrame()

        df = pd.DataFrame(hourly)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df = df.rename(columns=_PRESSURE_RENAME)
        # Keep only renamed columns + datetime
        keep = ["datetime"] + [c for c in _PRESSURE_RENAME.values() if c in df.columns]
        return df[keep]

    except Exception as e:
        logger.warning(f"Error obtenint pressure levels horari: {e}")
        return pd.DataFrame()
