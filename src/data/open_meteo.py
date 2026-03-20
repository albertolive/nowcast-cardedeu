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
from src.data._http import create_session

logger = logging.getLogger(__name__)

SESSION = create_session()


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
        import time as _time
        for attempt in range(6):
            r = SESSION.get(config.OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                logger.warning(f"Rate limited (429), esperant {wait}s...")
                _time.sleep(wait)
                continue
            break
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
    try:
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
            logger.warning(f"Open-Meteo: no hourly data in forecast response")
            return pd.DataFrame()

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["datetime"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        return df
    except Exception as e:
        logger.warning(f"Error obtenint forecast d'Open-Meteo: {e}")
        return pd.DataFrame()


def fetch_current_conditions() -> dict:
    """
    Obté les condicions actuals d'Open-Meteo (última hora + properes 2 hores).
    Retorna un diccionari amb les dades.
    """
    try:
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
    except Exception as e:
        logger.warning(f"Error obtenint condicions actuals d'Open-Meteo: {e}")
        return {}


# ── Dades a nivells de pressió (925/850/700/500/300 hPa) ──

PRESSURE_LEVEL_VARS = [
    # 925hPa — capa límit: low-level jet, inversions, flux d'humitat baix
    "temperature_925hPa",
    "relative_humidity_925hPa",
    "wind_speed_925hPa",
    "wind_direction_925hPa",
    # 850hPa — flux sinòptic, classificació de règims
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "temperature_850hPa",
    "relative_humidity_850hPa",
    # 700hPa — intrusió d'aire sec, capping
    "relative_humidity_700hPa",
    "temperature_700hPa",
    # 500hPa — aire fred, gradient tèrmic
    "temperature_500hPa",
    "geopotential_height_500hPa",
    # 300hPa — jet stream, cisalla profunda
    "wind_speed_300hPa",
    "wind_direction_300hPa",
    "geopotential_height_300hPa",
    # Convective parameters (available from Historical Forecast API, 2021-04+)
    "cape",
    "convective_inhibition",
    # Visibility & freezing level (Historical Forecast API only, 2021-04+)
    "visibility",
    "freezing_level_height",
    # Tier 2 — noves variables Historical Forecast (des d'abril 2021, ~44% cobertura)
    "lifted_index",                  # LI directe del NWP (millor que el derivat)
    "geopotential_height_850hPa",    # Topografia bàrica — depressió/anticicló
    "relative_humidity_500hPa",      # Humitat a 500hPa — intrusió seca mid-trop
    "wind_speed_700hPa",             # Vent de guia (steering level)
    "wind_direction_700hPa",         # Direcció del vent de guia
]

# Mapejat: noms de l'API → noms interns per al model
_PRESSURE_RENAME = {
    "temperature_925hPa": "temp_925",
    "relative_humidity_925hPa": "rh_925",
    "wind_speed_925hPa": "wind_925_speed",
    "wind_direction_925hPa": "wind_925_dir",
    "wind_speed_850hPa": "wind_850_speed",
    "wind_direction_850hPa": "wind_850_dir",
    "temperature_850hPa": "temp_850",
    "relative_humidity_850hPa": "rh_850",
    "relative_humidity_700hPa": "rh_700",
    "temperature_700hPa": "temp_700",
    "temperature_500hPa": "temp_500",
    "wind_speed_300hPa": "wind_300_speed",
    "wind_direction_300hPa": "wind_300_dir",
    "geopotential_height_300hPa": "gph_300",
    # Tier 2 — noves variables
    "geopotential_height_850hPa": "gph_850",
    "relative_humidity_500hPa": "rh_500",
    "wind_speed_700hPa": "wind_700_speed",
    "wind_direction_700hPa": "wind_700_dir",
}


def fetch_historical_pressure_levels(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Descarrega dades horàries de nivells de pressió (925/850/700/500/300 hPa)
    des de l'Historical Forecast API d'Open-Meteo.

    Disponible des d'abril 2021. Per dates anteriors retorna DataFrame buit.
    Les columnes es renomenen als noms interns del model.
    """
    # L'API té dades de pressure levels des de ~22-23 març 2021
    DATA_START = date(2021, 3, 23)
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
        import time as _time
        for attempt in range(3):
            r = SESSION.get(
                config.OPEN_METEO_HISTORICAL_FORECAST_URL,
                params=params,
                timeout=60,
            )
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning(f"Rate limited (429), esperant {wait}s...")
                _time.sleep(wait)
                continue
            break
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
        # Keep renamed pressure columns + passthrough columns (cape, convective_inhibition)
        _PASSTHROUGH_COLS = ["cape", "convective_inhibition", "visibility", "freezing_level_height", "lifted_index"]
        keep_cols = [c for c in _PRESSURE_RENAME.values() if c in df.columns]
        keep_cols += [c for c in _PASSTHROUGH_COLS if c in df.columns]
        df = df[["datetime"] + keep_cols]

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
    Obté dades a 5 nivells de pressió (925/850/700/500/300 hPa) del forecast d'Open-Meteo.

    925hPa (~750m): capa límit, low-level jet, inversions
    850hPa (~1500m): flux sinòptic, classificació de règims
    700hPa (~3000m): intrusió d'aire sec, capping
    500hPa (~5500m): aire fred, gradient tèrmic, VT/TT/LI
    300hPa (~9000m): jet stream, cisalla profunda
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

        # 925hPa — boundary layer
        temp_925 = _first("temperature_925hPa")
        rh_925 = _first("relative_humidity_925hPa")
        wind_925_speed = _first("wind_speed_925hPa")
        wind_925_dir = _first("wind_direction_925hPa")
        # 850hPa — synoptic flow
        wind_850_speed = _first("wind_speed_850hPa")
        wind_850_dir = _first("wind_direction_850hPa")
        temp_850 = _first("temperature_850hPa")
        rh_850 = _first("relative_humidity_850hPa")
        # 700hPa — dry air intrusion
        rh_700 = _first("relative_humidity_700hPa")
        temp_700 = _first("temperature_700hPa")
        # 500hPa — cold pool
        temp_500 = _first("temperature_500hPa")
        # 300hPa — jet stream
        wind_300_speed = _first("wind_speed_300hPa")
        wind_300_dir = _first("wind_direction_300hPa")
        gph_300 = _first("geopotential_height_300hPa")

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
            "temp_925": temp_925,
            "rh_925": rh_925,
            "wind_925_speed": wind_925_speed,
            "wind_925_dir": wind_925_dir,
            "wind_850_speed": wind_850_speed,
            "wind_850_dir": wind_850_dir,
            "temp_850": temp_850,
            "rh_850": rh_850,
            "rh_700": rh_700,
            "temp_700": temp_700,
            "temp_500": temp_500,
            "wind_300_speed": wind_300_speed,
            "wind_300_dir": wind_300_dir,
            "gph_300": gph_300,
            "vt_index": vt_index,
            "tt_index": tt_index,
            "li_index": li_index,
        }

        logger.info(
            f"Nivells pressió: 925={temp_925}°C, "
            f"850={wind_850_dir}°@{wind_850_speed}km/h T={temp_850}°C, "
            f"300={wind_300_speed}km/h, "
            f"VT={vt_index:.1f}, TT={tt_index:.1f}" if vt_index and tt_index else
            f"Nivells pressió: 850={wind_850_dir}°@{wind_850_speed}km/h"
        )
        return result

    except Exception as e:
        logger.warning(f"Error obtenint nivells de pressió: {e}")
        return {
            "temp_925": None,
            "rh_925": None,
            "wind_925_speed": None,
            "wind_925_dir": None,
            "wind_850_speed": None,
            "wind_850_dir": None,
            "temp_850": None,
            "rh_850": None,
            "rh_700": None,
            "temp_700": None,
            "temp_500": None,
            "wind_300_speed": None,
            "wind_300_dir": None,
            "gph_300": None,
            "vt_index": None,
            "tt_index": None,
            "li_index": None,
        }


def fetch_pressure_levels_hourly(hours_ahead: int = 48) -> pd.DataFrame:
    """
    Obté previsió horària de nivells de pressió (925/850/700/500/300 hPa).
    Retorna DataFrame amb totes les columnes de pressió renomenades.
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


# ── SST (Sea Surface Temperature) — Marine API ──

def fetch_sst_forecast() -> dict:
    """
    Obté la temperatura superficial del mar (SST) actual del Mediterrani
    proper a Cardedeu (costa Maresme) via l'API Marine d'Open-Meteo.

    Només disponible com a forecast (no històric).
    Les dades s'acumularan via feedback loop per a entrenament futur.
    """
    try:
        params = {
            "latitude": config.SEA_LATITUDE,
            "longitude": config.SEA_LONGITUDE,
            "hourly": "sea_surface_temperature",
            "timezone": "Europe/Madrid",
            "forecast_hours": 6,
        }
        r = SESSION.get(config.OPEN_METEO_MARINE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        vals = hourly.get("sea_surface_temperature", [])
        sst = vals[0] if vals and vals[0] is not None else None

        logger.info(f"SST Mediterrani: {sst}°C")
        return {"sst_med": sst}

    except Exception as e:
        logger.warning(f"Error obtenint SST: {e}")
        return {"sst_med": None}


# ── SST històric — NOAA OISST v2.1 (ERDDAP) ──

NOAA_ERDDAP_OISST_CSV = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"


def fetch_historical_sst(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Descarrega SST diària històrica del Mediterrani prop de Cardedeu
    des de NOAA OISST v2.1 via ERDDAP (gratuït, sense API key).

    Resolució: 0.25°, diària. Cobertura: 1981-present.
    Retorna DataFrame amb columnes: datetime, sst_med.
    Les dades diàries s'interpolen a horària al fer merge.
    """
    import time as _time
    import io

    all_rows = []
    chunk_start = start_date

    # Use 1-year chunks with CSV format (much faster than JSON)
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=365), end_date)

        lat = config.SEA_LATITUDE
        lon = config.SEA_LONGITUDE

        constraint = (
            f"?sst[({chunk_start.isoformat()}T00:00:00Z):1:({chunk_end.isoformat()}T00:00:00Z)]"
            f"[(0.0)][({lat})][({lon})]"
        )

        logger.info(f"NOAA OISST: {chunk_start} → {chunk_end}")

        r = None
        for attempt in range(3):
            try:
                r = requests.get(NOAA_ERDDAP_OISST_CSV + constraint, timeout=120)
                if r.status_code == 200:
                    break
                if r.status_code == 429:
                    _time.sleep(10 * (attempt + 1))
                    continue
                logger.warning(f"NOAA OISST HTTP {r.status_code} for {chunk_start}")
                break
            except Exception as e:
                logger.warning(f"NOAA OISST attempt {attempt+1}/3 error: {e}")
                _time.sleep(5)
                r = None

        try:
            if r is not None and r.status_code == 200:
                # CSV: skip units row (row 1), parse time/zlev/lat/lon/sst
                df_chunk = pd.read_csv(io.StringIO(r.text), skiprows=[1])
                df_chunk = df_chunk.rename(columns={"time": "datetime", "sst": "sst_med"})
                df_chunk = df_chunk[["datetime", "sst_med"]].dropna(subset=["sst_med"])
                all_rows.append(df_chunk)
        except Exception as e:
            logger.warning(f"Error parsing NOAA OISST response: {e}")

        chunk_start = chunk_end + timedelta(days=1)
        _time.sleep(0.5)

    if not all_rows:
        logger.warning("No SST data obtained from NOAA ERDDAP")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.sort_values("datetime").reset_index(drop=True)
    logger.info(f"NOAA SST: {len(result)} dies ({result['datetime'].min()} → {result['datetime'].max()})")
    return result
