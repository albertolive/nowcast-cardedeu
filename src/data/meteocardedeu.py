"""
Client per a l'API de meteocardedeu.net.
Obté dades en temps real (minut a minut) i històriques (NOAA).
"""
import re
import time
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session

logger = logging.getLogger(__name__)

SESSION = create_session()


def fetch_latest() -> dict:
    """Retorna l'últim registre disponible de l'estació."""
    try:
        r = SESSION.get(config.LATEST_URL, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Error obtenint dades actuals de MeteoCardedeu: {e}")
        return {}


def fetch_series(hours: int = 24) -> pd.DataFrame:
    """
    Retorna un DataFrame amb dades minut-a-minut de les últimes `hours` hores.
    Columnes: ts, dt_local, TEMP, HUM, VEL, DIR, DIR_DEG, BAR, PREC, PINT, SUN, UVI
    """
    try:
        params = {
            "slug": config.SLUG,
            "hours": hours,
            "vars": config.SERIES_VARS,
            "keys": config.SERIES_VARS,
            "nocache": int(time.time() * 1000),
        }
        r = SESSION.get(config.SERIES_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not data.get("ok") or "rows" not in data:
            logger.warning(f"MeteoCardedeu API error (series): {data}")
            return pd.DataFrame()

        df = pd.DataFrame(data["rows"])
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["dt_local"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Error obtenint sèrie de MeteoCardedeu: {e}")
        return pd.DataFrame()


def fetch_history_list() -> list[dict]:
    """Retorna la llista de fitxers històrics disponibles."""
    r = SESSION.get(config.HISTORY_LIST_URL, params={"slug": config.SLUG}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise ValueError(f"History list error: {data}")
    return data["files"]


def fetch_history_file(filename: str) -> str:
    """Descarrega el contingut TXT (format NOAA) d'un fitxer històric."""
    r = SESSION.get(
        config.HISTORY_FILE_URL,
        params={"slug": config.SLUG, "file": filename},
        timeout=30,
    )
    r.raise_for_status()
    return r.text


# ── Conversió de direccions del vent a graus ──
_WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSO": 202.5, "SO": 225, "OSO": 247.5,
    "O": 270, "ONO": 292.5, "NO": 315, "NNO": 337.5,
    # Variants en anglès
    "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}


def wind_dir_to_degrees(direction: str) -> Optional[float]:
    return _WIND_DIR_MAP.get(direction.strip().upper()) if direction else None


def parse_noaa_monthly(text: str, year: int, month: int) -> pd.DataFrame:
    """
    Parseja un fitxer NOAA mensual i retorna un DataFrame amb dades diàries.
    Format: DIA  MIT  MÀX  HORA  MÍN  HORA  CAL.  FRED  PLUJ  VEL_MIT  VEL_MÀX  HORA  DIR_DOM
    """
    rows = []
    lines = text.split("\n")

    # Buscar les línies de dades (comencen amb un número de dia 1-31)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Les línies de dades comencen amb el dia (1-31)
        match = re.match(
            r"^\s*(\d{1,2})\s+"          # dia
            r"([\d.-]+)\s+"               # temp mitja
            r"([\d.-]+)\s+"               # temp max
            r"(\d{1,2}:\d{2})\s+"         # hora max
            r"([\d.-]+)\s+"               # temp min
            r"(\d{1,2}:\d{2}|0?0:00)\s+"  # hora min
            r"([\d.-]+)\s+"               # graus calor
            r"([\d.-]+)\s+"               # graus fred
            r"([\d.-]+)\s+"               # pluja mm
            r"([\d.-]+)\s+"               # vel mitja vent
            r"([\d.-]+)\s+"               # vel max vent
            r"(\d{1,2}:\d{2}|0?0:00)\s+"  # hora max vent
            r"(\w+)",                     # dir dominant
            line,
        )
        if match:
            day = int(match.group(1))
            if day < 1 or day > 31:
                continue
            try:
                date = datetime(year, month, day)
            except ValueError:
                continue
            rows.append({
                "date": date,
                "temp_mean": float(match.group(2)),
                "temp_max": float(match.group(3)),
                "temp_min": float(match.group(5)),
                "heating_dd": float(match.group(7)),
                "cooling_dd": float(match.group(8)),
                "rain_mm": float(match.group(9)),
                "wind_mean_kmh": float(match.group(10)),
                "wind_max_kmh": float(match.group(11)),
                "wind_dir_dominant": match.group(13).strip(),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["wind_dir_deg"] = df["wind_dir_dominant"].apply(wind_dir_to_degrees)
    return df


def download_all_history(years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Descarrega tots els fitxers mensuals, els parseja i retorna un DataFrame
    amb dades diàries de tots els anys.
    """
    if years is None:
        years = config.HISTORY_YEARS

    file_list = fetch_history_list()
    monthly_files = [f for f in file_list if len(f["file"]) == 8]  # ex: "0325.TXT"

    all_dfs = []
    for finfo in monthly_files:
        fname = finfo["file"]
        # Extreure mes i any: "0325.TXT" → mes=3, any=2025
        match = re.match(r"^(\d{2})(\d{2})\.TXT$", fname)
        if not match:
            continue
        month = int(match.group(1))
        year_short = int(match.group(2))
        year = 2000 + year_short

        if year not in years:
            continue
        if month < 1 or month > 12:
            continue

        logger.info(f"Descarregant {fname} ({month:02d}/{year})...")
        try:
            text = fetch_history_file(fname)
            df = parse_noaa_monthly(text, year, month)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error processant {fname}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)
    result = result.drop_duplicates(subset=["date"], keep="last")
    return result
