#!/usr/bin/env python3
"""
Script 1: Descarrega dades històriques (incremental).
- NOAA daily data de meteocardedeu.net (sempre complet — fitxers petits)
- Hourly data d'Open-Meteo (incremental: només des de l'últim registre)
- Pressure levels (incremental)
- SST (incremental)
- Ensemble historical (via backfill_ensemble.py, ja incremental)

Mode incremental: si existeix el parquet, carrega'l, busca max(datetime),
i només descarrega des d'aquella data - OVERLAP_DAYS fins avui.
Primera execució: descàrrega completa.
"""
import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.meteocardedeu import download_all_history
from src.data.open_meteo import fetch_historical_hourly, fetch_historical_pressure_levels, fetch_historical_sst

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Dies de solapament per cobrir possibles correccions retroactives de l'API
OVERLAP_DAYS = 7


def _incremental_start(parquet_path: str, dt_col: str = "datetime") -> date | None:
    """Retorna la data d'inici incremental, o None si cal descàrrega completa."""
    if not os.path.exists(parquet_path):
        return None
    try:
        existing = pd.read_parquet(parquet_path, columns=[dt_col])
        if existing.empty:
            return None
        max_dt = pd.to_datetime(existing[dt_col]).max()
        start = max_dt.date() - timedelta(days=OVERLAP_DAYS)
        return start
    except Exception as e:
        logger.warning(f"No es pot llegir {parquet_path} per incremental: {e}")
        return None


def _merge_and_save(existing_path: str, new_df: pd.DataFrame,
                    dt_col: str = "datetime") -> pd.DataFrame:
    """Combina dades existents + noves, elimina duplicats, desa."""
    if os.path.exists(existing_path) and not new_df.empty:
        try:
            existing = pd.read_parquet(existing_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    if combined.empty:
        return combined

    combined[dt_col] = pd.to_datetime(combined[dt_col])
    combined = combined.drop_duplicates(subset=[dt_col], keep="last")
    combined = combined.sort_values(dt_col).reset_index(drop=True)
    combined.to_parquet(existing_path, index=False)
    return combined


def main():
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    full_start = date(min(config.HISTORY_YEARS), 1, 1)
    end = date.today() - timedelta(days=5)  # Archive API lags ~5 days

    # ── 1. Dades diàries de meteocardedeu (NOAA) ──
    # Sempre complet: són fitxers de text petits i l'últim mes canvia
    logger.info("=" * 60)
    logger.info("Descarregant històric NOAA de meteocardedeu.net...")
    logger.info("=" * 60)

    station_daily = download_all_history(config.HISTORY_YEARS)
    station_path = os.path.join(config.DATA_RAW_DIR, "station_daily.parquet")
    station_daily.to_parquet(station_path, index=False)
    logger.info(f"Dades estació: {len(station_daily)} dies → {station_path}")

    # ── 2. Dades horàries d'Open-Meteo (INCREMENTAL) ──
    logger.info("=" * 60)
    hourly_path = os.path.join(config.DATA_RAW_DIR, "open_meteo_hourly.parquet")
    incr_start = _incremental_start(hourly_path)

    if incr_start is not None:
        logger.info(f"Open-Meteo incremental: {incr_start} → {end}")
    else:
        incr_start = full_start
        logger.info(f"Open-Meteo complet: {incr_start} → {end}")

    hourly_new = fetch_historical_hourly(incr_start, end)
    hourly_data = _merge_and_save(hourly_path, hourly_new)
    logger.info(f"Dades Open-Meteo: {len(hourly_data)} hores → {hourly_path}")

    # ── 3. Dades de nivells de pressió (INCREMENTAL) ──
    logger.info("=" * 60)
    pressure_path = os.path.join(config.DATA_RAW_DIR, "pressure_levels_hourly.parquet")
    incr_start_pl = _incremental_start(pressure_path)

    if incr_start_pl is not None:
        logger.info(f"Pressure levels incremental: {incr_start_pl} → {end}")
    else:
        incr_start_pl = full_start
        logger.info(f"Pressure levels complet: {incr_start_pl} → {end}")

    pressure_new = fetch_historical_pressure_levels(incr_start_pl, end)
    if not pressure_new.empty:
        pressure_data = _merge_and_save(pressure_path, pressure_new)
        logger.info(f"Pressure levels: {len(pressure_data)} hores → {pressure_path}")
    else:
        logger.warning("No s'han obtingut noves dades de pressure levels")

    logger.info("=" * 60)
    logger.info("Descàrrega completada!")
    logger.info(f"  Estació: {station_daily['date'].min()} → {station_daily['date'].max()}")
    logger.info(f"  Open-Meteo: {hourly_data['datetime'].min()} → {hourly_data['datetime'].max()}")
    if os.path.exists(pressure_path):
        pl = pd.read_parquet(pressure_path)
        logger.info(f"  Pressure: {pl['datetime'].min()} → {pl['datetime'].max()}")
    logger.info("=" * 60)

    # ── 4. SST històric (INCREMENTAL) ──
    logger.info("=" * 60)
    sst_path = os.path.join(config.DATA_RAW_DIR, "sst_historical.parquet")
    incr_start_sst = _incremental_start(sst_path)

    if incr_start_sst is not None:
        logger.info(f"SST incremental: {incr_start_sst} → {end}")
    else:
        incr_start_sst = full_start
        logger.info(f"SST complet: {incr_start_sst} → {end}")

    sst_new = fetch_historical_sst(incr_start_sst, end)
    if not sst_new.empty:
        sst_data = _merge_and_save(sst_path, sst_new)
        logger.info(f"SST historical: {len(sst_data)} dies → {sst_path}")
    else:
        logger.warning("No s'han obtingut noves dades de SST històric")

    # ── 5. Ensemble historical (ja incremental per disseny) ──
    logger.info("=" * 60)
    logger.info("Descarregant ensemble historical (backfill_ensemble)...")
    logger.info("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "backfill_ensemble.py")],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            logger.info(f"  {line}")
    else:
        logger.warning(f"Ensemble backfill failed: {result.stderr[-200:]}")

    logger.info("=" * 60)
    logger.info("Tot completat!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
