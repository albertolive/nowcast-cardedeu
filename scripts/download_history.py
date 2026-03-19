#!/usr/bin/env python3
"""
Script 1: Descarrega totes les dades històriques.
- NOAA daily data de meteocardedeu.net (2015-2026)
- Hourly data d'Open-Meteo per al mateix període
- Ensemble historical (multi-model agreement) d'Open-Meteo
- Combina tot en un dataset únic per entrenar el model
"""
import logging
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.meteocardedeu import download_all_history
from src.data.open_meteo import fetch_historical_hourly, fetch_historical_pressure_levels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    # ── 1. Dades diàries de meteocardedeu (NOAA) ──
    logger.info("=" * 60)
    logger.info("Descarregant històric NOAA de meteocardedeu.net...")
    logger.info("=" * 60)

    station_daily = download_all_history(config.HISTORY_YEARS)
    station_path = os.path.join(config.DATA_RAW_DIR, "station_daily.parquet")
    station_daily.to_parquet(station_path, index=False)
    logger.info(f"Dades estació: {len(station_daily)} dies → {station_path}")

    # ── 2. Dades horàries d'Open-Meteo ──
    logger.info("=" * 60)
    logger.info("Descarregant històric horari d'Open-Meteo...")
    logger.info("=" * 60)

    # Open-Meteo archive va des de 1940, però el seu arxiu complet
    # acostuma a anar fins fa 5-7 dies (no inclou els últims dies)
    start = date(min(config.HISTORY_YEARS), 1, 1)
    end = date.today() - timedelta(days=5)  # Archive API lags ~5 days

    hourly_data = fetch_historical_hourly(start, end)
    hourly_path = os.path.join(config.DATA_RAW_DIR, "open_meteo_hourly.parquet")
    hourly_data.to_parquet(hourly_path, index=False)
    logger.info(f"Dades Open-Meteo: {len(hourly_data)} hores → {hourly_path}")

    # ── 3. Dades de nivells de pressió (Historical Forecast API) ──
    logger.info("=" * 60)
    logger.info("Descarregant nivells de pressió (850/700/500hPa)...")
    logger.info("=" * 60)

    pressure_data = fetch_historical_pressure_levels(start, end)
    pressure_path = os.path.join(config.DATA_RAW_DIR, "pressure_levels_hourly.parquet")
    if not pressure_data.empty:
        pressure_data.to_parquet(pressure_path, index=False)
        logger.info(f"Pressure levels: {len(pressure_data)} hores → {pressure_path}")
    else:
        logger.warning("No s'han obtingut dades de pressure levels")

    logger.info("=" * 60)
    logger.info("Descàrrega completada!")
    logger.info(f"  Estació: {station_daily['date'].min()} → {station_daily['date'].max()}")
    logger.info(f"  Open-Meteo: {hourly_data['datetime'].min()} → {hourly_data['datetime'].max()}")
    if not pressure_data.empty:
        logger.info(f"  Pressure: {pressure_data['datetime'].min()} → {pressure_data['datetime'].max()}")
    logger.info("=" * 60)

    # ── 4. Ensemble historical (multi-model agreement) ──
    logger.info("=" * 60)
    logger.info("Descarregant ensemble historical (backfill_ensemble)...")
    logger.info("=" * 60)
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "backfill_ensemble.py")],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        # Show last few lines of output
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
