#!/usr/bin/env python3
"""
Backfill de dades sentinella XEMA (Granollers + ETAP Cardedeu) al training dataset.

Descarrega dades diàries de temperatura, humitat i precipitació de l'API Meteocat XEMA.
Cada crida a fetch_variable_all_stations retorna TOTES les estacions (~180) per a una
variable i una data — per tant podem extreure múltiples estacions d'una sola crida.

Costos API: 3 crides/dia (temp, humidity, precip). Amb 750 crides/mes free tier
i ~200-400 crides/mes pel predict en temps real, queden ~350-550 crides/mes
per backfill = ~115-180 dies per mes (~10-15 dies de backfill per execució diària).

Ús:
    METEOCAT_API_KEY=xxx .venv/bin/python scripts/backfill_xema.py [--max-days N]

Característiques:
- Guarda progrés a data/raw/xema_sentinel_cache.parquet (reprèn si s'interromp)
- Descarrega des del més recent cap enrere (dades recents = més valor)
- Rate limiting: 0.5s entre crides API
- Default: 15 dies per execució (45 API calls) — segur dins del budget
"""
import argparse
import logging
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.meteocat import fetch_variable_all_stations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(config.DATA_PROCESSED_DIR, "xema_sentinel_cache.parquet")
API_DELAY = 0.5  # seconds between calls


def fetch_sentinel_day(target_date: date) -> pd.DataFrame:
    """
    Fetch temp, humidity, precip for sentinel (YM) + local (KX) stations for one day.
    Returns a DataFrame with hourly rows and sentinel feature columns.
    Uses 3 API calls.
    """
    var_map = {
        config.XEMA_VAR_TEMP: "temp",
        config.XEMA_VAR_HUMIDITY: "humidity",
        config.XEMA_VAR_PRECIP: "precip",
    }

    station_codes = {config.SENTINEL_STATION_CODE, config.LOCAL_RAIN_STATION_CODE}
    all_data = {}

    for var_code, var_name in var_map.items():
        df = fetch_variable_all_stations(var_code, target_date)
        time.sleep(API_DELAY)

        if df.empty:
            continue

        # Filter to our stations only
        df = df[df["station_code"].isin(station_codes)]
        if df.empty:
            continue

        for code in station_codes:
            station_df = df[df["station_code"] == code].copy()
            if station_df.empty:
                continue
            station_df = station_df.sort_values("datetime")
            # Resample to hourly (if sub-hourly readings exist)
            station_df = station_df.set_index("datetime")
            if var_name == "precip":
                hourly = station_df["value"].resample("1h").sum()
            else:
                hourly = station_df["value"].resample("1h").mean()
            hourly = hourly.reset_index()
            hourly.columns = ["datetime", "value"]

            prefix = "sentinel" if code == config.SENTINEL_STATION_CODE else "local"
            key = f"{prefix}_{var_name}"
            all_data[key] = hourly.set_index("datetime")["value"]

    if not all_data:
        return pd.DataFrame()

    # Combine all series into one DataFrame
    result = pd.DataFrame(all_data)
    result.index.name = "datetime"
    result = result.reset_index()

    # Compute derived features
    if "sentinel_temp" in result.columns:
        result["sentinel_temp_val"] = result["sentinel_temp"]
    if "sentinel_humidity" in result.columns:
        result["sentinel_humidity_val"] = result["sentinel_humidity"]
    if "sentinel_precip" in result.columns:
        result["sentinel_precip_val"] = result["sentinel_precip"]
        result["sentinel_raining"] = (result["sentinel_precip"] > 0).astype(int)
    if "local_precip" in result.columns:
        result["local_rain_xema"] = result["local_precip"]
        # 3h rolling sum
        result["local_rain_xema_3h"] = result["local_precip"].rolling(3, min_periods=1).sum()

    return result


def compute_diffs(sentinel_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentinel feature diffs using station temperature/humidity from Open-Meteo data.
    This merges sentinel data with the hourly training data's temperature_2m and
    relative_humidity_2m columns.
    """
    if sentinel_df.empty:
        return sentinel_df

    sentinel_df = sentinel_df.copy()

    # Merge with hourly station data to get temp/humidity for diff computation
    if hourly_df is not None and not hourly_df.empty:
        hourly_sub = hourly_df[["datetime", "temperature_2m", "relative_humidity_2m"]].copy()
        hourly_sub["datetime"] = pd.to_datetime(hourly_sub["datetime"])
        sentinel_df["datetime"] = pd.to_datetime(sentinel_df["datetime"])

        merged = sentinel_df.merge(hourly_sub, on="datetime", how="left")

        if "sentinel_temp_val" in merged.columns and "temperature_2m" in merged.columns:
            merged["sentinel_temp_diff"] = merged["temperature_2m"] - merged["sentinel_temp_val"]

        if "sentinel_humidity_val" in merged.columns and "relative_humidity_2m" in merged.columns:
            merged["sentinel_humidity_diff"] = merged["sentinel_humidity_val"] - merged["relative_humidity_2m"]

        return merged

    return sentinel_df


def main():
    parser = argparse.ArgumentParser(description="Backfill XEMA sentinel data")
    parser.add_argument("--max-days", type=int, default=15,
                        help="Max days to download per run (default: 15 = 45 API calls)")
    parser.add_argument("--start-date", type=str, default="2021-04-01",
                        help="Earliest date to backfill (default: 2021-04-01)")
    args = parser.parse_args()

    if not config.METEOCAT_API_KEY:
        logger.error("METEOCAT_API_KEY not set")
        sys.exit(1)

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)

    earliest = date.fromisoformat(args.start_date)
    latest = date.today() - timedelta(days=1)  # Yesterday

    # Load cache to find what's already downloaded
    if os.path.exists(CACHE_PATH):
        cached = pd.read_parquet(CACHE_PATH)
        cached["datetime"] = pd.to_datetime(cached["datetime"])
        cached_dates = set(cached["datetime"].dt.date.unique())
        logger.info(f"Cache: {len(cached)} rows, {len(cached_dates)} unique days")
    else:
        cached = pd.DataFrame()
        cached_dates = set()

    # Build list of dates to download (most recent first = more value)
    all_dates = []
    d = latest
    while d >= earliest:
        if d not in cached_dates:
            all_dates.append(d)
        d -= timedelta(days=1)

    if not all_dates:
        logger.info("All dates already cached")
        return

    # Limit to max_days
    dates_to_fetch = all_dates[:args.max_days]
    logger.info(f"Dates to download: {len(dates_to_fetch)} (of {len(all_dates)} remaining)")
    logger.info(f"API calls: {len(dates_to_fetch) * 3} (budget: ~350-550/month for backfill)")

    new_rows = []
    for i, target in enumerate(dates_to_fetch):
        logger.info(f"  [{i+1}/{len(dates_to_fetch)}] {target}")
        try:
            day_df = fetch_sentinel_day(target)
            if not day_df.empty:
                new_rows.append(day_df)
                logger.info(f"    → {len(day_df)} hourly rows")
            else:
                logger.warning(f"    → No data")
        except Exception as e:
            logger.warning(f"    → Error: {e}")

    if not new_rows:
        logger.info("No new data downloaded")
        return

    new_data = pd.concat(new_rows, ignore_index=True)

    # Merge with cache
    if not cached.empty:
        result = pd.concat([cached, new_data], ignore_index=True)
    else:
        result = new_data

    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    result.to_parquet(CACHE_PATH, index=False)

    n_days = result["datetime"].dt.date.nunique()
    logger.info(f"Saved: {len(result)} rows ({n_days} days) → {CACHE_PATH}")
    logger.info(f"  Period: {result['datetime'].min()} → {result['datetime'].max()}")
    remaining = len(all_dates) - len(dates_to_fetch)
    if remaining > 0:
        logger.info(f"  Remaining: {remaining} days (run again to continue)")


if __name__ == "__main__":
    main()
