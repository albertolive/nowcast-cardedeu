#!/usr/bin/env python3
"""
Backfill de dades d'ensemble (acord entre models) al training dataset.

Descarrega prediccions històriques de 4 models (ECMWF, GFS, ICON, AROME)
via l'Historical Forecast API d'Open-Meteo (gratuïta, sense API key).
Calcula les mateixes features d'ensemble que predict.py usa en temps real.

Disponibilitat:
  - GFS: des de 2021
  - ICON: des de 2022-12
  - AROME: des de 2024-01
  - ECMWF IFS 0.25°: des de 2024-06

L'script calcula agreement amb els models disponibles per cada data.

Ús:
    .venv/bin/python scripts/backfill_ensemble.py

Característiques:
- Guarda progrés a data/raw/ensemble_historical.parquet (reprèn si s'interromp)
- Rate limiting: 1s entre crides API
- Chunks de 90 dies per crida
"""
import logging
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NowcastCardedeu/1.0 (research)"})

CACHE_PATH = os.path.join(config.DATA_RAW_DIR, "ensemble_historical.parquet")
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global", "meteofrance_arome_france0025"]
RAIN_THRESHOLD = 0.1  # mm in 6h for a model to "predict rain"
CHUNK_DAYS = 90
API_DELAY = 1.0  # seconds between calls


def fetch_ensemble_chunk(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch multi-model precipitation+temperature for a date range."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "precipitation,temperature_2m",
        "timezone": "Europe/Madrid",
        "models": ",".join(ENSEMBLE_MODELS),
    }

    r = SESSION.get(HISTORICAL_FORECAST_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def compute_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ensemble agreement features from multi-model data.
    Mirrors the logic in src/data/ensemble.py's fetch_ensemble_agreement().
    For each hour, looks at the next 6 hours of each model's precipitation.
    """
    rows = []

    # Get precipitation columns for each model
    precip_cols = {}
    temp_cols = {}
    for model in ENSEMBLE_MODELS:
        pc = f"precipitation_{model}"
        tc = f"temperature_2m_{model}"
        if pc in df.columns:
            precip_cols[model] = pc
        if tc in df.columns:
            temp_cols[model] = tc

    n_available = len(precip_cols)
    if n_available == 0:
        return pd.DataFrame()

    datetimes = df["datetime"].values

    for i in range(len(df)):
        # For each hour, look at the next 6 hours of each model's precipitation
        end_idx = min(i + 6, len(df))

        model_precip_6h = []
        model_temps = []

        for model, col in precip_cols.items():
            vals = df[col].iloc[i:end_idx].values
            non_null = vals[~pd.isna(vals)]
            if len(non_null) > 0:
                model_precip_6h.append(float(np.sum(non_null)))

        for model, col in temp_cols.items():
            val = df[col].iloc[i]
            if pd.notna(val):
                model_temps.append(float(val))

        n_models = len(model_precip_6h)
        if n_models == 0:
            rows.append({
                "datetime": datetimes[i],
                "ensemble_rain_agreement": np.nan,
                "ensemble_precip_spread": np.nan,
                "ensemble_temp_spread": np.nan,
                "ensemble_max_precip": np.nan,
                "ensemble_min_precip": np.nan,
                "ensemble_models_rain": np.nan,
            })
            continue

        rain_models = sum(1 for p in model_precip_6h if p >= RAIN_THRESHOLD)

        rows.append({
            "datetime": datetimes[i],
            "ensemble_rain_agreement": rain_models / n_models,
            "ensemble_precip_spread": max(model_precip_6h) - min(model_precip_6h),
            "ensemble_temp_spread": float(np.std(model_temps)) if len(model_temps) >= 2 else 0.0,
            "ensemble_max_precip": max(model_precip_6h),
            "ensemble_min_precip": min(model_precip_6h),
            "ensemble_models_rain": rain_models,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)

    # Determine date range: GFS available from 2022-01, full 4 models from 2024-06
    # Use 2022-01-01 as start (2+ models available), up to 5 days ago
    data_start = date(2022, 1, 1)
    data_end = date.today() - timedelta(days=5)

    # Resume from cache if exists
    if os.path.exists(CACHE_PATH):
        cached = pd.read_parquet(CACHE_PATH)
        cached["datetime"] = pd.to_datetime(cached["datetime"])
        last_date = cached["datetime"].max().date()
        logger.info(f"Cache found: {len(cached)} rows up to {last_date}")
        data_start = last_date + timedelta(days=1)
        if data_start >= data_end:
            logger.info("Cache is up to date, nothing to download")
            return
    else:
        cached = pd.DataFrame()

    logger.info(f"Downloading ensemble data: {data_start} → {data_end}")

    all_chunks = []
    chunk_start = data_start

    while chunk_start < data_end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), data_end)
        logger.info(f"  Chunk: {chunk_start} → {chunk_end}")

        try:
            raw_df = fetch_ensemble_chunk(chunk_start, chunk_end)
            if raw_df.empty:
                logger.warning(f"  No data for {chunk_start} → {chunk_end}")
                chunk_start = chunk_end + timedelta(days=1)
                continue

            features_df = compute_ensemble_features(raw_df)
            if not features_df.empty:
                all_chunks.append(features_df)
                logger.info(f"  → {len(features_df)} hours processed")
        except Exception as e:
            logger.warning(f"  Error: {e}")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(API_DELAY)

    if not all_chunks:
        logger.info("No new data downloaded")
        return

    new_data = pd.concat(all_chunks, ignore_index=True)

    # Merge with cache
    if not cached.empty:
        result = pd.concat([cached, new_data], ignore_index=True)
    else:
        result = new_data

    result = result.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    result.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Saved: {len(result)} total rows → {CACHE_PATH}")
    logger.info(f"  Period: {result['datetime'].min()} → {result['datetime'].max()}")

    # Stats
    has_agreement = result["ensemble_rain_agreement"].notna().sum()
    rain_events = (result["ensemble_rain_agreement"] > 0).sum()
    logger.info(f"  Valid: {has_agreement}/{len(result)} hours")
    logger.info(f"  Rain predicted by any model: {rain_events} hours ({100*rain_events/max(has_agreement,1):.1f}%)")


if __name__ == "__main__":
    main()
