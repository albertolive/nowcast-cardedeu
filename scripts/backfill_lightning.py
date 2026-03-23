#!/usr/bin/env python3
"""
Backfill de dades de llamps (XDDE Meteocat) al training dataset.

Itera per cada dia únic del dataset, crida l'API XDDE per obtenir els llamps
d'aquell dia, i calcula les 7 features de llamps per cada hora.

Ús:
    METEOCAT_API_KEY=xxx .venv/bin/python scripts/backfill_lightning.py

Característiques:
- Guarda progrés a data/processed/lightning_cache.parquet (reprèn si s'interromp)
- Rate limiting: 0.3s entre crides API
- Gestiona fronteres de dia (hores 00-02 necessiten llamps del dia anterior)
"""
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data._http import create_session
from src.data._geo import _haversine_km, _bearing_deg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SESSION = create_session()
CACHE_PATH = os.path.join(config.DATA_PROCESSED_DIR, "lightning_cache.parquet")
DATASET_PATH = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
HOURS_BACK = 3.0
API_DELAY = 0.3  # Seconds between API calls


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


class QuotaExhaustedError(Exception):
    pass


def fetch_xdde_day(target_date: date) -> list[dict]:
    """Fetch all lightning strikes for a single day in Catalunya (all 24 hours)."""
    all_strikes = []
    for hour in range(24):
        url = (
            f"{config.METEOCAT_BASE_URL}/xdde/v1/catalunya/"
            f"{target_date.year}/{target_date.month:02d}/{target_date.day:02d}/{hour:02d}"
        )
        try:
            r = SESSION.get(url, headers=_headers(), timeout=30)
            if r.status_code == 404:
                continue
            if r.status_code == 429:
                raise QuotaExhaustedError(f"XDDE quota exhausted (429) at {target_date} {hour:02d}h")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                all_strikes.extend(data)
        except QuotaExhaustedError:
            raise
        except requests.exceptions.HTTPError as e:
            if "400" in str(e) or "404" in str(e):
                continue
            logger.warning(f"XDDE error ({target_date} {hour:02d}h): {e}")
        except Exception as e:
            logger.warning(f"XDDE error ({target_date} {hour:02d}h): {e}")
    return all_strikes


def parse_strikes(raw_strikes: list[dict]) -> list[dict]:
    """Parse raw API strikes into a standardized list with distances pre-computed."""
    cardedeu_lat = config.LATITUDE
    cardedeu_lon = config.LONGITUDE
    parsed = []
    for s in raw_strikes:
        coords = s.get("coordenades", {})
        lat = coords.get("latitud")
        lon = coords.get("longitud")
        if lat is None or lon is None:
            continue
        strike_time = s.get("data", "")
        try:
            st = datetime.fromisoformat(strike_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        dist = _haversine_km(cardedeu_lat, cardedeu_lon, lat, lon)
        if dist > config.RADAR_SCAN_RADIUS_KM:
            continue
        parsed.append({
            "timestamp": st.timestamp(),
            "dist_km": dist,
            "bearing": _bearing_deg(cardedeu_lat, cardedeu_lon, lat, lon),
            "cloud_ground": s.get("nuvolTerra", False),
            "current_ka": abs(s.get("correntPic", 0)),
        })
    return parsed


def compute_features_for_hour(
    hour_dt: datetime,
    nearby_strikes: list[dict],
) -> dict:
    """
    Compute lightning features for a specific hour using pre-parsed strikes.
    Only uses strikes in the [hour - HOURS_BACK, hour] window.
    """
    hour_ts = hour_dt.timestamp()
    cutoff_ts = hour_ts - (HOURS_BACK * 3600)

    # Filter to window
    window = [s for s in nearby_strikes if cutoff_ts <= s["timestamp"] <= hour_ts]

    if not window:
        return {
            "lightning_count_30km": 0,
            "lightning_count_15km": 0,
            "lightning_nearest_km": config.RADAR_SCAN_RADIUS_KM,
            "lightning_cloud_ground": 0,
            "lightning_max_current_ka": 0.0,
            "lightning_approaching": 0,
            "lightning_has_activity": 0,
        }

    window.sort(key=lambda x: x["dist_km"])
    nearest = window[0]
    count_30km = sum(1 for s in window if s["dist_km"] <= 30)
    count_15km = sum(1 for s in window if s["dist_km"] <= 15)
    cg_count = sum(1 for s in window if s["cloud_ground"])
    max_current = max(s["current_ka"] for s in window)

    # Approaching: compare recent (last hour) vs older strikes
    recent_cutoff = hour_ts - 3600
    recent = [s for s in window if s["timestamp"] >= recent_cutoff]
    older = [s for s in window if s["timestamp"] < recent_cutoff]

    approaching = 0
    if recent and older:
        avg_recent = sum(s["dist_km"] for s in recent) / len(recent)
        avg_older = sum(s["dist_km"] for s in older) / len(older)
        approaching = int(avg_recent < avg_older - 2)

    return {
        "lightning_count_30km": count_30km,
        "lightning_count_15km": count_15km,
        "lightning_nearest_km": round(nearest["dist_km"], 1),
        "lightning_cloud_ground": cg_count,
        "lightning_max_current_ka": round(max_current, 1),
        "lightning_approaching": approaching,
        "lightning_has_activity": 1,
    }


def main():
    if not config.METEOCAT_API_KEY:
        logger.error("METEOCAT_API_KEY no configurada! Ús: METEOCAT_API_KEY=xxx .venv/bin/python scripts/backfill_lightning.py")
        sys.exit(1)

    # Check XDDE quota before running
    from src.data.meteocat_cache import get_remaining
    remaining_quota = get_remaining("XDDE_250")
    if remaining_quota == 0:
        logger.warning("XDDE quota exhausted (0 remaining). Skipping backfill until next month.")
        return
    if remaining_quota > 0:
        # Each day = 24 API calls. Reserve 50 for real-time predictions.
        available = max(0, remaining_quota - 50)
        max_days_by_quota = available // 24
        logger.info(f"XDDE quota: {remaining_quota} remaining, {available} available for backfill = {max_days_by_quota} days")
        if max_days_by_quota == 0:
            logger.warning(f"XDDE quota too low for backfill ({remaining_quota} remaining, need 50 reserve).")
            return
    else:
        max_days_by_quota = None  # Unknown quota, proceed cautiously

    # Load training dataset
    logger.info("Carregant training dataset...")
    df = pd.read_parquet(DATASET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    logger.info(f"Dataset: {len(df)} mostres, {df['datetime'].dt.date.nunique()} dies únics")

    # Get unique dates we need to process
    all_dates = sorted(df["datetime"].dt.date.unique())

    # Load cache if exists (resume support)
    if os.path.exists(CACHE_PATH):
        cache_df = pd.read_parquet(CACHE_PATH)
        cache_df["datetime"] = pd.to_datetime(cache_df["datetime"])
        done_dates = set(cache_df["datetime"].dt.date.unique())
        logger.info(f"Reprèn: {len(done_dates)} dies ja processats, {len(all_dates) - len(done_dates)} pendents")
    else:
        cache_df = pd.DataFrame()
        done_dates = set()

    # Dates to process (limited by quota, most recent first = more training value)
    pending_dates = [d for d in reversed(all_dates) if d not in done_dates]
    if max_days_by_quota is not None and len(pending_dates) > max_days_by_quota:
        pending_dates = pending_dates[:max_days_by_quota]
        logger.info(f"Limited to {max_days_by_quota} days by XDDE quota")
    if not pending_dates:
        logger.info("Tots els dies ja processats!")
    else:
        logger.info(f"Processant {len(pending_dates)} dies ({pending_dates[0]} → {pending_dates[-1]})...")

    # Pre-fetch cache for previous day's strikes (needed for early morning hours)
    prev_day_strikes: dict[date, list[dict]] = {}
    batch_results = []
    errors = 0
    no_data_dates = 0

    for i, d in enumerate(pending_dates):
        # Progress
        if i % 100 == 0 and i > 0:
            logger.info(f"  Progrés: {i}/{len(pending_dates)} dies ({100*i/len(pending_dates):.1f}%), "
                        f"errors={errors}, sense dades={no_data_dates}")
            # Save intermediate progress
            if batch_results:
                _save_cache(cache_df, batch_results)
                cache_df = pd.read_parquet(CACHE_PATH)
                cache_df["datetime"] = pd.to_datetime(cache_df["datetime"])
                batch_results = []

        # Fetch today's strikes
        try:
            raw_today = fetch_xdde_day(d)
        except QuotaExhaustedError as e:
            logger.warning(f"Stopping backfill: {e}")
            break
        time.sleep(API_DELAY)

        strikes_today = parse_strikes(raw_today) if raw_today else []

        # Fetch previous day if needed (for hours 00-02 that look back 3h)
        prev_d = d - timedelta(days=1)
        if prev_d not in prev_day_strikes:
            if prev_d >= all_dates[0]:
                try:
                    raw_prev = fetch_xdde_day(prev_d)
                except QuotaExhaustedError as e:
                    logger.warning(f"Stopping backfill: {e}")
                    break
                time.sleep(API_DELAY)
                prev_day_strikes[prev_d] = parse_strikes(raw_prev) if raw_prev else []
            else:
                prev_day_strikes[prev_d] = []

        # Combine today + previous day strikes
        combined_strikes = strikes_today + prev_day_strikes.get(prev_d, [])

        #  Cache today for tomorrow's early morning
        prev_day_strikes[d] = strikes_today
        # Clean old entries to save memory
        old_keys = [k for k in prev_day_strikes if k < prev_d]
        for k in old_keys:
            del prev_day_strikes[k]

        if not raw_today:
            no_data_dates += 1

        # Get all hours for this date
        day_mask = df["datetime"].dt.date == d
        day_hours = df.loc[day_mask, "datetime"].tolist()

        for hour_dt in day_hours:
            # Make hour_dt timezone-aware (UTC) for timestamp comparison
            hour_dt_utc = hour_dt.replace(tzinfo=timezone.utc)
            features = compute_features_for_hour(hour_dt_utc, combined_strikes)
            features["datetime"] = hour_dt
            batch_results.append(features)

    # Final save
    if batch_results:
        _save_cache(cache_df, batch_results)
        cache_df = pd.read_parquet(CACHE_PATH)
        cache_df["datetime"] = pd.to_datetime(cache_df["datetime"])

    logger.info(f"Backfill completat: {len(cache_df)} files de features de llamps")
    logger.info(f"  Dies sense dades XDDE: {no_data_dates}/{len(pending_dates)}")

    # Merge into training dataset
    logger.info("Fusionant amb el training dataset...")
    lightning_cols = [
        "lightning_count_30km", "lightning_count_15km", "lightning_nearest_km",
        "lightning_cloud_ground", "lightning_max_current_ka",
        "lightning_approaching", "lightning_has_activity",
    ]

    # Drop existing lightning columns if any
    for col in lightning_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Merge on datetime
    cache_df["datetime"] = pd.to_datetime(cache_df["datetime"])
    df = df.merge(cache_df[["datetime"] + lightning_cols], on="datetime", how="left")

    # Stats
    has_activity = (df["lightning_has_activity"] == 1).sum()
    logger.info(f"  Hores amb activitat elèctrica: {has_activity} ({100*has_activity/len(df):.2f}%)")
    logger.info(f"  Mitjana llamps 30km (quan n'hi ha): {df.loc[df['lightning_has_activity']==1, 'lightning_count_30km'].mean():.1f}")

    # Save updated dataset
    df.to_parquet(DATASET_PATH, index=False)
    logger.info(f"Dataset actualitzat desat a {DATASET_PATH}")
    logger.info("Ara pots executar: .venv/bin/python scripts/train_model.py")


def _save_cache(existing_cache: pd.DataFrame, new_rows: list[dict]):
    """Save/append to the lightning cache parquet."""
    new_df = pd.DataFrame(new_rows)
    if not existing_cache.empty:
        combined = pd.concat([existing_cache, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(CACHE_PATH, index=False)


if __name__ == "__main__":
    main()
