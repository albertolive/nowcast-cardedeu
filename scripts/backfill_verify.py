#!/usr/bin/env python3
"""
One-shot: verify predictions that got stranded as `verified:false` while the
runtime verify window (≤48h) couldn't reach them.

Primary source: MeteoCardedeu daily history endpoint (minute granularity, no
API key, same sensor as the live verifier — so backfilled results are 100%
consistent with normal verification).

Fallback: XEMA KX historical via Meteocat /estacions/mesurades/KX/{Y}/{M}/{D}
(30-min granularity, requires METEOCAT_API_KEY, consumes quota).

Safe to re-run; already-verified entries are left alone.

Usage:
  python scripts/backfill_verify.py [--dry-run] [--since YYYY-MM-DD]
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.meteocardedeu import fetch_daily as fetch_meteocardedeu_daily
from src.data.meteocat import fetch_station_all_variables
from src.feedback.logger import load_predictions_log, save_predictions_log
from src.feedback.verify import _parse_log_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fetch_day_meteocardedeu(date) -> pd.DataFrame:
    """Fetch a day from MeteoCardedeu; return df with columns datetime + PREC.
    PREC is already converted to per-reading increment by fetch_daily."""
    df = fetch_meteocardedeu_daily(date)
    if df.empty or "PREC" not in df.columns or "datetime" not in df.columns:
        return pd.DataFrame()
    return df[["datetime", "PREC"]].copy()


def _fetch_day_xema_kx(date) -> pd.DataFrame:
    """Fallback: fetch a day from XEMA KX; return df with columns datetime + PREC.
    XEMA returns 30-min accumulation already, so `value` IS the increment."""
    if not config.METEOCAT_API_KEY:
        return pd.DataFrame()
    raw = fetch_station_all_variables(config.LOCAL_RAIN_STATION_CODE, date)
    if raw.empty:
        return pd.DataFrame()
    kx = raw[raw["variable_code"] == config.XEMA_VAR_PRECIP].copy()
    if kx.empty:
        return pd.DataFrame()
    kx["datetime"] = pd.to_datetime(kx["datetime"]).dt.tz_localize(None)
    kx["PREC"] = pd.to_numeric(kx["value"], errors="coerce").fillna(0)
    return kx[["datetime", "PREC"]].sort_values("datetime").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't write")
    parser.add_argument("--since", type=str, default=None, help="Only backfill entries on/after YYYY-MM-DD")
    args = parser.parse_args()

    entries = load_predictions_log()
    now = datetime.now()
    settle = timedelta(minutes=config.PREDICTION_HORIZON_MIN + 15)
    since = datetime.fromisoformat(args.since) if args.since else None

    pending = []
    for entry in entries:
        if entry.get("verified"):
            continue
        ts = _parse_log_timestamp(entry["timestamp"])
        if since and ts < since:
            continue
        if now - ts < settle:
            continue
        pending.append((ts, entry))

    if not pending:
        logger.info("Nothing to backfill.")
        return

    dates = sorted({ts.date() for ts, _ in pending})
    logger.info(f"{len(pending)} pending entries across {len(dates)} day(s): {dates[0]} → {dates[-1]}")

    # For each day: try MeteoCardedeu first (free, minute granularity, same
    # sensor as live verifier), fall back to XEMA KX if MC.net is empty.
    station_by_date: dict = {}
    source_counts = {"meteocardedeu": 0, "xema_kx": 0, "none": 0}
    for d in dates:
        df = _fetch_day_meteocardedeu(d)
        source = "meteocardedeu"
        if df.empty:
            df = _fetch_day_xema_kx(d)
            source = "xema_kx" if not df.empty else "none"
        if df.empty:
            logger.warning(f"  {d}: no data from either source, skipping.")
            source_counts["none"] += 1
            continue
        station_by_date[d] = (df, source)
        source_counts[source] += 1
        logger.info(f"  {d}: {len(df)} readings from {source}")

    logger.info(
        f"Day coverage — meteocardedeu: {source_counts['meteocardedeu']}, "
        f"xema_kx: {source_counts['xema_kx']}, no data: {source_counts['none']}"
    )

    verified_count = correct_count = wrong_count = uncertain_count = skipped = 0
    for ts, entry in pending:
        day_data = station_by_date.get(ts.date())
        if day_data is None:
            skipped += 1
            continue
        df, source = day_data
        window_end = ts + timedelta(minutes=config.PREDICTION_HORIZON_MIN)
        window = df[(df["datetime"] >= ts) & (df["datetime"] <= window_end)]
        if window.empty:
            skipped += 1
            continue

        rain_mm = float(pd.to_numeric(window["PREC"], errors="coerce").fillna(0).sum())
        actual_rain = rain_mm >= config.RAIN_THRESHOLD_MM

        prob = entry.get("probability")
        if prob is None and entry.get("probability_pct") is not None:
            prob = entry["probability_pct"] / 100.0
        rain_category = entry.get("rain_category")
        if rain_category is None and prob is not None:
            if prob >= config.DISPLAY_THRESHOLD_RAIN:
                rain_category = "probable"
            elif prob >= config.DISPLAY_THRESHOLD_UNCERTAIN:
                rain_category = "incert"
            else:
                rain_category = "sec"

        if rain_category == "incert":
            is_correct = None
            uncertain_count += 1
        elif rain_category is not None:
            is_correct = (rain_category == "probable") == actual_rain
            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1
        else:
            # No category and no probability — can't score; mark verified with observed rain only.
            is_correct = None

        entry["verified"] = True
        entry["actual_rain"] = actual_rain
        entry["actual_rain_mm"] = round(rain_mm, 2)
        entry["correct"] = is_correct
        entry["uncertain"] = rain_category == "incert"
        entry["rain_category"] = rain_category
        if prob is not None:
            entry["brier_component"] = round((prob - (1.0 if actual_rain else 0.0)) ** 2, 6)
        entry["verified_at"] = now.isoformat()
        entry["verification_source"] = f"{source}_backfill"
        verified_count += 1

    logger.info(
        f"Backfill: {verified_count} verified "
        f"({correct_count} correct, {wrong_count} wrong, {uncertain_count} uncertain), "
        f"{skipped} skipped (no station data in window)"
    )

    if args.dry_run:
        logger.info("DRY-RUN — no changes written.")
        return

    save_predictions_log(entries)
    logger.info("Log saved.")


if __name__ == "__main__":
    main()
