#!/usr/bin/env python3
"""
One-shot: verify predictions that got stranded as `verified:false` while the
runtime verify window (≤48h) couldn't reach them. Uses XEMA KX historical
data (Meteocat /variables/mesurades/{var}/{Y}/{M}/{D}), one API call per
distinct day of pending entries. Safe to re-run; already-verified entries
are left alone.

Usage:
  METEOCAT_API_KEY=xxx python scripts/backfill_verify.py [--dry-run] [--since YYYY-MM-DD]
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.meteocat import fetch_variable_all_stations
from src.feedback.logger import load_predictions_log, save_predictions_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't write")
    parser.add_argument("--since", type=str, default=None, help="Only backfill entries on/after YYYY-MM-DD")
    args = parser.parse_args()

    if not config.METEOCAT_API_KEY:
        logger.error("METEOCAT_API_KEY not set; cannot reach XEMA.")
        sys.exit(1)

    entries = load_predictions_log()
    now = datetime.now()
    settle = timedelta(minutes=config.PREDICTION_HORIZON_MIN + 15)
    since = datetime.fromisoformat(args.since) if args.since else None

    pending = []
    for entry in entries:
        if entry.get("verified"):
            continue
        ts = datetime.fromisoformat(entry["timestamp"])
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

    # Fetch XEMA KX precipitation once per day.
    station_by_date: dict = {}
    for d in dates:
        df = fetch_variable_all_stations(config.XEMA_VAR_PRECIP, d)
        if df.empty:
            logger.warning(f"{d}: XEMA returned no data, skipping.")
            continue
        kx = df[df["station_code"] == config.LOCAL_RAIN_STATION_CODE].copy()
        if kx.empty:
            logger.warning(f"{d}: XEMA has no KX readings, skipping.")
            continue
        kx["datetime"] = pd.to_datetime(kx["datetime"]).dt.tz_localize(None)
        kx = kx.sort_values("datetime").reset_index(drop=True)
        station_by_date[d] = kx
        logger.info(f"  {d}: {len(kx)} KX readings")

    verified_count = correct_count = wrong_count = uncertain_count = skipped = 0
    for ts, entry in pending:
        kx = station_by_date.get(ts.date())
        if kx is None:
            skipped += 1
            continue
        window_end = ts + timedelta(minutes=config.PREDICTION_HORIZON_MIN)
        window = kx[(kx["datetime"] >= ts) & (kx["datetime"] <= window_end)]
        if window.empty:
            skipped += 1
            continue

        rain_mm = float(pd.to_numeric(window["value"], errors="coerce").fillna(0).sum())
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
        entry["verification_source"] = "xema_kx_backfill"
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
