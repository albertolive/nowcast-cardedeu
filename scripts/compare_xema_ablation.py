#!/usr/bin/env python3
"""Offline diagnostic for the XEMA/no-XEMA comparison.

This script deliberately never imports the live data clients. It only inspects
prediction logs. A true quality comparison requires retraining a second model
without the XEMA features; this diagnostic refuses to pretend otherwise.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

XEMA_FIELDS = (
    "sentinel_temp_diff",
    "sentinel_humidity_diff",
    "sentinel_precip",
    "sentinel_raining",
    "local_rain_xema",
    "local_rain_xema_3h",
)


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=Path("data/predictions_log.jsonl"))
    args = ap.parse_args()
    if not args.path.exists():
        ap.error(f"log not found: {args.path}")
    rows = read_rows(args.path)
    dates = [r.get("timestamp") for r in rows if r.get("timestamp")]
    counts = Counter(
        field
        for row in rows
        for field in XEMA_FIELDS
        if row.get("feature_vector", {}).get(field) is not None
    )
    print(f"log={args.path}")
    print(f"rows={len(rows)}")
    print(f"first_timestamp={min(dates) if dates else 'n/a'}")
    print(f"last_timestamp={max(dates) if dates else 'n/a'}")
    for field in XEMA_FIELDS:
        print(f"non_null[{field}]={counts[field]}")
    if not rows:
        print("comparison=unavailable (empty log)")
    elif not any(counts.values()):
        print("comparison=unavailable (all logged XEMA features are null)")
        print("logged_no_xema_inference=identical_inputs_for_these_rows")
        print("true_quality_comparison=requires_a_separately_retrained_no_xema_model")
    else:
        print("comparison=requires_paired_retraining_and_out_of_sample_evaluation")


if __name__ == "__main__":
    main()
