#!/usr/bin/env python3
"""Restore as-issued measurements to verified forecasts without changing their labels."""
import argparse
from datetime import datetime, timedelta, timezone
import io
import json
import logging
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feedback.export import FEEDBACK_TRAINING_PATH, repair_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Examples: python scripts/repair_feedback.py --dry-run; "
        "python scripts/repair_feedback.py --since 2026-03-26T00:00:00Z --prior-ref e712a5d6fbbf"))
    parser.add_argument("--since", type=datetime.fromisoformat, help="first issue time to recover (default: 35 days ago)")
    parser.add_argument("--as-of", type=datetime.fromisoformat, help="UTC reference for freshness checks (default: now)")
    parser.add_argument("--ref", default="HEAD", help="git revision containing the issued forecasts")
    parser.add_argument("--prior-ref", help="also restore missing inputs from the verified parquet at this revision")
    parser.add_argument("--legacy-ref", help="April migration only: match offsetless snapshots by original wall time and issued probability")
    parser.add_argument("--dry-run", action="store_true", help="report coverage without replacing the parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    end = args.as_of or datetime.now(timezone.utc)
    prior = None
    if args.prior_ref:
        path = Path(FEEDBACK_TRAINING_PATH).relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", "--end-of-options", f"{args.prior_ref}:{path}"], cwd=ROOT)
        prior = pd.read_parquet(io.BytesIO(blob))
    stats = repair_feedback(since=args.since or end - timedelta(days=35), as_of=end,
                            ref=args.ref, repo=ROOT, prior=prior, legacy_ref=args.legacy_ref,
                            dry_run=args.dry_run)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
