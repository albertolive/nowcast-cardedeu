#!/usr/bin/env python3
"""Compare warnings for the first ten observable rain onsets after shadow activation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from scripts import evaluate_rain_warnings as ev


def issued_scores(snapshots, model_id, start, end):
    rows, times, counts = [], [], Counter()
    activation = last_score = None
    for row in snapshots:
        when = ev.utc_stamp(row["timestamp"])
        if not start <= when < end:
            continue
        shadow = row.get("shadow_probability")
        matching = row.get("shadow_model_id") == model_id and ev.probability(shadow)
        if matching:
            activation = activation or when
            last_score = when
        if activation is None:
            counts["before_activation"] += 1
            continue
        live, wet = row.get("probability"), row.get("station_raining_now")
        if not matching:
            reason = "wrong_candidate" if row.get("shadow_model_id") not in (None, model_id) else "missing_score"
        elif not ev.probability(live):
            reason = "missing_live_probability"
        elif row.get("station_available") is not True or type(wet) is not bool:
            reason = "unknown_station"
        else:
            reason = "matched"
        counts[reason] += 1
        # A known missing issue breaks all three simulated state machines.
        rows.append((live, shadow, wet) if reason == "matched" else (None, None, None))
        times.append(when)
    frame = pd.DataFrame(rows, columns=ev.NAMES, index=pd.DatetimeIndex(times, tz=ev.UTC))
    return frame, activation, last_score, dict(counts)


def false_warnings(gauge, episodes, watched, alerts, start, end):
    """Count only dry-at-issue transitions with a fully observed 60-minute target."""
    exposure = list(ev.intersects(list(ev.intersects(watched, [(start, end)])),
                                  [(a, b - ev.HOUR) for a, b in gauge if b - a > ev.HOUR]))
    seconds = sum((b - a).total_seconds() for a, b in exposure)
    tips = [tip for e in episodes for tip in e.tips]
    result = {}
    for name in ev.NAMES:
        false, ambiguous = [], []
        for when, wet in alerts[name]:
            if wet or not start <= when < end or not ev.inside(exposure, when, when):
                continue
            possible, certain = ev.window_rain(tips, when)
            if possible < config.RAIN_THRESHOLD_MM:
                false.append(ev.iso(when))
            elif certain < config.RAIN_THRESHOLD_MM:
                ambiguous.append(ev.iso(when))
        result[name] = dict(false_warnings=len(false), false_warning_times=false,
                            uncertain_windows=ambiguous,
                            false_warnings_per_week=round(len(false) * 7 * 86400 / seconds, 3) if seconds else None)
    return seconds, result


def trial_scores(gauge, episodes, frame, start, end):
    watched, alerts = ev.replay(frame)
    lead = timedelta(minutes=config.WARNING_TRIAL_LEAD_MIN)
    report = ev.score(gauge, episodes, watched, alerts, start, end, min_lead=lead)
    cases = [r for r in report["episodes"] if not r["onset_excluded"]]
    done = len(cases) >= config.WARNING_TRIAL_EPISODES
    if done:
        # End immediately after the tenth qualifying gauge reading. Later dry
        # weeks must not dilute the final false-warning rate.
        end = ev.utc_stamp(cases[config.WARNING_TRIAL_EPISODES - 1]["qualified_at"]) + timedelta(microseconds=1)
        report = ev.score(gauge, episodes, watched, alerts, start, end, min_lead=lead)
    seconds, false = false_warnings(gauge, episodes, watched, alerts, start, end)
    return dict(status="review_ready" if done else "collecting", scored_until=ev.iso(end),
                observable_onsets=min(len(cases), config.WARNING_TRIAL_EPISODES),
                observed_days=round(seconds / 86400, 5), episodes=report["episodes"],
                excluded_onsets=report["excluded_onsets"],
                comparison={n: {**report["onset_comparison"][n], **false[n]} for n in ev.NAMES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python scripts/monitor_rain_warnings.py --fetch-station --out trial.json\n"
               "  python scripts/monitor_rain_warnings.py --ref origin/main --station-dir /tmp/gauge")
    parser.add_argument("--ref", default="HEAD", help="local revision with production snapshot history")
    parser.add_argument("--station-dir", type=Path, default=ev.STATION_DIR / "issued-trial")
    parser.add_argument("--fetch-station", action="store_true", help="fetch public gauge data; refresh partial days")
    parser.add_argument("--out", type=Path, help="write the full JSON report")
    args = parser.parse_args(argv)
    start = ev.utc_stamp(config.WARNING_TRIAL_NOT_BEFORE)
    end = datetime.now(ev.UTC)
    sha = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                   args.ref + "^{commit}"], cwd=ev.ROOT, text=True).strip()
    first_commit = subprocess.check_output(["git", "rev-list", "--first-parent", "--max-parents=0",
                                            "--timestamp", sha], cwd=ev.ROOT, text=True).split()
    if not first_commit or datetime.fromtimestamp(int(first_commit[0]), ev.UTC) >= start:
        parser.exit(2, "Insufficient Git history: fetch the first-parent history through the trial start.\n")
    model_id = config.WARNING_TRIAL_MODEL_ID
    snapshots = ev.iter_snapshots(start, end, ref=sha, repo=ev.ROOT) if start < end else ()
    frame, activation, last, counts = issued_scores(snapshots, model_id, start, end)
    report = dict(status="awaiting_activation", model_id=model_id, not_before=ev.iso(start),
                  first_issued_at=ev.iso(activation) if activation else None,
                  last_shadow_issue=ev.iso(last) if last else None, input_counts=counts,
                  git_commit=sha, checked_at=ev.iso(end), goal=config.WARNING_TRIAL_EPISODES,
                  minimum_lead_minutes=config.WARNING_TRIAL_LEAD_MIN, observable_onsets=0,
                  policy=dict(up=config.ALERT_THRESHOLD_UP, down=config.ALERT_THRESHOLD_DOWN,
                              cooldown_minutes=config.NOTIFICATION_COOLDOWN_MIN,
                              horizon_minutes=config.PREDICTION_HORIZON_MIN, threshold_mm=config.RAIN_THRESHOLD_MM),
                  notice="Only originally issued matching scores. Simulated rain alerts, not Telegram deliveries. "
                         "Review manually when complete; no promotion.")
    if activation:
        # Report that the model is recording today, but don't score an open
        # gauge day or freeze a result before its false-warning labels mature.
        midnight = datetime.combine(end.astimezone(ev.MADRID).date(), time(), ev.MADRID).astimezone(ev.UTC)
        settled = max(activation, midnight - ev.HOUR)
        days = ev.station_days(activation - ev.HOUR, midnight, args.station_dir, args.fetch_station)
        gauge, episodes, files = ev.gauge_episodes(days) if settled > activation else ([], [], {})
        report.update(trial_scores(gauge, episodes, frame, activation, settled), station_day_sha256=files)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if activation and report["status"] != "review_ready" and end - last > 2 * ev.HOUR:
        parser.exit(1, "The trial candidate has not issued a score in over two hours.\n")


if __name__ == "__main__":
    main()
