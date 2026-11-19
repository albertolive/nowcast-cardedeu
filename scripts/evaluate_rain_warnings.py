#!/usr/bin/env python3
"""Replay rain-incoming notifications from archived forecasts, not delivery logs.

Scores use the same issue times and require a known live station state. A gauge tip
dates rain to (previous reading, current reading], not necessarily the current minute.
Only dry-at-issue alerts strictly before that interval can earn advance credit.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
from statistics import median
import subprocess
import sys
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src.feedback.archive import iter_snapshots

UTC = timezone.utc
MADRID = ZoneInfo("Europe/Madrid")
HOUR = timedelta(minutes=config.PREDICTION_HORIZON_MIN)
COOLDOWN = timedelta(minutes=config.NOTIFICATION_COOLDOWN_MIN)
STATION_DIR = Path(config.DATA_RAW_DIR) / "rain_warning_station"
# Daily archive is nominally 5-minute (observed jitter of 1–2 minutes).
MAX_STATION_GAP = timedelta(minutes=8)
MAX_ISSUE_GAP = timedelta(minutes=15)
NAMES = ("issued", "shadow", "persistence")


def station_days(start: datetime, end: datetime, directory: Path, fetch: bool = False):
    """Cache only the fetch_daily interval columns, never cumulative totals."""
    day = start.astimezone(MADRID).date()
    last_day = (end - timedelta(microseconds=1)).astimezone(MADRID).date()
    while day <= last_day:
        path = directory / f"{day}.csv"
        if fetch and not _closed_station_day(path, day):
            from src.data.meteocardedeu import fetch_daily

            frame = fetch_daily(day, preserve_missing=True)
            if not frame.empty and {"datetime", "PREC"} <= set(frame):
                directory.mkdir(exist_ok=True, parents=True)
                with NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as tmp:
                    temp = Path(tmp.name)
                try:
                    frame.to_csv(temp, columns=["datetime", "PREC"], index=False)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
            else:
                logging.getLogger(__name__).warning("No fresh gauge archive for %s", day)
        yield day, path
        day += timedelta(days=1)


def _closed_station_day(path, day):
    midnight = datetime.combine(day, time())
    tomorrow = midnight + timedelta(days=1)
    if not path.is_file() or datetime.now(MADRID).date() <= day:
        return False
    try:
        frame = pd.read_csv(path, usecols=["datetime", "PREC"])
    except (OSError, ValueError):
        return False  # A damaged cache file is fetched again.
    stamps = pd.to_datetime(frame["datetime"], errors="coerce")
    return (len(stamps) > 1 and stamps.notna().all()
            and midnight <= stamps.iloc[0] <= midnight + MAX_STATION_GAP
            and tomorrow - MAX_STATION_GAP <= stamps.iloc[-1] < tomorrow
            and stamps.diff().iloc[1:].between(timedelta(0), MAX_STATION_GAP, inclusive="right").all()
            and frame["PREC"].iloc[1:].notna().all())


@dataclass(frozen=True)
class Episode:
    # First tip occurred in (lower, onset]; a forecast inside that interval
    # cannot prove advance warning. Lead time uses the conservative lower bound.
    lower: datetime
    onset: datetime
    last_tip: datetime
    mm: float
    complete: bool
    qualified_at: datetime | None = None
    tips: tuple = field(default=(), compare=False, repr=False)


def gauge_episodes(days: Iterable[tuple[date, Path]]):
    """Return observed spans and wet episodes, without bridging station outages."""
    spans, episodes, files, wet_rows = [], [], {}, []
    for day, path in days:
        if not path.exists():
            files[str(day)] = None
            continue
        contents = path.read_bytes()
        files[str(day)] = hashlib.sha256(contents).hexdigest()
        frame = pd.read_csv(BytesIO(contents), usecols=["datetime", "PREC"])
        local = pd.to_datetime(frame["datetime"], errors="coerce")
        # Do not guess which folded hour a naive stamp belongs to at DST end.
        index = pd.DatetimeIndex(local).tz_localize(MADRID, ambiguous="NaT", nonexistent="NaT")
        rain = pd.Series(pd.to_numeric(frame["PREC"], errors="coerce").to_numpy(), index=index)
        rain = rain[(~index.isna()) & (index.date == day)].sort_index().tz_convert(UTC)
        rain.loc[rain.index.duplicated(keep=False)] = np.nan
        rain = rain[~rain.index.duplicated()]
        # The first reading is a baseline, never evidence of a dry midnight.
        delta = rain.index.to_series().diff()
        valid = np.isfinite(rain) & rain.ge(0) & delta.le(MAX_STATION_GAP)
        # A tip across a long gap proves rain, but may not date its onset.
        wet = rain[(rain > 0) & np.isfinite(rain) & delta.notna()]
        wet_rows.extend(((t - delta[t]).to_pydatetime(), t.to_pydatetime(), float(mm))
                        for t, mm in wet.items())
        for _, group in rain[valid].groupby((~valid).cumsum()[valid]):
            first = (group.index[0] - delta[group.index[0]]).to_pydatetime()
            last = group.index[-1].to_pydatetime()
            spans.append((first, last))
    groups = []
    for tip in wet_rows:
        if not groups or tip[0] - groups[-1][-1][1] >= HOUR:
            groups.append([])
        groups[-1].append(tip)
    for tips in groups:
        lower, onset, _ = tips[0]
        last_tip = tips[-1][1]
        amount, qualified = 0., None
        for _, when, mm in tips:
            amount += mm
            if qualified is None and round(amount, 6) >= config.RAIN_THRESHOLD_MM:
                qualified = when
        episodes.append(Episode(lower, onset, last_tip, round(amount, 6),
                                inside(spans, lower - HOUR, last_tip + HOUR), qualified, tuple(tips)))
    return spans, episodes, files


def utc_stamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Missing UTC offset: {value}")
    return stamp.astimezone(UTC)


def iso(stamp):
    return stamp.isoformat().replace("+00:00", "Z")


def probability(value):
    return type(value) in (float, int) and 0 <= value <= 1


def archived_floor(row):
    from src.model.predict import _apply_physical_constraints

    # The vector contains storm_approaching (not radar.approaching) and
    # ensemble_min_precip. Diagnostics retain freeze flags, ETA and AEMET streak.
    inputs = dict(row["feature_vector"])
    for source in ("radar", "aemet_radar", "aemet", "sentinel", "ensemble"):
        inputs.update({f"{source}_{k}": v for k, v in (row.get(source) or {}).items()})
    floor, _ = _apply_physical_constraints(
        0., inputs, inputs, aemet_radar_data=inputs, aemet_forecast=inputs, ensemble=inputs,
        current={"PINT": int(row["station_raining_now"])}, lightning=inputs)
    return max(floor, row["probability"] if row["physical_adjustments"] else 0)


def aligned_scores(snapshots: Iterable[dict], bundle, model_id: str | None):
    model, names, calibrator, _ = bundle
    times, vectors, issued, raining, floors, shadow = [], [], [], [], [], []
    counts = Counter()
    for row in snapshots:
        counts["archived"] += 1
        p, vector = row.get("probability"), row.get("feature_vector")
        wet, adjustments = row.get("station_raining_now"), row.get("physical_adjustments")
        if row.get("station_available") is not True or type(wet) is not bool:
            counts["station_unknown"] += 1
        elif not probability(p) or not isinstance(adjustments, list):
            counts["issued_or_floor_unknown"] += 1
        elif not isinstance(vector, dict) or not all(k in vector for k in names):
            counts["missing_feature_vector"] += 1
        else:
            times.append(utc_stamp(row["timestamp"]))
            vectors.append(vector)
            issued.append(p)
            raining.append(wet)
            floor = archived_floor(row)
            floors.append(floor)
            if adjustments:
                counts["issued_floor_recorded"] += 1
            elif floor > 0:
                counts["recovered_masked_floor"] += 1
            if round(floor, 4) > p:
                counts["recovered_floor_above_issued"] += 1
            archived = row.get("shadow_probability")
            shadow.append(archived if model_id and row.get("shadow_model_id") == model_id
                          and probability(archived) else np.nan)

    frame = pd.DataFrame(dict(issued=issued, shadow=shadow, persistence=raining),
                         index=pd.DatetimeIndex(times, tz=UTC))
    if vectors:
        x = pd.DataFrame(vectors, columns=names, index=frame.index).apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        valid = x.notna().any(axis=1)
        counts["empty_feature_vector"] = int((~valid).sum())
        replay = valid & frame["shadow"].isna()
        counts["logged_shadow"] = int((valid & ~replay).sum())
        if replay.any():
            raw = model.predict_proba(x.loc[replay])[:, 1]
            p = calibrator.predict(raw) if calibrator is not None else raw
            if not (np.isfinite(p) & (p >= 0) & (p <= 1)).all():
                raise ValueError("Invalid candidate probability")
            frame.loc[replay, "shadow"] = [round(float(v), 4) for v in np.maximum(p, np.array(floors)[replay])]
        frame = frame.loc[valid]
    counts["aligned"] = len(frame)
    return frame, dict(counts)


def replay(frame: pd.DataFrame):
    """Synchronize after every archive gap without assuming an initial state."""
    spans, alerts = [], {name: [] for name in NAMES}
    missing = frame.isna().any(axis=1)
    blocks = (missing | frame.index.to_series().diff().gt(MAX_ISSUE_GAP)).cumsum()
    for _, block in frame.loc[~missing].groupby(blocks):
        # After two cooldowns of consistently low (or high) scores the state
        # and eligibility are known, regardless of state before the block.
        states, anchors, last, ready = {}, {}, {}, None
        for row in block.itertuples():
            now = row.Index.to_pydatetime()
            for name in NAMES:
                p = getattr(row, name)
                high = p >= config.ALERT_THRESHOLD_UP
                low = p <= config.ALERT_THRESHOLD_DOWN
                if name not in states:
                    old, since, forced = anchors.get(name, (None, now, None))
                    target = high if high or low else None
                    if target is None or old != target:
                        anchors[name] = (target, now, None)
                    elif forced is None and now - since >= COOLDOWN:
                        # The first observed extremal score after the unknown
                        # cooldown expired forces the target. Its own cooldown
                        # starts at that *issue*, not at since + COOLDOWN.
                        anchors[name] = (target, since, now)
                    elif forced is not None and now - forced >= COOLDOWN:
                        states[name] = target
                        last[name] = forced
                    continue
                if now - last[name] < COOLDOWN:
                    continue
                if (high and not states[name]) or (low and states[name]):
                    states[name], last[name] = high, now
                    if high:
                        alerts[name].append((now, row.persistence))
            if ready is None and len(states) == len(NAMES):
                ready = now
        if ready is not None:
            spans.append((ready, block.index[-1].to_pydatetime()))
    for name in NAMES:
        alerts[name] = [(ts, wet) for ts, wet in alerts[name] if inside(spans, ts, ts)]
    return spans, alerts


def intersects(a, b):
    """Intersect two sorted collections of disjoint time spans."""
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            yield lo, hi
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1


def inside(spans, left, right):
    i = bisect_right(spans, left, key=lambda span: span[0]) - 1
    return i >= 0 and right <= spans[i][1]


def window_rain(tips, when):
    """Return (possible, certain) gauge mm in [when, when + 1h); boundary tips are only possible."""
    stop, possible, certain = when + HOUR, 0., 0.
    for lower, upper, amount in tips[bisect_left(tips, when, key=lambda tip: tip[1]):]:
        if lower >= stop:
            break
        if upper > when:
            possible += amount
            if lower >= when and upper <= stop:
                certain += amount
    return round(possible, 6), round(certain, 6)


def score(spans, episodes, score_spans, alerts, start, end, *, min_lead=timedelta(0)):
    watched = list(intersects(score_spans, [(start, end)]))
    tips = [tip for e in episodes for tip in e.tips]
    # Only count false-alert exposure where a full forward horizon has labels.
    exposure = list(intersects(watched, [(a, b - HOUR) for a, b in spans
                                         if b - a > HOUR]))
    weeks = sum((b - a).total_seconds() for a, b in exposure) / (7 * 86400)
    records, eligible, onsets = [], set(), set()
    for i, e in enumerate(episodes):
        if not start <= e.onset < end:
            continue
        onset_reason = None
        first = e.qualified_at
        if first is None:
            onset_reason = "below_rain_threshold"
        elif e.lower < start or first >= end or not inside(spans, e.lower - HOUR, first):
            onset_reason = "station_gap_or_boundary"
        elif not inside(score_spans, e.lower - HOUR, first):
            onset_reason = "archive_gap_or_warmup"
        if onset_reason is None:
            onsets.add(i)
        reason = onset_reason
        if reason is None and (not e.complete or e.last_tip >= end):
            reason = "station_gap_or_boundary"
        if reason is None and not inside(score_spans, e.lower - HOUR, e.last_tip + HOUR):
            reason = "archive_gap_or_warmup"
        if reason is None:
            eligible.add(i)
        records.append(dict(onset=iso(e.onset), after=iso(e.lower), last_tip=iso(e.last_tip),
                            qualified_at=iso(first) if first is not None else None, mm=e.mm,
                            excluded=reason, onset_excluded=onset_reason, outcomes={}, onset_outcomes={}))
    by_onset = {r["onset"]: r for r in records}
    summary, onset_summary, audit = {}, {}, {}
    for name in NAMES:
        advances, confirmations, short, uncertain, log = {}, {}, {}, set(), []
        for t, wet in alerts[name]:
            # A dry issue can warn the next storm even during the previous
            # storm's closing dry hour. Never match an issue to both.
            i = bisect_right(episodes, t, key=lambda e: e.lower)
            if i and episodes[i - 1].lower == t:
                i -= 1
            upcoming = i < len(episodes) and t >= episodes[i].lower - HOUR
            previous = episodes[i - 1] if i else None
            ongoing = previous and (t <= previous.last_tip or
                                    (wet and t < previous.last_tip + HOUR))
            if upcoming or ongoing:
                i = i if upcoming and not (wet and ongoing) else i - 1
                # A trace of rain below the target doesn't make a dry alert correct.
                if (episodes[i].qualified_at is None and not wet
                        and window_rain(tips, t)[0] < config.RAIN_THRESHOLD_MM):
                    upcoming = ongoing = False
            if upcoming or ongoing:
                e = episodes[i]
                if i not in onsets:
                    kind = "unscored_episode"
                elif t == e.lower:
                    kind = "timing_uncertain"
                elif t < e.lower and not wet and e.qualified_at > t + HOUR:
                    kind = "timing_uncertain"
                elif t < e.lower and not wet and e.lower - t < min_lead:
                    kind = "short_lead"
                    short.setdefault(i, round((e.lower - t).total_seconds() / 60, 2))
                elif t < e.lower and not wet:
                    kind = "repeat" if i in advances else "advance"
                    advances.setdefault(i, round((e.lower - t).total_seconds() / 60, 2))
                elif wet and t < e.lower:
                    kind = "station_disagreement"
                elif t >= e.onset or wet:
                    kind = "confirmation"
                    if i in eligible:
                        confirmations.setdefault(i, iso(t))
                else:
                    kind = "timing_uncertain"
                if kind == "timing_uncertain":
                    uncertain.add(i)
            elif start <= t < end and inside(exposure, t, t):
                kind = "station_disagreement" if wet else "false"
            else:
                kind = "unscored_window"
            if start <= t < end or kind == "advance":
                full_kind = "unscored_episode" if (upcoming or ongoing) and i not in eligible else kind
                log.append(dict(time=iso(t), station_wet=wet, kind=full_kind, onset_kind=kind))
        for i in onsets:
            row = by_onset[iso(episodes[i].onset)]
            row["onset_outcomes"][name] = dict(lead_minutes=advances.get(i), timing_uncertain=i in uncertain)
            if min_lead:
                row["onset_outcomes"][name]["short_lead_minutes"] = short.get(i)
            if i in eligible:
                row["outcomes"][name] = dict(lead_minutes=advances.get(i), confirmation=confirmations.get(i))
        onset_summary[name] = dict(storms=len(onsets), advance=len(advances),
                                   missed_advance=len(onsets) - len(advances),
                                   timing_uncertain=len(uncertain - advances.keys()),
                                   lead_minutes_median=median(advances.values()) if advances else None)
        if min_lead:
            onset_summary[name]["short_lead_only"] = len(short.keys() - advances.keys())
        advances = {i: lead for i, lead in advances.items() if i in eligible}
        false = sum(a["kind"] == "false" for a in log)
        summary[name] = dict(
            storms=len(eligible), advance=len(advances), missed_advance=len(eligible) - len(advances),
            confirmation_only=len(confirmations.keys() - advances.keys()),
            missed_entirely=len(eligible - advances.keys() - confirmations.keys()),
            lead_minutes_median=median(advances.values()) if advances else None,
            false_alerts=false, false_alerts_per_week=round(false / weeks, 3) if weeks else None,
            alert_counts=dict(Counter(a["kind"] for a in log)),
        )
        audit[name] = log
    return dict(exposure_days=round(7 * weeks, 5), episodes=records, comparison=summary,
                onset_comparison=onset_summary, alerts=audit,
                excluded_episodes=dict(Counter(r["excluded"] for r in records if r["excluded"])),
                excluded_onsets=dict(Counter(r["onset_excluded"] for r in records if r["onset_excluded"])))


def run(args):
    from src.model.train import load_model

    start = datetime.combine(args.start, time(), UTC)
    end = datetime.combine(args.end, time(), UTC)
    if end <= start or end > datetime.now(UTC):
        raise ValueError("--end must be after --start and in the past (exclusive UTC)")
    metrics = json.loads((args.model_dir / "metrics.json").read_text())
    if not metrics.get("training_cutoff"):
        raise ValueError("metrics.json must declare training_cutoff; train using --before")
    cutoff = utc_stamp(metrics["training_cutoff"])
    if cutoff > start:
        raise ValueError("Holdout overlaps training_cutoff; move --start or retrain using --before")
    warm = max(cutoff, start - HOUR - 2 * COOLDOWN)
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                        args.ref + "^{commit}"], cwd=ROOT, text=True).strip()
    bundle = load_model(model_dir=args.model_dir)
    if getattr(bundle[0], "nowcast_model_id", None) != metrics.get("model_id"):
        raise ValueError("The candidate changed while loading; run again")
    frame, counts = aligned_scores(iter_snapshots(warm, end, ref=revision, repo=ROOT),
                                   bundle, metrics.get("model_id"))
    days = station_days(start - HOUR, end + HOUR, args.station_dir, args.fetch_station)
    station, episodes, files = gauge_episodes(days)
    spans, alerts = replay(frame)
    report = score(station, episodes, spans, alerts, start, end, min_lead=timedelta(minutes=args.min_lead))
    report.update(start=iso(start), end=iso(end), warmup_start=iso(warm),
                  training_cutoff=iso(cutoff), model_id=metrics.get("model_id"),
                  git_commit=revision, station_day_sha256=files, input_counts=counts,
                  policy=dict(notification="simulated rain_incoming, no regime/daily alerts or delivery logs",
                              up=config.ALERT_THRESHOLD_UP, down=config.ALERT_THRESHOLD_DOWN,
                              cooldown_minutes=config.NOTIFICATION_COOLDOWN_MIN,
                              horizon_minutes=config.PREDICTION_HORIZON_MIN,
                              station_gap_minutes=MAX_STATION_GAP.total_seconds() / 60,
                              issue_gap_minutes=MAX_ISSUE_GAP.total_seconds() / 60,
                              lead="minutes before earliest possible first gauge tip; dry station at issue",
                              minimum_lead_minutes=args.min_lead,
                              storms="first tip after 60 dry minutes; minimum total mm in rain_threshold_mm; "
                                     "require observed warning window, whole episode and dry tail",
                              onset_comparison="require the same dry prelude and coverage through first "
                                               "qualifying gauge evidence, not the subsequent storm tail",
                              rain_threshold_mm=config.RAIN_THRESHOLD_MM,
                              persistence="as-issued station_raining_now, when station_available is true",
                              shadow="logged matching model ID, else calibrated archived features plus "
                                     "physical rules from as-issued diagnostics; issued probability is also "
                                     "a floor only with physical_adjustments. A masked 1h lightning floor "
                                     "is unrecoverable when only 3h counts were archived."))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python scripts/evaluate_rain_warnings.py --start 2026-11-01 --end 2026-11-18 "
               "--fetch-station --out warnings.json\n  # Omit --fetch-station for an offline replay.")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="first UTC date (inclusive)")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="first UTC date not scored")
    parser.add_argument("--model-dir", type=Path, default=Path(config.MODELS_DIR) / "shadow")
    parser.add_argument("--ref", default="HEAD", help="local Git revision containing as-issued forecasts")
    parser.add_argument("--station-dir", type=Path, default=STATION_DIR,
                        help="daily YYYY-MM-DD.csv files: datetime (Madrid wall clock), PREC (interval mm)")
    parser.add_argument("--fetch-station", action="store_true", help="fetch missing public gauge archive days")
    parser.add_argument("--min-lead", type=float, default=config.WARNING_TRIAL_LEAD_MIN,
                        help="minutes of lead required for an advance warning (default: same as the trial)")
    parser.add_argument("--out", type=Path, help="save full JSON including per-storm leads and alert times")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps({k: report[k] for k in ("input_counts", "exposure_days", "onset_comparison",
                                             "comparison")}, indent=2))
    if not report["exposure_days"]:
        parser.exit(1, "No matched observation time; check --ref, station cache and --fetch-station.\n")


if __name__ == "__main__":
    main()
