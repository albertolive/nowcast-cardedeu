"""Merge verified outcomes with the measurements used to issue each forecast."""
from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile

import pandas as pd

import config
from src.features.engineering import FEATURE_COLUMNS
from src.feedback.archive import iter_snapshots
from src.feedback.logger import load_predictions_log

logger = logging.getLogger(__name__)
FEEDBACK_TRAINING_PATH = os.path.join(config.DATA_PROCESSED_DIR, "feedback_verified.parquet")
_METADATA = ("verified", "actual_rain", "actual_rain_mm", "correct", "probability_pct",
             "probability", "raw_probability", "threshold", "rain_gate_open", "uncertain",
             "brier_component", "station_available", "station_raining_now")


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))
    frame = pd.read_parquet(path).set_index("datetime")
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f"{path}: missing or duplicate issue time")
    return frame


def _merge_verified(existing: pd.DataFrame, entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return existing
    labels = pd.DataFrame([{k: e[k] for k in _METADATA if k in e} for e in entries])
    if "actual_rain" not in labels or not labels["actual_rain"].isin([True, False]).all():
        raise ValueError("A verified forecast has no binary observed outcome")
    labels = labels.apply(pd.to_numeric)
    labels["will_rain"] = labels["actual_rain"].astype(int)
    # Keep predicted probability and verification fields out of the predictors.
    inputs = pd.DataFrame([e.get("features") or {} for e in entries], columns=FEATURE_COLUMNS)
    if any(pd.Timestamp(e["timestamp"]).tzinfo is None for e in entries):
        raise ValueError("A verified forecast has no offset in its issue time")
    issued = pd.to_datetime([e["timestamp"] for e in entries], format="ISO8601", utc=True)
    updates = pd.concat([labels, inputs.apply(pd.to_numeric)], axis=1)
    updates.index = issued.tz_localize(None).rename("datetime")
    if updates.index.hasnans:
        raise ValueError("A verified forecast has no issue time")
    combined = pd.concat([existing, updates]) if len(existing) else updates
    # Newest available measurement wins; an absent field in a dashboard row
    # must never erase the original reading. An explicitly unscored outcome,
    # on the other hand, must be allowed to replace a previous correctness flag.
    merged = combined.groupby(level=0, sort=True).last()
    if "correct" in labels:
        latest = updates[~updates.index.duplicated(keep="last")]
        merged.loc[latest.index, "correct"] = latest["correct"]
    if "brier_component" in merged:
        # A compact correction has no Brier field; don't inherit a score for a different outcome.
        changed = combined["actual_rain"].groupby(level=0).nunique().gt(1)
        corrected = merged.loc[changed]
        if not corrected.empty:
            p = corrected.get("probability", pd.Series(index=corrected.index, dtype=float))
            pct = corrected.get("probability_pct", pd.Series(index=corrected.index, dtype=float))
            prob = p.fillna(pct / 100)
            merged.loc[changed, "brier_component"] = (prob.where(prob.between(0, 1)) - corrected["actual_rain"]).pow(2).round(6)
    return merged


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        frame.copy().reset_index().to_parquet(temp, index=False)
        temp.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def export_verified_for_training() -> int:
    entries = [e for e in load_predictions_log() if e.get("verified")]
    if not entries:
        logger.info("Cap predicció verificada per exportar.")
        return 0
    path = Path(FEEDBACK_TRAINING_PATH)
    existing = _read_existing(path)
    merged = _merge_verified(existing, entries)
    if not merged.equals(existing):
        _save(merged, path)
    logger.info("Exportades %d prediccions verificades a %s", len(merged), path)
    return len(merged)


def repair_feedback(*, since: datetime, as_of: datetime, ref: str = "HEAD",
                    repo: str | Path = ".", prior: pd.DataFrame | None = None,
                    legacy_ref: str | None = None, dry_run: bool = False) -> dict:
    """Recover exact issue-time inputs; validate the weekly batch before replacing it."""
    if since.tzinfo is None or as_of.tzinfo is None or since >= as_of:
        raise ValueError("Expected an increasing interval with explicit UTC offsets")
    path = Path(FEEDBACK_TRAINING_PATH)
    existing = _read_existing(path)
    entries = [e for e in load_predictions_log() if e.get("verified")]
    if not entries:
        raise ValueError("The verification log contains no verified forecasts")
    frame = _merge_verified(existing, entries)
    end = pd.Timestamp(as_of).tz_convert(None)
    start = pd.Timestamp(since).tz_convert(None)
    last_verified = pd.to_datetime([e["timestamp"] for e in entries], utc=True, format="ISO8601").max()
    if last_verified < as_of - pd.Timedelta(days=config.FEEDBACK_MAX_AGE_DAYS) or frame.index.max() > end:
        raise ValueError("No current verified feedback: check the nightly verification log and UTC cutoff")
    inputs = frame.reindex(columns=FEATURE_COLUMNS)
    missing = inputs.isna().all(axis=1)
    before = int(missing.sum())
    if prior is not None:
        old_inputs = prior.set_index("datetime").reindex(columns=FEATURE_COLUMNS)
        if not old_inputs.index.is_unique or old_inputs.index.hasnans:
            raise ValueError("Prior feedback has a missing or duplicate issue time")
        matched = inputs.index[missing].intersection(old_inputs.dropna(how="all").index)
        inputs.loc[matched] = old_inputs.loc[matched]
        missing.loc[matched] = False
    from_prior = before - int(missing.sum())

    wanted = set(inputs.index[missing & (inputs.index >= start) & (inputs.index < end)])
    recovered = {}
    if wanted:
        for snapshot in iter_snapshots(min(wanted).tz_localize("UTC").to_pydatetime(),
                                       as_of, ref=ref, repo=repo):
            ts = pd.Timestamp(snapshot["timestamp"]).tz_convert(None)
            if ts in wanted and snapshot.get("feature_vector"):
                recovered[ts] = snapshot["feature_vector"]
        if legacy_ref:
            # At this revision both the log and exporter used the identical
            # naive wall-clock string. Check the issued probability as well.
            for snapshot in iter_snapshots(min(wanted).to_pydatetime(), end.to_pydatetime(),
                                           ref=legacy_ref, repo=repo, legacy_wall_time=True):
                ts = pd.Timestamp(snapshot["timestamp"])
                if (ts in wanted and ts not in recovered and snapshot.get("feature_vector")
                        and frame.at[ts, "probability_pct"] == snapshot.get("probability_pct")):
                    recovered[ts] = snapshot["feature_vector"]
    if recovered:
        measured = pd.DataFrame.from_dict(recovered, orient="index").reindex(columns=FEATURE_COLUMNS)
        measured = measured.apply(pd.to_numeric).dropna(how="all")
        inputs.loc[measured.index] = measured
        missing.loc[measured.index] = False
    columns = frame.columns.union(inputs.columns, sort=False)
    frame = pd.concat([frame.drop(columns=FEATURE_COLUMNS, errors="ignore"), inputs], axis=1)[columns]
    recent = frame.index[(frame.index >= end - pd.Timedelta(days=7)) & (frame.index < end)]
    new = frame.index.difference(existing.index)
    for name, batch in (("last seven days", recent), ("new verified rows", new[new >= start])):
        if len(batch) and (1 - missing.loc[batch].mean()) < config.FEEDBACK_MIN_INPUT_COVERAGE:
            raise ValueError(f"{name}: {missing.loc[batch].sum()}/{len(batch)} lack issued inputs; "
                             "check git history depth before retraining")
    stats = {"verified": len(frame), "new": len(new), "from_prior": from_prior,
             "from_snapshots": before - from_prior - int(missing.sum()),
             "without_inputs": int(missing.sum()), "recent_rows": len(recent),
             "recent_without_inputs": int(missing.loc[recent].sum())}
    if missing.any():
        logger.warning("%d verified rows kept for history only; %d wet. First: %s; last: %s",
                       missing.sum(), frame.loc[missing, "will_rain"].sum(),
                       frame.index[missing].min(), frame.index[missing].max())
    if not dry_run and not frame.equals(existing):
        _save(frame, path)
    return stats
