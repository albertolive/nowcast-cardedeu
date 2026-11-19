"""
Tests per a src/feedback/export.py.

Comprova que export_verified_for_training és APPEND-ONLY: les files
verificades acumulades al parquet no es perden quan el JSONL es retalla.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feedback import export as export_mod
from src.feedback import logger as logger_mod


def _write_jsonl(path: str, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _make_entry(ts: str, prob: float, actual_rain: bool, verified: bool = True) -> dict:
    return {
        "timestamp": ts if pd.Timestamp(ts).tzinfo is not None else ts + "Z",
        "probability_pct": prob,
        "rain_category": "probable" if prob >= 50 else "sec",
        "verified": verified,
        "actual_rain": actual_rain,
        "actual_rain_mm": 1.2 if actual_rain else 0.0,
        "correct": True,
        "features": {"radar_dbz": 12.5, "temperature_2m": 0.0},
    }


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    jsonl = tmp_path / "predictions_log.jsonl"
    parquet = tmp_path / "feedback_verified.parquet"
    monkeypatch.setattr(logger_mod, "PREDICTIONS_LOG", str(jsonl))
    monkeypatch.setattr(export_mod, "FEEDBACK_TRAINING_PATH", str(parquet))
    return jsonl, parquet


def test_first_call_creates_parquet(tmp_paths):
    jsonl, parquet = tmp_paths
    _write_jsonl(str(jsonl), [
        _make_entry("2026-01-01T10:00:00", 80.0, True),
        _make_entry("2026-01-01T10:10:00", 20.0, False),
    ])

    n = export_mod.export_verified_for_training()

    assert n == 2
    assert parquet.exists()
    df = pd.read_parquet(parquet)
    assert len(df) == 2
    assert set(df["datetime"].astype(str)) == {
        "2026-01-01 10:00:00",
        "2026-01-01 10:10:00",
    }


def test_second_call_appends_new_rows(tmp_paths):
    jsonl, parquet = tmp_paths
    _write_jsonl(str(jsonl), [_make_entry("2026-01-01T10:00:00", 80.0, True)])
    export_mod.export_verified_for_training()

    # New tick: JSONL now has the original row + a new one
    _write_jsonl(str(jsonl), [
        _make_entry("2026-01-01T10:00:00", 80.0, True),
        _make_entry("2026-01-01T10:10:00", 20.0, False),
    ])
    export_mod.export_verified_for_training()

    df = pd.read_parquet(parquet)
    assert len(df) == 2  # not 3 — dedup on datetime


def test_trimmed_jsonl_does_not_lose_parquet_rows(tmp_paths):
    """The whole point: after JSONL is trimmed, old verified rows remain in parquet."""
    jsonl, parquet = tmp_paths
    # Initial export with 3 rows
    _write_jsonl(str(jsonl), [
        _make_entry("2026-01-01T10:00:00", 80.0, True),
        _make_entry("2026-01-01T10:10:00", 20.0, False),
        _make_entry("2026-01-01T10:20:00", 90.0, True),
    ])
    export_mod.export_verified_for_training()

    # Simulate JSONL trim: only the most recent row remains, plus a brand-new one
    _write_jsonl(str(jsonl), [
        _make_entry("2026-01-01T10:20:00", 90.0, True),
        _make_entry("2026-01-01T10:30:00", 75.0, True),
    ])
    export_mod.export_verified_for_training()

    df = pd.read_parquet(parquet)
    timestamps = set(df["datetime"].astype(str))
    # All 4 unique timestamps survive (3 from initial + 1 new); none lost
    assert timestamps == {
        "2026-01-01 10:00:00",
        "2026-01-01 10:10:00",
        "2026-01-01 10:20:00",
        "2026-01-01 10:30:00",
    }
    assert len(df) == 4


def test_unverified_entries_skipped(tmp_paths):
    jsonl, parquet = tmp_paths
    _write_jsonl(str(jsonl), [
        _make_entry("2026-01-01T10:00:00", 80.0, True, verified=True),
        _make_entry("2026-01-01T10:10:00", 20.0, False, verified=False),
    ])

    n = export_mod.export_verified_for_training()

    assert n == 1
    df = pd.read_parquet(parquet)
    assert len(df) == 1


def test_empty_jsonl_returns_zero(tmp_paths):
    jsonl, parquet = tmp_paths
    _write_jsonl(str(jsonl), [])
    n = export_mod.export_verified_for_training()
    assert n == 0
    assert not parquet.exists()


def test_dedup_keeps_latest_version(tmp_paths):
    """If the same timestamp appears twice with different ground-truth, keep the latest."""
    jsonl, parquet = tmp_paths
    _write_jsonl(str(jsonl), [_make_entry("2026-01-01T10:00:00", 80.0, True)])
    export_mod.export_verified_for_training()

    # Re-verification corrects the row (e.g., late-arriving station data)
    corrected = _make_entry("2026-01-01T10:00:00", 80.0, False)
    corrected["actual_rain_mm"] = 0.05  # below threshold
    _write_jsonl(str(jsonl), [corrected])
    export_mod.export_verified_for_training()

    df = pd.read_parquet(parquet)
    assert len(df) == 1
    # latest version wins
    assert bool(df.iloc[0]["actual_rain"]) is False


def test_slim_reverification_keeps_issued_readings(tmp_paths):
    log, path = tmp_paths
    entry = _make_entry("2026-11-14T20:50:44+01:00", 75, False)
    entry.update(probability=0.7543, brier_component=round(0.7543**2, 6))
    _write_jsonl(log, [entry])
    export_mod.export_verified_for_training()
    entry.pop("features")
    entry.pop("brier_component")
    entry.pop("probability")
    entry.update(timestamp="2026-11-14T19:50:44+00:00", actual_rain=True, actual_rain_mm=3.6)
    _write_jsonl(log, [entry])
    assert export_mod.export_verified_for_training() == 1
    row = pd.read_parquet(path).iloc[0]
    assert (row["radar_dbz"], row["temperature_2m"]) == (12.5, 0)
    assert (row["will_rain"], row["actual_rain_mm"]) == (1, 3.6)
    assert row["brier_component"] == round((0.7543 - 1)**2, 6)


def test_reverification_without_correct_keeps_previous_value(tmp_paths):
    log, path = tmp_paths
    first = _make_entry("2026-11-14T19:50:44Z", 75, True)
    _write_jsonl(log, [first])
    export_mod.export_verified_for_training()
    other = _make_entry("2026-11-14T20:00:44Z", 10, False)
    other["correct"] = False
    compact = {k: first[k] for k in ("timestamp", "verified", "actual_rain")}
    _write_jsonl(log, [compact, other])
    export_mod.export_verified_for_training()
    rows = pd.read_parquet(path).set_index("datetime")["correct"]
    assert list(rows) == [True, False]
    flipped = dict(compact, actual_rain=False)
    _write_jsonl(log, [flipped])
    export_mod.export_verified_for_training()
    assert pd.isna(pd.read_parquet(path).set_index("datetime")["correct"].iloc[0])


def test_corrupt_previous_export_is_not_replaced(tmp_paths):
    log, path = tmp_paths
    path.write_bytes(b"not a parquet")
    _write_jsonl(log, [_make_entry("2026-11-14T19:50:44Z", 75, True)])
    with pytest.raises(ValueError):
        export_mod.export_verified_for_training()
    assert path.read_bytes() == b"not a parquet"


def test_missing_offset_does_not_match_a_utc_reading(tmp_paths):
    log, path = tmp_paths
    entry = _make_entry("2026-11-14T19:50:44Z", 75, True)
    _write_jsonl(log, [dict(entry, timestamp="2026-11-14T19:50:44")])
    with pytest.raises(ValueError, match="no offset"):
        export_mod.export_verified_for_training()
    assert not path.exists()
