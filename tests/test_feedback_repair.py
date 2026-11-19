import json
import os
import subprocess

import pandas as pd
import pytest

from src.feedback import export
from src.feedback.archive import iter_snapshots
from src.feedback import logger as feedback_log


@pytest.fixture
def archive(tmp_path, monkeypatch):
    log = tmp_path / "verified.jsonl"
    parquet = tmp_path / "verified.parquet"
    monkeypatch.setattr(feedback_log, "PREDICTIONS_LOG", str(log))
    monkeypatch.setattr(export, "FEEDBACK_TRAINING_PATH", str(parquet))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "data/latest_prediction.json"
    path.parent.mkdir()

    def publish(snapshot):
        path.write_text(json.dumps(snapshot))
        subprocess.run(["git", "add", "data/latest_prediction.json"], cwd=tmp_path, check=True)
        published = "2026-11-18T18:00:00Z"
        subprocess.run(["git", "-c", "user.name=weather", "-c", "user.email=weather@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "forecast"],
                       env={**os.environ, "GIT_AUTHOR_DATE": published, "GIT_COMMITTER_DATE": published},
                       cwd=tmp_path, check=True)
    return tmp_path, log, parquet, publish


def test_archive_uses_issue_time_and_first_publication(archive):
    repo, _, _, publish = archive
    for ts, val in [("2026-11-17T22:00:00Z", 5), ("2026-11-17T20:01:00Z", 1),
                    ("2026-11-17T21:00:00+01:00", 0), ("2026-11-17T20:00:00Z", 9),
                    ("2026-11-17T20:02:00", 12)]:
        publish({"timestamp": ts, "value": val})
    rows = iter_snapshots(pd.Timestamp("2026-11-17T20:00Z"), pd.Timestamp("2026-11-17T22:00Z"), repo=repo)
    assert [s["value"] for s in rows] == [0, 1]


def test_repair_uses_exact_inputs_and_rejects_an_empty_new_batch(archive):
    repo, log, parquet, publish = archive
    start = pd.Timestamp("2026-11-17T00:00:00Z")
    end = pd.Timestamp("2026-11-18T18:30:00Z")
    times = pd.date_range(start, periods=201, freq="10min")
    entries = [{"timestamp": t.isoformat(), "verified": True, "actual_rain": False,
                "probability_pct": 5, "features": {"temperature_2m": 10}} for t in times]
    del entries[0]["features"], entries[1]["features"]
    entries[0].update(actual_rain=True)
    publish({"timestamp": times[0].isoformat(), "feature_vector": {"radar_dbz": 0, "will_rain": 0},
             "actual_rain": False})
    # One microsecond is a different forecast. A later gauge reading isn't a feature either.
    publish({"timestamp": (times[1] + pd.Timedelta(microseconds=1)).isoformat(),
             "feature_vector": {"radar_dbz": 32}, "actual_rain_mm": 4.0})

    def write_log():
        log.write_text("\n".join(map(json.dumps, entries)))

    write_log()
    result = export.repair_feedback(since=start, as_of=end, repo=repo, dry_run=True)
    assert result["from_snapshots"] == 1 and not parquet.exists()
    export.repair_feedback(since=start, as_of=end, repo=repo)
    data = pd.read_parquet(parquet).set_index("datetime")
    assert data.iloc[0]["radar_dbz"] == 0 and data.iloc[0]["will_rain"] == 1
    assert pd.isna(data.iloc[1]["radar_dbz"]) and data.iloc[1]["will_rain"] == 0
    original = parquet.read_bytes()
    assert export.repair_feedback(since=start, as_of=end, repo=repo)["new"] == 0
    assert original == parquet.read_bytes()
    entries.append({"timestamp": "2026-11-18T18:00:00Z", "verified": True, "actual_rain": True})
    write_log()
    with pytest.raises(ValueError, match="new verified rows: 1/1 lack issued inputs"):
        export.repair_feedback(since=start, as_of=end, repo=repo)
    assert original == parquet.read_bytes()
    with pytest.raises(ValueError, match="No current verified feedback"):
        export.repair_feedback(since=start, as_of=end + pd.Timedelta(days=3), repo=repo)
    assert original == parquet.read_bytes()


def test_wall_clock_recovery_requires_matching_issued_probability(archive):
    repo, log, parquet, publish = archive
    legacy = {"timestamp": "2026-04-20T19:04:56.495644", "verified": True,
              "actual_rain": True, "probability_pct": 0.5}
    recent = dict(legacy, timestamp="2026-11-18T18:00:00Z", features={"temperature_2m": 9})
    log.write_text(json.dumps(recent))
    pd.DataFrame([{"datetime": pd.Timestamp(legacy["timestamp"]), "will_rain": 1,
                   "verified": True, "actual_rain": True, "probability_pct": 1.5}]).to_parquet(parquet)
    forecast = dict(legacy, feature_vector={"radar_dbz": 30})
    publish(forecast)
    args = dict(since=pd.Timestamp("2026-04-01T00:00:00Z"),
                as_of=pd.Timestamp("2026-11-18T18:30:00Z"), repo=repo)
    assert export.repair_feedback(**args)["from_snapshots"] == 0
    assert export.repair_feedback(**args, legacy_ref="HEAD")["from_snapshots"] == 0
    corrected = pd.read_parquet(parquet)
    corrected.loc[corrected.datetime == legacy["timestamp"], "probability_pct"] = 0.5
    corrected.to_parquet(parquet)
    assert export.repair_feedback(**args, legacy_ref="HEAD")["from_snapshots"] == 1
    assert pd.read_parquet(parquet).iloc[0]["radar_dbz"] == 30
