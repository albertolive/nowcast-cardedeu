import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HISTORY_DIR = ROOT / "docs/history"
PREDICTIONS_LOG = ROOT / "docs/predictions_log.jsonl"
APP_JS = ROOT / "docs/app.js"
PARQUET = ROOT / "data/processed/feedback_verified.parquet"

def test_shards_exist_and_have_may_start():
    # May 2026 is project start, must exist and have July 1-24 (3400) not just 25-31
    for ym in ["2026-05", "2026-06", "2026-07", "2026-08"]:
        p = HISTORY_DIR / f"{ym}.jsonl"
        assert p.exists(), f"missing shard {ym}"
        lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) > 3000, f"{ym} too small {len(lines)} <3000, would be truncated 921 bug"
        # Check rain_category present (fix for parquet-derived shards)
        assert "rain_category" in lines[0], f"{ym} missing rain_category"
        assert lines[0]["rain_category"] in ("sec","incert","probable")
        # Check first timestamp for May and July
        if ym == "2026-05":
            assert lines[0]["timestamp"].startswith("2026-05-01")
            assert lines[-1]["timestamp"].startswith("2026-05-31")
        if ym == "2026-07":
            assert lines[0]["timestamp"].startswith("2026-07-01"), f"July shard must start 07-01, got {lines[0]['timestamp']}"
            # July 1-24 should be present (3400)
            july_1_24 = [r for r in lines if "2026-07-01" <= r["timestamp"][:10] <= "2026-07-24"]
            assert len(july_1_24) >= 3000, f"July 1-24 missing {len(july_1_24)}"

def test_predictions_log_trimmed_but_not_too_much():
    lines = [l for l in PREDICTIONS_LOG.read_text().splitlines() if l.strip()]
    assert 5000 <= len(lines) <= 6000, f"predictions_log should be 5k-6k, got {len(lines)}"
    first = json.loads(lines[0])["timestamp"]
    # Should be within 35-40 days of now (July 25 for Aug 30)
    assert first.startswith("2026-07-"), f"first should be July after trim, got {first}"

def test_app_js_shard_logic():
    txt = APP_JS.read_text()
    assert "HISTORY_SHARDS" in txt, "app.js missing HISTORY_SHARDS"
    assert "fetchShard" in txt, "app.js missing fetchShard"
    assert "SHARD_CACHE" in txt
    # July fix: should not have the buggy guard that skipped July when 25-31 already in dayMap
    assert "Fix: fetch even if dayMap already has some days" in txt or "HISTORY_SHARDS.includes(ym) && !SHARD_CACHE.has(ym)" in txt
    assert "fullHistory" in txt, "app.js must merge shards for metrics from May start"
    assert "recallPct" in txt and "precisionPct" in txt, "app.js missing recall/precision definitions"

def test_parquet_has_may():
    # Ensure parquet still has May for future shard builds
    import sys
    sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))
    import pyarrow.parquet as pq
    df = pq.read_table(PARQUET).to_pandas()
    assert len(df) > 15000, f"parquet too small {len(df)}"
    # Check May exists
    has_may = df["datetime"].astype(str).str.startswith("2026-05").any()
    assert has_may, "parquet missing May 2026"
