#!/usr/bin/env python3
"""
Build monthly history shards from feedback_verified.parquet + current predictions_log.jsonl.
Keeps docs/predictions_log.jsonl at 5000 (35d fast path) and writes docs/history/YYYY-MM.jsonl (slim 7 fields) for every month since May.
Long-term: Vercel initial load stays 823K (~50ms parse), May 2026 -> ... history lazy-loaded on demand.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data/processed/feedback_verified.parquet"
JSONL = ROOT / "data/predictions_log.jsonl"
OUT_DIR = ROOT / "docs/history"
KEEP = {"timestamp","probability_pct","rain_category","verified","actual_rain","actual_rain_mm","correct"}

def load_parquet():
    try:
        sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))
        import pyarrow.parquet as pq
        df = pq.read_table(PARQUET).to_pandas()
        # parquet has 'datetime' column, rename to timestamp iso
        if "datetime" in df.columns:
            df["timestamp"] = df["datetime"].astype(str)
            # ensure iso with tz
            df["timestamp"] = df["timestamp"].str.replace(" ", "T") + "+00:00"
        return df
    except Exception as e:
        print(f"parquet load failed: {e}", file=sys.stderr)
        return None

def rain_cat(pct):
    if pct is None or pct != pct: return None
    if pct < 30: return "sec"
    if pct < 65: return "incert"
    return "probable"

def write_shards():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load from parquet (full history since March) + recent jsonl tail for unverified
    shards = {}
    if PARQUET.exists():
        df = load_parquet()
        if df is not None:
            for _, row in df.iterrows():
                ts = str(row.get("timestamp",""))[:10]  # YYYY-MM-DD
                if len(ts) < 7: continue
                ym = ts[:7]
                obj = {k: row[k] for k in KEEP if k in row and row[k] == row[k]}  # not NaN
                # parquet has no rain_category (only will_rain/probability_pct), derive for slim
                if "rain_category" not in obj and "probability_pct" in obj:
                    obj["rain_category"] = rain_cat(obj["probability_pct"])
                # normalize timestamp to iso
                if "timestamp" in obj:
                    obj["timestamp"] = str(obj["timestamp"])
                shards.setdefault(ym, []).append(obj)
    # Also include unverified recent from jsonl that may not yet be in parquet
    if JSONL.exists():
        for line in open(JSONL):
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
                ts=str(obj.get("timestamp",""))[:7]
                if len(ts)<7: continue
                slim={k: obj[k] for k in KEEP if k in obj}
                # dedup by timestamp
                shards.setdefault(ts, [])
                if not any(x.get("timestamp")==slim.get("timestamp") for x in shards[ts]):
                    shards[ts].append(slim)
            except: pass

    for ym, rows in sorted(shards.items()):
        if ym < "2026-05": continue  # project start May
        rows.sort(key=lambda x: x.get("timestamp",""))
        out = OUT_DIR / f"{ym}.jsonl"
        out.write_text("\n".join(json.dumps(r, separators=(",",":")) for r in rows))
        print(f"{ym}: {len(rows)} -> {out} ({out.stat().st_size/1024:.0f}K)")

if __name__ == "__main__":
    write_shards()
