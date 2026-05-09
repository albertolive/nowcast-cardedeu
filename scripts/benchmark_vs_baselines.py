"""
Honest benchmark: local nowcast vs the global baselines that were already
recorded alongside each historical prediction.

Reads data/predictions_log.jsonl, where each verified entry carries the
predictor outputs we want to compare:
  - probability             → local XGBoost (calibrated, 0..1)
  - features.aemet_prob_precip   → AEMET hourly prob (0..100, gated)
  - features.ensemble_rain_agreement → fraction of 4 NWP members predicting
                                       rain in the next hour (0..1)

Output: a markdown report (`docs/benchmark_report.md` by default) with
Brier, AUC, log loss, accuracy at the operating threshold, calibration
bins, and bootstrap 95% confidence intervals.

WHY BOOTSTRAP CIs — at the time of writing the log holds ~3.8k verified
predictions but only ~43 positives spread across 4 distinct days. Any
single-number comparison (e.g. "Brier 0.014 vs 0.018") at this base rate
is dominated by sampling noise; the CI tells you whether the difference
survives resampling. If the CIs overlap heavily, no winner can be
claimed honestly.

Run:
  python3 scripts/benchmark_vs_baselines.py [--log PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class Sample:
    y: int          # 1 if rain occurred in the verification window, else 0
    p_local: float  # local model probability (0..1)
    p_aemet: float | None
    p_ensemble: float | None
    radar_near_km: float | None
    station_raining: bool | None  # at prediction time
    timestamp: str


def load_samples(log_path: Path) -> list[Sample]:
    out: list[Sample] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if not e.get("verified"):
                continue
            actual = e.get("actual_rain")
            if actual is None:
                continue
            prob = e.get("probability")
            if prob is None and e.get("probability_pct") is not None:
                prob = e["probability_pct"] / 100.0
            if prob is None:
                continue
            features = e.get("features") or {}
            aemet = features.get("aemet_prob_precip")
            p_aemet = aemet / 100.0 if aemet is not None else None
            p_ens = features.get("ensemble_rain_agreement")
            out.append(
                Sample(
                    y=int(bool(actual)),
                    p_local=float(prob),
                    p_aemet=p_aemet,
                    p_ensemble=float(p_ens) if p_ens is not None else None,
                    radar_near_km=features.get("radar_nearest_echo_km"),
                    station_raining=e.get("station_raining_now"),
                    timestamp=e.get("timestamp", ""),
                )
            )
    return out


# ── Metrics ────────────────────────────────────────────────────────────

def brier(ys: list[int], ps: list[float]) -> float:
    return mean((p - y) ** 2 for y, p in zip(ys, ps))


def log_loss(ys: list[int], ps: list[float], eps: float = 1e-12) -> float:
    losses = []
    for y, p in zip(ys, ps):
        p = min(max(p, eps), 1 - eps)
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return mean(losses)


def auc(ys: list[int], ps: list[float]) -> float | None:
    """ROC AUC via the rank-sum identity. Returns None if degenerate."""
    pos = [p for y, p in zip(ys, ps) if y == 1]
    neg = [p for y, p in zip(ys, ps) if y == 0]
    if not pos or not neg:
        return None
    pairs = 0
    wins = 0.0
    # O(n*m) — fine for our sample sizes.
    for pp in pos:
        for pn in neg:
            pairs += 1
            if pp > pn:
                wins += 1
            elif pp == pn:
                wins += 0.5
    return wins / pairs


def accuracy_at(ys: list[int], ps: list[float], threshold: float) -> float:
    return mean(int((p >= threshold) == bool(y)) for y, p in zip(ys, ps))


def calibration_bins(ys: list[int], ps: list[float], n_bins: int = 5):
    """Returns (bin_lo, bin_hi, n, predicted_mean, observed_freq)."""
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [(y, p) for y, p in zip(ys, ps) if lo <= p < hi or (i == n_bins - 1 and p == 1.0)]
        if not in_bin:
            bins.append((lo, hi, 0, None, None))
            continue
        pred_mean = mean(p for _, p in in_bin)
        obs_mean = mean(y for y, _ in in_bin)
        bins.append((lo, hi, len(in_bin), pred_mean, obs_mean))
    return bins


def bootstrap_ci(
    ys: list[int],
    ps: list[float],
    metric_fn,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float] | None:
    rng = random.Random(seed)
    n = len(ys)
    if n == 0:
        return None
    point = metric_fn(ys, ps)
    if point is None:
        return None
    estimates = []
    for _ in range(n_iter):
        idxs = [rng.randrange(n) for _ in range(n)]
        bys = [ys[i] for i in idxs]
        bps = [ps[i] for i in idxs]
        v = metric_fn(bys, bps)
        if v is not None:
            estimates.append(v)
    if not estimates:
        return None
    estimates.sort()
    lo = estimates[int((alpha / 2) * len(estimates))]
    hi = estimates[int((1 - alpha / 2) * len(estimates))]
    return point, lo, hi


# ── Reporting ──────────────────────────────────────────────────────────

def evaluate(name: str, ys: list[int], ps: list[float]) -> dict:
    res: dict = {"name": name, "n": len(ys), "n_pos": sum(ys)}
    res["base_rate"] = (sum(ys) / len(ys)) if ys else None
    res["brier"] = bootstrap_ci(ys, ps, brier)
    res["log_loss"] = bootstrap_ci(ys, ps, log_loss)
    res["auc"] = bootstrap_ci(ys, ps, auc)
    res["acc@0.5"] = bootstrap_ci(ys, ps, lambda y, p: accuracy_at(y, p, 0.5))
    res["acc@0.65"] = bootstrap_ci(ys, ps, lambda y, p: accuracy_at(y, p, 0.65))
    res["calibration"] = calibration_bins(ys, ps)
    return res


def fmt_ci(triple) -> str:
    if triple is None:
        return "—"
    point, lo, hi = triple
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def render_table(rows: list[dict]) -> str:
    cols = ["name", "n", "n_pos", "base_rate", "brier", "log_loss", "auc", "acc@0.5", "acc@0.65"]
    headers = ["predictor", "N", "N+", "base", "Brier ↓", "LogLoss ↓", "AUC ↑", "Acc@0.5", "Acc@0.65"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for c, h in zip(cols, headers):
            v = r.get(c)
            if c == "base_rate":
                cells.append(f"{v:.3f}" if v is not None else "—")
            elif c in ("n", "n_pos"):
                cells.append(str(v))
            elif c == "name":
                cells.append(v)
            else:
                cells.append(fmt_ci(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_calibration(name: str, bins) -> str:
    out = [f"### {name} — calibration"]
    out.append("| bin | N | predicted mean | observed freq |")
    out.append("|---|---|---|---|")
    for lo, hi, n, pmean, ofreq in bins:
        if n == 0:
            out.append(f"| {lo:.2f}–{hi:.2f} | 0 | — | — |")
        else:
            out.append(f"| {lo:.2f}–{hi:.2f} | {n} | {pmean:.3f} | {ofreq:.3f} |")
    return "\n".join(out)


# ── Stratifications ────────────────────────────────────────────────────

def filter_full_baselines(samples: list[Sample]) -> list[Sample]:
    return [s for s in samples if s.p_aemet is not None and s.p_ensemble is not None]


def filter_anticipation(samples: list[Sample]) -> list[Sample]:
    """Strict anticipation: not raining at station AND no echo within 15 km."""
    out = []
    for s in samples:
        if s.station_raining is True:
            continue
        if s.radar_near_km is None or s.radar_near_km < 15:
            continue
        out.append(s)
    return out


def make_arrays(samples: list[Sample], predictor: str) -> tuple[list[int], list[float]]:
    ys, ps = [], []
    for s in samples:
        p = {"local": s.p_local, "aemet": s.p_aemet, "ensemble": s.p_ensemble}[predictor]
        if p is None:
            continue
        ys.append(s.y)
        ps.append(p)
    return ys, ps


def evaluate_all(samples: list[Sample], label: str) -> str:
    rows = []
    for predictor, pretty in [("local", "Local (XGBoost)"), ("aemet", "AEMET prob_precip"),
                               ("ensemble", "Ensemble agreement (4 NWP)")]:
        ys, ps = make_arrays(samples, predictor)
        if not ys:
            continue
        rows.append(evaluate(pretty, ys, ps))
    if not rows:
        return f"\n## {label}\n\n_No samples available._\n"
    out = [f"\n## {label}"]
    out.append(f"\nN={len(samples)}, positives={sum(s.y for s in samples)}")
    out.append("")
    out.append(render_table(rows))
    out.append("")
    for r in rows:
        out.append(render_calibration(r["name"], r["calibration"]))
        out.append("")
    return "\n".join(out)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/predictions_log.jsonl")
    ap.add_argument("--out", default="docs/benchmark_report.md")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    samples = load_samples(log_path)
    full = filter_full_baselines(samples)
    anticipation = filter_anticipation(full)

    n_total = len(samples)
    n_full = len(full)
    n_anti = len(anticipation)
    pos_total = sum(s.y for s in samples)
    pos_full = sum(s.y for s in full)
    pos_anti = sum(s.y for s in anticipation)
    distinct_rain_days = len({s.timestamp[:10] for s in samples if s.y == 1})

    parts = []
    parts.append("# Benchmark vs baselines")
    parts.append("")
    parts.append("> ⚠️ **PRELIMINARY** — sample size is small. The headline finding is "
                 f"that there are only **{pos_total} rain events across {distinct_rain_days} "
                 "distinct days** in the verified log. Confidence intervals below are wide; "
                 "any 'winner' claim with overlapping CIs is not supported.")
    parts.append("")
    parts.append(f"- Source log: `{log_path}`")
    parts.append(f"- Verified predictions: **{n_total}** (positives: **{pos_total}**)")
    parts.append(f"- Predictions with both AEMET and ensemble baselines: **{n_full}** "
                 f"(positives: **{pos_full}**)")
    parts.append(f"- Strict anticipation subset (station dry + radar > 15 km): **{n_anti}** "
                 f"(positives: **{pos_anti}**)")
    parts.append("")
    parts.append("Metrics:")
    parts.append("- **Brier ↓**: mean squared error between predicted prob and outcome. Lower is better.")
    parts.append("- **LogLoss ↓**: penalises confident mistakes harder. Lower is better.")
    parts.append("- **AUC ↑**: rank-discrimination. 0.5 = chance, 1.0 = perfect ordering.")
    parts.append("- **Acc@τ**: accuracy when the predictor is thresholded at τ. Sensitive to base rate at low N.")
    parts.append("- **Calibration**: predicted prob within a bin should match observed frequency.")
    parts.append("")
    parts.append("All intervals are 95% bootstrap CIs over the per-row sample (1000 iterations).")

    parts.append(evaluate_all(samples, "Global — local on full sample"))
    parts.append(evaluate_all(full, "Apples-to-apples — entries with all three predictors"))
    parts.append(evaluate_all(anticipation, "Anticipation only — station dry & radar > 15 km"))

    parts.append("\n## How to read this\n")
    parts.append("If the local model's Brier CI fully sits below the baselines' CIs, the "
                 "local model is calibrated better. If they overlap (which is likely at this "
                 "sample size), conclude **inconclusive** — the data isn't there yet.\n")
    parts.append("AEMET and ensemble predictions are only present when the rain gate was open "
                 "(see `predict.py:265-275`). The full-baseline subset therefore over-represents "
                 "moments where rain was already plausible — a selection bias inflating all "
                 "predictors' apparent accuracy. Treat absolute numbers cautiously and focus "
                 "on relative ordering.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  total verified: {n_total}, positives: {pos_total}, rainy days: {distinct_rain_days}")
    print(f"  full-baseline subset: {n_full}, positives: {pos_full}")
    print(f"  anticipation subset: {n_anti}, positives: {pos_anti}")


if __name__ == "__main__":
    main()
