# AEMET `prob_precip` investigation

> Follow-up to the preliminary benchmark in `docs/benchmark_report.md`, which
> showed AEMET with a Brier of 0.350 and a calibration bin where 553 entries
> at "98%" had only 3.1% station-level rain. That number sounded
> catastrophic. After tracing the data path the picture is more nuanced —
> AEMET is doing what it advertises, the benchmark was comparing different
> products.

## What `aemet_prob_precip` actually represents

`src/data/aemet.py:fetch_hourly_forecast` documents itself plainly:

```python
aemet_prob_precip: màxima prob. de precipitació properes 6h (0-100)
```

The implementation at `aemet.py:92-102` walks the AEMET API response, which
emits probabilities in **6-hour periods** (`"0006"`, `"0612"`, `"1218"`,
`"1824"`), and takes the **max over any period whose window overlaps
`[now, now + 6h]`**. So the feature is:

> *"Maximum probability of any rain occurring at some moment during the
> next ~6 hours, anywhere within the Cardedeu municipality polygon."*

The benchmark compared this against `actual_rain`, which is a binary "did
the local station accumulate ≥ `RAIN_THRESHOLD_MM` in the next 60 min".
**Different products**:

| | AEMET `prob_precip` | benchmark target |
|---|---|---|
| Time horizon | next ~6 h | next 60 min |
| Spatial scope | municipality (~130 km²) | single station |
| Threshold | "any precip" | ≥ `RAIN_THRESHOLD_MM` accumulated |
| Update cadence | every ~6-12 h | every 10 min |

A direct Brier comparison is structurally unfair to AEMET.

## Empirical patterns from the log

`AEMET prob distribution (1660 verified entries)`:

| Value | N |
|---|---|
| 0% | 826 |
| 100% | 454 |
| 95% | 59 |
| 40% | 54 |
| 20% | 52 |
| 35% | 49 |
| 45% | 39 |
| 10% | 34 |
| 80% | 26 |
| 90% | 19 |
| other | 38 |

AEMET emits **discrete** values, not a continuous distribution. ~77% of
entries are 0% or ≥80%.

### Per-day picture

When AEMET says ≥ 80%, did the station actually record rain that day?

| Date | max AEMET | N entries ≥80% | Station rain? |
|---|---|---|---|
| 2026-04-26 | 100% | 1 | ❌ |
| 2026-04-27 | 100% | 42 | ❌ |
| 2026-04-28 | 80% | 7 | ❌ |
| 2026-04-29 | 100% | 52 | ❌ |
| 2026-04-30 | 100% | 61 | ❌ |
| 2026-05-01 | 80% | 12 | ❌ |
| 2026-05-02 | 90% | 23 | ❌ |
| 2026-05-03 | 100% | 69 | ✅ |
| 2026-05-04 | 100% | 64 | ✅ |
| 2026-05-05 | 100% | 59 | ✅ |
| 2026-05-06 | 100% | 68 | ❌ |
| 2026-05-07 | 100% | 61 | ❌ |
| 2026-05-09 | 100% | 39 | ✅ |

**4 of 13 days = ~31% per-day hit rate when AEMET says ≥80%.**

That's the right number to interpret AEMET's signal: when AEMET goes
high for the municipality, roughly 1 in 3 days has rain at this specific
station. The remaining 2/3 are days where rain occurred elsewhere in the
municipality (or below the verification threshold), or AEMET was simply
wrong at regional resolution.

### Hit rate at 6h (the windowing AEMET actually advertises)

For each verified prediction, "did it rain anywhere in the next 6h log
window?":

| AEMET bin | N | rain in next 6h |
|---|---|---|
| 0-20% | 867 | 7.8% |
| 20-40% | 101 | 0.0% |
| 40-60% | 107 | 0.0% |
| 60-80% | 32 | 0.0% |
| 80-100% | 553 | 17.9% |

Even on its own time horizon AEMET is poorly calibrated for this station
— the 80-100% bin should observe rain ≥ 80% of the time if calibrated.
The middle bins observing 0% is partly a discrete-value artifact (AEMET
rarely emits 20-80%, and when it does the few entries fall on dry days).

## What this means for the project

1. **AEMET is not catastrophically broken.** It's a regional 6h forecast
   product, not a station-level 60-min nowcast. The benchmark's headline
   "Brier 0.35" reflects the apples-to-oranges comparison, not AEMET's
   actual quality on its own terms.

2. **AEMET should not be a primary baseline in the benchmark.** It's
   structurally lower-resolution. Honest fair baselines for the local
   model are:
   - Ensemble agreement of the 4 NWPs (already in the benchmark)
   - Open-Meteo's `precipitation_probability` at 1h resolution (already
     stored in features as part of the per-model prob, but not currently
     surfaced as an aggregate baseline)

3. **AEMET's role is correctly that of a feature, not a target.** The
   XGBoost uses `aemet_prob_precip` as one of 209 inputs — it gives a
   regional context signal that sometimes correlates with station-level
   rain (the 4 hit days). The model learns from when this regional signal
   does and doesn't translate.

4. **The period filter at `aemet.py:97-102` is correct** — initial
   reading flagged it as a "micro-bug" but a second look (and a
   reviewer's pushback) shows it's doing the right thing. The filter
   `h_start <= current_hour + 6 and h_end > current_hour` keeps **any**
   6h AEMET period that overlaps the [now, now + 6h] forecast window,
   and the max over them is conservative — pessimistic in either edge
   of the window. Tightening it to "current period only" would, at
   11:00, exclude the 12-18 period that covers 5 of the next 6 hours.
   The conservative max is the intended semantic.

   The only remaining design question is whether a duration-weighted
   average (instead of max) would be more meaningful. That's a feature
   redesign, not a bug fix.

## Recommended follow-ups

| Action | Where | Priority |
|---|---|---|
| Update `benchmark_vs_baselines.py` to disclose AEMET's window mismatch and de-emphasise it as a baseline | `scripts/` | High |
| Add a per-day hit-rate metric (more honest for AEMET) | `scripts/` | Medium |
| Add Open-Meteo per-hour `precipitation_probability` as a true 60-min baseline | features pipeline + script | Medium |
| (withdrawn — period filter is correct, see point 4 above) | — | — |
| Re-run the benchmark when more rain events accumulate (4 days isn't enough) | quarterly | High |

## What this does NOT change

The local XGBoost still leads the benchmark in Brier on the apples-to-apples
subset (0.039 [0.033, 0.044] vs 0.136 [0.123, 0.147] for ensemble), even
without AEMET in the picture. The "local model wins" headline survives —
it just has one less baseline to claim victory over.
