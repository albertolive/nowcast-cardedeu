#!/usr/bin/env python3
"""Deep gap analysis of the training dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/training_dataset.parquet")
print("=" * 80)
print("DEEP GAP ANALYSIS")
print("=" * 80)
print(f"Rows: {len(df):,}")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print()

# NaN rates for key columns
print("NaN RATES (the gaps):")
print("-" * 80)
key_cols = [
    "wind_850_dir", "wind_850_speed", "temp_850", "temp_500",
    "rh_850", "rh_700", "temp_700",
    "vt_index", "tt_index", "li_index",
    "wind_shear_speed", "wind_shear_dir",
    "is_tramuntana", "is_llevantada", "is_garbi", "is_ponent", "is_migjorn",
    "is_sea_breeze",
    "ensemble_rain_agreement", "ensemble_precip_spread", "ensemble_models_rain",
    "sentinel_temp_diff", "sentinel_humidity_diff", "sentinel_precip",
    "local_rain_xema", "local_rain_xema_3h",
    "model_predicts_precip", "weather_code", "precipitation",
    "wind_u", "wind_v",
    "tramuntana_strength", "tramuntana_moisture",
    "llevantada_strength", "llevantada_moisture",
]
for c in key_cols:
    if c not in df.columns:
        print(f"  {c:35s}  MISSING from dataset")
        continue
    nan_ct = df[c].isna().sum()
    nan_pct = 100 * nan_ct / len(df)
    zero_ct = (df[c] == 0).sum()
    zero_pct = 100 * zero_ct / len(df)
    if nan_pct > 0.1:
        print(f"  {c:35s}  {nan_ct:6d} NaN ({nan_pct:5.1f}%)  |  {zero_ct:6d} zeros ({zero_pct:5.1f}%)")
    else:
        print(f"  {c:35s}  OK ({nan_pct:.1f}% NaN)  |  {zero_ct:6d} zeros ({zero_pct:5.1f}%)")

# When do pressure levels appear?
print()
print("PRESSURE LEVEL DATA BY YEAR:")
print("-" * 80)
df["year"] = pd.to_datetime(df["datetime"]).dt.year
for yr in sorted(df["year"].unique()):
    yr_mask = df["year"] == yr
    yr_n = yr_mask.sum()
    w850 = df.loc[yr_mask, "wind_850_dir"].notna().mean() * 100
    ens = df.loc[yr_mask, "ensemble_rain_agreement"].notna().mean() * 100 if "ensemble_rain_agreement" in df.columns else 0
    sent = df.loc[yr_mask, "sentinel_temp_diff"].notna().mean() * 100 if "sentinel_temp_diff" in df.columns else 0
    print(f"  {yr}: {yr_n:6,} rows  |  850hPa: {w850:5.1f}%  |  Ensemble: {ens:5.1f}%  |  Sentinel: {sent:5.1f}%")

# Where is rain happening vs data availability?
print()
print("RAIN EVENTS vs DATA AVAILABILITY BY YEAR:")
print("-" * 80)
TARGET = "will_rain"
for yr in sorted(df["year"].unique()):
    yr_mask = df["year"] == yr
    rain_ct = df.loc[yr_mask, TARGET].sum()
    rain_pct = 100 * df.loc[yr_mask, TARGET].mean()
    has_wind = df.loc[yr_mask, "wind_850_dir"].notna()
    rain_with_wind = df.loc[yr_mask & has_wind, TARGET].sum()
    rain_without_wind = rain_ct - rain_with_wind
    print(f"  {yr}: {int(rain_ct):4d} rain events ({rain_pct:4.1f}%)  "
          f"|  {int(rain_with_wind):4d} with 850hPa  |  {int(rain_without_wind):4d} WITHOUT 850hPa")

# is_sea_breeze analysis — what regimes does it overlap with?
print()
print("is_sea_breeze OVERLAP WITH WIND REGIMES:")
print("-" * 80)
sb = df["is_sea_breeze"] == 1
for flag in ["is_tramuntana", "is_llevantada", "is_garbi", "is_ponent", "is_migjorn"]:
    if flag not in df.columns:
        continue
    overlap = (sb & (df[flag] == 1)).sum()
    if sb.sum() > 0:
        pct = 100 * overlap / sb.sum()
        print(f"  is_sea_breeze + {flag:20s}  {overlap:5d} ({pct:5.1f}% of sea breezes)")
# How many sea breezes are NOT any regime?
any_regime = ((df.get("is_tramuntana", 0) == 1) | (df.get("is_llevantada", 0) == 1) |
              (df.get("is_garbi", 0) == 1) | (df.get("is_ponent", 0) == 1) |
              (df.get("is_migjorn", 0) == 1))
sb_no_regime = (sb & ~any_regime).sum()
print(f"  is_sea_breeze + NO regime:        {sb_no_regime:5d} ({100*sb_no_regime/sb.sum():.1f}% of sea breezes)")

# Model dominance analysis
print()
print("MODEL DOMINANCE ANALYSIS:")
print("-" * 80)
# When model says rain, how often does it actually rain?
mp = df["model_predicts_precip"] == 1
print(f"  model_predicts_precip=1: {mp.sum():,} samples ({100*mp.mean():.1f}%)")
print(f"    Rain rate when model says rain: {100*df.loc[mp, TARGET].mean():.1f}%")
print(f"    Rain rate when model says dry:  {100*df.loc[~mp, TARGET].mean():.1f}%")
# Weather codes that predict rain
wc = df["weather_code"]
rain_codes = wc.isin([51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99])
print(f"  Rain weather codes: {rain_codes.sum():,} ({100*rain_codes.mean():.1f}%)")
print(f"    Rain rate with rain code: {100*df.loc[rain_codes, TARGET].mean():.1f}%")
print(f"    Rain rate without rain code: {100*df.loc[~rain_codes, TARGET].mean():.1f}%")

# The key question: when model is WRONG, what local features could help?
print()
print("NWP ERROR ANALYSIS (where local features matter):")
print("-" * 80)
fp = mp & (df[TARGET] == 0)  # False positives: model says rain but no rain
fn = ~mp & (df[TARGET] == 1)  # False negatives: model says dry but it rained
tp = mp & (df[TARGET] == 1)   # True positives
tn = ~mp & (df[TARGET] == 0)   # True negatives
print(f"  True Positives (model=rain, actual=rain):  {tp.sum():5d}")
print(f"  False Positives (model=rain, actual=dry):  {fp.sum():5d}")
print(f"  True Negatives (model=dry, actual=dry):    {tn.sum():5d}")
print(f"  False Negatives (model=dry, actual=rain):  {fn.sum():5d}")
print()
print(f"  Model precision: {100*tp.sum()/(tp.sum()+fp.sum()):.1f}%")
print(f"  Model recall:    {100*tp.sum()/(tp.sum()+fn.sum()):.1f}%")
print(f"  Model accuracy:  {100*(tp.sum()+tn.sum())/len(df):.1f}%")

# For false negatives: what wind regimes?
print()
print("  FALSE NEGATIVES (missed rain) by wind regime:")
for regime in ["Tramuntana (N)", "NE", "Llevantada (E/SE)", "Migjorn (S)",
               "Garbi (SW)", "Ponent (W/NW)", "Unknown"]:
    mask = (df.get("wind_regime", pd.Series(["Unknown"]*len(df))) == regime) if "wind_regime" in df.columns else pd.Series([False]*len(df))
    # Recompute wind regime
fn_count = fn.sum()
print(f"  Total false negatives (model missed rain): {int(fn_count)}")
print(f"  FN with 850hPa data: {(fn & df['wind_850_dir'].notna()).sum()}")
print(f"  FN without 850hPa data: {(fn & df['wind_850_dir'].isna()).sum()}")

# Check pressure levels file directly
print()
print("RAW DATA FILES:")
print("-" * 80)
import os
for f in ["data/open_meteo_hourly.parquet", "data/pressure_levels_hourly.parquet",
          "data/processed/ensemble_historical.parquet", "data/processed/xema_sentinel_cache.parquet"]:
    if os.path.exists(f):
        d = pd.read_parquet(f)
        print(f"  {f}")
        print(f"    Rows: {len(d):,}  Cols: {len(d.columns)}")
        if "datetime" in d.columns or "date" in d.columns:
            dt_col = "datetime" if "datetime" in d.columns else "date"
            print(f"    Range: {d[dt_col].min()} to {d[dt_col].max()}")
        print(f"    Columns: {list(d.columns)}")
        print()
    else:
        print(f"  {f}: NOT FOUND")
