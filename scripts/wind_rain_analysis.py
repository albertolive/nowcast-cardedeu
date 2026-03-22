#!/usr/bin/env python3
"""Analyze wind direction vs rain events in the training dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/training_dataset.parquet")
TARGET = "will_rain"
print(f"Dataset: {len(df)} rows")
print(f"Rain events ({TARGET}=1): {int(df[TARGET].sum())} ({100*df[TARGET].mean():.1f}%)")
print()

# Wind direction at 850hPa (synoptic level)
if "wind_850_dir" in df.columns:
    wd = df["wind_850_dir"]
    print("Using wind_850_dir (synoptic level)")
else:
    wd = (270 - np.degrees(np.arctan2(df["wind_v"], df["wind_u"]))) % 360
    print("Using surface wind (u/v)")


def classify_wind(d):
    if pd.isna(d):
        return "Unknown"
    d = d % 360
    if d >= 340 or d < 30:
        return "Tramuntana (N)"
    elif 30 <= d < 60:
        return "NE"
    elif 60 <= d < 150:
        return "Llevantada (E/SE)"
    elif 150 <= d < 190:
        return "Migjorn (S)"
    elif 190 <= d < 250:
        return "Garbi (SW)"
    elif 250 <= d < 340:
        return "Ponent (W/NW)"
    return "Unknown"


df["wind_regime"] = wd.apply(classify_wind)

print()
print("=" * 80)
print("WIND REGIME vs RAIN ANALYSIS")
print("=" * 80)
print(f"{'Regime':25s}  {'Samples':>8s}  {'Rain':>6s}  {'Rain%':>6s}  {'Share of all rain':>18s}")
print("-" * 80)

total_rain = df[TARGET].sum()
for regime in ["Tramuntana (N)", "NE", "Llevantada (E/SE)", "Migjorn (S)",
               "Garbi (SW)", "Ponent (W/NW)", "Unknown"]:
    mask = df["wind_regime"] == regime
    n = mask.sum()
    if n == 0:
        continue
    rain = df.loc[mask, TARGET].sum()
    rain_pct = 100 * rain / n
    rain_share = 100 * rain / total_rain
    print(f"{regime:25s}  {n:8d}  {int(rain):6d}  {rain_pct:5.1f}%  {rain_share:17.1f}%")

print(f"{'TOTAL':25s}  {len(df):8d}  {int(total_rain):6d}  {100*total_rain/len(df):5.1f}%  {'100.0%':>18s}")

# Using boolean flags
print()
print("=" * 80)
print("USING BOOLEAN FLAGS FROM FEATURE ENGINEERING")
print("=" * 80)
for flag in ["is_tramuntana", "is_llevantada", "is_garbi", "is_ponent",
             "is_migjorn", "is_sea_breeze"]:
    if flag not in df.columns:
        print(f"  {flag}: NOT in dataset")
        continue
    mask = df[flag] == 1
    n = mask.sum()
    if n == 0:
        print(f"  {flag}: 0 samples (all zeros)")
        continue
    rain = df.loc[mask, TARGET].sum()
    rain_pct = 100 * rain / n
    rain_share = 100 * rain / total_rain
    print(f"  {flag:25s}  {n:6d} samples  {int(rain):5d} rain  "
          f"{rain_pct:5.1f}% rain rate  {rain_share:5.1f}% of all rain")

# Conditional rain rate comparison
print()
print("=" * 80)
print("RAIN RATE BY REGIME (which wind brings most rain?)")
print("=" * 80)
regime_stats = []
for regime in ["Tramuntana (N)", "NE", "Llevantada (E/SE)", "Migjorn (S)",
               "Garbi (SW)", "Ponent (W/NW)"]:
    mask = df["wind_regime"] == regime
    n = mask.sum()
    if n == 0:
        continue
    rain_rate = 100 * df.loc[mask, TARGET].mean()
    regime_stats.append((regime, rain_rate, n))

regime_stats.sort(key=lambda x: -x[1])
for regime, rate, n in regime_stats:
    bar = "█" * int(rate * 2)
    print(f"  {regime:25s}  {rate:5.1f}%  {bar}  (n={n})")
