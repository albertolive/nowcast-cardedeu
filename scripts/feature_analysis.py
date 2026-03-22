#!/usr/bin/env python3
"""Feature importance analysis for the trained XGBoost model."""
import json
import sys
import os

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Load metrics for sample count
metrics_path = os.path.join(os.path.dirname(config.MODEL_PATH), "metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)
n_samples = metrics.get("n_samples", "?")

# Load model
model = xgb.XGBClassifier()
model.load_model(config.MODEL_PATH)
with open(config.FEATURE_NAMES_PATH) as f:
    feature_names = json.load(f)

# Get importance (3 types)
gain = model.get_booster().get_score(importance_type="gain")
weight = model.get_booster().get_score(importance_type="weight")
cover = model.get_booster().get_score(importance_type="cover")

# Build comprehensive DataFrame — booster uses actual feature names as keys
rows = []
for i, fname in enumerate(feature_names):
    rows.append({
        "feature": fname,
        "gain": gain.get(fname, 0),
        "weight": weight.get(fname, 0),
        "cover": cover.get(fname, 0),
        "sklearn_importance": model.feature_importances_[i],
    })

df = pd.DataFrame(rows)

# Normalize gain for percentage
total_gain = df["gain"].sum()
df["gain_pct"] = 100 * df["gain"] / total_gain

# Sort by gain (most meaningful importance type)
df = df.sort_values("gain", ascending=False)

print("=" * 90)
print(f"FEATURE IMPORTANCE ANALYSIS — {len(feature_names)} features, {n_samples} samples")
print("=" * 90)
print()

# Top features
print("TOP 20 FEATURES (by information gain):")
header = f"{'Feature':35s} {'Gain%':>7s} {'Splits':>7s} {'Cover':>9s}  Bar"
print(header)
print("-" * 90)
for _, r in df.head(20).iterrows():
    gp = r["gain_pct"] if not pd.isna(r["gain_pct"]) else 0
    w = int(r["weight"]) if not pd.isna(r["weight"]) else 0
    c = r["cover"] if not pd.isna(r["cover"]) else 0
    bar = "█" * int(gp * 2)
    print(f"{r['feature']:35s} {gp:6.2f}% {w:>7d} {c:9.1f}  {bar}")

print()

# Bottom features (potential dead weight)
print("BOTTOM 15 FEATURES (lowest gain — candidates for removal):")
header2 = f"{'Feature':35s} {'Gain%':>7s} {'Splits':>7s} {'Cover':>9s}"
print(header2)
print("-" * 90)
for _, r in df.tail(15).iterrows():
    gp = r['gain_pct'] if not pd.isna(r['gain_pct']) else 0
    w = int(r['weight']) if not pd.isna(r['weight']) else 0
    c = r['cover'] if not pd.isna(r['cover']) else 0
    print(f"{r['feature']:35s} {gp:6.2f}% {w:>7d} {c:9.1f}")

print()

# Zero importance features
zero = df[df["gain"] == 0]
print(f"ZERO IMPORTANCE FEATURES: {len(zero)}")
for _, r in zero.iterrows():
    print(f"  ✗ {r['feature']}")

print()

# Category analysis
categories = {
    "Temporal": ["hour_sin", "hour_cos", "month_sin", "month_cos"],
    "Pressure": ["pressure_msl", "pressure_change_1h", "pressure_change_3h",
                  "pressure_change_6h", "pressure_accel_3h"],
    "Humidity": ["relative_humidity_2m", "dew_point", "dew_point_depression",
                 "humidity_change_1h", "humidity_change_3h"],
    "Wind raw": ["wind_speed_10m", "wind_u", "wind_v", "wind_speed_change_1h",
                 "wind_speed_change_3h", "wind_dir_change_3h"],
    "Wind regime": ["is_sea_breeze", "is_tramuntana", "is_llevantada", "is_migjorn",
                    "is_garbi", "is_ponent", "llevantada_strength",
                    "llevantada_moisture", "garbi_strength"],
    "Rain": ["precipitation", "rain_accum_3h", "rain_accum_6h", "rained_last_3h"],
    "Cloud/Radiation": ["cloud_cover", "cloud_change_1h", "cloud_change_3h",
                        "is_overcast", "shortwave_radiation"],
    "Pressure levels": ["wind_850_speed", "wind_850_dir", "temp_850", "temp_500",
                         "rh_850", "rh_700", "temp_700", "vt_index", "tt_index",
                         "li_index", "wind_shear_speed", "wind_shear_dir",
                         "cold_500_moderate", "cold_500_strong",
                         "li_unstable", "li_very_unstable"],
    "Model/CAPE": ["cape", "cape_high", "cape_very_high", "weather_code",
                   "model_predicts_precip", "model_predicts_showers"],
}
print("GAIN BY CATEGORY:")
header3 = f"{'Category':20s} {'Total Gain%':>12s} {'# Features':>12s} {'Avg Gain%':>12s}"
print(header3)
print("-" * 60)
for cat, feats in sorted(
    categories.items(),
    key=lambda x: -df[df["feature"].isin(x[1])]["gain_pct"].sum(),
):
    cat_df = df[df["feature"].isin(feats)]
    total = cat_df["gain_pct"].sum()
    avg = total / max(len(cat_df), 1)
    print(f"{cat:20s} {total:11.2f}% {len(cat_df):>12d} {avg:11.2f}%")

# Cumulative importance
print()
df_sorted = df.sort_values("gain_pct", ascending=False)
df_sorted["cum_gain"] = df_sorted["gain_pct"].cumsum()
n_80 = (df_sorted["cum_gain"] <= 80).sum() + 1
n_90 = (df_sorted["cum_gain"] <= 90).sum() + 1
n_95 = (df_sorted["cum_gain"] <= 95).sum() + 1
print(f"CUMULATIVE GAIN:")
print(f"  80% of gain covered by top {n_80} features")
print(f"  90% of gain covered by top {n_90} features")
print(f"  95% of gain covered by top {n_95} features")
print(f"  Remaining {len(feature_names) - n_95} features contribute only {100 - df_sorted.head(n_95)['gain_pct'].sum():.1f}% of gain")

# Dataset reality check
from src.features.engineering import FEATURE_COLUMNS
n_defined = len(FEATURE_COLUMNS)
n_missing = n_defined - len(feature_names)
missing_features = [f for f in FEATURE_COLUMNS if f not in feature_names]

print()
print("=" * 90)
print("REALITY CHECK")
print("=" * 90)
print(f"  FEATURE_COLUMNS defined in code:  {n_defined}")
print(f"  Features actually in model:       {len(feature_names)}")
print(f"  Features MISSING from model:      {n_missing}")
if missing_features:
    print(f"  Missing: {', '.join(missing_features)}")
