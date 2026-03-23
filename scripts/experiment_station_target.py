#!/usr/bin/env python3
"""
Experiment: Compara el target bassat en ERA5 (actual) vs un target corregit
amb les dades diàries de l'estació MeteoCardedeu.

Hipòtesi: el target ERA5 té soroll (FP/FN vs la realitat a Cardedeu).
Corregir el target amb les observacions de l'estació podria millorar el model.

Estratègia de correcció conservadora:
- Si ERA5 diu pluja horària PERÒ l'estació diu 0mm aquell dia → flip a no-rain (FP fix)
- Si ERA5 diu no-pluja horàia PERÒ l'estació diu pluja aquell dia → NO tocar
  (no sabem a quina hora va ploure, millor no inventar)
- Si ambdós coincideixen → mantenir ERA5

Això només elimina FP del target, no afegeix FN. És conservador perquè
sabem segur que si l'estació no va mesurar res, qualsevol pluja ERA5 és soroll.

Execució:
    python scripts/experiment_station_target.py
"""
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.features.engineering import (
    build_features_from_hourly,
    build_target_column,
    FEATURE_COLUMNS,
)
from src.model.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega i prepara les dades com fa build_dataset.py."""
    hourly_path = os.path.join(config.DATA_RAW_DIR, "open_meteo_hourly.parquet")
    station_path = os.path.join(config.DATA_RAW_DIR, "station_daily.parquet")
    pressure_path = os.path.join(config.DATA_RAW_DIR, "pressure_levels_hourly.parquet")
    ensemble_path = os.path.join(config.DATA_RAW_DIR, "ensemble_historical.parquet")
    sst_path = os.path.join(config.DATA_RAW_DIR, "sst_historical.parquet")

    hourly_df = pd.read_parquet(hourly_path)
    station_daily = pd.read_parquet(station_path)

    # Merge pressure levels
    if os.path.exists(pressure_path):
        pressure_df = pd.read_parquet(pressure_path)
        pressure_df["datetime"] = pd.to_datetime(pressure_df["datetime"])
        hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
        overlap_cols = [c for c in pressure_df.columns if c in hourly_df.columns and c != "datetime"]
        if overlap_cols:
            hourly_df = hourly_df.drop(columns=overlap_cols)
        hourly_df = hourly_df.merge(pressure_df, on="datetime", how="left")

    # Merge ensemble
    if os.path.exists(ensemble_path):
        ensemble_df = pd.read_parquet(ensemble_path)
        ensemble_df["datetime"] = pd.to_datetime(ensemble_df["datetime"])
        hourly_df = hourly_df.merge(ensemble_df, on="datetime", how="left")

    # Merge SST
    if os.path.exists(sst_path):
        sst_df = pd.read_parquet(sst_path)
        sst_df["datetime"] = pd.to_datetime(sst_df["datetime"], utc=True).dt.tz_localize(None)
        sst_df = sst_df.set_index("datetime").resample("1h").ffill().reset_index()
        hourly_df = hourly_df.merge(sst_df, on="datetime", how="left")

    # Merge XEMA sentinel
    xema_path = os.path.join(config.DATA_PROCESSED_DIR, "xema_sentinel_cache.parquet")
    if os.path.exists(xema_path):
        xema_df = pd.read_parquet(xema_path)
        xema_df["datetime"] = pd.to_datetime(xema_df["datetime"], utc=True).dt.tz_localize(None)
        if "sentinel_temp_val" in xema_df.columns and "temperature_2m" in hourly_df.columns:
            xema_merged = xema_df.merge(
                hourly_df[["datetime", "temperature_2m", "relative_humidity_2m"]],
                on="datetime", how="left"
            )
            xema_merged["sentinel_temp_diff"] = xema_merged["temperature_2m"] - xema_merged["sentinel_temp_val"]
            xema_merged["sentinel_humidity_diff"] = xema_merged["sentinel_humidity_val"] - xema_merged["relative_humidity_2m"]
            sentinel_feature_cols = ["datetime", "sentinel_temp_diff", "sentinel_humidity_diff",
                                     "sentinel_precip_val", "sentinel_raining",
                                     "local_rain_xema", "local_rain_xema_3h"]
            sentinel_feature_cols = [c for c in sentinel_feature_cols if c in xema_merged.columns]
            xema_features = xema_merged[sentinel_feature_cols].copy()
            xema_features = xema_features.rename(columns={"sentinel_precip_val": "sentinel_precip"})
            hourly_df = hourly_df.merge(xema_features, on="datetime", how="left")

    return hourly_df, station_daily


def build_corrected_target(df: pd.DataFrame, station_daily: pd.DataFrame) -> pd.Series:
    """
    Crea un target corregit: elimina falsos positius d'ERA5 comparant amb
    les observacions diàries de l'estació.

    Conservador: només eliminem FP (ERA5 diu pluja, estació diu no).
    No afegim FN (no sabem l'hora exacta de la pluja de l'estació).
    """
    df = df.copy()
    df["date_only"] = df["datetime"].dt.date

    station_daily = station_daily.copy()
    station_daily["date_only"] = pd.to_datetime(station_daily["date"]).dt.date
    station_daily["station_rained"] = (station_daily["rain_mm"] >= config.RAIN_THRESHOLD_MM).astype(int)

    # Merge daily station data
    df = df.merge(
        station_daily[["date_only", "rain_mm", "station_rained"]],
        on="date_only",
        how="left",
    )

    # Build ERA5 target first
    future_rain = df["precipitation"].rolling(1, min_periods=1).sum().shift(-1)
    era5_target = (future_rain >= config.RAIN_THRESHOLD_MM).astype(int)

    # Correcció: si ERA5 diu pluja PERÒ l'estació diu 0mm aquell dia → no pluja
    # Nota: on station_rained és NaN (sense dades), mantenim ERA5 tal qual
    fp_mask = (era5_target == 1) & (df["station_rained"] == 0)
    corrected_target = era5_target.copy()
    corrected_target[fp_mask] = 0

    n_flipped = fp_mask.sum()
    n_era5_rain = (era5_target == 1).sum()
    logger.info(f"Target correction: {n_flipped} FP flipped to 0 "
                f"({100*n_flipped/n_era5_rain:.1f}% of ERA5 rain hours)")
    logger.info(f"  ERA5 rain hours: {n_era5_rain} → Corrected: {(corrected_target == 1).sum()}")

    return corrected_target


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara la matriu de features."""
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[[c for c in FEATURE_COLUMNS if c in X.columns]]
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def run_experiment():
    """Executa l'experiment A/B: ERA5 target vs target corregit."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT: ERA5 target vs Station-corrected target")
    logger.info("=" * 70)

    # Load data (same for both)
    logger.info("\nCarregant dades...")
    hourly_df, station_daily = load_and_prepare_data()

    # Feature engineering (same for both)
    logger.info("Feature engineering...")
    featured_df = build_features_from_hourly(hourly_df)
    featured_df["datetime"] = pd.to_datetime(featured_df["datetime"])

    # ── Experiment A: ERA5 target (baseline — current model) ──
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT A: ERA5 target (baseline)")
    logger.info("=" * 70)

    df_a = build_target_column(featured_df.copy(), "precipitation", horizon=1)
    df_a = df_a.dropna(subset=["will_rain"])
    X_a = prepare_X(df_a)
    y_a = df_a["will_rain"].copy()

    logger.info(f"  Samples: {len(y_a)}, Rain: {y_a.sum()} ({100*y_a.mean():.1f}%)")

    t0 = time.time()
    model_a, metrics_a, calibrator_a = train_model(X_a, y_a, n_splits=5)
    time_a = time.time() - t0

    # ── Experiment B: Station-corrected target ──
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT B: Station-corrected target")
    logger.info("=" * 70)

    df_b = featured_df.copy()
    corrected_y = build_corrected_target(df_b, station_daily)
    # Need to align: drop last row (shifted target) and NaN
    valid_mask = corrected_y.notna()
    df_b = df_b[valid_mask].copy()
    df_b["will_rain"] = corrected_y[valid_mask].values

    df_b = df_b.dropna(subset=["will_rain"])
    X_b = prepare_X(df_b)
    y_b = df_b["will_rain"].copy()

    logger.info(f"  Samples: {len(y_b)}, Rain: {y_b.sum()} ({100*y_b.mean():.1f}%)")

    t0 = time.time()
    model_b, metrics_b, calibrator_b = train_model(X_b, y_b, n_splits=5)
    time_b = time.time() - t0

    # ── Compare ──
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 70)

    comparison_metrics = [
        ("cv_auc_mean", "AUC (CV mean)", "+"),
        ("cv_auc_std", "AUC (CV std)", "-"),
        ("cv_f1_mean", "Cal F1 (CV mean)", "+"),
        ("cv_f1_std", "Cal F1 (CV std)", "-"),
        ("final_auc", "AUC (final)", "+"),
        ("optimal_threshold", "Threshold", ""),
    ]

    results = []
    for key, name, direction in comparison_metrics:
        va = metrics_a.get(key, 0)
        vb = metrics_b.get(key, 0)
        diff = vb - va
        if direction == "+":
            better = "B ✓" if diff > 0.001 else ("A ✓" if diff < -0.001 else "≈")
        elif direction == "-":
            better = "B ✓" if diff < -0.001 else ("A ✓" if diff > 0.001 else "≈")
        else:
            better = ""
        results.append((name, va, vb, diff, better))
        logger.info(f"  {name:25s}  A={va:.4f}  B={vb:.4f}  Δ={diff:+.4f}  {better}")

    logger.info(f"\n  Training time: A={time_a:.0f}s  B={time_b:.0f}s")
    logger.info(f"  Samples:       A={len(y_a)}  B={len(y_b)}")
    logger.info(f"  Rain events:   A={int(y_a.sum())}  B={int(y_b.sum())}")

    # Check how many labels actually changed
    n_different = int((y_a.values[:len(y_b)] != y_b.values[:len(y_a)]).sum()) if len(y_a) == len(y_b) else -1
    if n_different >= 0:
        logger.info(f"  Labels changed: {n_different} ({100*n_different/len(y_a):.2f}%)")

    logger.info("\n" + "=" * 70)
    key_metric = metrics_b.get("cv_f1_mean", 0) - metrics_a.get("cv_f1_mean", 0)
    if key_metric > 0.005:
        logger.info(f"CONCLUSION: Station-corrected target WINS (Cal F1 +{key_metric:.4f})")
        logger.info("Recommendation: integrate station correction into build_dataset.py")
    elif key_metric < -0.005:
        logger.info(f"CONCLUSION: ERA5 target WINS (Cal F1 {key_metric:+.4f})")
        logger.info("Recommendation: keep current ERA5-based target")
    else:
        logger.info(f"CONCLUSION: No significant difference (Cal F1 Δ={key_metric:+.4f})")
        logger.info("Recommendation: keep current ERA5-based target (simpler)")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_experiment()
