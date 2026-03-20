#!/usr/bin/env python3
"""
Script 2: Construeix el dataset d'entrenament.
Creua les dades horàries d'Open-Meteo (features + precipitació real d'Open-Meteo)
amb les dades diàries de l'estació (validació de pluja real).
Aplica feature engineering i genera X, y per a l'entrenament.
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.features.engineering import build_features_from_hourly, build_target_column, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def enhance_target_with_station(hourly_df: pd.DataFrame, station_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Millora el target de pluja creuant les dades horàries d'Open-Meteo
    amb les mesures reals de l'estació de Cardedeu.

    Estratègia:
    - Si Open-Meteo diu pluja i l'estació confirma pluja aquell dia → definitely rain
    - Si Open-Meteo diu pluja però l'estació diu 0mm → possible falsa alarma del model
    - Si Open-Meteo NO diu pluja però l'estació sí → el model va fallar (important per aprendre!)

    Creem una feature extra: "station_rain_daily" que indica si l'estació va mesurar pluja.
    """
    hourly_df = hourly_df.copy()
    hourly_df["date_only"] = hourly_df["datetime"].dt.date

    station_daily = station_daily.copy()
    station_daily["date_only"] = pd.to_datetime(station_daily["date"]).dt.date
    station_daily["station_rain_mm"] = station_daily["rain_mm"]
    station_daily["station_rained"] = (station_daily["rain_mm"] >= config.RAIN_THRESHOLD_MM).astype(int)

    # Merge per data
    merged = hourly_df.merge(
        station_daily[["date_only", "station_rain_mm", "station_rained",
                       "temp_mean", "wind_mean_kmh", "wind_dir_deg"]],
        on="date_only",
        how="left",
    )

    # Feature extra: discrepància entre model i estació
    if "precipitation" in merged.columns:
        merged["daily_precip_model"] = merged.groupby("date_only")["precipitation"].transform("sum")
        merged["model_vs_station_rain"] = merged["daily_precip_model"] - merged["station_rain_mm"].fillna(0)

    # Usar la precipitació de l'estació per millorar el target
    # Si tenim dades de l'estació, ponderar el target
    # Target hybrid: combina Open-Meteo horari + confirmació diària de l'estació
    if "station_rained" in merged.columns:
        merged["station_confirmation"] = merged["station_rained"].fillna(0)

    merged = merged.drop(columns=["date_only"], errors="ignore")
    return merged


def main():
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    # Carregar dades descarregades
    hourly_path = os.path.join(config.DATA_RAW_DIR, "open_meteo_hourly.parquet")
    station_path = os.path.join(config.DATA_RAW_DIR, "station_daily.parquet")

    if not os.path.exists(hourly_path) or not os.path.exists(station_path):
        logger.error("Primer executa scripts/download_history.py!")
        sys.exit(1)

    logger.info("Carregant dades...")
    hourly_df = pd.read_parquet(hourly_path)
    station_daily = pd.read_parquet(station_path)

    logger.info(f"Open-Meteo: {len(hourly_df)} registres horaris")
    logger.info(f"Estació: {len(station_daily)} dies")

    # ── Afegir nivells de pressió si existeixen ──
    pressure_path = os.path.join(config.DATA_RAW_DIR, "pressure_levels_hourly.parquet")
    if os.path.exists(pressure_path):
        pressure_df = pd.read_parquet(pressure_path)
        pressure_df["datetime"] = pd.to_datetime(pressure_df["datetime"])
        hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
        # CAPE/CIN come from both Archive (all NULL) and Historical Forecast (2021+).
        # Drop the Archive NULL columns before merging to avoid _x/_y conflicts.
        overlap_cols = [c for c in pressure_df.columns if c in hourly_df.columns and c != "datetime"]
        if overlap_cols:
            hourly_df = hourly_df.drop(columns=overlap_cols)
            logger.info(f"Dropped Archive columns superseded by Historical Forecast: {overlap_cols}")
        hourly_df = hourly_df.merge(pressure_df, on="datetime", how="left")
        logger.info(f"Pressure levels merged: {len(pressure_df)} registres, "
                    f"columnes afegides: {[c for c in pressure_df.columns if c != 'datetime']}")
    else:
        logger.warning("No s'han trobat dades de pressure levels (executa download_history.py)")

    # ── Afegir dades d'ensemble històric si existeixen ──
    ensemble_path = os.path.join(config.DATA_RAW_DIR, "ensemble_historical.parquet")
    if os.path.exists(ensemble_path):
        ensemble_df = pd.read_parquet(ensemble_path)
        ensemble_df["datetime"] = pd.to_datetime(ensemble_df["datetime"])
        hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
        ensemble_cols = [c for c in ensemble_df.columns if c != "datetime"]
        hourly_df = hourly_df.merge(ensemble_df, on="datetime", how="left")
        n_valid = hourly_df[ensemble_cols[0]].notna().sum() if ensemble_cols else 0
        logger.info(f"Ensemble merged: {len(ensemble_df)} registres, "
                    f"{n_valid}/{len(hourly_df)} hores amb dades ({100*n_valid/len(hourly_df):.1f}%)")
    else:
        logger.info("No ensemble data (run: .venv/bin/python scripts/backfill_ensemble.py)")

    # ── Afegir SST històric (NOAA OISST) si existeix ──
    sst_path = os.path.join(config.DATA_RAW_DIR, "sst_historical.parquet")
    if os.path.exists(sst_path):
        sst_df = pd.read_parquet(sst_path)
        sst_df["datetime"] = pd.to_datetime(sst_df["datetime"], utc=True).dt.tz_localize(None)
        # SST is daily — forward-fill to hourly resolution
        hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
        sst_df = sst_df.set_index("datetime").resample("1h").ffill().reset_index()
        hourly_df = hourly_df.merge(sst_df, on="datetime", how="left")
        n_valid = hourly_df["sst_med"].notna().sum() if "sst_med" in hourly_df.columns else 0
        logger.info(f"SST merged: {len(sst_df)} registres, "
                    f"{n_valid}/{len(hourly_df)} hores amb dades ({100*n_valid/len(hourly_df):.1f}%)")
    else:
        logger.info("No SST data (run download_history.py to fetch NOAA OISST)")

    # ── Afegir dades sentinella XEMA si existeixen ──
    xema_path = os.path.join(config.DATA_PROCESSED_DIR, "xema_sentinel_cache.parquet")
    if os.path.exists(xema_path):
        xema_df = pd.read_parquet(xema_path)
        xema_df["datetime"] = pd.to_datetime(xema_df["datetime"], utc=True).dt.tz_localize(None)
        hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
        # Compute sentinel diffs vs station data
        if "sentinel_temp_val" in xema_df.columns and "temperature_2m" in hourly_df.columns:
            xema_merged = xema_df.merge(
                hourly_df[["datetime", "temperature_2m", "relative_humidity_2m"]],
                on="datetime", how="left"
            )
            xema_merged["sentinel_temp_diff"] = xema_merged["temperature_2m"] - xema_merged["sentinel_temp_val"]
            xema_merged["sentinel_humidity_diff"] = xema_merged["sentinel_humidity_val"] - xema_merged["relative_humidity_2m"]
            # Keep only the features we need
            sentinel_feature_cols = ["datetime", "sentinel_temp_diff", "sentinel_humidity_diff",
                                     "sentinel_precip_val", "sentinel_raining",
                                     "local_rain_xema", "local_rain_xema_3h"]
            sentinel_feature_cols = [c for c in sentinel_feature_cols if c in xema_merged.columns]
            xema_features = xema_merged[sentinel_feature_cols].copy()
            # Rename to match FEATURE_COLUMNS
            xema_features = xema_features.rename(columns={"sentinel_precip_val": "sentinel_precip"})
            hourly_df = hourly_df.merge(xema_features, on="datetime", how="left")
            n_valid = xema_features["sentinel_temp_diff"].notna().sum() if "sentinel_temp_diff" in xema_features.columns else 0
            logger.info(f"XEMA sentinel merged: {len(xema_df)} registres, "
                        f"{n_valid}/{len(hourly_df)} hores amb dades ({100*n_valid/len(hourly_df):.1f}%)")
        else:
            logger.warning("XEMA data missing expected columns, skipping sentinel merge")
    else:
        logger.info("No XEMA sentinel data (run: METEOCAT_API_KEY=xxx .venv/bin/python scripts/backfill_xema.py)")

    # ── Feature engineering ──
    logger.info("Aplicant feature engineering...")
    featured_df = build_features_from_hourly(hourly_df)

    # ── Enriquir amb dades de l'estació ──
    logger.info("Creuant amb dades de l'estació...")
    enhanced_df = enhance_target_with_station(featured_df, station_daily)

    # ── Construir target ──
    logger.info("Construint variable target (will_rain)...")
    final_df = build_target_column(enhanced_df, "precipitation", horizon=1)

    # Eliminar files sense target o sense suficients features
    final_df = final_df.dropna(subset=["will_rain"])

    # ── Estadístiques ──
    available_features = [c for c in FEATURE_COLUMNS if c in final_df.columns]
    n_rain = int(final_df["will_rain"].sum())
    n_total = len(final_df)

    logger.info("=" * 60)
    logger.info(f"Dataset final: {n_total} mostres")
    logger.info(f"  Pluja: {n_rain} ({100*n_rain/n_total:.1f}%)")
    logger.info(f"  No pluja: {n_total - n_rain} ({100*(n_total-n_rain)/n_total:.1f}%)")
    logger.info(f"  Features disponibles: {len(available_features)}/{len(FEATURE_COLUMNS)}")
    logger.info(f"  Període: {final_df['datetime'].min()} → {final_df['datetime'].max()}")
    logger.info("=" * 60)

    # Mostrar features disponibles i les que falten
    missing = [c for c in FEATURE_COLUMNS if c not in final_df.columns]
    if missing:
        logger.info(f"Features no disponibles (s'ompliran amb NaN): {missing}")
        # Add missing columns as NaN so the model trains on ALL 112 features.
        # This allows feedback rows (with radar/lightning data) to be absorbed directly.
        for col in missing:
            final_df[col] = np.nan

    # Desar
    output_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    final_df.to_parquet(output_path, index=False)
    logger.info(f"Dataset desat a {output_path}")


if __name__ == "__main__":
    main()
