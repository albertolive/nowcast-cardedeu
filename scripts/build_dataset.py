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

    # Desar
    output_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    final_df.to_parquet(output_path, index=False)
    logger.info(f"Dataset desat a {output_path}")


if __name__ == "__main__":
    main()
