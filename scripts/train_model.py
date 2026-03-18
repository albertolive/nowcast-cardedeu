#!/usr/bin/env python3
"""
Script 3: Entrena el model XGBoost.
Llegeix el dataset processat, entrena amb validació creuada temporal
i desa el model + mètriques.
"""
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.train import prepare_training_data, train_model, save_model, get_feature_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    dataset_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    if not os.path.exists(dataset_path):
        logger.error("Primer executa scripts/build_dataset.py!")
        sys.exit(1)

    logger.info("Carregant dataset...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset: {len(df)} mostres")

    logger.info("Preparant dades d'entrenament...")
    X, y = prepare_training_data(df)
    logger.info(f"Matriu de features: {X.shape}")

    logger.info("=" * 60)
    logger.info("Entrenant XGBoost...")
    logger.info("=" * 60)

    model, metrics = train_model(X, y, n_splits=5)

    # Importància de features
    fi = get_feature_importance(model, list(X.columns))
    logger.info("\n🔑 Top 15 features més importants:")
    for _, row in fi.head(15).iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    # Desar model
    save_model(model, list(X.columns), metrics)

    logger.info("=" * 60)
    logger.info("Entrenament completat!")
    logger.info(f"  AUC mitjà (CV): {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    logger.info(f"  F1 mitjà (CV):  {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    logger.info(f"  AUC final:      {metrics['final_auc']:.4f}")
    logger.info(f"  Model desat a:  {config.MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
