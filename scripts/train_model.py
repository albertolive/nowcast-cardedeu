#!/usr/bin/env python3
"""
Script 3: Entrena el model XGBoost.
Amb --shadow, llegeix els parquets ja preparats i desa el candidat a models/shadow-next/.
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.features.engineering import FEATURE_COLUMNS
from src.feedback.export import FEEDBACK_TRAINING_PATH
from src.model.train import (
    InsufficientShadowData, get_feature_importance, prepare_training_data, save_model,
    save_shadow_model, train_model, train_shadow_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena el model de pluja de Cardedeu.",
        epilog="Exemples:\n  python scripts/train_model.py --shadow\n"
               "  python scripts/train_model.py --shadow --before 2026-11-01T00:00:00Z",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shadow", action="store_true", help="Desa el candidat a models/shadow-next/ sense modificar els models en ús.")
    parser.add_argument("--before", metavar="ISO8601", help="Límit exclusiu de les etiquetes, amb fus horari (per defecte: ara).")
    args = parser.parse_args(argv)
    if args.before is not None and not args.shadow:
        parser.error("--before requereix --shadow.")
    if args.shadow:
        try:
            cutoff = datetime.fromisoformat(args.before) if args.before is not None else datetime.now(timezone.utc)
            if cutoff.tzinfo is None:
                raise ValueError
        except ValueError:
            parser.error("--before requereix una data amb fus horari, p. ex. 2026-11-01T00:00:00Z.")
        if not os.path.isfile(FEEDBACK_TRAINING_PATH):
            parser.error(f"Falta {FEEDBACK_TRAINING_PATH}. Executa primer scripts/repair_feedback.py.")

    dataset_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    if not os.path.exists(dataset_path):
        logger.error("Primer executa scripts/build_dataset.py!")
        sys.exit(1)

    logger.info("Carregant dataset...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset base: {len(df)} mostres")

    if args.shadow:
        try:
            model, metrics, calibrator = train_shadow_model(df, pd.read_parquet(FEEDBACK_TRAINING_PATH), cutoff)
        except InsufficientShadowData as exc:
            # Una quinzena seca és normal: no hi ha candidat, però no és una fallada.
            logger.warning("Sense candidat aquesta setmana: %s", exc)
            sys.exit(3)
        save_shadow_model(model, metrics, calibrator)
        logger.info("Candidat desat a models/shadow-next/ (tall %s, AUC %.4f, F1 %.4f, Brier %.4f).",
                    metrics["training_cutoff"], metrics["holdout_auc"], metrics["holdout_f1"], metrics["holdout_brier"])
        return

    # La reparació/exportació del parquet és un pas separat, mai un efecte de l'entrenament.
    if os.path.isfile(FEEDBACK_TRAINING_PATH):
        feedback_df = pd.read_parquet(FEEDBACK_TRAINING_PATH)
        features, labels = prepare_training_data(feedback_df)
        has_predictors = features.notna().any(axis=1)
        df = pd.concat([df, features.loc[has_predictors, FEATURE_COLUMNS].assign(will_rain=labels[has_predictors])],
                       ignore_index=True)
        logger.info("  + %d prediccions verificades amb predictors", has_predictors.sum())
    logger.info(f"Dataset total: {len(df)} mostres")

    logger.info("Preparant dades d'entrenament...")
    X, y = prepare_training_data(df)
    logger.info(f"Matriu de features: {X.shape}")

    logger.info("=" * 60)
    logger.info("Entrenant XGBoost...")
    logger.info("=" * 60)

    model, metrics, calibrator = train_model(X, y, n_splits=5)

    # Importància de features
    fi = get_feature_importance(model, list(X.columns))
    logger.info("\n🔑 Top 15 features més importants:")
    for _, row in fi.head(15).iterrows():
        bar = "█" * int(row["importance"] * 50)
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

    # Desar model + calibrador
    save_model(model, list(X.columns), metrics, calibrator=calibrator)

    logger.info("=" * 60)
    logger.info("Entrenament completat!")
    logger.info(f"  AUC mitjà (CV): {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    logger.info(f"  F1 mitjà (CV):  {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    logger.info(f"  AUC final:      {metrics['final_auc']:.4f}")
    logger.info(f"  Llindar òptim:  {metrics['optimal_threshold']:.4f}")
    logger.info(f"  Model desat a:  {config.MODEL_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
