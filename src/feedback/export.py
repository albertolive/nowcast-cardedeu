"""
Converteix prediccions verificades en files d'entrenament addicionals.
Aquestes files es fusionen amb el dataset històric durant el retrain setmanal,
permetent que el model aprengui dels seus propis errors recents.
"""
import logging
import os

import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.feedback.logger import load_predictions_log

logger = logging.getLogger(__name__)

FEEDBACK_TRAINING_PATH = os.path.join(config.DATA_PROCESSED_DIR, "feedback_verified.parquet")


def export_verified_for_training() -> int:
    """
    Exporta les prediccions verificades com a dades de feedback.
    Retorna el nombre de files exportades.

    Cada fila verificada inclou:
    - Les features que va usar el model en aquell moment
    - El resultat real (actual_rain) com a nova ground truth
    - Útil perquè conté dades de radar + sentinella reals
    """
    entries = load_predictions_log()
    verified = [e for e in entries if e.get("verified")]

    if not verified:
        logger.info("Cap predicció verificada per exportar.")
        return 0

    df = pd.DataFrame(verified)
    df["will_rain"] = df["actual_rain"].astype(int)  # Ground truth real
    df["datetime"] = pd.to_datetime(df["timestamp"])

    # Expandir el vector de features complet (guardat pel logger) en columnes
    if "features" in df.columns:
        # Filter out rows where features is not a dict (e.g., NaN/float)
        valid_mask = df["features"].apply(lambda x: isinstance(x, dict))
        if valid_mask.any():
            features_df = pd.json_normalize(df.loc[valid_mask, "features"])
            for col in features_df.columns:
                df.loc[valid_mask, col] = features_df[col].values
        df = df.drop(columns=["features"])

    # Normalitzar columnes booleanes a tipus consistent
    for col in df.columns:
        if df[col].dtype == object:
            # Convert bool-like objects to int, then try numeric
            try:
                df[col] = df[col].map(lambda x: int(x) if isinstance(x, bool) else x)
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    # Drop columns that are still object type (nested dicts, strings, etc.)
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)

    # Append-only merge with any existing parquet so we never lose verified history
    # when the source JSONL is trimmed for repo size. Dedup on the parsed `datetime`
    # column (the raw `timestamp` string was dropped above as an object dtype).
    os.makedirs(os.path.dirname(FEEDBACK_TRAINING_PATH), exist_ok=True)
    if os.path.exists(FEEDBACK_TRAINING_PATH):
        try:
            existing = pd.read_parquet(FEEDBACK_TRAINING_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["datetime"], keep="last")
        except Exception as e:
            logger.warning(f"No s'ha pogut llegir parquet existent ({e}); reescrivint des de zero.")

    df.to_parquet(FEEDBACK_TRAINING_PATH, index=False)

    logger.info(f"Exportades {len(df)} prediccions verificades a {FEEDBACK_TRAINING_PATH}")
    return len(df)
