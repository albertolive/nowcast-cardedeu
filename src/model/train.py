"""
Pipeline d'entrenament del model XGBoost per a nowcasting de pluja.
"""
import json
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.features.engineering import FEATURE_COLUMNS, build_features_from_hourly, build_target_column

logger = logging.getLogger(__name__)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X (features) i y (target) per a l'entrenament.
    Elimina files amb NaN al target i gestiona NaN a les features.
    """
    # Construir target
    df = build_target_column(df, "precipitation", horizon=1)

    # Filtrar files on no tenim target (últimes hores sense futur conegut)
    df = df.dropna(subset=["will_rain"])

    # Seleccionar features disponibles
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Features disponibles: {len(available_features)}/{len(FEATURE_COLUMNS)}")

    X = df[available_features].copy()
    y = df["will_rain"].copy()

    # Convertir totes les columnes a numèric (algunes poden ser 'object' per valors mixtos)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Reemplaçar infinits per NaN (XGBoost gestiona NaN natívament)
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Entrena XGBoost amb validació creuada temporal (TimeSeriesSplit).
    Retorna el model entrenat i les mètriques.
    """
    # Calcular el pes de les classes (la pluja és minoritària)
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    scale_pos_weight = n_negative / max(n_positive, 1)

    logger.info(f"Dataset: {len(y)} mostres, {n_positive} pluja ({100*n_positive/len(y):.1f}%), "
                f"scale_pos_weight={scale_pos_weight:.2f}")

    # Paràmetres XGBoost optimitzats per nowcasting
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",  # Area Under Precision-Recall (millor per dades desbalancejades)
        random_state=42,
        early_stopping_rounds=30,
        enable_categorical=False,
    )

    # Validació creuada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= config.ALERT_PROBABILITY_THRESHOLD).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba) if y_val.nunique() > 1 else 0
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cv_scores.append(auc)
        cv_f1_scores.append(f1)
        logger.info(f"  Fold {fold+1}: AUC={auc:.4f}, F1={f1:.4f}")

    # Entrenar model final amb totes les dades
    # Utilitzem les últimes 10% com a eval_set per early stopping
    split_idx = int(len(X) * 0.9)
    model.fit(
        X.iloc[:split_idx], y.iloc[:split_idx],
        eval_set=[(X.iloc[split_idx:], y.iloc[split_idx:])],
        verbose=False,
    )

    # Mètriques finals al conjunt de validació
    y_final_proba = model.predict_proba(X.iloc[split_idx:])[:, 1]
    y_final_pred = (y_final_proba >= config.ALERT_PROBABILITY_THRESHOLD).astype(int)
    y_final_true = y.iloc[split_idx:]

    metrics = {
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
        "cv_f1_mean": float(np.mean(cv_f1_scores)),
        "cv_f1_std": float(np.std(cv_f1_scores)),
        "final_auc": float(roc_auc_score(y_final_true, y_final_proba)) if y_final_true.nunique() > 1 else 0,
        "final_report": classification_report(y_final_true, y_final_pred, output_dict=True, zero_division=0),
        "n_samples": len(y),
        "n_positive": int(n_positive),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
    }

    logger.info(f"\nResultats CV: AUC={metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f}, "
                f"F1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
    logger.info(f"Model final: AUC={metrics['final_auc']:.4f}")

    return model, metrics


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Retorna les importàncies de les features ordenades."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return fi


def save_model(model: xgb.XGBClassifier, feature_names: list[str], metrics: dict) -> None:
    """Desa el model, noms de features i mètriques."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    model.save_model(config.MODEL_PATH)
    logger.info(f"Model desat a {config.MODEL_PATH}")

    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)

    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Mètriques desades a {metrics_path}")


def load_model() -> tuple[xgb.XGBClassifier, list[str]]:
    """Carrega el model entrenat i els noms de features."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model no trobat: {config.MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)

    with open(config.FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    return model, feature_names
