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
from sklearn.isotonic import IsotonicRegression

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

    # Use ALL FEATURE_COLUMNS so the model can learn from feedback data
    # that includes radar/lightning/sentinel features.
    # Missing columns are added as NaN (XGBoost handles this natively).
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    logger.info(f"Features disponibles: {len(available_features)}/{len(FEATURE_COLUMNS)}")

    X = df[available_features].copy()
    # Add any missing FEATURE_COLUMNS as NaN so the model trains on all 112
    for col in missing_features:
        X[col] = np.nan
    # Reorder to match FEATURE_COLUMNS canonical order
    X = X[[c for c in FEATURE_COLUMNS if c in X.columns]]
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
    Inclou calibratge isotònic de les probabilitats i cerca del llindar òptim.
    Retorna el model entrenat i les mètriques.
    """
    n_positive = y.sum()
    n_negative = len(y) - n_positive

    logger.info(f"Dataset: {len(y)} mostres, {n_positive} pluja ({100*n_positive/len(y):.1f}%)")

    # Paràmetres XGBoost optimitzats per nowcasting (199 features)
    # Tuned via 5-fold CV grid search (2026-03-20):
    #   - Removed scale_pos_weight: calibration + threshold search handles class imbalance
    #     better than upweighting minority class (+0.0108 F1 OOF)
    #   - colsample 0.7→0.6, reg_alpha 0.1→0.2, reg_lambda 1.0→1.5: stronger regularization
    #     helps generalize with 55% NaN pressure level features (+0.0020 F1 OOF)
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        early_stopping_rounds=75,
        enable_categorical=False,
    )

    # Validació creuada temporal — recollir prediccions out-of-fold per calibratge
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_f1_scores = []
    oof_proba = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = y_pred_proba

        auc = roc_auc_score(y_val, y_pred_proba) if y_val.nunique() > 1 else 0
        # Find per-fold optimal threshold for F1
        best_f1 = 0
        for t in np.arange(0.15, 0.65, 0.01):
            f = f1_score(y_val, (y_pred_proba >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
        cv_scores.append(auc)
        cv_f1_scores.append(best_f1)
        logger.info(f"  Fold {fold+1}: AUC={auc:.4f}, F1={best_f1:.4f}")

    # ── Calibratge isotònic sobre les prediccions out-of-fold ──
    oof_mask = ~np.isnan(oof_proba)
    oof_y = y.values[oof_mask]
    oof_p = oof_proba[oof_mask]

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(oof_p, oof_y)
    logger.info(f"Calibratge isotònic ajustat amb {oof_mask.sum()} prediccions OOF")

    # Trobar llindar òptim (F1) sobre les probabilitats calibrades OOF
    oof_calibrated = calibrator.predict(oof_p)
    precisions, recalls, thresholds = precision_recall_curve(oof_y, oof_calibrated)
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    optimal_idx = np.argmax(f1s)
    optimal_threshold = float(thresholds[optimal_idx])
    logger.info(f"Llindar òptim (F1={f1s[optimal_idx]:.4f}): {optimal_threshold:.4f}")

    # Recalcular CV F1 amb llindar òptim calibrat per comparar
    oof_cal_pred = (oof_calibrated >= optimal_threshold).astype(int)
    cal_f1 = f1_score(oof_y, oof_cal_pred, zero_division=0)
    uncal_f1 = f1_score(oof_y, (oof_p >= config.ALERT_PROBABILITY_THRESHOLD).astype(int), zero_division=0)
    logger.info(f"F1 OOF sense calibratge (llindar={config.ALERT_PROBABILITY_THRESHOLD}): {uncal_f1:.4f}")
    logger.info(f"F1 OOF amb calibratge  (llindar={optimal_threshold:.4f}): {cal_f1:.4f}")

    # Entrenar model final amb totes les dades
    # Utilitzem les últimes 10% com a eval_set per early stopping
    split_idx = int(len(X) * 0.9)
    model.fit(
        X.iloc[:split_idx], y.iloc[:split_idx],
        eval_set=[(X.iloc[split_idx:], y.iloc[split_idx:])],
        verbose=False,
    )

    # Mètriques finals al conjunt de validació (amb calibratge)
    y_final_proba_raw = model.predict_proba(X.iloc[split_idx:])[:, 1]
    y_final_proba = calibrator.predict(y_final_proba_raw)
    y_final_pred = (y_final_proba >= optimal_threshold).astype(int)
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
        "calibrated": True,
        "optimal_threshold": optimal_threshold,
        "calibration_oof_f1_uncalibrated": float(uncal_f1),
        "calibration_oof_f1_calibrated": float(cal_f1),
    }

    logger.info(f"\nResultats CV: AUC={metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f}, "
                f"F1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")
    logger.info(f"Model final (calibrat): AUC={metrics['final_auc']:.4f}, llindar={optimal_threshold:.4f}")

    return model, metrics, calibrator


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Retorna les importàncies de les features ordenades."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return fi


def save_model(model: xgb.XGBClassifier, feature_names: list[str], metrics: dict,
               calibrator: Optional[IsotonicRegression] = None) -> None:
    """Desa el model, calibrador, noms de features i mètriques."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    model.save_model(config.MODEL_PATH)
    logger.info(f"Model desat a {config.MODEL_PATH}")

    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)

    if calibrator is not None:
        joblib.dump(calibrator, config.CALIBRATOR_PATH)
        logger.info(f"Calibrador desat a {config.CALIBRATOR_PATH}")

    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Mètriques desades a {metrics_path}")


def load_model() -> tuple[xgb.XGBClassifier, list[str], Optional[IsotonicRegression], float]:
    """Carrega el model entrenat, calibrador i llindar òptim."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model no trobat: {config.MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)

    with open(config.FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)

    # Carregar calibrador (si existeix)
    calibrator = None
    if os.path.exists(config.CALIBRATOR_PATH):
        calibrator = joblib.load(config.CALIBRATOR_PATH)

    # Carregar llindar òptim de metrics.json (si existeix)
    threshold = config.ALERT_PROBABILITY_THRESHOLD
    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        if "optimal_threshold" in metrics:
            threshold = metrics["optimal_threshold"]

    return model, feature_names, calibrator, threshold
