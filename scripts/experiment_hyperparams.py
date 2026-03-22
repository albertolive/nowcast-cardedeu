#!/usr/bin/env python3
"""
Experiment: prova múltiples configuracions d'hiperparàmetres per trobar la millor.
Carrega les dades una sola vegada i entrena amb TimeSeriesSplit CV per a cada config.
"""
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.train import prepare_training_data
from src.feedback.export import export_verified_for_training, FEEDBACK_TRAINING_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameter configurations to test ──
CONFIGS = {
    # ROUND 2: Refine around diversity_50 winner
    "div50_base": {
        "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.015,
        "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.5,
    },
    # More trees - let the model converge further with diversity
    "div50_1500trees": {
        "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.012,
        "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.5,
    },
    # More trees + slightly higher regularization
    "div50_1500_reg": {
        "n_estimators": 1500, "max_depth": 6, "learning_rate": 0.012,
        "subsample": 0.75, "colsample_bytree": 0.5, "min_child_weight": 5,
        "gamma": 0.15, "reg_alpha": 0.3, "reg_lambda": 2.0,
    },
    # Depth 7 + diversity 50 (allow more complex interactions)
    "div50_deep7": {
        "n_estimators": 1200, "max_depth": 7, "learning_rate": 0.012,
        "subsample": 0.75, "colsample_bytree": 0.5, "min_child_weight": 6,
        "gamma": 0.15, "reg_alpha": 0.3, "reg_lambda": 2.0,
    },
    # colsample_bynode instead of bytree (diversity at split level)
    "div50_rowsamp": {
        "n_estimators": 1200, "max_depth": 6, "learning_rate": 0.012,
        "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.5,
    },
    # Sweet spot attempt: moderate everything
    "sweet_spot": {
        "n_estimators": 1200, "max_depth": 6, "learning_rate": 0.015,
        "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 5,
        "gamma": 0.1, "reg_alpha": 0.25, "reg_lambda": 1.8,
    },
}


def evaluate_config(name: str, params: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Entrena amb CV i retorna mètriques."""
    t0 = time.time()
    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=max(75, int(params["n_estimators"] * 0.08)),
        enable_categorical=False,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_aucs, cv_f1s = [], []
    oof_proba = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = y_pred

        auc = roc_auc_score(y_val, y_pred) if y_val.nunique() > 1 else 0
        best_f1 = max(
            f1_score(y_val, (y_pred >= t).astype(int), zero_division=0)
            for t in np.arange(0.15, 0.65, 0.01)
        )
        cv_aucs.append(auc)
        cv_f1s.append(best_f1)

    # Full OOF calibration + threshold search
    mask = ~np.isnan(oof_proba)
    oof_y, oof_p = y.values[mask], oof_proba[mask]
    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    cal.fit(oof_p, oof_y)
    oof_cal = cal.predict(oof_p)

    prec, rec, thr = precision_recall_curve(oof_y, oof_cal)
    f1_arr = np.where((prec[:-1] + rec[:-1]) > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    best_idx = np.argmax(f1_arr)
    opt_thr = float(thr[best_idx])
    cal_f1 = float(f1_arr[best_idx])

    elapsed = time.time() - t0

    # Feature importance from last fold model
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_3_gain = fi.head(3).sum()
    nonzero_features = (fi > 0).sum()

    return {
        "name": name,
        "cv_auc": f"{np.mean(cv_aucs):.4f}±{np.std(cv_aucs):.4f}",
        "cv_f1": f"{np.mean(cv_f1s):.4f}±{np.std(cv_f1s):.4f}",
        "cal_f1": f"{cal_f1:.4f}",
        "threshold": f"{opt_thr:.4f}",
        "top3_gain%": f"{top_3_gain*100:.1f}%",
        "active_features": nonzero_features,
        "time_s": f"{elapsed:.0f}",
        # Raw values for sorting
        "_auc": np.mean(cv_aucs),
        "_f1": np.mean(cv_f1s),
        "_cal_f1": cal_f1,
    }


def main():
    # Load data once
    dataset_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    if not os.path.exists(dataset_path):
        logger.error("Primer executa scripts/build_dataset.py!")
        sys.exit(1)

    logger.info("Carregant dataset...")
    df = pd.read_parquet(dataset_path)

    # Feedback loop
    n_feedback = export_verified_for_training()
    if n_feedback > 0 and os.path.exists(FEEDBACK_TRAINING_PATH):
        feedback_df = pd.read_parquet(FEEDBACK_TRAINING_PATH)
        common_cols = [c for c in df.columns if c in feedback_df.columns]
        if "will_rain" in common_cols and len(common_cols) > 2:
            df = pd.concat([df, feedback_df[common_cols]], ignore_index=True)
            logger.info(f"  + {n_feedback} feedback rows added")

    logger.info("Preparant features...")
    X, y = prepare_training_data(df)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.sum():.0f} rain ({100*y.mean():.1f}%)")

    # Run all experiments
    logger.info("=" * 80)
    logger.info("EXPERIMENT GRID: 6 hyperparameter configurations")
    logger.info("=" * 80)

    results = []
    for name, params in CONFIGS.items():
        logger.info(f"\n{'─'*60}")
        logger.info(f"Running: {name} (colsample={params['colsample_bytree']}, "
                    f"depth={params['max_depth']}, trees={params['n_estimators']}, "
                    f"lr={params['learning_rate']})")
        logger.info(f"{'─'*60}")

        result = evaluate_config(name, params, X, y)
        results.append(result)

        logger.info(f"  → AUC={result['cv_auc']}, F1={result['cv_f1']}, "
                    f"Cal F1={result['cal_f1']}, Top3={result['top3_gain%']}, "
                    f"Active={result['active_features']}, Time={result['time_s']}s")

    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Config':<16} {'CV AUC':>14} {'CV F1':>14} {'Cal F1':>8} {'Thr':>7} {'Top3':>7} {'Active':>7} {'Time':>5}")
    logger.info("-" * 80)

    results.sort(key=lambda r: r["_cal_f1"], reverse=True)
    for r in results:
        marker = " ★" if r == results[0] else ""
        logger.info(f"{r['name']:<16} {r['cv_auc']:>14} {r['cv_f1']:>14} {r['cal_f1']:>8} "
                    f"{r['threshold']:>7} {r['top3_gain%']:>7} {r['active_features']:>7} {r['time_s']:>5}{marker}")

    best = results[0]
    logger.info(f"\n🏆 Best config: {best['name']} (Cal F1={best['cal_f1']})")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
