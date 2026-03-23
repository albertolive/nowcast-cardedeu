#!/usr/bin/env python3
"""
Experiment: investigar 3 possibles millores al model actual.

1. Feature interactions: nwp_precip_severity × altres features
2. Model stacking: XGBoost + LightGBM + CatBoost blend
3. Sequence modeling: features temporals (lag-6h) com a input extra

Cada experiment utilitza TimeSeriesSplit CV + IsotonicRegression + threshold search
(idèntic al pipeline de producció) per comparar amb el baseline de forma justa.
"""
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.train import prepare_training_data
from src.feedback.export import export_verified_for_training, FEEDBACK_TRAINING_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5


def calibrate_and_score(y_true, oof_proba):
    """Apply isotonic calibration to OOF predictions and find optimal F1 threshold."""
    mask = ~np.isnan(oof_proba)
    y = y_true[mask]
    p = oof_proba[mask]

    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    cal.fit(p, y)
    p_cal = cal.predict(p)

    prec, rec, thr = precision_recall_curve(y, p_cal)
    f1_arr = np.where(
        (prec[:-1] + rec[:-1]) > 0,
        2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
        0,
    )
    best_idx = np.argmax(f1_arr)
    return {
        "cal_f1": float(f1_arr[best_idx]),
        "threshold": float(thr[best_idx]),
        "auc": float(roc_auc_score(y, p_cal)),
    }


def run_xgboost_cv(X, y, params=None):
    """Run XGBoost with TimeSeriesSplit CV, return OOF predictions."""
    if params is None:
        params = {
            "n_estimators": 1200, "max_depth": 7, "learning_rate": 0.012,
            "subsample": 0.75, "colsample_bytree": 0.7, "colsample_bynode": 0.7,
            "min_child_weight": 6, "gamma": 0.15, "reg_alpha": 0.3, "reg_lambda": 2.0,
        }
    model = xgb.XGBClassifier(
        **params, objective="binary:logistic", eval_metric="logloss",
        random_state=42, early_stopping_rounds=96, enable_categorical=False,
    )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    aucs = []

    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        aucs.append(roc_auc_score(y.iloc[val_idx], p))

    return oof, np.mean(aucs), model


def run_lightgbm_cv(X, y):
    """Run LightGBM with TimeSeriesSplit CV, return OOF predictions."""
    model = lgb.LGBMClassifier(
        n_estimators=1200, max_depth=7, learning_rate=0.012,
        subsample=0.75, colsample_bytree=0.7,
        min_child_weight=6, reg_alpha=0.3, reg_lambda=2.0,
        objective="binary", metric="binary_logloss",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    aucs = []

    for train_idx, val_idx in tscv.split(X):
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(96, verbose=False), lgb.log_evaluation(-1)],
        )
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        aucs.append(roc_auc_score(y.iloc[val_idx], p))

    return oof, np.mean(aucs), model


def run_catboost_cv(X, y):
    """Run CatBoost with TimeSeriesSplit CV, return OOF predictions."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof = np.full(len(y), np.nan)
    aucs = []

    for train_idx, val_idx in tscv.split(X):
        model = cb.CatBoostClassifier(
            iterations=1200, depth=7, learning_rate=0.012,
            subsample=0.75, colsample_bylevel=0.7,
            l2_leaf_reg=2.0, random_seed=42,
            eval_metric="Logloss", verbose=0,
            early_stopping_rounds=96,
        )
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = p
        aucs.append(roc_auc_score(y.iloc[val_idx], p))

    return oof, np.mean(aucs), model


def load_data():
    """Load dataset + feedback, return X, y."""
    dataset_path = os.path.join(config.DATA_PROCESSED_DIR, "training_dataset.parquet")
    df = pd.read_parquet(dataset_path)

    n_feedback = export_verified_for_training()
    if n_feedback > 0 and os.path.exists(FEEDBACK_TRAINING_PATH):
        feedback_df = pd.read_parquet(FEEDBACK_TRAINING_PATH)
        common_cols = [c for c in df.columns if c in feedback_df.columns]
        if "will_rain" in common_cols and len(common_cols) > 2:
            df = pd.concat([df, feedback_df[common_cols]], ignore_index=True)
            logger.info(f"  + {n_feedback} feedback rows added")

    X, y = prepare_training_data(df)
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.sum():.0f} rain ({100*y.mean():.1f}%)")
    return X, y, df


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Feature Interactions
# ═══════════════════════════════════════════════════════════════
def experiment_interactions(X, y):
    """Test if hand-crafted feature interactions improve over XGBoost depth=7."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1: FEATURE INTERACTIONS")
    logger.info("=" * 80)

    # Baseline: current XGBoost
    t0 = time.time()
    oof_base, auc_base, _ = run_xgboost_cv(X, y)
    base_score = calibrate_and_score(y.values, oof_base)
    logger.info(f"  Baseline: Cal F1={base_score['cal_f1']:.4f}, AUC={auc_base:.4f} ({time.time()-t0:.0f}s)")

    # Add interaction features
    X_int = X.copy()

    # Key interactions that XGBoost might struggle with at depth=7:
    # 1. nwp_precip_severity × radar proximity (NWP confidence × radar confirmation)
    if "nwp_precip_severity" in X.columns and "radar_nearest_echo_km" in X.columns:
        X_int["severity_x_radar_near"] = X["nwp_precip_severity"] * (30 - X["radar_nearest_echo_km"].clip(0, 30)) / 30

    # 2. nwp_precip_severity × ensemble agreement (NWP + multi-model confirmation)
    if "nwp_precip_severity" in X.columns and "ensemble_rain_agreement" in X.columns:
        X_int["severity_x_ensemble"] = X["nwp_precip_severity"] * X["ensemble_rain_agreement"]

    # 3. tcwv × pressure_change_3h (moisture loading + dynamics)
    if "tcwv" in X.columns and "pressure_change_3h" in X.columns:
        X_int["tcwv_x_pressure_drop"] = X["tcwv"] * (-X["pressure_change_3h"]).clip(lower=0)

    # 4. nwp_precip_severity × humidity (drizzle codes + dry air = FP)
    if "nwp_precip_severity" in X.columns and "relative_humidity_2m" in X.columns:
        X_int["severity_x_humidity"] = X["nwp_precip_severity"] * X["relative_humidity_2m"] / 100

    # 5. llevantada_strength × tcwv (Llevantada regime + moisture)
    if "llevantada_strength" in X.columns and "tcwv" in X.columns:
        X_int["llevantada_x_tcwv"] = X["llevantada_strength"] * X["tcwv"]

    # 6. cape × humidity convergence (instability + trigger)
    if "cape" in X.columns and "cloud_humidity_convergence" in X.columns:
        X_int["cape_x_convergence"] = X["cape"] * X["cloud_humidity_convergence"]

    # 7. nwp_rain_persistence × severity (persistent + severe = frontal)
    if "nwp_rain_persistence_6h" in X.columns and "nwp_precip_severity" in X.columns:
        X_int["persistence_x_severity"] = X["nwp_rain_persistence_6h"] * X["nwp_precip_severity"]

    # 8. Pressure drop × humidity (frontal signature)
    if "pressure_change_3h" in X.columns and "relative_humidity_2m" in X.columns:
        X_int["pressure_drop_x_humidity"] = (-X["pressure_change_3h"]).clip(lower=0) * X["relative_humidity_2m"] / 100

    n_new = X_int.shape[1] - X.shape[1]
    logger.info(f"  Added {n_new} interaction features (total: {X_int.shape[1]})")

    t0 = time.time()
    oof_int, auc_int, model_int = run_xgboost_cv(X_int, y)
    int_score = calibrate_and_score(y.values, oof_int)
    logger.info(f"  With interactions: Cal F1={int_score['cal_f1']:.4f}, AUC={auc_int:.4f} ({time.time()-t0:.0f}s)")

    delta = int_score["cal_f1"] - base_score["cal_f1"]
    logger.info(f"  Delta Cal F1: {delta:+.4f}")

    # Check if any interaction features gained importance
    fi = pd.Series(model_int.feature_importances_, index=X_int.columns).sort_values(ascending=False)
    interaction_cols = [c for c in X_int.columns if c not in X.columns]
    int_importance = fi[interaction_cols].sort_values(ascending=False)
    logger.info(f"  Interaction feature importance:")
    for feat, imp in int_importance.items():
        logger.info(f"    {feat}: {imp:.6f} {'✓' if imp > 0 else '✗'}")

    return {
        "name": "interactions",
        "baseline_f1": base_score["cal_f1"],
        "new_f1": int_score["cal_f1"],
        "delta": delta,
        "n_new_features": n_new,
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Model Stacking
# ═══════════════════════════════════════════════════════════════
def experiment_stacking(X, y):
    """Test if blending XGBoost + LightGBM + CatBoost improves predictions."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2: MODEL STACKING (XGBoost + LightGBM + CatBoost)")
    logger.info("=" * 80)

    # Run all 3 models and collect OOF predictions
    t0 = time.time()
    logger.info("  Training XGBoost...")
    oof_xgb, auc_xgb, _ = run_xgboost_cv(X, y)
    xgb_score = calibrate_and_score(y.values, oof_xgb)
    logger.info(f"  XGBoost: Cal F1={xgb_score['cal_f1']:.4f}, AUC={auc_xgb:.4f}")

    logger.info("  Training LightGBM...")
    oof_lgb, auc_lgb, _ = run_lightgbm_cv(X, y)
    lgb_score = calibrate_and_score(y.values, oof_lgb)
    logger.info(f"  LightGBM: Cal F1={lgb_score['cal_f1']:.4f}, AUC={auc_lgb:.4f}")

    logger.info("  Training CatBoost...")
    oof_cb, auc_cb, _ = run_catboost_cv(X, y)
    cb_score = calibrate_and_score(y.values, oof_cb)
    logger.info(f"  CatBoost: Cal F1={cb_score['cal_f1']:.4f}, AUC={auc_cb:.4f}")

    elapsed_models = time.time() - t0
    logger.info(f"  All 3 models trained in {elapsed_models:.0f}s")

    # Find common OOF mask (all 3 models have predictions)
    mask = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgb) & ~np.isnan(oof_cb)
    y_common = y.values[mask]

    # --- Blend method 1: Simple average ---
    oof_avg = (oof_xgb[mask] + oof_lgb[mask] + oof_cb[mask]) / 3
    avg_score = calibrate_and_score(y_common, oof_avg)
    logger.info(f"\n  Blend (simple avg): Cal F1={avg_score['cal_f1']:.4f}")

    # --- Blend method 2: Weighted average (optimize on OOF) ---
    # Grid search weights
    best_w_score = 0
    best_weights = (1/3, 1/3, 1/3)
    for w1 in np.arange(0.2, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            oof_w = w1 * oof_xgb[mask] + w2 * oof_lgb[mask] + w3 * oof_cb[mask]
            try:
                s = calibrate_and_score(y_common, oof_w)
                if s["cal_f1"] > best_w_score:
                    best_w_score = s["cal_f1"]
                    best_weights = (w1, w2, w3)
            except Exception:
                pass

    oof_weighted = best_weights[0] * oof_xgb[mask] + best_weights[1] * oof_lgb[mask] + best_weights[2] * oof_cb[mask]
    weighted_score = calibrate_and_score(y_common, oof_weighted)
    logger.info(f"  Blend (weighted {best_weights[0]:.2f}/{best_weights[1]:.2f}/{best_weights[2]:.2f}): Cal F1={weighted_score['cal_f1']:.4f}")

    # --- Blend method 3: Logistic regression stacking ---
    # Use OOF predictions as meta-features with time-aware split
    meta_X = np.column_stack([oof_xgb[mask], oof_lgb[mask], oof_cb[mask]])
    # Use last 40% as test for the meta-learner (since these are already OOF)
    split_point = int(len(y_common) * 0.6)
    meta_train_X, meta_test_X = meta_X[:split_point], meta_X[split_point:]
    meta_train_y, meta_test_y = y_common[:split_point], y_common[split_point:]

    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(meta_train_X, meta_train_y)
    meta_proba = lr.predict_proba(meta_test_X)[:, 1]
    lr_score = calibrate_and_score(meta_test_y, meta_proba)
    logger.info(f"  Blend (logistic stacking): Cal F1={lr_score['cal_f1']:.4f}")
    logger.info(f"  Logistic weights: XGB={lr.coef_[0][0]:.3f}, LGB={lr.coef_[0][1]:.3f}, CB={lr.coef_[0][2]:.3f}")

    # --- Check diversity: correlation between models ---
    corr_xgb_lgb = np.corrcoef(oof_xgb[mask], oof_lgb[mask])[0, 1]
    corr_xgb_cb = np.corrcoef(oof_xgb[mask], oof_cb[mask])[0, 1]
    corr_lgb_cb = np.corrcoef(oof_lgb[mask], oof_cb[mask])[0, 1]
    logger.info(f"\n  Model diversity (prediction correlation):")
    logger.info(f"    XGBoost ↔ LightGBM: {corr_xgb_lgb:.4f}")
    logger.info(f"    XGBoost ↔ CatBoost: {corr_xgb_cb:.4f}")
    logger.info(f"    LightGBM ↔ CatBoost: {corr_lgb_cb:.4f}")

    # --- Check where models disagree and who's right ---
    # High-disagreement samples: where the models differ most
    disagreement = np.std(np.column_stack([oof_xgb[mask], oof_lgb[mask], oof_cb[mask]]), axis=1)
    high_disagree_mask = disagreement > np.percentile(disagreement, 90)
    n_disagree = high_disagree_mask.sum()
    rain_rate_disagree = y_common[high_disagree_mask].mean()
    rain_rate_overall = y_common.mean()
    logger.info(f"  High-disagreement samples (top 10%): {n_disagree}, rain rate: {rain_rate_disagree:.1%} (vs {rain_rate_overall:.1%} overall)")

    best_blend_f1 = max(avg_score["cal_f1"], weighted_score["cal_f1"])
    delta_vs_xgb = best_blend_f1 - xgb_score["cal_f1"]

    return {
        "name": "stacking",
        "xgb_f1": xgb_score["cal_f1"],
        "lgb_f1": lgb_score["cal_f1"],
        "cb_f1": cb_score["cal_f1"],
        "avg_blend_f1": avg_score["cal_f1"],
        "weighted_blend_f1": weighted_score["cal_f1"],
        "best_weights": best_weights,
        "lr_stacking_f1": lr_score["cal_f1"],
        "delta_vs_xgb": delta_vs_xgb,
        "corr_xgb_lgb": corr_xgb_lgb,
        "corr_xgb_cb": corr_xgb_cb,
        "corr_lgb_cb": corr_lgb_cb,
        "time_s": elapsed_models,
    }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Sequence Features (Temporal Lag Window)
# ═══════════════════════════════════════════════════════════════
def experiment_sequence(X, y):
    """Test if adding lag features (t-1, t-2, ..., t-6) improves predictions.

    Instead of a full LSTM, test the hypothesis that temporal patterns matter
    by creating lag features — if these help XGBoost, then sequence modeling
    would help even more. If they don't help, LSTM won't help either.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3: SEQUENCE FEATURES (LAG WINDOW)")
    logger.info("=" * 80)
    logger.info("  Hypothesis: if lag features help XGBoost, an LSTM would help more.")
    logger.info("  If they DON'T help, an LSTM is not worth the complexity.")

    # Baseline
    t0 = time.time()
    oof_base, auc_base, _ = run_xgboost_cv(X, y)
    base_score = calibrate_and_score(y.values, oof_base)
    logger.info(f"  Baseline: Cal F1={base_score['cal_f1']:.4f}, AUC={auc_base:.4f} ({time.time()-t0:.0f}s)")

    # Select the most informative features to lag (top by importance + physically motivated)
    key_features_to_lag = [
        "precipitation",        # Most direct signal
        "pressure_change_3h",   # Dynamic signal
        "relative_humidity_2m", # Moisture trend
        "cloud_cover",          # Cloud evolution
        "nwp_precip_severity",  # NWP confidence evolution
        "tcwv",                 # Moisture loading evolution
    ]
    available_lag = [f for f in key_features_to_lag if f in X.columns]
    logger.info(f"  Lagging {len(available_lag)} features: {available_lag}")

    # Create lag features (t-1 through t-3 — 3 hours back)
    X_lag = X.copy()
    for feat in available_lag:
        for lag in [1, 2, 3]:
            col_name = f"{feat}_lag{lag}h"
            X_lag[col_name] = X[feat].shift(lag)

    # Also add rate-of-change features (derivative)
    for feat in available_lag:
        if feat in X.columns:
            X_lag[f"{feat}_roc_1h"] = X[feat] - X[feat].shift(1)  # first derivative
            X_lag[f"{feat}_roc_accel"] = X_lag[f"{feat}_roc_1h"] - X_lag[f"{feat}_roc_1h"].shift(1)  # acceleration

    n_new = X_lag.shape[1] - X.shape[1]
    logger.info(f"  Added {n_new} lag/derivative features (total: {X_lag.shape[1]})")

    t0 = time.time()
    oof_lag, auc_lag, model_lag = run_xgboost_cv(X_lag, y)
    lag_score = calibrate_and_score(y.values, oof_lag)
    logger.info(f"  With lags: Cal F1={lag_score['cal_f1']:.4f}, AUC={auc_lag:.4f} ({time.time()-t0:.0f}s)")

    delta = lag_score["cal_f1"] - base_score["cal_f1"]
    logger.info(f"  Delta Cal F1: {delta:+.4f}")

    # Check importance of lag features
    fi = pd.Series(model_lag.feature_importances_, index=X_lag.columns).sort_values(ascending=False)
    lag_cols = [c for c in X_lag.columns if c not in X.columns]
    lag_importance = fi[lag_cols].sort_values(ascending=False)
    top_lags = lag_importance.head(10)
    logger.info(f"  Top 10 lag feature importance:")
    for feat, imp in top_lags.items():
        logger.info(f"    {feat}: {imp:.6f} {'✓' if imp > 0 else '✗'}")

    nonzero_lags = (lag_importance > 0).sum()
    logger.info(f"  Lag features with nonzero importance: {nonzero_lags}/{len(lag_cols)}")

    # --- Extended test: 6-hour lag window ---
    logger.info("\n  Extended test: 6-hour lag window...")
    X_lag6 = X.copy()
    for feat in available_lag:
        for lag in [1, 2, 3, 4, 5, 6]:
            X_lag6[f"{feat}_lag{lag}h"] = X[feat].shift(lag)

    n_new6 = X_lag6.shape[1] - X.shape[1]
    logger.info(f"  Added {n_new6} lag features for 6h window (total: {X_lag6.shape[1]})")

    t0 = time.time()
    oof_lag6, auc_lag6, _ = run_xgboost_cv(X_lag6, y)
    lag6_score = calibrate_and_score(y.values, oof_lag6)
    logger.info(f"  With 6h lags: Cal F1={lag6_score['cal_f1']:.4f}, AUC={auc_lag6:.4f} ({time.time()-t0:.0f}s)")

    delta6 = lag6_score["cal_f1"] - base_score["cal_f1"]
    logger.info(f"  Delta Cal F1 (6h): {delta6:+.4f}")

    lstm_verdict = "YES" if max(delta, delta6) > 0.002 else "NO"
    logger.info(f"\n  LSTM worth pursuing? {lstm_verdict}")
    if lstm_verdict == "NO":
        logger.info("  → Lag features don't help → temporal patterns are already captured")
        logger.info("  → XGBoost's trend features (change_1h, change_3h, etc.) already encode this")
    else:
        logger.info("  → Lag features help → an LSTM could capture even more complex patterns")

    return {
        "name": "sequence",
        "baseline_f1": base_score["cal_f1"],
        "lag3h_f1": lag_score["cal_f1"],
        "lag6h_f1": lag6_score["cal_f1"],
        "delta_3h": delta,
        "delta_6h": delta6,
        "lstm_verdict": lstm_verdict,
        "nonzero_lag_features": nonzero_lags,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("Loading data...")
    X, y, df = load_data()

    results = []

    # Run all 3 experiments
    r1 = experiment_interactions(X, y)
    results.append(r1)

    r2 = experiment_stacking(X, y)
    results.append(r2)

    r3 = experiment_sequence(X, y)
    results.append(r3)

    # ── Final Summary ──
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n  Current baseline (XGBoost 210 features): Cal F1 = {r1['baseline_f1']:.4f}")

    logger.info(f"\n  1. FEATURE INTERACTIONS:")
    logger.info(f"     Cal F1 = {r1['new_f1']:.4f} (delta: {r1['delta']:+.4f})")
    logger.info(f"     Verdict: {'✓ WORTH IT' if r1['delta'] > 0.0005 else '✗ Not worth it'}")

    logger.info(f"\n  2. MODEL STACKING:")
    logger.info(f"     XGBoost alone:    {r2['xgb_f1']:.4f}")
    logger.info(f"     LightGBM alone:   {r2['lgb_f1']:.4f}")
    logger.info(f"     CatBoost alone:   {r2['cb_f1']:.4f}")
    logger.info(f"     Best blend:       {max(r2['avg_blend_f1'], r2['weighted_blend_f1']):.4f} "
                f"(delta: {r2['delta_vs_xgb']:+.4f})")
    logger.info(f"     Correlation: XGB↔LGB={r2['corr_xgb_lgb']:.3f}, XGB↔CB={r2['corr_xgb_cb']:.3f}")
    gains_stacking = r2["delta_vs_xgb"] > 0.001
    logger.info(f"     Verdict: {'✓ WORTH IT' if gains_stacking else '✗ Not worth it'} "
                f"(but 3x training cost + complexity)")

    logger.info(f"\n  3. SEQUENCE MODELING:")
    logger.info(f"     Lag 3h:  {r3['lag3h_f1']:.4f} (delta: {r3['delta_3h']:+.4f})")
    logger.info(f"     Lag 6h:  {r3['lag6h_f1']:.4f} (delta: {r3['delta_6h']:+.4f})")
    logger.info(f"     LSTM worth pursuing? {r3['lstm_verdict']}")
    logger.info(f"     Active lag features: {r3['nonzero_lag_features']}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
