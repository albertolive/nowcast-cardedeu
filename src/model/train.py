"""
Pipeline d'entrenament del model XGBoost per a nowcasting de pluja.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.isotonic import IsotonicRegression

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.features.engineering import FEATURE_COLUMNS, build_target_column

logger = logging.getLogger(__name__)


class InsufficientShadowData(ValueError):
    """No hi ha prou pluja observada per jutjar un candidat; no és una avaria."""


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara X (features) i y (target) per a l'entrenament.
    Elimina files amb NaN al target i gestiona NaN a les features.
    """
    # Construir el target NOMÉS si no existeix. Les files de feedback arriben
    # amb will_rain verificat amb l'estació; recalcular-lo des de la columna
    # 'precipitation' (valor NWP) les reetiquetava: el 2026-08-26 va esborrar
    # la meitat dels events de pluja verificats (116/222) i va invertir-ne 46
    # de secs en plujosos — ensenyant al model a desconfiar exactament dels
    # patrons locals que ha d'aprendre.
    if "will_rain" not in df.columns:
        df = build_target_column(df, "precipitation", horizon=1)
    elif df["will_rain"].isna().any():
        raise ValueError(
            "will_rain present però amb NaN: reconstruir-lo parcialment corrompria "
            "les etiquetes verificades (build_target_column usa rolling sobre el "
            "df sencer). Passa el dataset amb el target complet o sense ell."
        )

    # Filtrar files on no tenim target (últimes hores sense futur conegut)
    df = df.dropna(subset=["will_rain"])

    # Use ALL FEATURE_COLUMNS so the model can learn from feedback data
    # that includes radar/lightning/sentinel features.
    # Missing columns are added as NaN (XGBoost handles this natively).
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Features disponibles: {len(available_features)}/{len(FEATURE_COLUMNS)}")

    X = df.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    y = df["will_rain"].copy()

    # Reemplaçar infinits per NaN (XGBoost gestiona NaN natívament)
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def _new_classifier() -> xgb.XGBClassifier:
    # Paràmetres XGBoost optimitzats per nowcasting (208 features)
    # Tuned via 2-round grid search (2026-03-22):
    #   - colsample 0.6→0.5: forces feature diversity — reduces NWP dominance from
    #     82% to 74%, letting XGBoost learn from tail features (NWP error patterns)
    #   - max_depth 6→7: captures more complex NWP×surface condition interactions
    #   - n_estimators 800→1200, lr 0.02→0.012: more trees compensate for lower lr + diversity
    #   - subsample 0.8→0.75, gamma 0.1→0.15, reg_alpha 0.2→0.3, reg_lambda 1.5→2.0:
    #     stronger regularization matches deeper trees to prevent overfitting
    #   - min_child_weight 5→6: higher to compensate for depth=7
    #   Result: Cal F1 0.7033→0.7061 (+0.0028), top-3 gain 82%→74%
    # Round 3 (2026-03-22): 210 features (cape + nwp_precip_severity)
    #   - colsample_bytree 0.5→0.7, colsample_bynode=0.7: combined diversity at tree+split level
    #     Top-3 gain 80%→74%, more features contribute
    #   Result: Cal F1 0.7061→0.7070 (+0.0009)
    return xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.012,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bynode=0.7,
        min_child_weight=6,
        gamma=0.15,
        reg_alpha=0.3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=96,
        enable_categorical=False,
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[xgb.XGBClassifier, dict, IsotonicRegression]:
    """Entrena el model històric amb validació creuada temporal i calibratge."""
    n_positive = y.sum()
    logger.info(f"Dataset: {len(y)} mostres, {n_positive} pluja ({100*n_positive/len(y):.1f}%)")
    model = _new_classifier()

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
    # Full OOF: isotonic regression is monotonically constrained, can't overfit.
    # More data points = finer step function = better calibration in production.
    oof_mask = ~np.isnan(oof_proba)
    oof_y = y.values[oof_mask]
    oof_p = oof_proba[oof_mask]

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(oof_p, oof_y)
    logger.info(f"Calibratge isotònic ajustat amb {len(oof_p)} prediccions OOF")

    # Trobar llindar òptim (F1) sobre les prediccions calibrades
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


def _shadow_splits(
    history: pd.DataFrame, feedback: pd.DataFrame, before,
) -> tuple[dict[str, pd.DataFrame], dict]:
    cutoff = pd.Timestamp(before)
    if pd.isna(cutoff) or cutoff.tz is None:
        raise ValueError("La data de tall ha d'incloure el fus horari (p. ex. 2026-11-01T00:00:00Z).")
    cutoff = cutoff.tz_convert("UTC")
    horizon = pd.Timedelta(minutes=config.PREDICTION_HORIZON_MIN)
    window = pd.Timedelta(days=14)
    holdout_start = cutoff - window
    calibration_start = holdout_start - window

    required = {"datetime", "verified", "actual_rain", "will_rain"}
    if missing := required.difference(feedback.columns):
        raise ValueError(f"Falten columnes al parquet de verificacions: {sorted(missing)}")
    times = pd.to_datetime(feedback["datetime"], format="ISO8601", utc=True)
    if times.isna().any():
        raise ValueError("Hi ha verificacions sense data d'emissió.")
    before_migration = times < pd.Timestamp(config.FEEDBACK_UTC_FROM)
    eligible = (times + horizon < cutoff) & ~before_migration
    pre_utc = int((before_migration & (times + horizon < cutoff)).sum())
    station = feedback.loc[eligible].copy()
    station.index = pd.DatetimeIndex(times[eligible])
    if station.index.has_duplicates:
        raise ValueError("Hi ha verificacions duplicades per a una mateixa emissió.")
    if (not station["verified"].isin([True]).all()
            or not station["actual_rain"].isin([0, 1]).all()
            or not station["will_rain"].isin([0, 1]).all()
            or not station["will_rain"].eq(station["actual_rain"]).all()):
        raise ValueError("El parquet ha de contenir etiquetes binàries verificades amb l'estació.")
    X, y = prepare_training_data(station)
    has_predictors = X.notna().any(axis=1)
    recent = X.index >= calibration_start
    if not recent.any() or has_predictors[recent].mean() < config.FEEDBACK_MIN_INPUT_COVERAGE:
        raise ValueError(f"Menys del {config.FEEDBACK_MIN_INPUT_COVERAGE * 100:g} % de les verificacions "
                         "recents tenen predictors. Repara el parquet.")
    excluded = int((~has_predictors).sum())
    logger.info("Verificacions sense cap predictor excloses de l'entrenament: %d", excluded)
    station = X.loc[has_predictors].assign(will_rain=y[has_predictors]).sort_index()

    # L'històric d'Open-Meteo usa hora local, el parquet verificat usa UTC.
    # A les dues transicions anuals no podem situar l'hora local 02:00 amb certesa.
    times = pd.to_datetime(history["datetime"], format="ISO8601")
    if times.isna().any():
        raise ValueError("Hi ha files de l'històric sense data.")
    if times.dt.tz is None:
        times = times.dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="NaT")
    times = times.dt.tz_convert("UTC")
    eligible = times + horizon < calibration_start
    base = history.loc[eligible].copy()
    base.index = pd.DatetimeIndex(times[eligible])
    if "will_rain" not in base or not base["will_rain"].isin([0, 1]).all():
        raise ValueError("L'històric processat ha de contenir etiquetes will_rain binàries.")
    X, y = prepare_training_data(base)
    # El conjunt històric va creuar hores UTC de XEMA amb hores locals
    # d'Open-Meteo. Només les lectures emeses en directe tenen una hora fiable.
    xema = ["sentinel_temp_diff", "sentinel_humidity_diff", "sentinel_precip", "sentinel_raining",
            "local_rain_xema", "local_rain_xema_3h"]
    X.loc[:, xema] = np.nan
    base = X.assign(will_rain=y)

    train_station = station.loc[station.index + horizon < calibration_start]
    # Si les dues fonts coincideixen exactament, preval l'etiqueta de l'estació.
    train = pd.concat([base.loc[~base.index.isin(train_station.index)], train_station])
    splits = {
        "train": train.sort_index(kind="stable"),
        "calibration": station.loc[(station.index >= calibration_start)
                                   & (station.index + horizon < holdout_start)],
        "holdout": station.loc[station.index >= holdout_start],
    }
    stats = {}
    for name, frame in splits.items():
        observed = train_station if name == "train" else frame
        # A deu minuts per emissió, sis positius poden ser una sola finestra
        # de pluja. Exigim també dades de mitja quinzena, no només una tempesta.
        n_wet = int(observed.will_rain.sum())
        if (min(n_wet, len(observed) - n_wet) < 6
                or observed.index.normalize().nunique() < 7):
            raise InsufficientShadowData(f"{name}: cobertura insuficient amb predictors i etiquetes d'estació "
                             f"({len(observed)} files, {n_wet} amb pluja).")
        stats[name] = {
            "first_issue": frame.index.min().isoformat(),
            "last_issue": frame.index.max().isoformat(),
            "last_label_end": (frame.index.max() + horizon).isoformat(),
            "n_samples": len(frame),
            "n_positive": int(frame.will_rain.sum()),
            "station_rows": len(observed),
            "station_positive": n_wet,
        }
    return splits, {
        "training_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "embargo_minutes": config.PREDICTION_HORIZON_MIN,
        "feedback_without_predictors": excluded,
        "feedback_before_utc_migration": pre_utc,
        "history_columns_masked": xema,
        "splits": stats,
    }


def train_shadow_model(
    history: pd.DataFrame, feedback: pd.DataFrame, before,
) -> tuple[xgb.XGBClassifier, dict, IsotonicRegression]:
    """Ajusta el candidat sense fer servir la quinzena final per triar cap paràmetre."""
    splits, metrics = _shadow_splits(history, feedback, before)
    train, calibration, holdout = (splits[key] for key in ("train", "calibration", "holdout"))
    model = _new_classifier()
    model.fit(train[FEATURE_COLUMNS], train.will_rain,
              eval_set=[(calibration[FEATURE_COLUMNS], calibration.will_rain)], verbose=False)
    raw = model.predict_proba(calibration[FEATURE_COLUMNS])[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw, calibration.will_rain)
    precision, recall, thresholds = precision_recall_curve(calibration.will_rain, calibrator.predict(raw))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])

    probabilities = calibrator.predict(model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1])
    labels = holdout.will_rain
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, probabilities))
    if not np.isfinite(auc) or auc <= 0.5 or tp == 0:
        raise ValueError(f"El candidat no distingeix la pluja a la validació: AUC={auc:.4f}, TP={tp}.")
    metrics.update({
        "calibrated": True,
        "optimal_threshold": threshold,
        "holdout_auc": auc,
        "holdout_f1": float(f1_score(labels, predictions)),
        "holdout_brier": float(brier_score_loss(labels, probabilities)),
        "holdout_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "best_iteration": model.best_iteration,
    })
    return model, metrics, calibrator


def get_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Retorna les importàncies de les features ordenades."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return fi


def _artifact_paths(model_dir):
    directory = Path(config.MODELS_DIR if model_dir is None else model_dir)
    paths = (Path(config.MODEL_PATH), Path(config.FEATURE_NAMES_PATH), Path(config.CALIBRATOR_PATH))
    if model_dir is not None:
        paths = tuple(directory / path.name for path in paths)
    return (*paths, directory / "metrics.json")


def _model_fingerprint(metrics, contents):
    identity = hashlib.sha256(json.dumps([metrics["training_cutoff"], metrics["optimal_threshold"]]).encode())
    for data in contents:
        identity.update(data)
    return identity.hexdigest()


class _ClippedIsotonic:
    """Portable monotone interpolation; the live sklearn calibrator is untouched."""
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (self.x.ndim != 1 or self.x.shape != self.y.shape or not self.x.size
                or not np.isfinite(self.x).all() or not np.isfinite(self.y).all()
                or not (np.diff(self.x) > 0).all() or not (np.diff(self.y) >= 0).all()
                or self.y[0] < 0 or self.y[-1] > 1):
            raise ValueError("Invalid isotonic knots")

    def predict(self, raw):
        return np.interp(raw, self.x, self.y)


def save_model(model: xgb.XGBClassifier, feature_names: list[str], metrics: dict,
               calibrator: Optional[IsotonicRegression] = None, *, model_dir=None) -> None:
    """Desa el model, calibrador, noms de features i mètriques."""
    if "training_cutoff" in metrics and (model_dir is None or calibrator is None):
        raise ValueError("A shadow model requires an isolated directory and a fitted calibrator")
    model_path, names_path, cal_path, metrics_path = _artifact_paths(model_dir)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(model_path)
    names_path.write_text(json.dumps(feature_names))
    if "training_cutoff" in metrics:
        cal_path = cal_path.with_suffix(".json")
        cal_path.write_text(json.dumps({"x": calibrator.X_thresholds_.tolist(),
                                        "y": calibrator.y_thresholds_.tolist()}, allow_nan=False))
        contents = (p.read_bytes() for p in (model_path, names_path, cal_path))
        metrics = {**metrics, "model_id": _model_fingerprint(metrics, contents)}
    elif calibrator is not None:
        joblib.dump(calibrator, cal_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str, allow_nan=False))
    logger.info("Model i mètriques desats a %s", metrics_path.parent)


def save_shadow_model(model, metrics, calibrator) -> None:
    """Desa un candidat per revisar sense substituir el que puntua en producció."""
    root = Path(config.MODELS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "shadow-next"
    with TemporaryDirectory(prefix=".shadow-", dir=root) as work:
        stage, previous = Path(work) / "new", Path(work) / "previous"
        save_model(model, FEATURE_COLUMNS, metrics, calibrator, model_dir=stage)
        load_model(model_dir=stage)
        try:
            if target.exists():
                target.rename(previous)
            stage.rename(target)
        finally:
            if previous.exists() and not target.exists():
                previous.rename(target)


def load_model(model_dir=None) -> tuple[xgb.XGBClassifier, list[str], IsotonicRegression | _ClippedIsotonic | None, float]:
    """Carrega els quatre fitxers del candidat o, per defecte, el model de producció."""
    model_path, names_path, cal_path, metrics_path = _artifact_paths(model_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model no trobat: {model_path}")

    model = xgb.XGBClassifier()
    if model_dir is not None and (metrics_path.exists() or Path(model_dir).name == "shadow"):
        metrics = json.loads(metrics_path.read_text())
        if "training_cutoff" in metrics or Path(model_dir).name == "shadow":
            # El desplegament pot canviar els fitxers entre lectures. Deserialitzem
            # els mateixos bytes que hem verificat, sense tornar a obrir els fitxers.
            cal_path = cal_path.with_suffix(".json")
            contents = tuple(p.read_bytes() for p in (model_path, names_path, cal_path))
            fingerprint = _model_fingerprint(metrics, contents)
            if metrics.get("model_id") != fingerprint:
                raise ValueError(f"L'empremta del candidat no coincideix amb els fitxers de {model_dir}.")
            model.load_model(bytearray(contents[0]))
            feature_names = json.loads(contents[1])
            calibrator = _ClippedIsotonic(**json.loads(contents[2]))
            threshold = metrics["optimal_threshold"]
            if not 0 <= threshold <= 1 or feature_names != model.get_booster().feature_names:
                raise ValueError(f"Fitxers de model incompatibles: {model_dir}")
            model.nowcast_model_id = fingerprint
            return model, feature_names, calibrator, threshold
    model.load_model(model_path)
    feature_names = json.loads(names_path.read_text())
    calibrator = None
    if cal_path.exists():
        calibrator = joblib.load(cal_path)

    threshold = config.ALERT_PROBABILITY_THRESHOLD
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        threshold = metrics.get("optimal_threshold", threshold)

    return model, feature_names, calibrator, threshold
