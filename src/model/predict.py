"""
Predicció en temps real: combina dades locals + models i executa XGBoost.
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.meteocardedeu import fetch_series, fetch_latest
from src.data.open_meteo import fetch_forecast
from src.data.rainviewer import fetch_radar_at_cardedeu
from src.data.meteocat import fetch_sentinel_latest, compute_sentinel_features
from src.data.ensemble import fetch_ensemble_agreement, compute_forecast_bias
from src.data.aemet import fetch_hourly_forecast as fetch_aemet_hourly
from src.features.engineering import build_features_from_realtime, FEATURE_COLUMNS
from src.model.train import load_model

logger = logging.getLogger(__name__)


def predict_now() -> dict:
    """
    Executa una predicció en temps real.
    Retorna un diccionari amb:
    - probability: probabilitat de pluja (0-1)
    - will_rain: bool
    - confidence: str (Alta/Mitja/Baixa)
    - conditions: dict amb les condicions actuals
    - timestamp: str
    """
    logger.info("Obtenint dades en temps real de l'estació...")
    station_df = fetch_series(hours=24)

    logger.info("Obtenint previsió d'Open-Meteo...")
    forecast_df = fetch_forecast(hours_ahead=48)

    logger.info("Obtenint acord entre models (Ensemble)...")
    ensemble_data = fetch_ensemble_agreement()

    logger.info("Obtenint dades de radar (RainViewer)...")
    radar_data = fetch_radar_at_cardedeu()
    logger.info(f"  Radar: dBZ={radar_data['radar_dbz']}, echo={radar_data['radar_has_echo']}, "
                f"approaching={radar_data['radar_approaching']}")

    # ── AEMET: probabilitats de precipitació i tempesta ──
    aemet_data = {"aemet_prob_precip": np.nan, "aemet_prob_storm": np.nan, "aemet_precip_today": np.nan}
    if config.AEMET_API_KEY:
        logger.info("Obtenint previsió AEMET (probTormenta)...")
        aemet_data = fetch_aemet_hourly()
    else:
        logger.info("AEMET no configurat (sense AEMET_API_KEY)")

    # ── Rain gate: només consultar Meteocat si hi ha senyals de pluja ──
    rain_signals = (
        ensemble_data.get("ensemble_rain_agreement", 0) >= config.RAIN_GATE_ENSEMBLE_PROB
        or radar_data.get("radar_has_echo", False)
        or (not np.isnan(aemet_data.get("aemet_prob_storm", 0) or 0)
            and (aemet_data.get("aemet_prob_storm", 0) or 0) >= config.RAIN_GATE_AEMET_STORM)
    )
    # Also check CAPE from forecast
    cape_vals = forecast_df["cape"].dropna() if "cape" in forecast_df.columns else pd.Series()
    cape_max_6h = float(cape_vals.head(6).max()) if not cape_vals.empty else 0
    rain_signals = rain_signals or cape_max_6h >= config.RAIN_GATE_CAPE_THRESHOLD

    sentinel_data = {}
    if rain_signals:
        logger.info("🚨 Rain gate OBERT — consultant Meteocat XEMA...")
        sentinel_data = fetch_sentinel_latest()
    else:
        logger.info("✅ Rain gate tancat — no cal Meteocat (estalvi d'API)")
        sentinel_data = {"sentinel_temp": None, "sentinel_humidity": None, "sentinel_precip": None}
    logger.info(f"  Sentinella: temp={sentinel_data.get('sentinel_temp')}, "
                f"precip={sentinel_data.get('sentinel_precip')}")

    logger.info("Construint features...")
    features_df = build_features_from_realtime(station_df, forecast_df)

    if features_df.empty:
        raise ValueError("No s'han pogut construir features")

    # Agafar l'última fila (moment actual)
    latest = features_df.iloc[-1:].copy()

    # Afegir features de radar a l'última fila
    for k, v in radar_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir features sentinella
    latest_data_station = fetch_latest()
    current = latest_data_station.get("dades_act", {})
    station_temp = float(current.get("TEMP", 0) or 0)
    station_hum = int(current.get("HUM", 0) or 0)
    sentinel_features = compute_sentinel_features(sentinel_data, station_temp, station_hum)
    for k, v in sentinel_features.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir features d'ensemble (acord entre models)
    for k, v in ensemble_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir bias del forecast vs observació
    bias_data = compute_forecast_bias(station_temp, station_hum, forecast_df)
    for k, v in bias_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir features AEMET
    for k, v in aemet_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    logger.info("Carregant model...")
    model, feature_names = load_model()

    # Preparar el vector de features (alinear amb les que el model espera)
    X = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in latest.columns:
            X[col] = latest[col].values
        else:
            X[col] = [np.nan]

    X = X.replace([np.inf, -np.inf], np.nan)

    # Predicció
    probability = float(model.predict_proba(X)[:, 1][0])
    will_rain = probability >= config.ALERT_PROBABILITY_THRESHOLD

    # Nivell de confiança
    if probability >= 0.85:
        confidence = "Molt Alta"
    elif probability >= 0.70:
        confidence = "Alta"
    elif probability >= 0.50:
        confidence = "Mitjana"
    elif probability >= 0.30:
        confidence = "Baixa"
    else:
        confidence = "Molt Baixa"

    # Condicions actuals
    result = {
        "probability": round(probability, 4),
        "probability_pct": round(probability * 100, 1),
        "will_rain": will_rain,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "conditions": {
            "temperature": current.get("TEMP"),
            "humidity": current.get("HUM"),
            "pressure": current.get("BAR"),
            "wind_speed": current.get("VEL"),
            "wind_dir": current.get("DIR"),
            "rain_today": current.get("PINT", "0"),
            "solar_radiation": current.get("SUN"),
        },
        "radar": {
            "dbz": radar_data["radar_dbz"],
            "rain_rate_mmh": radar_data["radar_rain_rate"],
            "has_echo": radar_data["radar_has_echo"],
            "approaching": radar_data["radar_approaching"],
        },
        "sentinel": {
            "station": config.SENTINEL_STATION_NAME,
            "temp": sentinel_data.get("sentinel_temp"),
            "humidity": sentinel_data.get("sentinel_humidity"),
            "precip": sentinel_data.get("sentinel_precip"),
            "raining": sentinel_features.get("sentinel_raining", 0),
        },
        "ensemble": {
            "rain_agreement": ensemble_data.get("ensemble_rain_agreement"),
            "precip_spread_mm": ensemble_data.get("ensemble_precip_spread"),
            "models_rain": ensemble_data.get("ensemble_models_rain"),
            "total_models": 4,  # ECMWF + GFS + ICON + AROME
        },
        "aemet": {
            "prob_precip": aemet_data.get("aemet_prob_precip"),
            "prob_storm": aemet_data.get("aemet_prob_storm"),
        },
        "bias": {
            "temp": bias_data.get("forecast_temp_bias"),
            "humidity": bias_data.get("forecast_humidity_bias"),
        },
        "wind_regime": {
            "is_llevantada": bool(latest.get("is_llevantada", pd.Series([0])).values[0]),
            "is_garbi": bool(latest.get("is_garbi", pd.Series([0])).values[0]),
            "is_ponent": bool(latest.get("is_ponent", pd.Series([0])).values[0]),
            "llevantada_strength": float(latest.get("llevantada_strength", pd.Series([0])).values[0]),
            "wind_dir_change_3h": float(latest.get("wind_dir_change_3h", pd.Series([0])).values[0]) if pd.notna(latest.get("wind_dir_change_3h", pd.Series([np.nan])).values[0]) else None,
        },
        "rain_gate_open": rain_signals,
        "features_used": len(feature_names),
        "threshold": config.ALERT_PROBABILITY_THRESHOLD,
    }

    logger.info(
        f"Predicció: {result['probability_pct']}% probabilitat de pluja "
        f"(confiança: {result['confidence']}, alerta: {result['will_rain']})"
    )

    return result
