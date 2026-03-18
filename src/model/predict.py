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

    logger.info("Obtenint dades de radar (RainViewer)...")
    radar_data = fetch_radar_at_cardedeu()
    logger.info(f"  Radar: dBZ={radar_data['radar_dbz']}, echo={radar_data['radar_has_echo']}, "
                f"approaching={radar_data['radar_approaching']}")

    logger.info("Obtenint dades sentinella (Meteocat XEMA)...")
    sentinel_data = fetch_sentinel_latest()
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
        "features_used": len(feature_names),
        "threshold": config.ALERT_PROBABILITY_THRESHOLD,
    }

    logger.info(
        f"Predicció: {result['probability_pct']}% probabilitat de pluja "
        f"(confiança: {result['confidence']}, alerta: {result['will_rain']})"
    )

    return result
