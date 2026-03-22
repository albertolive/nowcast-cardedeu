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
from src.data.open_meteo import fetch_forecast, fetch_pressure_levels, fetch_sst_forecast
from src.data.rainviewer import fetch_radar_at_cardedeu
from src.data.meteocat import fetch_sentinel_latest, compute_sentinel_features
from src.data.meteocat_xdde import compute_lightning_features
from src.data.meteocat_prediccio import fetch_municipal_hourly_forecast
from src.data.aemet_radar import fetch_aemet_radar
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

    logger.info("Obtenint dades a 850hPa (flux sinòptic)...")
    pressure_data = fetch_pressure_levels()

    logger.info("Obtenint SST Mediterrani (Marine API)...")
    sst_data = fetch_sst_forecast()

    logger.info("Obtenint dades de radar (RainViewer)...")
    # Passar la direcció del vent a 850hPa per escaneig del sector de sobrevent
    wind_from_dir = pressure_data.get("wind_850_dir")
    radar_data = fetch_radar_at_cardedeu(wind_from_dir=wind_from_dir)
    logger.info(f"  Radar: dBZ={radar_data['radar_dbz']}, echo={radar_data['radar_has_echo']}, "
                f"approaching={radar_data['radar_approaching']}")

    # ── Radar AEMET Barcelona (complement professional al RainViewer) ──
    aemet_radar_data = {"aemet_radar_dbz": 0.0, "aemet_radar_has_echo": False,
                        "aemet_radar_nearest_echo_km": config.RADAR_SCAN_RADIUS_KM,
                        "aemet_radar_max_dbz_20km": 0.0, "aemet_radar_coverage_20km": 0.0,
                        "aemet_radar_echoes_found": False, "aemet_radar_available": False}
    if config.AEMET_API_KEY:
        logger.info("Obtenint radar AEMET Barcelona...")
        aemet_radar_data = fetch_aemet_radar()
    else:
        logger.info("AEMET radar no configurat (sense AEMET_API_KEY)")

    # ── AEMET: probabilitats de precipitació i tempesta ──
    aemet_data = {"aemet_prob_precip": np.nan, "aemet_prob_storm": np.nan, "aemet_precip_today": np.nan}
    if config.AEMET_API_KEY:
        logger.info("Obtenint previsió AEMET (probTormenta)...")
        aemet_data = fetch_aemet_hourly()
    else:
        logger.info("AEMET no configurat (sense AEMET_API_KEY)")

    # ── Rain gate: només consultar Meteocat si hi ha senyals de pluja ──
    # Checked BEFORE any Meteocat call to stay within separate quotas:
    #   XDDE: 250/month, Predicció: 100/month, XEMA: 750/month
    rain_signals = (
        ensemble_data.get("ensemble_rain_agreement", 0) >= config.RAIN_GATE_ENSEMBLE_PROB
        or radar_data.get("radar_has_echo", False)
        or radar_data.get("radar_nearest_echo_km", 30) < config.RAIN_GATE_RADAR_NEARBY_KM
        or (not np.isnan(aemet_data.get("aemet_prob_storm", 0) or 0)
            and (aemet_data.get("aemet_prob_storm", 0) or 0) >= config.RAIN_GATE_AEMET_STORM)
        or aemet_radar_data.get("aemet_radar_has_echo", False)
    )
    # Also check CAPE from forecast
    cape_vals = forecast_df["cape"].dropna() if "cape" in forecast_df.columns else pd.Series()
    cape_max_6h = float(cape_vals.head(6).max()) if not cape_vals.empty else 0
    rain_signals = rain_signals or cape_max_6h >= config.RAIN_GATE_CAPE_THRESHOLD

    # ── Meteocat calls: ALL behind rain gate (XDDE 250/mo, Predicció 100/mo, XEMA 750/mo) ──
    lightning_data = {}
    smc_forecast = {"smc_prob_precip_1h": np.nan, "smc_prob_precip_6h": np.nan,
                    "smc_precip_intensity": np.nan, "smc_temp_forecast": np.nan,
                    "smc_weather_symbol": np.nan}
    sentinel_data = {"sentinel_temp": None, "sentinel_humidity": None, "sentinel_precip": None}

    if rain_signals:
        logger.info("🚨 Rain gate OBERT — consultant Meteocat (XDDE + Predicció + XEMA)...")
        lightning_data = compute_lightning_features()
        if config.METEOCAT_API_KEY:
            smc_forecast = fetch_municipal_hourly_forecast()
        sentinel_data = fetch_sentinel_latest()
    else:
        logger.info("✅ Rain gate tancat — no cal Meteocat (estalvi d'API)")
    logger.info(f"  Sentinella: temp={sentinel_data.get('sentinel_temp')}, "
                f"precip={sentinel_data.get('sentinel_precip')}")

    logger.info("Construint features...")
    # Afegir dades de nivells de pressió al forecast_df ABANS de build_features
    # perquè _add_wind_regime_features pugui usar wind_850_dir (sinòptic)
    # en lloc del vent de superfície (10m), que és distorsionat per orografia
    if pressure_data and not forecast_df.empty:
        forecast_df = forecast_df.copy()
        for k, v in pressure_data.items():
            if k not in forecast_df.columns:
                forecast_df[k] = v
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
    if not current and not station_df.empty:
        # Fallback: usar última lectura de la sèrie temporal si fetch_latest() falla
        last_row = station_df.iloc[-1]
        logger.warning("Usant última lectura de sèrie com a fallback per dades actuals")
        current = {
            "TEMP": last_row.get("TEMP"),
            "HUM": last_row.get("HUM"),
            "BAR": last_row.get("BAR"),
            "VEL": last_row.get("VEL"),
            "DIR": last_row.get("DIR"),
            "PINT": last_row.get("PINT"),
            "SUN": last_row.get("SUN"),
        }
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

    # Afegir features de llamps (XDDE Meteocat)
    for k, v in lightning_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir features de radar AEMET Barcelona
    for k, v in aemet_radar_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir features de predicció municipal SMC (Meteocat)
    for k, v in smc_forecast.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir dades de nivells de pressió (850hPa, 500hPa)
    for k, v in pressure_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    # Afegir SST Mediterrani
    for k, v in sst_data.items():
        if k in FEATURE_COLUMNS:
            latest[k] = v

    logger.info("Carregant model...")
    model, feature_names, calibrator, threshold = load_model()

    # Preparar el vector de features (alinear amb les que el model espera)
    X = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in latest.columns:
            X[col] = latest[col].values
        else:
            X[col] = [np.nan]

    X = X.replace([np.inf, -np.inf], np.nan)
    # Ensure all columns are numeric (None from closed rain gate → object dtype)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Predicció amb calibratge
    raw_probability = float(model.predict_proba(X)[:, 1][0])
    if calibrator is not None:
        probability = float(calibrator.predict([raw_probability])[0])
    else:
        probability = raw_probability
    will_rain = probability >= threshold

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
            "nearest_echo_km": radar_data.get("radar_nearest_echo_km"),
            "nearest_echo_compass": radar_data.get("radar_nearest_echo_compass"),
            "max_dbz_20km": radar_data.get("radar_max_dbz_20km"),
            "coverage_20km": radar_data.get("radar_coverage_20km"),
            "upwind_nearest_echo_km": radar_data.get("radar_upwind_nearest_echo_km"),
            "storm_velocity_kmh": radar_data.get("radar_storm_velocity_kmh"),
            "storm_eta_min": radar_data.get("radar_storm_eta_min"),
            "quadrants": {
                "max_dbz_N": radar_data.get("radar_quadrant_max_dbz_N", 0.0),
                "max_dbz_E": radar_data.get("radar_quadrant_max_dbz_E", 0.0),
                "max_dbz_S": radar_data.get("radar_quadrant_max_dbz_S", 0.0),
                "max_dbz_W": radar_data.get("radar_quadrant_max_dbz_W", 0.0),
                "coverage_N": radar_data.get("radar_quadrant_coverage_N", 0.0),
                "coverage_E": radar_data.get("radar_quadrant_coverage_E", 0.0),
                "coverage_S": radar_data.get("radar_quadrant_coverage_S", 0.0),
                "coverage_W": radar_data.get("radar_quadrant_coverage_W", 0.0),
            },
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
            "level": "850hPa" if pressure_data.get("wind_850_dir") is not None else "10m",
            "is_tramuntana": bool(latest.get("is_tramuntana", pd.Series([0])).values[0]),
            "is_llevantada": bool(latest.get("is_llevantada", pd.Series([0])).values[0]),
            "is_migjorn": bool(latest.get("is_migjorn", pd.Series([0])).values[0]),
            "is_garbi": bool(latest.get("is_garbi", pd.Series([0])).values[0]),
            "is_ponent": bool(latest.get("is_ponent", pd.Series([0])).values[0]),
            "llevantada_strength": float(latest.get("llevantada_strength", pd.Series([0])).values[0]),
            "wind_dir_change_3h": float(latest.get("wind_dir_change_3h", pd.Series([0])).values[0]) if pd.notna(latest.get("wind_dir_change_3h", pd.Series([np.nan])).values[0]) else None,
        },
        "pressure_levels": {
            "temp_925": pressure_data.get("temp_925"),
            "rh_925": pressure_data.get("rh_925"),
            "wind_925_speed_kmh": pressure_data.get("wind_925_speed"),
            "wind_925_dir": pressure_data.get("wind_925_dir"),
            "wind_850_dir": pressure_data.get("wind_850_dir"),
            "wind_850_speed_kmh": pressure_data.get("wind_850_speed"),
            "temp_850": pressure_data.get("temp_850"),
            "rh_850": pressure_data.get("rh_850"),
            "rh_700": pressure_data.get("rh_700"),
            "temp_700": pressure_data.get("temp_700"),
            "temp_500": pressure_data.get("temp_500"),
            "wind_300_speed_kmh": pressure_data.get("wind_300_speed"),
            "wind_300_dir": pressure_data.get("wind_300_dir"),
            "gph_300": pressure_data.get("gph_300"),
            "vt_index": pressure_data.get("vt_index"),
            "tt_index": pressure_data.get("tt_index"),
            "li_index": pressure_data.get("li_index"),
        },
        "sst": {
            "sst_med": sst_data.get("sst_med"),
        },
        "rain_gate_open": rain_signals,
        "features_used": len(feature_names),
        "threshold": threshold,
        "calibrated": calibrator is not None,
        "raw_probability": round(raw_probability, 4),
        # Save ALL 131 FEATURE_COLUMNS (not just model's feature_names)
        # so the feedback loop accumulates radar/lightning/sentinel data for retraining
        "feature_vector": {
            col: (float(latest[col].values[0]) if col in latest.columns and pd.notna(latest[col].values[0]) else None)
            for col in FEATURE_COLUMNS
        },
        "pressure_change_3h": float(latest.get("pressure_change_3h", pd.Series([np.nan])).values[0])
            if pd.notna(latest.get("pressure_change_3h", pd.Series([np.nan])).values[0]) else None,
    }

    logger.info(
        f"Predicció: {result['probability_pct']}% probabilitat de pluja "
        f"(confiança: {result['confidence']}, alerta: {result['will_rain']})"
    )

    return result


def predict_hourly_forecast(hours_ahead: int = 48) -> list[dict]:
    """
    Executa el model ML sobre cada hora futura del forecast.
    Usa les features disponibles del forecast Open-Meteo + pressure levels + SMC.
    Features de radar/sentinella/bias queden com NaN (XGBoost ho gestiona).

    Retorna llista de dicts: [{"datetime": ..., "probability": ..., "will_rain": ...}, ...]
    """
    from src.data.open_meteo import fetch_forecast as _fetch_forecast
    from src.data.open_meteo import fetch_pressure_levels_hourly
    from src.data.meteocat_prediccio import fetch_smc_hourly_df
    from src.features.engineering import build_features_from_forecast, FEATURE_COLUMNS

    logger.info(f"Generant forecast ML per a les properes {hours_ahead}h...")

    forecast_df = _fetch_forecast(hours_ahead=hours_ahead)
    if forecast_df.empty:
        logger.warning("No forecast data available")
        return []

    pressure_df = fetch_pressure_levels_hourly(hours_ahead=hours_ahead)

    # SMC municipal forecast (72h, Cardedeu-specific)
    smc_df = pd.DataFrame()
    try:
        smc_df = fetch_smc_hourly_df()
    except Exception as e:
        logger.warning(f"SMC hourly forecast no disponible: {e}")

    features_df = build_features_from_forecast(forecast_df, pressure_df, smc_df)
    if features_df.empty:
        logger.warning("No features built from forecast")
        return []

    model, feature_names, calibrator, threshold = load_model()

    # Build X matrix aligned with model features
    X = pd.DataFrame(columns=feature_names, index=features_df.index)
    for col in feature_names:
        if col in features_df.columns:
            X[col] = features_df[col].values
        else:
            X[col] = np.nan

    X = X.replace([np.inf, -np.inf], np.nan).astype(float)

    raw_probs = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict(raw_probs)
    else:
        probs = raw_probs

    results = []
    for i, row in features_df.iterrows():
        prob = float(probs[i] if i < len(probs) else 0)
        results.append({
            "datetime": row["datetime"],
            "probability": round(prob, 4),
            "will_rain": prob >= threshold,
        })

    rain_hours = sum(1 for r in results if r["will_rain"])
    logger.info(f"  ML forecast: {rain_hours}/{len(results)} hores amb pluja prevista")

    return results
