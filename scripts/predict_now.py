#!/usr/bin/env python3
"""
Script 4: Predicció en temps real.
Usat per GitHub Actions cada 10 minuts.
Obté dades actuals, fa la predicció i gestiona notificacions
basades en transicions d'estat (no spam) + canvis de règim atmosfèric.
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now
from src.notify.telegram import send_rain_incoming, send_rain_clearing, send_regime_change
from src.notify.state import (
    load_state, save_state, should_notify, should_notify_regime, update_state,
)
from src.features.regime import detect_regime_change
from src.feedback.logger import log_prediction, _NumpyEncoder, _sanitize_nans
from src.feedback.verify import verify_pending_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model no trobat a {config.MODEL_PATH}")
        logger.error("Primer executa: download_history → build_dataset → train_model")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🌦️  Nowcast Cardedeu — Predicció en temps real")
    logger.info("=" * 60)

    try:
        result = predict_now()
    except Exception as e:
        logger.error(f"Error en la predicció: {e}", exc_info=True)
        sys.exit(1)

    probability = result["probability"] if "probability" in result else result["probability_pct"] / 100

    # Mostrar resultat
    logger.info(f"\n📊 Resultat:")
    logger.info(f"  Probabilitat de pluja: {result['probability_pct']}%")
    logger.info(f"  Confiança: {result['confidence']}")
    logger.info(f"  Condicions: {json.dumps(result['conditions'], indent=4, cls=_NumpyEncoder)}")
    logger.info(f"  Radar: {json.dumps(result.get('radar', {}), indent=4, cls=_NumpyEncoder)}")
    logger.info(f"  Sentinella: {json.dumps(result.get('sentinel', {}), indent=4, cls=_NumpyEncoder)}")

    # Desar resultat en JSON (per GitHub Actions artifact)
    output_path = os.path.join(config.PROJECT_ROOT, "data", "latest_prediction.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_sanitize_nans(result), f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    logger.info(f"  Resultat desat a {output_path}")

    # ── Registrar predicció al log de feedback ──
    log_prediction(result)

    # ── Verificar prediccions passades (60+ min enrere) ──
    logger.info("🔍 Verificant prediccions anteriors...")
    try:
        verification = verify_pending_predictions()
    except Exception as e:
        logger.error(f"verify_pending_predictions va fallar: {e}", exc_info=True)
        verification = {"verified_count": 0, "skipped": 0, "error": str(e)[:200]}
    if verification.get("verified_count", 0) > 0:
        acc = verification.get("accuracy", "?")
        logger.info(f"  Verificades: {verification['verified_count']} | Accuracy: {acc}%")

    # Embed diagnostics so they land in data/latest_prediction.json (pushed every cycle).
    result["verification"] = verification
    with open(output_path, "w") as f:
        json.dump(_sanitize_nans(result), f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # ── Notificacions basades en transicions d'estat ──
    state = load_state()
    logger.info(f"  Estat actual: {state['current_state']} | Prob: {result['probability_pct']}%")

    wind_regime = result.get("wind_regime", {})
    threshold = result.get("threshold", config.ALERT_PROBABILITY_THRESHOLD)
    notification_type = should_notify(probability, state)

    if notification_type == "rain_incoming":
        logger.info("⚠️  Transició: clear → rain_alert. Enviant alerta de pluja...")
        send_rain_incoming(result)
        update_state(state, "rain_incoming", probability, wind_regime=wind_regime)
    elif notification_type == "rain_clearing":
        logger.info("☀️  Transició: rain_alert → clear. Enviant avís de millora...")
        send_rain_clearing(result)
        update_state(state, "rain_clearing", probability, wind_regime=wind_regime)
    else:
        logger.info(f"  Sense canvi d'estat de pluja. (estat={state['current_state']})")

        # ── Detecció de canvi de règim atmosfèric ──
        regime_change = detect_regime_change(result, state)
        if regime_change and probability < threshold:
            logger.info(
                f"🌬️  Canvi de règim detectat: {regime_change['type']} "
                f"— però model diu {result['probability_pct']}% (< llindar {threshold:.0%}), no s'envia."
            )
            regime_change = None
        if regime_change:
            logger.info(f"🌬️  Canvi de règim detectat: {regime_change['type']} ({regime_change['severity']})")
            if should_notify_regime(regime_change, state):
                logger.info(f"  Enviant alerta de règim: {regime_change['title']}")
                send_regime_change(result, regime_change)
                update_state(
                    state, "regime_change", probability,
                    wind_regime=wind_regime,
                    regime_alert_type=regime_change["type"],
                )
            else:
                logger.info("  Canvi de règim detectat però no s'envia (cooldown/repetit).")
                state["last_probability"] = probability
                state["last_wind_regime"] = wind_regime
                save_state(state)
        else:
            # Actualitzar la probabilitat i règim sense notificació
            state["last_probability"] = probability
            state["last_wind_regime"] = wind_regime
            save_state(state)

    # Output per GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"probability={result['probability_pct']}\n")
            f.write(f"will_rain={str(result.get('will_rain', False)).lower()}\n")
            f.write(f"confidence={result['confidence']}\n")
            f.write(f"notification={notification_type or 'none'}\n")

    return result


if __name__ == "__main__":
    main()
