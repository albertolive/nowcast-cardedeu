#!/usr/bin/env python3
"""
Script 4: Predicció en temps real.
Usat per GitHub Actions cada 15 minuts.
Obté dades actuals, fa la predicció i gestiona notificacions
basades en transicions d'estat (no spam).
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now
from src.notify.telegram import send_rain_incoming, send_rain_clearing
from src.notify.state import load_state, save_state, should_notify, update_state

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
    logger.info(f"  Condicions: {json.dumps(result['conditions'], indent=4)}")
    logger.info(f"  Radar: {json.dumps(result.get('radar', {}), indent=4)}")
    logger.info(f"  Sentinella: {json.dumps(result.get('sentinel', {}), indent=4)}")

    # Desar resultat en JSON (per GitHub Actions artifact)
    output_path = os.path.join(config.PROJECT_ROOT, "data", "latest_prediction.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"  Resultat desat a {output_path}")

    # ── Notificacions basades en transicions d'estat ──
    state = load_state()
    logger.info(f"  Estat actual: {state['current_state']} | Prob: {result['probability_pct']}%")

    notification_type = should_notify(probability, state)

    if notification_type == "rain_incoming":
        logger.info("⚠️  Transició: clear → rain_alert. Enviant alerta de pluja...")
        send_rain_incoming(result)
        update_state(state, "rain_incoming", probability)
    elif notification_type == "rain_clearing":
        logger.info("☀️  Transició: rain_alert → clear. Enviant avís de millora...")
        send_rain_clearing(result)
        update_state(state, "rain_clearing", probability)
    else:
        logger.info(f"  Sense canvi d'estat. No es notifica. (estat={state['current_state']})")
        # Actualitzar la probabilitat sense canviar l'estat
        state["last_probability"] = probability
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
