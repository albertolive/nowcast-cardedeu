#!/usr/bin/env python3
"""
Script 4: Predicció en temps real.
Usat per GitHub Actions cada 15 minuts.
Obté dades actuals, fa la predicció i envia alerta si cal.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now
from src.notify.telegram import send_prediction_alert

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

    # Mostrar resultat
    logger.info(f"\n📊 Resultat:")
    logger.info(f"  Probabilitat de pluja: {result['probability_pct']}%")
    logger.info(f"  Confiança: {result['confidence']}")
    logger.info(f"  Alerta: {'SÍ' if result['will_rain'] else 'NO'}")
    logger.info(f"  Condicions: {json.dumps(result['conditions'], indent=4)}")
    logger.info(f"  Radar: {json.dumps(result.get('radar', {}), indent=4)}")
    logger.info(f"  Sentinella: {json.dumps(result.get('sentinel', {}), indent=4)}")

    # Desar resultat en JSON (per GitHub Actions artifact)
    output_path = os.path.join(config.PROJECT_ROOT, "data", "latest_prediction.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"  Resultat desat a {output_path}")

    # Enviar alerta per Telegram si cal
    if result["will_rain"]:
        logger.info("⚠️  Enviant alerta de pluja per Telegram...")
        send_prediction_alert(result)
    else:
        logger.info("✅ No es preveu pluja. No s'envia alerta.")

    # Output per GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"probability={result['probability_pct']}\n")
            f.write(f"will_rain={str(result['will_rain']).lower()}\n")
            f.write(f"confidence={result['confidence']}\n")

    return result


if __name__ == "__main__":
    main()
