#!/usr/bin/env python3
"""
Script: Resum diari del matí.
Executat per GitHub Actions a les 7:00 cada dia.
Fa una predicció i envia un resum per Telegram.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now
from src.notify.telegram import send_daily_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model no trobat a {config.MODEL_PATH}")
        sys.exit(1)

    logger.info("📋 Nowcast Cardedeu — Resum diari")

    try:
        result = predict_now()
    except Exception as e:
        logger.error(f"Error en la predicció: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Probabilitat: {result['probability_pct']}% ({result['confidence']})")

    send_daily_summary(result)
    logger.info("✅ Resum diari enviat per Telegram")

    return result


if __name__ == "__main__":
    main()
