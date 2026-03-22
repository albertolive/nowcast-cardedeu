#!/usr/bin/env python3
"""
Script: Informe setmanal d'accuracy.
Executat per GitHub Actions cada dilluns a les 8:00.
Calcula mètriques de rendiment i envia report per Telegram.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feedback.accuracy import compute_accuracy, format_accuracy_report
from src.notify.telegram import send_telegram_message
from src.ai.enricher import generate_accuracy_narrative

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("📊 Nowcast Cardedeu — Informe setmanal d'accuracy")

        # Mètriques dels últims 7 dies
        metrics_week = compute_accuracy(days=7)
        logger.info(f"Últims 7 dies: {metrics_week.get('verified', 0)} verificades, "
                    f"accuracy={metrics_week.get('accuracy', '?')}%")

        # Mètriques totals
        metrics_all = compute_accuracy()
        logger.info(f"Total: {metrics_all.get('verified', 0)} verificades, "
                    f"accuracy={metrics_all.get('accuracy', '?')}%")

        # Enviar report per Telegram
        report = format_accuracy_report(metrics_week)
        if metrics_all.get("verified", 0) > metrics_week.get("verified", 0):
            report += (
                f"\n\n📈 <b>Total acumulat:</b> "
                f"{metrics_all['accuracy']}% accuracy "
                f"({metrics_all['verified']} prediccions)"
            )

        # Narrativa IA (opcional, 1 crida/setmana, fallback graciós)
        try:
            ai_narrative = generate_accuracy_narrative(metrics_week, metrics_all)
            if ai_narrative:
                report += f"\n\n💬 <i>{ai_narrative}</i>"
                logger.info(f"Narrativa IA afegida a l'informe")
        except Exception as e:
            logger.warning(f"Error generant narrativa IA (no bloquejant): {e}")

        send_telegram_message(report)
        logger.info("✅ Informe enviat per Telegram")

    except Exception as e:
        logger.error(f"Error generant informe d'accuracy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
