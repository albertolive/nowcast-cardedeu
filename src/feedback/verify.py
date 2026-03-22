"""
Verificador de prediccions — compara prediccions passades amb la realitat.
60 minuts després de cada predicció, comprova si realment va ploure.
"""
import logging
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.meteocardedeu import fetch_series
from src.feedback.logger import load_predictions_log, save_predictions_log

logger = logging.getLogger(__name__)


def verify_pending_predictions() -> dict:
    """
    Revisa les prediccions no verificades i comprova si va ploure
    en els 60 minuts següents a cada predicció.

    Retorna un resum: {verified_count, correct_count, wrong_count, skipped}
    """
    entries = load_predictions_log()
    if not entries:
        logger.info("Cap predicció al log per verificar.")
        return {"verified_count": 0, "correct_count": 0, "wrong_count": 0, "skipped": 0}

    now = datetime.now()
    # Obtenim les últimes 3h de dades de l'estació (suficient per verificar)
    try:
        station_df = fetch_series(hours=3)
    except Exception as e:
        logger.error(f"Error obtenint dades de l'estació per verificar: {e}")
        return {"verified_count": 0, "correct_count": 0, "wrong_count": 0, "skipped": 0}

    if station_df.empty or "PREC" not in station_df.columns:
        logger.warning("Dades de l'estació buides o sense PREC.")
        return {"verified_count": 0, "correct_count": 0, "wrong_count": 0, "skipped": 0}

    station_df["datetime"] = station_df["datetime"].dt.tz_localize(None)

    verified_count = 0
    correct_count = 0
    wrong_count = 0
    skipped = 0

    for entry in entries:
        if entry.get("verified"):
            continue

        pred_time = datetime.fromisoformat(entry["timestamp"])
        verification_window_end = pred_time + timedelta(minutes=config.PREDICTION_HORIZON_MIN)

        # Només verificar si ja han passat 60 min + 15 min de marge
        if now < verification_window_end + timedelta(minutes=15):
            skipped += 1
            continue

        # Buscar pluja real en la finestra [pred_time, pred_time + 60min]
        mask = (
            (station_df["datetime"] >= pred_time)
            & (station_df["datetime"] <= verification_window_end)
        )
        window = station_df.loc[mask]

        if window.empty:
            # No tenim dades per aquest període (pot passar si l'estació estava offline)
            skipped += 1
            continue

        # PREC és acumulat en cada minut, sumem per obtenir total de pluja en la finestra
        rain_mm = float(window["PREC"].astype(float).sum())
        actual_rain = rain_mm >= config.RAIN_THRESHOLD_MM

        # Comparar predicció vs realitat
        predicted_rain = entry["will_rain"]
        is_correct = predicted_rain == actual_rain

        entry["verified"] = True
        entry["actual_rain"] = actual_rain
        entry["actual_rain_mm"] = round(rain_mm, 2)
        entry["correct"] = is_correct
        entry["verified_at"] = now.isoformat()

        verified_count += 1
        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        # Log detallat per a cada verificació
        symbol = "✅" if is_correct else "❌"
        logger.info(
            f"  {symbol} {pred_time.strftime('%H:%M')} → "
            f"Predit: {'PLUJA' if predicted_rain else 'SEC'} ({entry['probability_pct']}%) | "
            f"Real: {'PLUJA' if actual_rain else 'SEC'} ({rain_mm:.1f}mm)"
        )

    # Desar les actualitzacions
    save_predictions_log(entries)

    summary = {
        "verified_count": verified_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "skipped": skipped,
    }
    if verified_count > 0:
        summary["accuracy"] = round(correct_count / verified_count * 100, 1)
    logger.info(f"Verificació: {verified_count} noves, {correct_count} correctes, {wrong_count} erronis")
    return summary
