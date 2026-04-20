"""
Verificador de prediccions — compara prediccions passades amb la realitat.
60 minuts després de cada predicció, comprova si realment va ploure.
Usa MeteoCardedeu.net com a font primària i XEMA KX (La Roca - ETAP Cardedeu) com a fallback.
"""
import logging
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.meteocardedeu import fetch_series
from src.data.meteocat import fetch_kx_precipitation_series
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
    # Cover at least the oldest pending entry's verification window so a short
    # upstream outage doesn't permanently strand old predictions as "Pendent".
    oldest_pending = min(
        (datetime.fromisoformat(e["timestamp"]) for e in entries if not e.get("verified")),
        default=now,
    )
    needed_hours = max(3, int((now - oldest_pending).total_seconds() // 3600) + 2)
    lookup_hours = min(needed_hours, 48)
    # Font primària: MeteoCardedeu.net (minut a minut)
    # Fallback: XEMA KX La Roca - ETAP Cardedeu (cada 30min, professional SMC)
    verification_source = "meteocardedeu"
    mc_error = None
    try:
        station_df = fetch_series(hours=lookup_hours)
    except Exception as e:
        logger.warning(f"MeteoCardedeu no disponible per verificar: {e}")
        station_df = None
        mc_error = str(e)[:200]

    if station_df is None or station_df.empty or "PREC" not in station_df.columns:
        logger.info("MeteoCardedeu sense dades — provant fallback XEMA KX (La Roca)...")
        station_df = fetch_kx_precipitation_series(hours=lookup_hours)
        verification_source = "xema_kx"
        if station_df.empty or "PREC" not in station_df.columns:
            logger.warning("Ni MeteoCardedeu ni XEMA KX disponibles per verificar.")
            return {
                "verified_count": 0, "correct_count": 0, "wrong_count": 0, "skipped": 0,
                "source": "none",
                "lookup_hours": lookup_hours,
                "station_rows": 0,
                "mc_error": mc_error,
                "pending_total": sum(1 for e in entries if not e.get("verified")),
            }
        logger.info(f"Usant XEMA KX (La Roca) com a font de verificació ({len(station_df)} lectures)")

    # Assegurar que datetime no té timezone (KX ja arriba sense, MC.net pot tenir-ne)
    if station_df["datetime"].dt.tz is not None:
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

        # PREC és l'increment de pluja per lectura (convertit a fetch_series).
        # Per XEMA KX, PREC ja és l'acumulat per interval de 30min.
        # En ambdós casos, .sum() dona el total de pluja real en la finestra.
        rain_mm = float(window["PREC"].astype(float).sum())
        actual_rain = rain_mm >= config.RAIN_THRESHOLD_MM

        # Comparar predicció vs realitat
        # Verificació justa: la zona incerta (30-65%) no es puntua com encert/error
        # Slim log entries (pushed daily to git) lack `probability` and `will_rain`;
        # derive them from `probability_pct` / `rain_category` so verify still runs.
        prob = entry.get("probability")
        if prob is None and entry.get("probability_pct") is not None:
            prob = entry["probability_pct"] / 100.0
        rain_category = entry.get("rain_category")
        if rain_category is None:
            if prob is None:
                skipped += 1
                continue
            if prob >= config.DISPLAY_THRESHOLD_RAIN:
                rain_category = "probable"
            elif prob >= config.DISPLAY_THRESHOLD_UNCERTAIN:
                rain_category = "incert"
            else:
                rain_category = "sec"
            entry["rain_category"] = rain_category
        is_uncertain = rain_category == "incert"

        if is_uncertain:
            # Zona incerta: registrem el resultat però no comptem com a encert/error
            is_correct = None  # ni encert ni error
        else:
            # Sec (<30%) o probable (>65%): verificació binària justa
            display_predicted_rain = rain_category == "probable"
            is_correct = display_predicted_rain == actual_rain

        # Brier score component: (probabilitat - resultat real)²
        brier_component = ((prob if prob is not None else 0.0) - (1.0 if actual_rain else 0.0)) ** 2

        entry["verified"] = True
        entry["actual_rain"] = actual_rain
        entry["actual_rain_mm"] = round(rain_mm, 2)
        entry["correct"] = is_correct
        entry["uncertain"] = is_uncertain
        entry["brier_component"] = round(brier_component, 6)
        entry["verified_at"] = now.isoformat()
        entry["verification_source"] = verification_source

        verified_count += 1
        if is_correct is True:
            correct_count += 1
        elif is_correct is False:
            wrong_count += 1
        # is_correct is None (uncertain): don't count either way

        # Log detallat per a cada verificació
        if is_uncertain:
            symbol = "🔸"
            label = "INCERT"
        elif is_correct:
            symbol = "✅"
            label = "PLUJA" if entry.get("rain_category") == "probable" else "SEC"
        else:
            symbol = "❌"
            label = "PLUJA" if entry.get("rain_category") == "probable" else "SEC"
        pct = entry.get("probability_pct", round((prob or 0.0) * 100, 1))
        logger.info(
            f"  {symbol} {pred_time.strftime('%H:%M')} → "
            f"Predit: {label} ({pct}%) | "
            f"Real: {'PLUJA' if actual_rain else 'SEC'} ({rain_mm:.1f}mm)"
        )

    # Desar les actualitzacions
    save_predictions_log(entries)

    summary = {
        "verified_count": verified_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "skipped": skipped,
        "source": verification_source,
        "lookup_hours": lookup_hours,
        "station_rows": int(len(station_df)),
        "station_first_ts": station_df["datetime"].iloc[0].isoformat() if len(station_df) else None,
        "station_last_ts": station_df["datetime"].iloc[-1].isoformat() if len(station_df) else None,
        "pending_total": sum(1 for e in entries if not e.get("verified")),
    }
    if verified_count > 0:
        summary["accuracy"] = round(correct_count / verified_count * 100, 1)
    logger.info(f"Verificació: {verified_count} noves, {correct_count} correctes, {wrong_count} erronis")
    return summary
