#!/usr/bin/env python3
"""
Script: Previsió diària millorada del matí.
Executat per GitHub Actions a les 7:00 cada dia.
Fa una predicció actual, i executa el model ML sobre les pròximes 48h
de forecast per generar franges (matí/tarda/nit) amb probabilitats locals.
"""
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now, predict_hourly_forecast
from src.data.open_meteo import fetch_forecast
from src.notify.telegram import send_daily_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_hourly_outlook(
    ml_hourly: list[dict],
    forecast_df: pd.DataFrame,
) -> list[dict]:
    """
    Construeix un outlook per franges horàries combinant:
    - Probabilitats del model ML local (entrenat amb dades de meteocardedeu)
    - Temperatures del forecast Open-Meteo (el model no prediu temperatura)

    Franges: Matí (7-13h), Tarda (13-19h), Nit (19-1h).
    """
    if not ml_hourly:
        return []

    # Convertir a DataFrame per facilitar l'agregació
    ml_df = pd.DataFrame(ml_hourly)
    ml_df["datetime"] = pd.to_datetime(ml_df["datetime"])
    ml_df["hour"] = ml_df["datetime"].dt.hour

    now = datetime.now()
    today = now.date()
    today_ts = pd.Timestamp(today)

    # Filtrar per avui en endavant
    ml_df = ml_df[ml_df["datetime"] >= today_ts]

    # Merge amb temperatures del forecast
    if not forecast_df.empty:
        fdf = forecast_df[["datetime", "temperature_2m"]].copy()
        fdf["datetime"] = pd.to_datetime(fdf["datetime"])
        ml_df = pd.merge_asof(
            ml_df.sort_values("datetime"),
            fdf.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )

    if ml_df.empty:
        return []

    # Definir franges amb datetime explícit per evitar barrejar
    # temperatures de demà al matí amb avui
    slots = [
        {"label": "Matí (7-13h)", "start": today_ts + pd.Timedelta(hours=7), "end": today_ts + pd.Timedelta(hours=13)},
        {"label": "Tarda (13-19h)", "start": today_ts + pd.Timedelta(hours=13), "end": today_ts + pd.Timedelta(hours=19)},
        {"label": "Nit (19-1h)", "start": today_ts + pd.Timedelta(hours=19), "end": today_ts + pd.Timedelta(hours=25)},
    ]

    outlook = []
    for slot in slots:
        slot_df = ml_df[
            (ml_df["datetime"] >= slot["start"]) &
            (ml_df["datetime"] < slot["end"])
        ]
        if slot_df.empty:
            continue

        # Probabilitat ML: màxima de la franja (si algun hora diu pluja, avisa)
        max_prob = float(slot_df["probability"].max()) * 100
        rain_hours = int(slot_df["will_rain"].sum())
        total_hours = len(slot_df)

        # Rang de temperatura
        temp_range = ""
        if "temperature_2m" in slot_df.columns:
            temps = slot_df["temperature_2m"].dropna()
            if not temps.empty:
                t_min = temps.min()
                t_max = temps.max()
                if t_min == t_max:
                    temp_range = f"{t_min:.0f}°C"
                else:
                    temp_range = f"{t_min:.0f}-{t_max:.0f}°C"

        outlook.append({
            "label": slot["label"],
            "max_prob": max_prob,
            "temp_range": temp_range,
            "rain_hours": rain_hours,
            "total_hours": total_hours,
        })

    return outlook


def _next_rain_text(ml_hourly: list[dict]) -> str:
    """
    Busca la propera hora on el model ML prediu pluja.
    Retorna un text descriptiu o None.
    """
    if not ml_hourly:
        return None

    now = datetime.now()

    for h in ml_hourly:
        dt = pd.Timestamp(h["datetime"])
        if dt <= pd.Timestamp(now):
            continue
        if h["will_rain"]:
            delta_h = (dt - pd.Timestamp(now)).total_seconds() / 3600
            if delta_h < 3:
                return f"Pluja prevista d'aquí {delta_h:.0f}h"
            elif dt.date() == now.date():
                hour = dt.hour
                if hour < 13:
                    return "Pluja prevista aquest matí"
                elif hour < 19:
                    return "Pluja prevista aquesta tarda"
                else:
                    return "Pluja prevista aquesta nit"
            else:
                hour = dt.hour
                if hour < 13:
                    return "Propera pluja: demà al matí"
                elif hour < 19:
                    return "Propera pluja: demà a la tarda"
                else:
                    return "Propera pluja: demà a la nit"

    return "Cap pluja prevista en 48h"


def main():
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model no trobat a {config.MODEL_PATH}")
        sys.exit(1)

    logger.info("📋 Nowcast Cardedeu — Previsió diària amb ML")

    # Predicció actual (nowcast amb totes les dades: radar, sentinella, etc.)
    try:
        result = predict_now()
    except Exception as e:
        logger.error(f"Error en la predicció: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Probabilitat actual: {result['probability_pct']}% ({result['confidence']})")

    # Forecast ML horari (model entrenat amb dades locals sobre forecast Open-Meteo)
    ml_hourly = []
    try:
        ml_hourly = predict_hourly_forecast(hours_ahead=48)
        logger.info(f"ML forecast: {len(ml_hourly)} hores generades")
    except Exception as e:
        logger.warning(f"Error generant forecast ML: {e}")

    # Temperatures del forecast (el model no prediu temperatura)
    forecast_df = pd.DataFrame()
    try:
        forecast_df = fetch_forecast(hours_ahead=48)
    except Exception as e:
        logger.warning(f"Error obtenint forecast: {e}")

    # Construir franges amb probabilitats ML + temperatures
    hourly_outlook = build_hourly_outlook(ml_hourly, forecast_df)
    for slot in hourly_outlook:
        logger.info(f"  {slot['label']}: {slot['max_prob']:.0f}% pluja ML")

    # Propera pluja (basada en ML, no en forecast cru)
    next_rain = _next_rain_text(ml_hourly) if ml_hourly else None
    if next_rain:
        logger.info(f"  {next_rain}")

    send_daily_forecast(result, hourly_outlook, next_rain_text=next_rain)
    logger.info("✅ Previsió diària enviada per Telegram")

    return result

    return result


if __name__ == "__main__":
    main()
