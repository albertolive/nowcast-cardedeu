#!/usr/bin/env python3
"""
Script: Previsió diària millorada del matí.
Executat per GitHub Actions a les 7:00 cada dia.
Fa una predicció, obté el forecast horari d'Open-Meteo, i envia
un resum per franges (matí/tarda/nit) amb probabilitats corregides.
"""
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model.predict import predict_now
from src.data.open_meteo import fetch_forecast
from src.data.ensemble import fetch_ensemble_agreement
from src.notify.telegram import send_daily_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_hourly_outlook(forecast_df: pd.DataFrame) -> list[dict]:
    """
    Construeix un outlook per franges horàries a partir del forecast d'Open-Meteo.
    Franges: Matí (7-13h), Tarda (13-19h), Nit (19-1h), Matinada (1-7h).
    """
    if forecast_df.empty:
        return []

    df = forecast_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour

    now = datetime.now()
    today = now.date()

    # Filtrar per avui + demà matinada
    df = df[df["datetime"].dt.date >= today].head(24)

    if df.empty:
        return []

    # Definir franges
    slots = [
        {"label": "Matí (7-13h)", "hours": range(7, 13)},
        {"label": "Tarda (13-19h)", "hours": range(13, 19)},
        {"label": "Nit (19-1h)", "hours": list(range(19, 24)) + [0]},
    ]

    outlook = []
    for slot in slots:
        slot_df = df[df["hour"].isin(slot["hours"])]
        if slot_df.empty:
            continue

        # Precipitació acumulada a la franja
        precip_cols = [c for c in ["precipitation", "rain"] if c in slot_df.columns]
        precip_mm = float(slot_df[precip_cols[0]].sum()) if precip_cols else 0

        # Probabilitat basada en weather code + precipitació prevista
        rain_hours = 0
        total_hours = len(slot_df)
        if "weather_code" in slot_df.columns:
            rain_hours = int((slot_df["weather_code"] >= 50).sum())

        # Hores amb precipitació > 0.1mm (potser weather_code no ho marca)
        if precip_cols:
            precip_hours = int((slot_df[precip_cols[0]] > 0.1).sum())
            rain_hours = max(rain_hours, precip_hours)

        max_prob = (rain_hours / total_hours * 100) if total_hours > 0 else 0

        # CAPE alt → augmentar probabilitat de tempesta
        if "cape" in slot_df.columns:
            max_cape = float(slot_df["cape"].max())
            if max_cape > 800:
                max_prob = min(100, max_prob + 15)

        # Rang de temperatura
        temp_range = ""
        if "temperature_2m" in slot_df.columns:
            t_min = slot_df["temperature_2m"].min()
            t_max = slot_df["temperature_2m"].max()
            if t_min == t_max:
                temp_range = f"{t_min:.0f}°C"
            else:
                temp_range = f"{t_min:.0f}-{t_max:.0f}°C"

        outlook.append({
            "label": slot["label"],
            "max_prob": max_prob,
            "precip_mm": precip_mm,
            "temp_range": temp_range,
            "rain_hours": rain_hours,
            "total_hours": total_hours,
        })

    return outlook


def _next_rain_text(forecast_df: pd.DataFrame) -> str:
    """
    Busca la propera hora amb pluja prevista en les pròximes 48h.
    Retorna un text descriptiu o None si no es preveu pluja.
    """
    if forecast_df.empty:
        return None

    df = forecast_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    now = datetime.now()

    # Buscar primera hora amb precipitació > 0.1mm o weather_code >= 50
    precip_col = "precipitation" if "precipitation" in df.columns else "rain" if "rain" in df.columns else None
    if precip_col is None:
        return None

    future = df[df["datetime"] > pd.Timestamp(now)]
    rain_mask = future[precip_col] > 0.1
    if "weather_code" in future.columns:
        rain_mask = rain_mask | (future["weather_code"] >= 50)

    rain_hours = future[rain_mask]
    if rain_hours.empty:
        return "Cap pluja prevista en 48h"

    first_rain = rain_hours.iloc[0]["datetime"]
    delta_h = (first_rain - pd.Timestamp(now)).total_seconds() / 3600

    if delta_h < 3:
        return f"Pluja prevista d'aquí {delta_h:.0f}h"
    elif first_rain.date() == now.date():
        hour = first_rain.hour
        if hour < 13:
            return "Pluja prevista aquest matí"
        elif hour < 19:
            return "Pluja prevista aquesta tarda"
        else:
            return "Pluja prevista aquesta nit"
    else:
        hour = first_rain.hour
        if hour < 13:
            return "Propera pluja: demà al matí"
        elif hour < 19:
            return "Propera pluja: demà a la tarda"
        else:
            return "Propera pluja: demà a la nit"


def main():
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model no trobat a {config.MODEL_PATH}")
        sys.exit(1)

    logger.info("📋 Nowcast Cardedeu — Previsió diària millorada")

    try:
        result = predict_now()
    except Exception as e:
        logger.error(f"Error en la predicció: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Probabilitat actual: {result['probability_pct']}% ({result['confidence']})")

    # Obtenir forecast horari per a les properes 48h
    try:
        forecast_df = fetch_forecast(hours_ahead=48)
        hourly_outlook = build_hourly_outlook(forecast_df)
        logger.info(f"Franges horàries: {len(hourly_outlook)}")
        for slot in hourly_outlook:
            logger.info(f"  {slot['label']}: {slot['max_prob']:.0f}% pluja, {slot['precip_mm']:.1f}mm")
    except Exception as e:
        logger.warning(f"Error obtenint forecast horari: {e}")
        forecast_df = pd.DataFrame()
        hourly_outlook = None

    # Propera pluja estimada (48h)
    next_rain = None
    try:
        if not forecast_df.empty:
            next_rain = _next_rain_text(forecast_df)
            if next_rain:
                logger.info(f"  Propera pluja: {next_rain}")
    except Exception as e:
        logger.warning(f"Error calculant propera pluja: {e}")

    send_daily_forecast(result, hourly_outlook, next_rain_text=next_rain)
    logger.info("✅ Previsió diària enviada per Telegram")

    return result


if __name__ == "__main__":
    main()
