"""
Notificacions via Telegram.
"""
import logging

import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def send_telegram_message(text: str) -> bool:
    """Envia un missatge via Telegram bot."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurat (falten TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID)")
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Missatge Telegram enviat correctament")
        return True
    except Exception as e:
        logger.error(f"Error enviant missatge Telegram: {e}")
        return False


def format_rain_alert(prediction: dict) -> str:
    """Formata el missatge d'alerta de pluja per Telegram."""
    prob = prediction["probability_pct"]
    confidence = prediction["confidence"]
    conditions = prediction["conditions"]

    emoji = "🌧️" if prediction["will_rain"] else "⛅"

    lines = [
        f"{emoji} <b>Nowcast Cardedeu</b>",
        "",
        f"🎯 Probabilitat de pluja: <b>{prob}%</b>",
        f"📊 Confiança: <b>{confidence}</b>",
        "",
        "📡 <b>Condicions actuals:</b>",
        f"  🌡️ Temp: {conditions.get('temperature', '?')}°C",
        f"  💧 Humitat: {conditions.get('humidity', '?')}%",
        f"  📊 Pressió: {conditions.get('pressure', '?')} hPa",
        f"  💨 Vent: {conditions.get('wind_speed', '?')} km/h {conditions.get('wind_dir', '')}",
        f"  ☀️ Radiació: {conditions.get('solar_radiation', '?')} W/m²",
    ]

    # Radar
    radar = prediction.get("radar", {})
    if radar:
        lines.append("")
        lines.append("📡 <b>Radar:</b>")
        lines.append(f"  Eco: {'SÍ' if radar.get('has_echo') else 'NO'} ({radar.get('dbz', 0)} dBZ)")
        if radar.get("rain_rate_mmh", 0) > 0:
            lines.append(f"  Intensitat: {radar['rain_rate_mmh']} mm/h")
        if radar.get("approaching"):
            lines.append("  ⚡ Pluja acostant-se!")

    # Sentinella
    sentinel = prediction.get("sentinel", {})
    if sentinel and sentinel.get("raining"):
        lines.append("")
        lines.append(f"🔭 <b>{sentinel.get('station', 'Sentinella')}:</b> Plovent!")

    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")

    if prediction["will_rain"]:
        lines.insert(2, "⚠️ <b>ALERTA: Pluja imminent en els propers 60 min!</b>")

    return "\n".join(lines)


def send_prediction_alert(prediction: dict) -> bool:
    """
    Envia l'alerta de predicció si supera el llindar,
    o un resum si s'ha demanat explícitament.
    """
    message = format_rain_alert(prediction)
    return send_telegram_message(message)
