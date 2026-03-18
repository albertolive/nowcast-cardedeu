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


def _format_conditions(prediction: dict) -> list[str]:
    """Genera les línies de condicions actuals, radar i sentinella."""
    conditions = prediction.get("conditions", {})
    lines = [
        "📡 <b>Condicions actuals:</b>",
        f"  🌡️ Temp: {conditions.get('temperature', '?')}°C",
        f"  💧 Humitat: {conditions.get('humidity', '?')}%",
        f"  📊 Pressió: {conditions.get('pressure', '?')} hPa",
        f"  💨 Vent: {conditions.get('wind_speed', '?')} km/h {conditions.get('wind_dir', '')}",
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

    return lines


def format_rain_incoming(prediction: dict) -> str:
    """Missatge quan la pluja s'acosta (clear → rain_alert)."""
    prob = prediction["probability_pct"]
    lines = [
        "🌧️ <b>Nowcast Cardedeu</b>",
        "",
        "⚠️ <b>ALERTA: Pluja imminent en els propers 60 min!</b>",
        "",
        f"🎯 Probabilitat: <b>{prob}%</b>",
        f"📊 Confiança: <b>{prediction['confidence']}</b>",
        "",
    ]
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def format_rain_clearing(prediction: dict) -> str:
    """Missatge quan la pluja s'allunya (rain_alert → clear)."""
    prob = prediction["probability_pct"]
    lines = [
        "☀️ <b>Nowcast Cardedeu</b>",
        "",
        "✅ <b>La pluja s'allunya!</b>",
        f"Probabilitat tornada a <b>{prob}%</b>",
        "",
    ]
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def format_daily_summary(prediction: dict) -> str:
    """Resum diari al matí."""
    prob = prediction["probability_pct"]
    confidence = prediction["confidence"]

    if prob >= 65:
        outlook = "🌧️ Pluja probable avui"
    elif prob >= 40:
        outlook = "🌥️ Possibilitat de pluja"
    else:
        outlook = "☀️ No es preveu pluja"

    lines = [
        "📋 <b>Nowcast Cardedeu — Resum del matí</b>",
        "",
        f"{outlook}",
        f"🎯 Probabilitat: <b>{prob}%</b> ({confidence})",
        "",
    ]
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def format_rain_alert(prediction: dict) -> str:
    """Compat: format antic per alertes simples."""
    return format_rain_incoming(prediction)


def send_rain_incoming(prediction: dict) -> bool:
    """Envia alerta de pluja imminent."""
    return send_telegram_message(format_rain_incoming(prediction))


def send_rain_clearing(prediction: dict) -> bool:
    """Envia avís de pluja que s'allunya."""
    return send_telegram_message(format_rain_clearing(prediction))


def send_daily_summary(prediction: dict) -> bool:
    """Envia el resum diari del matí."""
    return send_telegram_message(format_daily_summary(prediction))


def send_prediction_alert(prediction: dict) -> bool:
    """Compat: envia alerta genèrica."""
    return send_telegram_message(format_rain_incoming(prediction))
