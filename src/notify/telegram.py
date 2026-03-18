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

    # Radar (puntual + espacial)
    radar = prediction.get("radar", {})
    if radar:
        lines.append("")
        lines.append("📡 <b>Radar:</b>")
        if radar.get("has_echo"):
            lines.append(f"  ⚡ Eco sobre Cardedeu: {radar.get('dbz', 0)} dBZ ({radar.get('rain_rate_mmh', 0)} mm/h)")
        else:
            # Mostrar ecos propers si existeixen
            nearest_km = radar.get("nearest_echo_km")
            compass = radar.get("nearest_echo_compass")
            if nearest_km is not None and nearest_km < 30:
                lines.append(f"  Eco més proper: {nearest_km} km {compass}")
                eta = radar.get("storm_eta_min")
                if eta is not None:
                    lines.append(f"  ⏱️ ETA estimat: ~{eta} min")
                velocity = radar.get("storm_velocity_kmh", 0)
                if velocity > 5:
                    lines.append(f"  Velocitat: {velocity} km/h")
            else:
                lines.append("  Sense ecos en 30 km")

        coverage = radar.get("coverage_20km", 0)
        if coverage > 0:
            lines.append(f"  Cobertura 20 km: {coverage:.0%}")

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


def format_regime_change(prediction: dict, regime_change: dict) -> str:
    """Missatge d'alerta de canvi de règim atmosfèric."""
    severity_icon = "⚠️" if regime_change["severity"] == "warning" else "👁️"
    prob = prediction["probability_pct"]
    ensemble = prediction.get("ensemble", {})

    lines = [
        f"🌦️ <b>Nowcast Cardedeu — Canvi de règim</b>",
        "",
        f"{severity_icon} <b>{regime_change['title']}</b>",
        "",
        regime_change["description"],
        "",
        f"🎯 Probabilitat pluja 60 min: <b>{prob}%</b>",
    ]

    # Models
    models_rain = ensemble.get("models_rain", 0)
    total_models = ensemble.get("total_models", 4)
    if models_rain is not None:
        lines.append(f"🔮 Models: {models_rain}/{total_models} prediuen pluja")

    lines.append("")
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def format_daily_forecast(prediction: dict, hourly_outlook: list[dict] = None) -> str:
    """
    Resum diari millorat amb previsió hora per hora (matí/tarda/nit)
    corregida pel model ML de Cardedeu.
    """
    prob = prediction["probability_pct"]
    confidence = prediction["confidence"]

    if prob >= 65:
        outlook = "🌧️ Pluja probable avui"
    elif prob >= 40:
        outlook = "🌥️ Possibilitat de pluja"
    else:
        outlook = "☀️ No es preveu pluja"

    # Règim eòlic actual
    wind_regime = prediction.get("wind_regime", {})
    pressure_levels = prediction.get("pressure_levels", {})
    regime_text = _format_regime_text(wind_regime, pressure_levels)

    lines = [
        "📋 <b>Nowcast Cardedeu — Previsió del dia</b>",
        "",
        f"{outlook}",
        f"🎯 Probabilitat ara: <b>{prob}%</b> ({confidence})",
        f"🌬️ Règim: {regime_text}",
        "",
    ]

    # Previsió per franges horàries
    if hourly_outlook:
        lines.append("📅 <b>Previsió per franges:</b>")
        for slot in hourly_outlook:
            icon = _rain_icon(slot.get("max_prob", 0))
            label = slot["label"]
            max_prob = slot.get("max_prob", 0)
            precip = slot.get("precip_mm", 0)
            temp_range = slot.get("temp_range", "")

            line = f"  {icon} <b>{label}</b>: {max_prob:.0f}% pluja"
            if precip > 0.1:
                line += f" ({precip:.1f} mm)"
            if temp_range:
                line += f" · {temp_range}"
            lines.append(line)
        lines.append("")

    # Ensemble models
    ensemble = prediction.get("ensemble", {})
    models_rain = ensemble.get("models_rain", 0)
    total_models = ensemble.get("total_models", 4)
    if models_rain is not None:
        lines.append(f"🔮 Models: {models_rain}/{total_models} prediuen pluja")

    lines.append("")
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def _format_regime_text(wind_regime: dict, pressure_levels: dict) -> str:
    """Text curt del règim eòlic actual."""
    if wind_regime.get("is_llevantada"):
        speed = pressure_levels.get("wind_850_speed_kmh", "?")
        return f"🌊 Llevantada ({speed} km/h a 850hPa)"
    elif wind_regime.get("is_garbi"):
        return "🌀 Garbí (SW)"
    elif wind_regime.get("is_tramuntana"):
        return "❄️ Tramuntana (N)"
    elif wind_regime.get("is_migjorn"):
        return "🌡️ Migjorn (S)"
    elif wind_regime.get("is_ponent"):
        return "🏔️ Ponent (W/NW)"
    else:
        return "Variable"


def _rain_icon(prob: float) -> str:
    if prob >= 65:
        return "🌧️"
    elif prob >= 40:
        return "🌥️"
    elif prob >= 20:
        return "⛅"
    else:
        return "☀️"


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
    """Envia el resum diari del matí (format bàsic)."""
    return send_telegram_message(format_daily_summary(prediction))


def send_daily_forecast(prediction: dict, hourly_outlook: list[dict] = None) -> bool:
    """Envia la previsió diària millorada."""
    return send_telegram_message(format_daily_forecast(prediction, hourly_outlook))


def send_regime_change(prediction: dict, regime_change: dict) -> bool:
    """Envia alerta de canvi de règim atmosfèric."""
    return send_telegram_message(format_regime_change(prediction, regime_change))


def send_prediction_alert(prediction: dict) -> bool:
    """Compat: envia alerta genèrica."""
    return send_telegram_message(format_rain_incoming(prediction))
