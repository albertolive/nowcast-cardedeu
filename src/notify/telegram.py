"""
Notificacions via Telegram.
"""
import logging

import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def _round_prob_5(prob_pct: float) -> int:
    """Arrodoneix la probabilitat al 5% més proper per evitar falsa precisió."""
    return int(5 * round(prob_pct / 5))


def _format_drivers(top_drivers: list) -> list[str]:
    """Formata els 2-3 drivers principals que empugen cap a pluja."""
    if not top_drivers:
        return []
    rain_drivers = [d for d in top_drivers if d.get("direction") == "pluja"
                    and d.get("group") != "Base (climatologia)"]
    if not rain_drivers:
        return []
    parts = [f"{d['icon']} {d['group']}" for d in rain_drivers[:3]]
    return [f"📊 Per què: {' · '.join(parts)}"]


def _format_physical_adjustments(prediction: dict) -> list[str]:
    """Mostra els ajustos de restriccions físiques si s'han aplicat."""
    adjustments = prediction.get("physical_adjustments", [])
    if not adjustments:
        return []
    return [f"⚡ {adj}" for adj in adjustments]


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


def _pressure_trend_arrow(change_3h) -> str:
    """Retorna fletxa de tendència de pressió."""
    if change_3h is None:
        return ""
    if change_3h >= 1.5:
        return " ↑↑"
    elif change_3h >= 0.5:
        return " ↑"
    elif change_3h > -0.5:
        return " →"
    elif change_3h > -1.5:
        return " ↓"
    else:
        return " ↓↓"


def _format_conditions(prediction: dict) -> list[str]:
    """Genera les línies de condicions actuals, radar i sentinella."""
    conditions = prediction.get("conditions", {})
    pressure = conditions.get('pressure', '?')
    pressure_change = prediction.get("pressure_change_3h")
    trend = _pressure_trend_arrow(pressure_change)

    lines = [
        "📡 <b>Condicions actuals:</b>",
        f"  🌡️ Temp: {conditions.get('temperature', '?')}°C",
        f"  💧 Humitat: {conditions.get('humidity', '?')}%",
        f"  📊 Pressió: {pressure} hPa{trend}",
        f"  💨 Vent: {conditions.get('wind_speed', '?')} km/h {conditions.get('wind_dir', '')}",
    ]

    # Radar (només mostrar si hi ha informació rellevant)
    radar = prediction.get("radar", {})
    if radar:
        if radar.get("has_echo"):
            lines.append("")
            lines.append("📡 <b>Radar:</b>")
            lines.append(f"  ⚡ Eco sobre Cardedeu: {radar.get('dbz', 0)} dBZ ({radar.get('rain_rate_mmh', 0)} mm/h)")
        else:
            nearest_km = radar.get("nearest_echo_km")
            compass = radar.get("nearest_echo_compass")
            if nearest_km is not None and nearest_km < 25:
                lines.append("")
                lines.append("📡 <b>Radar:</b>")
                lines.append(f"  Eco més proper: {nearest_km} km {compass}")
                eta = radar.get("storm_eta_min")
                if eta is not None:
                    lines.append(f"  ⏱️ ETA estimat: ~{eta} min")
                velocity = radar.get("storm_velocity_kmh", 0)
                if velocity > 5:
                    lines.append(f"  Velocitat: {velocity} km/h")
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
    prob = _round_prob_5(prediction["probability_pct"])
    lines = [
        "🌧️ <b>Nowcast Cardedeu</b>",
        "",
        "⚠️ <b>ALERTA: Pluja imminent en els propers 60 min!</b>",
        "",
        f"🎯 Probabilitat: <b>{prob}%</b>",
        f"📊 Confiança: <b>{prediction['confidence']}</b>",
    ]
    # Drivers: per què el model prediu pluja
    driver_lines = _format_drivers(prediction.get("top_drivers", []))
    if driver_lines:
        lines.extend(driver_lines)
    # Ajustos físics (radar/sentinella override)
    adj_lines = _format_physical_adjustments(prediction)
    if adj_lines:
        lines.extend(adj_lines)
    lines.append("")
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def format_rain_clearing(prediction: dict) -> str:
    """Missatge quan la pluja s'allunya (rain_alert → clear)."""
    prob = _round_prob_5(prediction["probability_pct"])
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
    prob = _round_prob_5(prediction["probability_pct"])
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
    prob = _round_prob_5(prediction["probability_pct"])
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

    # Drivers: per què el model prediu pluja
    driver_lines = _format_drivers(prediction.get("top_drivers", []))
    if driver_lines:
        lines.extend(driver_lines)
    # Ajustos físics
    adj_lines = _format_physical_adjustments(prediction)
    if adj_lines:
        lines.extend(adj_lines)

    lines.append("")
    lines.extend(_format_conditions(prediction))
    lines.append("")
    lines.append(f"⏰ {prediction['timestamp'][:19]}")
    return "\n".join(lines)


def _format_timestamp(iso_ts: str) -> str:
    """Converteix ISO timestamp a format llegible en català."""
    from datetime import datetime
    months = ['gen', 'feb', 'mar', 'abr', 'mai', 'jun',
              'jul', 'ago', 'set', 'oct', 'nov', 'des']
    try:
        dt = datetime.fromisoformat(iso_ts[:19])
        return f"{dt.day} {months[dt.month-1]} {dt.year}, {dt.hour:02d}:{dt.minute:02d}"
    except Exception:
        return iso_ts[:19]


def _dir_to_compass(deg) -> str:
    """Converteix graus a punt cardinal abreujat (8 punts)."""
    if deg is None:
        return "?"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def _format_radar_summary(radar: dict) -> str:
    """Resum intel·ligent del radar per la previsió diària."""
    if not radar:
        return "Radar net"

    if radar.get("has_echo"):
        dbz = radar.get("dbz", 0)
        rate = radar.get("rain_rate_mmh", 0)
        line = f"Eco actiu: {dbz:.0f} dBZ ({rate:.1f} mm/h)"
        if radar.get("approaching"):
            eta = radar.get("storm_eta_min")
            if eta:
                line += f" — ETA ~{eta} min"
        return line

    nearest = radar.get("nearest_echo_km", config.RADAR_SCAN_RADIUS_KM)
    coverage = radar.get("coverage_20km", 0)

    if nearest < 10 and coverage > 0.05:
        compass = radar.get("nearest_echo_compass", "")
        line = f"Eco a {nearest:.0f} km {compass}"
        if coverage > 0.1:
            line += f" · {coverage:.0%} cobertura"
        if radar.get("approaching"):
            eta = radar.get("storm_eta_min")
            if eta:
                line += f" — ETA ~{eta} min"
        return line

    return "Radar net"


def format_daily_forecast(prediction: dict, hourly_outlook: list[dict] = None,
                          next_rain_text: str = None, ai_narrative: str = None) -> str:
    """
    Resum diari millorat amb previsió hora per hora (matí/tarda/nit)
    corregida pel model ML de Cardedeu.
    Dissenyat per doble audiència: públic general (part superior) i
    entusiastes de la meteorologia (secció tècnica).
    """
    prob = _round_prob_5(prediction["probability_pct"])
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
    ]

    # Només mostrar règim si és informatiu (no "Variable")
    if regime_text != "Variable":
        lines.append(f"🌬️ {regime_text}")

    # Narrativa IA (generada per OpenRouter, opcional)
    if ai_narrative:
        lines.append("")
        lines.append(f"💬 <i>{ai_narrative}</i>")

    lines.append("")

    # ── Previsió per franges (públic general) ──
    if hourly_outlook:
        lines.append("📅 <b>Previsió per franges:</b>")
        for slot in hourly_outlook:
            icon = _rain_icon(slot.get("max_prob", 0))
            label = slot["label"]
            max_prob = _round_prob_5(slot.get("max_prob", 0))
            temp_range = slot.get("temp_range", "")

            line = f"  {icon} <b>{label}</b>: {max_prob}% pluja"
            if temp_range:
                line += f" · {temp_range}"
            lines.append(line)
        lines.append("")

    # Propera pluja estimada (48h)
    if next_rain_text:
        lines.append(f"🔭 {next_rain_text}")
        lines.append("")

    # ── Condicions actuals (compacte) ──
    conditions = prediction.get("conditions", {})
    pressure = conditions.get("pressure", "?")
    pressure_change = prediction.get("pressure_change_3h")
    trend = _pressure_trend_arrow(pressure_change)

    fv = prediction.get("feature_vector", {})
    dew_point = fv.get("dew_point")
    cloud_cover = fv.get("cloud_cover")

    lines.append("📡 <b>Condicions actuals:</b>")

    temp_line = f"  🌡️ {conditions.get('temperature', '?')}°C · 💧 {conditions.get('humidity', '?')}%"
    if dew_point is not None:
        temp_line += f" · Rosada {dew_point:.1f}°C"
    lines.append(temp_line)

    pres_line = f"  📊 {pressure} hPa"
    if pressure_change is not None:
        sign = "+" if pressure_change >= 0 else ""
        pres_line += f"{trend}({sign}{pressure_change:.1f}/3h)"
    lines.append(pres_line)

    wind_line = f"  💨 {conditions.get('wind_speed', '?')} km/h {conditions.get('wind_dir', '')}"
    if cloud_cover is not None:
        wind_line += f" · ☁️ {cloud_cover:.0f}%"
    lines.append(wind_line)

    lines.append("")

    # ── Detall tècnic (entusiastes meteo) ──
    lines.append("🔬 <b>Detall tècnic:</b>")

    # Ensemble
    ensemble = prediction.get("ensemble", {})
    models_rain = ensemble.get("models_rain")
    total_models = ensemble.get("total_models", 4)
    if models_rain is not None:
        lines.append(f"  🔮 Ensemble: {models_rain}/{total_models} models pluja")

    # 850hPa wind + temperature
    if pressure_levels:
        wind_dir_850 = pressure_levels.get("wind_850_dir")
        wind_spd_850 = pressure_levels.get("wind_850_speed_kmh")
        temp_850 = pressure_levels.get("temp_850")
        rh_850 = pressure_levels.get("rh_850")
        rh_700 = pressure_levels.get("rh_700")
        if wind_dir_850 is not None and wind_spd_850 is not None:
            compass_850 = _dir_to_compass(wind_dir_850)
            pl_line = f"  🌬️ 850hPa: {compass_850} {wind_spd_850:.0f} km/h"
            if temp_850 is not None:
                pl_line += f" · T850 {temp_850:.1f}°C"
            if rh_850 is not None and rh_700 is not None:
                pl_line += f" · RH {rh_850:.0f}/{rh_700:.0f}%"
            lines.append(pl_line)

        # Instability indices
        tt = pressure_levels.get("tt_index")
        li = pressure_levels.get("li_index")
        vt = pressure_levels.get("vt_index")
        if tt is not None:
            idx_parts = [f"TT {tt:.1f}"]
            if li is not None:
                sign = "+" if li >= 0 else ""
                idx_parts.append(f"LI {sign}{li:.1f}")
            if vt is not None:
                idx_parts.append(f"VT {vt:.1f}")
            lines.append(f"  📊 {' · '.join(idx_parts)}")

    # Radar summary
    radar = prediction.get("radar", {})
    radar_line = _format_radar_summary(radar)
    lines.append(f"  📡 {radar_line}")

    lines.append("")
    lines.append(f"⏰ {_format_timestamp(prediction['timestamp'])}")
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


def send_daily_forecast(prediction: dict, hourly_outlook: list[dict] = None,
                        next_rain_text: str = None, ai_narrative: str = None) -> bool:
    """Envia la previsió diària millorada."""
    return send_telegram_message(format_daily_forecast(prediction, hourly_outlook, next_rain_text, ai_narrative))


def send_regime_change(prediction: dict, regime_change: dict) -> bool:
    """Envia alerta de canvi de règim atmosfèric."""
    return send_telegram_message(format_regime_change(prediction, regime_change))


def send_prediction_alert(prediction: dict) -> bool:
    """Compat: envia alerta genèrica."""
    return send_telegram_message(format_rain_incoming(prediction))
