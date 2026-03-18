"""
Gestió de l'estat de notificacions.
Implementa lògica de transicions d'estat amb histèresi i cooldown
per evitar spam de notificacions.

Estats possibles:
  - clear: No es preveu pluja
  - rain_alert: S'ha enviat alerta de pluja
"""
import json
import logging
import os
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(config.PROJECT_ROOT, "data", "notification_state.json")

DEFAULT_STATE = {
    "current_state": "clear",        # clear | rain_alert
    "last_alert_time": 0,            # Unix timestamp
    "last_alert_type": None,         # rain_incoming | rain_clearing | daily_summary
    "last_probability": 0.0,
    "consecutive_high": 0,           # Quantes prediccions seguides > threshold_up
    "consecutive_low": 0,            # Quantes prediccions seguides < threshold_down
}


def load_state() -> dict:
    """Carrega l'estat de notificacions des del fitxer."""
    if not os.path.exists(STATE_FILE):
        return DEFAULT_STATE.copy()
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        # Assegurar que tots els camps existeixen
        for key, default in DEFAULT_STATE.items():
            if key not in state:
                state[key] = default
        return state
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error llegint estat, reinicialitzant: {e}")
        return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    """Desa l'estat de notificacions."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def should_notify(probability: float, state: dict) -> str | None:
    """
    Determina si cal enviar una notificació basant-se en la probabilitat
    actual i l'estat anterior. Retorna el tipus de notificació o None.

    Lògica de transicions amb histèresi:
      clear → rain_alert:  quan probability > THRESHOLD_UP (65%)
      rain_alert → clear:  quan probability < THRESHOLD_DOWN (30%)

    El gap entre 30% i 65% evita flip-flopping.
    Cooldown de 30 min entre alertes del mateix tipus.
    """
    now = time.time()
    current_state = state.get("current_state", "clear")
    last_alert_time = state.get("last_alert_time", 0)
    cooldown_seconds = config.NOTIFICATION_COOLDOWN_MIN * 60

    # Cooldown: no notificar si l'última alerta és massa recent
    time_since_last = now - last_alert_time
    if time_since_last < cooldown_seconds:
        logger.info(
            f"Cooldown actiu ({int(time_since_last)}s / {cooldown_seconds}s). "
            f"No es notifica."
        )
        return None

    # Transició: clear → rain_alert
    if current_state == "clear" and probability >= config.ALERT_THRESHOLD_UP:
        return "rain_incoming"

    # Transició: rain_alert → clear
    if current_state == "rain_alert" and probability <= config.ALERT_THRESHOLD_DOWN:
        return "rain_clearing"

    return None


def update_state(state: dict, notification_type: str, probability: float) -> dict:
    """Actualitza l'estat després d'enviar una notificació."""
    now = time.time()
    state["last_alert_time"] = now
    state["last_alert_type"] = notification_type
    state["last_probability"] = probability

    if notification_type == "rain_incoming":
        state["current_state"] = "rain_alert"
    elif notification_type == "rain_clearing":
        state["current_state"] = "clear"
    elif notification_type == "daily_summary":
        pass  # No canvia l'estat base

    save_state(state)
    return state
