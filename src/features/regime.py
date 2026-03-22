"""
Detecció de canvis de règim atmosfèric per a Cardedeu.

Analitza les condicions actuals (vent, pressió, humitat, inestabilitat)
i detecta transicions cap a configuracions que històricament produeixen
pluja a Cardedeu (Vallès Oriental).

Règims d'alerta:
  - Llevantada humida (E/SE + HR alta): patró #1 de pluja a Cardedeu
  - Garbí inestable (SW + CAPE/TT alts): tempestes convectives
  - Caiguda de pressió ràpida: aproximació de front o baixa
  - Backing wind + humitat: aproximació de front càlid
"""
import logging
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def detect_regime_change(prediction: dict, previous_state: dict) -> Optional[dict]:
    """
    Analitza el resultat de predicció actual i l'estat anterior per
    detectar canvis de règim atmosfèric significatius.

    Args:
        prediction: Resultat de predict_now() amb wind_regime, pressure_levels, conditions
        previous_state: Estat de notificació (notification_state.json) amb last_wind_regime

    Returns:
        Dict amb informació del canvi de règim si n'hi ha, o None.
        {
            "type": "llevantada_onset" | "garbi_inestable" | "pressure_drop" | "backing_wind",
            "severity": "watch" | "warning",
            "title": str (titular curt),
            "description": str (explicació),
            "details": dict (dades tècniques),
        }
    """
    wind_regime = prediction.get("wind_regime", {})
    pressure_levels = prediction.get("pressure_levels", {})
    conditions = prediction.get("conditions", {})
    radar = prediction.get("radar", {})
    ensemble = prediction.get("ensemble", {})

    prev_regime = previous_state.get("last_wind_regime", {})

    # Condicions actuals
    humidity = _safe_float(conditions.get("humidity"), 0)
    pressure_change_3h = _get_pressure_change(prediction)

    is_llevantada = wind_regime.get("is_llevantada", False)
    is_garbi = wind_regime.get("is_garbi", False)
    llevantada_strength = wind_regime.get("llevantada_strength", 0)
    wind_dir_change = wind_regime.get("wind_dir_change_3h")

    was_llevantada = prev_regime.get("is_llevantada", False)
    was_garbi = prev_regime.get("is_garbi", False)

    # Nivells de pressió
    tt_index = _safe_float(pressure_levels.get("tt_index"), 0)
    li_index = _safe_float(pressure_levels.get("li_index"), 10)
    temp_500 = _safe_float(pressure_levels.get("temp_500"), 0)
    rh_850 = _safe_float(pressure_levels.get("rh_850"), 0)

    # Radar espacial
    nearest_echo_km = radar.get("nearest_echo_km")
    storm_eta = radar.get("storm_eta_min")

    # ── 1. LLEVANTADA ONSET: vent gira a E/SE + humitat alta ──
    if is_llevantada and not was_llevantada:
        if humidity >= config.REGIME_HUMIDITY_THRESHOLD or rh_850 >= 80:
            severity = "warning" if llevantada_strength > 30 else "watch"
            details = {
                "wind_850_dir": pressure_levels.get("wind_850_dir"),
                "wind_850_speed": pressure_levels.get("wind_850_speed_kmh"),
                "humidity": humidity,
                "rh_850": rh_850,
                "llevantada_strength": llevantada_strength,
            }

            # Afegir context de radar si escau
            radar_context = ""
            if nearest_echo_km and nearest_echo_km < config.RADAR_SCAN_RADIUS_KM:
                compass = radar.get("nearest_echo_compass", "?")
                radar_context = f" Radar: ecos a {nearest_echo_km}km {compass}."
                if storm_eta:
                    radar_context += f" ETA ≈{storm_eta} min."

            return {
                "type": "llevantada_onset",
                "severity": severity,
                "title": "🌊 Llevantada: entrada d'humitat mediterrània",
                "description": (
                    f"El vent ha girat a Llevantada (E/SE) amb humitat del {humidity}%. "
                    f"Històricament, aquest patró produeix pluja a Cardedeu "
                    f"la majoria dels cops.{radar_context}"
                ),
                "details": details,
            }

    # ── 2. GARBÍ INESTABLE: vent SW + inestabilitat ──
    if is_garbi and not was_garbi:
        is_unstable = tt_index > 44 or li_index < -2 or temp_500 < -17
        if is_unstable:
            severity = "warning" if tt_index > 50 or li_index < -4 else "watch"
            return {
                "type": "garbi_inestable",
                "severity": severity,
                "title": "🌀 Garbí amb inestabilitat: risc de tempestes",
                "description": (
                    f"Entrada de Garbí (SW) amb senyals d'inestabilitat "
                    f"(TT={tt_index:.0f}, LI={li_index:.1f}). "
                    f"Configuració favorable a tempestes convectives."
                ),
                "details": {
                    "tt_index": tt_index,
                    "li_index": li_index,
                    "temp_500": temp_500,
                },
            }

    # ── 3. CAIGUDA RÀPIDA DE PRESSIÓ ──
    if pressure_change_3h is not None and pressure_change_3h <= config.REGIME_PRESSURE_DROP_3H:
        # Només alertar si estem en un règim humit (Llevantada o Garbí)
        is_moist_regime = is_llevantada or is_garbi or humidity >= 75
        if is_moist_regime:
            severity = "warning" if pressure_change_3h <= -4 else "watch"
            return {
                "type": "pressure_drop",
                "severity": severity,
                "title": "📉 Pressió baixant ràpidament",
                "description": (
                    f"La pressió ha caigut {abs(pressure_change_3h):.1f} hPa "
                    f"en 3 hores amb humitat del {humidity}%. "
                    f"Indica aproximació de front o baixa."
                ),
                "details": {
                    "pressure_change_3h": pressure_change_3h,
                    "humidity": humidity,
                },
            }

    # ── 4. BACKING WIND (gir antihorari) + HUMITAT ──
    if wind_dir_change is not None and wind_dir_change < -20:
        if humidity >= 70 or rh_850 >= 75:
            return {
                "type": "backing_wind",
                "severity": "watch",
                "title": "🔄 Vent girant (backing): possible front actiu",
                "description": (
                    f"El vent ha girat {abs(wind_dir_change):.0f}° en sentit antihorari "
                    f"en 3h. Amb humitat del {humidity}%, "
                    f"indica un front en aproximació."
                ),
                "details": {
                    "wind_dir_change_3h": wind_dir_change,
                    "humidity": humidity,
                },
            }

    return None


def get_current_regime_summary(prediction: dict) -> str:
    """
    Retorna un string curt que descriu el règim actual.
    Útil per al resum diari.
    """
    wind_regime = prediction.get("wind_regime", {})
    pressure_levels = prediction.get("pressure_levels", {})

    if wind_regime.get("is_llevantada"):
        speed = pressure_levels.get("wind_850_speed_kmh", "?")
        return f"🌊 Llevantada (ENE {speed} km/h a 850hPa)"
    elif wind_regime.get("is_garbi"):
        return "🌀 Garbí (SW)"
    elif wind_regime.get("is_tramuntana"):
        return "❄️ Tramuntana (N)"
    elif wind_regime.get("is_migjorn"):
        return "🌡️ Migjorn (S)"
    elif wind_regime.get("is_ponent"):
        return "🏔️ Ponent (W/NW)"
    else:
        wind_dir = pressure_levels.get("wind_850_dir")
        if wind_dir is not None:
            return f"Variable ({wind_dir:.0f}° a 850hPa)"
        return "Indeterminat"


def _safe_float(val, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


def _get_pressure_change(prediction: dict) -> Optional[float]:
    """Extreu el canvi de pressió en 3h del resultat de predicció."""
    # El canvi de pressió es calcula a engineering.py des de les dades de l'estació
    # Com que no el tenim directament, l'estimem des del conditions
    # o el passem explícitament si està disponible
    return prediction.get("pressure_change_3h")
