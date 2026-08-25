"""
Enriquiment amb IA via l'endpoint hostatjat d'albertolive/ai-gateway.
El `model` és un nom de CASCADE ("general"), no un id de proveïdor: el
gateway el resol a provider/model/clau i fa failover en servidor (models
AND comptes). Una rotació de models a ai-gateway/models.json arriba aquí
sense cap canvi — abans es llegia models.json en calent i es duplicava la
cadena de failover al client, amb claus de proveïdor pròpies.
GitHub Models (retirat 30/07/2026) ja no s'usa.
Patró adaptat de gencat-cultural-agenda/src/ai/enricher.ts.

Dissenyat per a ús de baixa freqüència (1 crida/dia al resum diari,
1 crida/setmana a l'informe d'accuracy). Mai al camí crític d'alertes.
"""
import logging
import time

import requests

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

# Endpoint hostatjat, compatible OpenAI. Override amb AI_GATEWAY_URL.
_GATEWAY_URL = os.environ.get(
    "AI_GATEWAY_URL",
    "https://ai-gateway-livid-eight.vercel.app/api/chat/completions",
)
_CASCADE = os.environ.get("AI_GATEWAY_CASCADE", "general")

_exhausted_run = False


def _is_transient_error(e: Exception) -> bool:
    """429/5xx/timeout valen la pena reintentar; 401/403/400 no."""
    msg = str(e).lower()
    return any(w in msg for w in ("429", "rate limit", "too many requests", "quota",
                                  "503", "502", "500", "service unavailable",
                                  "bad gateway", "timeout", "timed out"))


def _call_gateway(messages: list[dict], temperature: float,
                  max_tokens: int) -> str | None:
    """Una crida al gateway amb retry exponencial sobre errors transitoris."""
    if not config.GATEWAY_TOKEN:
        return None
    headers = {
        "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _CASCADE,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(config.AI_MAX_RETRIES + 1):
        try:
            resp = requests.post(_GATEWAY_URL, headers=headers, json=payload,
                                 timeout=120)
            if resp.status_code in (429, 500, 502, 503, 504):
                # Transitori: rellança ConnectionError per al flux de retry.
                raise ConnectionError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.ok:
                logger.error(f"Gateway HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            return (data.get("choices", [{}])[0].get("message", {})
                    .get("content", "").strip() or None)
        except ConnectionError as e:
            if attempt < config.AI_MAX_RETRIES:
                delay = config.AI_RETRY_BASE_DELAY_MS / 1000 * (2 ** attempt)
                logger.warning(
                    f"Error transitori del gateway ({e}), reintent "
                    f"{attempt + 1}/{config.AI_MAX_RETRIES} en {delay:.0f}s...")
                time.sleep(delay)
                continue
            logger.warning(f"Gateway exhaurit despr\u00e9s de {attempt + 1} intents: {e}")
            return None
        except Exception as e:
            logger.warning(f"Crida al gateway ha fallat: {e}")
            return None

    return None


def _call_with_retry_and_fallback(messages: list[dict], temperature: float = 0.3,
                                  max_tokens: int = 500) -> str | None:
    """
    Manté la signatura històrica. Sense GATEWAY_TOKEN no hi ha IA: les
    narratives retornen None i els scripts ho gestionen sense excepció.
    """
    global _exhausted_run
    if _exhausted_run or not config.GATEWAY_TOKEN:
        if not config.GATEWAY_TOKEN:
            logger.info("GATEWAY_TOKEN no configurat — saltant narrativa IA")
            _exhausted_run = True
        return None

    result = _call_gateway(messages, temperature, max_tokens)
    if result is None:
        # Gateway caigut: no insisteixis més durant aquesta execució.
        _exhausted_run = True
        return None
    return result


def generate_daily_narrative(prediction: dict, hourly_outlook: list[dict],
                              next_rain_text: str | None) -> str | None:
    """
    Genera una narrativa en català per al resum diari (7:00).
    Retorna un paràgraf fluid o None si la IA no està disponible.
    """
    prob = prediction.get("probability_pct", 0)
    confidence = prediction.get("confidence", "?")
    conditions = prediction.get("conditions", {})
    radar = prediction.get("radar", {})
    ensemble = prediction.get("ensemble", {})
    pressure_levels = prediction.get("pressure_levels", {})
    wind_regime = prediction.get("wind_regime", {})

    # Determinar règim eòlic actiu
    regime = "variable"
    for name in ("llevantada", "garbi", "tramuntana", "migjorn", "ponent"):
        if wind_regime.get(f"is_{name}"):
            regime = name
            break

    # Construir contexte per al prompt
    slots_text = ""
    if hourly_outlook:
        for s in hourly_outlook:
            slots_text += f"  - {s['label']}: {s.get('max_prob', 0):.0f}% pluja, {s.get('temp_range', '?')}\n"

    context = f"""Dades actuals de Cardedeu (Vallès Oriental):
- Probabilitat de pluja 60 min: {prob}% (confiança: {confidence})
- Temperatura: {conditions.get('temperature', '?')}°C, humitat: {conditions.get('humidity', '?')}%
- Pressió: {conditions.get('pressure', '?')} hPa, canvi 3h: {prediction.get('pressure_change_3h', '?')} hPa
- Vent: {conditions.get('wind_speed', '?')} km/h {conditions.get('wind_dir', '')}
- Règim eòlic (850hPa): {regime}
- Radar: {'eco detectat' if radar.get('has_echo') else f"eco més proper a {radar.get('nearest_echo_km', '?')} km" if radar.get('nearest_echo_km') else 'net'}
- Ensemble: {ensemble.get('models_rain', '?')}/{ensemble.get('total_models', 4)} models prediuen pluja
- Índex TT: {pressure_levels.get('tt_index', '?')}, LI: {pressure_levels.get('li_index', '?')}
Franges previstes:
{slots_text if slots_text else '  No disponibles'}
Propera pluja: {next_rain_text or "cap prevista en 48h"}"""

    messages = [
        {
            "role": "system",
            "content": (
                "Ets un meteoròleg local de Cardedeu (Vallès Oriental, Catalunya). "
                "Escriu un paràgraf curt (3-4 frases) en català explicant la previsió del dia "
                "d'una manera natural i entenedora per al públic general. "
                "Menciona les causes principals (règim eòlic, pressió, radar, ensemble) "
                "només si són rellevants. No repeteixis números exactes, sinó interpreta'ls. "
                "To informal però informatiu, com un amic que entén el temps. "
                "No facis servir emojis ni formatatge HTML."
            ),
        },
        {"role": "user", "content": context},
    ]

    return _call_with_retry_and_fallback(messages, temperature=0.4, max_tokens=300)


def generate_accuracy_narrative(metrics_week: dict, metrics_all: dict) -> str | None:
    """
    Genera una narrativa en català per a l'informe setmanal d'accuracy.
    Retorna 2-3 frases interpretant les mètriques o None.
    """
    if metrics_week.get("verified", 0) == 0:
        return None

    cm = metrics_week.get("confusion", {})
    no_rain = (cm.get("tp", 0) + cm.get("fn", 0)) == 0

    context = f"""Mètriques setmanals del model de predicció de pluja a Cardedeu:
- Prediccions verificades: {metrics_week.get('verified', 0)}
- Accuracy: {metrics_week.get('accuracy', '?')}%
- Precision: {metrics_week.get('precision', 'N/A')}% (de les alertes, quantes van ser pluja real)
- Recall: {metrics_week.get('recall', 'N/A')}% (de les pluges reals, quantes vam detectar)
- F1: {metrics_week.get('f1', 'N/A')}%
- True Positives: {cm.get('tp', 0)}, False Positives: {cm.get('fp', 0)}
- True Negatives: {cm.get('tn', 0)}, False Negatives: {cm.get('fn', 0)}
- Ha plogut aquesta setmana? {'NO — TP+FN=0, no hi ha hagut pluja real. Recall no es pot avaluar.' if no_rain else 'SÍ — hi ha hagut episodis de pluja.'}
Total acumulat: {metrics_all.get('verified', '?')} prediccions, {metrics_all.get('accuracy', '?')}% accuracy"""

    by_conf = metrics_week.get("by_confidence", {})
    if by_conf:
        context += "\nAccuracy per confiança:"
        for level, data in by_conf.items():
            context += f"\n  {level}: {data['accuracy']}% ({data['total']} prediccions)"

    messages = [
        {
            "role": "system",
            "content": (
                "Ets un analista del model de predicció de pluja de Cardedeu. "
                "Escriu 2-3 frases curtes en català interpretant les mètriques setmanals. "
                "IMPORTANT: si TP+FN=0 vol dir que NO ha plogut — no diguis que el model "
                "ha fallat en detectar pluja, sinó que no hi ha hagut pluja per avaluar el recall. "
                "En aquest cas, centra't en els falsos positius (alertes innecessàries) i en "
                "les condicions meteorològiques que els van provocar. "
                "Destaca: on ha fallat el model, tendència general. Sigues concís i directe. "
                "No repeteixis tots els números, interpreta'ls. "
                "No facis servir emojis ni formatatge HTML."
            ),
        },
        {"role": "user", "content": context},
    ]

    return _call_with_retry_and_fallback(messages, temperature=0.3, max_tokens=200)
