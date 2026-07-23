"""
Enriquiment amb IA: genera narratives en català a partir de dades meteorològiques.
Proveïdor/model gestionats centralment al cascade "general" d'
albertolive/ai-gateway (models.json), llegit en calent a cada crida.
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

_GATEWAY_CONFIG_URL = "https://raw.githubusercontent.com/albertolive/ai-gateway/main/models.json"

_exhausted: set[str] = set()


def _fetch_gateway_cascade(cascade: str = "general") -> tuple[dict, list[dict]]:
    """Llegeix providers + cascade d'ai-gateway/models.json (font compartida amb tot el fleet)."""
    try:
        resp = requests.get(_GATEWAY_CONFIG_URL, timeout=10)
        resp.raise_for_status()
        cfg = resp.json()
        return cfg["providers"], cfg["cascades"].get(cascade, [])
    except Exception as e:
        logger.warning(f"No s'ha pogut llegir ai-gateway/models.json ({e})")
        return {}, []


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(w in msg for w in ("429", "rate limit", "too many requests", "quota", "capacity"))


def _is_provider_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(w in msg for w in ("402", "provider returned error", "503", "502", "500",
                                   "service unavailable", "bad gateway", "internal server error"))


def _call_api(url: str, api_key: str, model: str, messages: list[dict],
              temperature: float, max_tokens: int, extra_headers: dict = None) -> str | None:
    """Crida genèrica a qualsevol API compatible amb OpenAI."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None


def _build_provider_chain() -> list[dict]:
    """
    Construeix la cadena de proveïdors des del cascade "general" d'ai-gateway,
    filtrant als que tenen la clau configurada (OPENROUTER_API_KEY /
    GEMINI_API_KEY / GROQ_API_KEY). Buida si cap secret està configurat o el
    fetch falla — generate_daily_narrative/generate_accuracy_narrative
    retornen None en aquest cas, mai exception.
    """
    providers, cascade = _fetch_gateway_cascade("general")
    chain = []
    for entry in cascade:
        p = providers.get(entry["provider"])
        if not p:
            continue
        key = os.environ.get(p["key_env"], "")
        if not key:
            continue
        chain.append({
            "provider": entry["provider"],
            "url": f"{p['url'].rstrip('/')}/chat/completions",
            "key": key,
            "model": entry["model"],
            "extra_headers": {
                "HTTP-Referer": "https://github.com/nowcast-cardedeu",
                "X-Title": "Nowcast Cardedeu",
            } if entry["provider"] == "openrouter" else {},
        })
    return chain


def _call_with_retry_and_fallback(messages: list[dict], temperature: float = 0.3,
                                   max_tokens: int = 500) -> str | None:
    """
    Crida l'API amb retry exponencial + fallback entre proveïdors/models.
    GitHub Models primer, OpenRouter free com a fallback.
    """
    chain = [e for e in _build_provider_chain()
             if f"{e['provider']}:{e['model']}" not in _exhausted]

    if not chain:
        logger.info("Cap proveïdor d'IA configurat o tots exhaurits — saltant narrativa")
        return None

    max_retries = config.AI_MAX_RETRIES

    for entry in chain:
        model_key = f"{entry['provider']}:{entry['model']}"

        for attempt in range(max_retries + 1):
            try:
                result = _call_api(
                    entry["url"], entry["key"], entry["model"],
                    messages, temperature, max_tokens, entry.get("extra_headers"),
                )
                if result:
                    logger.info(f"Resposta IA rebuda de {entry['provider']}/{entry['model']}")
                return result
            except Exception as e:
                if _is_provider_error(e):
                    logger.warning(f"Error de proveïdor amb {model_key}: {e}. Provant següent...")
                    _exhausted.add(model_key)
                    break
                elif _is_rate_limit_error(e):
                    if attempt < max_retries:
                        delay = config.AI_RETRY_BASE_DELAY_MS / 1000 * (3 ** attempt)
                        logger.warning(f"Rate limit a {model_key}, reintent {attempt + 1}/{max_retries} en {delay:.0f}s...")
                        time.sleep(delay)
                    else:
                        logger.warning(f"{model_key} exhaurit després de {max_retries} reintents")
                        _exhausted.add(model_key)
                else:
                    logger.warning(f"Error no recuperable amb {model_key}: {e}")
                    _exhausted.add(model_key)
                    break

    logger.warning("Tots els proveïdors d'IA exhaurits en aquesta execució")
    return None


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
