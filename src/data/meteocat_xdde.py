"""
Client per a l'API del Meteocat (XDDE).
Obté dades de descàrregues elèctriques (llamps) prop de Cardedeu.
La detecció de llamps és un indicador directe d'activitat convectiva
i pluja imminent.
Documentació: https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xdde/
"""
import logging
import math
from datetime import datetime, date, timezone
from typing import Optional

import numpy as np
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distància entre dos punts en km (fórmula de Haversine)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rumb (bearing) del punt 1 al punt 2 en graus (0=N, 90=E)."""
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r) -
         math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_to_compass(bearing: float) -> str:
    """Converteix graus (0-360) a punt cardinal."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(bearing / 22.5) % 16
    return directions[idx]


def fetch_lightning_data(target_date: Optional[date] = None) -> list[dict]:
    """
    Obté les descàrregues elèctriques de Catalunya per un dia.
    Endpoint: /xdde/v1/Catalunya/{YYYY}/{MM}/{DD}

    Retorna llista de dicts amb:
      - id, data (timestamp), lat, lon, correntPic, nuvolTerra
    """
    if not _is_configured():
        logger.warning("Meteocat API key no configurada per XDDE")
        return []

    if target_date is None:
        target_date = date.today()

    url = (
        f"{config.METEOCAT_BASE_URL}/xdde/v1/Catalunya/"
        f"{target_date.year}/{target_date.month:02d}/{target_date.day:02d}"
    )
    try:
        r = SESSION.get(url, headers=_headers(), timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Meteocat XDDE error ({target_date}): {e}")
        return []


def compute_lightning_features(
    radius_km: float = 30.0,
    hours_back: float = 3.0,
) -> dict:
    """
    Calcula features de llamps prop de Cardedeu.

    Args:
        radius_km: Radi de cerca en km al voltant de Cardedeu.
        hours_back: Hores enrere per comptar llamps recents.

    Retorna dict amb:
      - lightning_count_30km: nombre de llamps dins del radi en les últimes N hores
      - lightning_count_15km: llamps dins de 15km
      - lightning_nearest_km: distància al llamp més proper
      - lightning_nearest_compass: direcció del llamp més proper
      - lightning_cloud_ground: nombre de llamps núvol-terra (més perillosos)
      - lightning_max_current_ka: corrent pic màxim (kA)
      - lightning_approaching: True si la tempesta s'acosta (llamps recents més propers)
    """
    result = _empty_lightning_result(radius_km)

    if not _is_configured():
        return result

    try:
        strikes = fetch_lightning_data()
    except Exception as e:
        logger.warning(f"Error obtenint XDDE: {e}")
        return result

    if not strikes:
        logger.info("  XDDE: cap descàrrega avui a Catalunya")
        return result

    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - (hours_back * 3600)

    cardedeu_lat = config.LATITUDE
    cardedeu_lon = config.LONGITUDE

    nearby = []
    for s in strikes:
        coords = s.get("coordenades", {})
        lat = coords.get("latitud")
        lon = coords.get("longitud")
        if lat is None or lon is None:
            continue

        # Parsejar la data del llamp
        strike_time = s.get("data", "")
        try:
            # Format ISO: "2024-08-15T14:23:45.123Z"
            st = datetime.fromisoformat(strike_time.replace("Z", "+00:00"))
            ts = st.timestamp()
        except (ValueError, AttributeError):
            continue

        if ts < cutoff:
            continue

        dist = _haversine_km(cardedeu_lat, cardedeu_lon, lat, lon)
        if dist <= radius_km:
            bearing = _bearing_deg(cardedeu_lat, cardedeu_lon, lat, lon)
            nearby.append({
                "dist_km": dist,
                "bearing": bearing,
                "timestamp": ts,
                "cloud_ground": s.get("nuvolTerra", False),
                "current_ka": abs(s.get("correntPic", 0)),
            })

    if not nearby:
        logger.info(f"  XDDE: cap llamp dins de {radius_km}km en les últimes {hours_back}h")
        return result

    # Ordenar per distància
    nearby.sort(key=lambda x: x["dist_km"])

    nearest = nearby[0]
    count_30km = sum(1 for s in nearby if s["dist_km"] <= 30)
    count_15km = sum(1 for s in nearby if s["dist_km"] <= 15)
    cg_count = sum(1 for s in nearby if s["cloud_ground"])
    max_current = max(s["current_ka"] for s in nearby)

    # Detectar si la tempesta s'acosta:
    # Comparar la distància mitjana dels llamps recents (última hora)
    # vs els anteriors
    recent_cutoff = now.timestamp() - 3600  # última hora
    older_cutoff = now.timestamp() - (hours_back * 3600)
    recent = [s for s in nearby if s["timestamp"] >= recent_cutoff]
    older = [s for s in nearby if s["timestamp"] < recent_cutoff]

    approaching = False
    if recent and older:
        avg_recent = sum(s["dist_km"] for s in recent) / len(recent)
        avg_older = sum(s["dist_km"] for s in older) / len(older)
        approaching = avg_recent < avg_older - 2  # ≥2km més a prop

    result = {
        "lightning_count_30km": count_30km,
        "lightning_count_15km": count_15km,
        "lightning_nearest_km": round(nearest["dist_km"], 1),
        "lightning_nearest_bearing": round(nearest["bearing"]),
        "lightning_nearest_compass": _bearing_to_compass(nearest["bearing"]),
        "lightning_cloud_ground": cg_count,
        "lightning_max_current_ka": round(max_current, 1),
        "lightning_approaching": approaching,
        "lightning_has_activity": True,
    }

    logger.info(
        f"  XDDE: {count_30km} llamps dins 30km, {count_15km} dins 15km, "
        f"més proper a {nearest['dist_km']:.1f}km {_bearing_to_compass(nearest['bearing'])}, "
        f"{'s\'acosta' if approaching else 'estable/s\'allunya'}"
    )

    return result


def _empty_lightning_result(radius_km: float = 30.0) -> dict:
    return {
        "lightning_count_30km": 0,
        "lightning_count_15km": 0,
        "lightning_nearest_km": radius_km,
        "lightning_nearest_bearing": None,
        "lightning_nearest_compass": None,
        "lightning_cloud_ground": 0,
        "lightning_max_current_ka": 0.0,
        "lightning_approaching": False,
        "lightning_has_activity": False,
    }
