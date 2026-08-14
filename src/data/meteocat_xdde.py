"""
Client per a l'API del Meteocat (XDDE): dades de descàrregues elèctriques.
"""
import logging
from datetime import datetime, date, timezone
from typing import Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session
from src.data._geo import _haversine_km, _bearing_deg, _bearing_to_compass

logger = logging.getLogger(__name__)
SESSION = create_session(retry_429=False)


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def _is_429(exc, response=None) -> bool:
    return (getattr(getattr(exc, "response", None), "status_code", None) == 429
            or getattr(response, "status_code", None) == 429)


def _fetch_lightning_hour(target_date: date, hour: int) -> list[dict]:
    """Fetch one XDDE hour with persistent per-service breaker and bounded fallback."""
    from src.data.meteocat_cache import (
        get_cached, get_stale_cached, is_meteocat_rate_limited,
        mark_meteocat_rate_limited, set_cached,
    )
    cache_key = f"xdde_{target_date}_{hour:02d}"
    now_utc = datetime.now(timezone.utc)
    is_current_hour = target_date == now_utc.date() and hour == now_utc.hour
    ttl = config.METEOCAT_CACHE_TTL_XDDE if is_current_hour else 1440
    cached = get_cached(cache_key, ttl)
    if cached is not None:
        return cached
    if is_meteocat_rate_limited("xdde"):
        stale = get_stale_cached(cache_key, config.METEOCAT_XDDE_STALE_MAX_MIN)
        if stale is not None:
            logger.warning("XDDE cooldown active — using bounded stale hour %s %02dh", target_date, hour)
            return stale
        return []
    url = (f"{config.METEOCAT_BASE_URL}/xdde/v1/catalunya/"
           f"{target_date.year}/{target_date.month:02d}/{target_date.day:02d}/{hour:02d}")
    response = None
    try:
        response = SESSION.get(url, headers=_headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
        result = data if isinstance(data, list) else []
        set_cached(cache_key, result)
        return result
    except Exception as exc:
        if _is_429(exc, response):
            mark_meteocat_rate_limited("xdde")
        stale = get_stale_cached(cache_key, config.METEOCAT_XDDE_STALE_MAX_MIN)
        if stale is not None:
            logger.warning("XDDE request failed — using bounded stale data: %s", exc)
            return stale
        logger.debug("XDDE hour error (%s %02dh): %s", target_date, hour, exc)
        return []


def fetch_lightning_data(target_date: Optional[date] = None, hours: Optional[list[int]] = None) -> list[dict]:
    """Fetch the last four requested XDDE hours, returning a list on failure."""
    if not _is_configured():
        logger.warning("Meteocat API key no configurada per XDDE")
        return []
    if target_date is None:
        target_date = date.today()
    if hours is None:
        current_hour = datetime.now(timezone.utc).hour
        hours = [(current_hour - i) % 24 for i in range(4)]
    all_strikes = []
    for h in hours:
        all_strikes.extend(_fetch_lightning_hour(target_date, h))
    logger.info("XDDE: %s descàrregues obtingudes (%s hores consultades)", len(all_strikes), len(hours))
    return all_strikes


def compute_lightning_features(radius_km: float = config.RADAR_SCAN_RADIUS_KM, hours_back: float = 3.0) -> dict:
    """Calcula features de llamps prop de Cardedeu."""
    result = _empty_lightning_result(radius_km)
    if not _is_configured():
        return result
    from src.data.meteocat_cache import get_cached, set_cached
    cache_key = f"lightning_features_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}"
    cached = get_cached(cache_key, config.METEOCAT_CACHE_TTL_XDDE)
    if cached is not None:
        logger.info("XDDE: using cached lightning features")
        return cached
    try:
        strikes = fetch_lightning_data()
    except Exception as exc:
        logger.warning("Error obtenint XDDE: %s", exc)
        return result
    if not strikes:
        set_cached(cache_key, result)
        return result
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - hours_back * 3600
    nearby = []
    for strike in strikes:
        coords = strike.get("coordenades", {})
        lat, lon = coords.get("latitud"), coords.get("longitud")
        if lat is None or lon is None:
            continue
        try:
            timestamp = datetime.fromisoformat(strike.get("data", "").replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if timestamp < cutoff:
            continue
        dist = _haversine_km(config.LATITUDE, config.LONGITUDE, lat, lon)
        if dist <= radius_km:
            bearing = _bearing_deg(config.LATITUDE, config.LONGITUDE, lat, lon)
            nearby.append({"dist_km": dist, "bearing": bearing, "timestamp": timestamp,
                           "cloud_ground": strike.get("nuvolTerra", False),
                           "current_ka": abs(strike.get("correntPic", 0))})
    if not nearby:
        set_cached(cache_key, result)
        return result
    nearby.sort(key=lambda item: item["dist_km"])
    nearest = nearby[0]
    recent_cutoff = now.timestamp() - 3600
    recent = [item for item in nearby if item["timestamp"] >= recent_cutoff]
    older = [item for item in nearby if item["timestamp"] < recent_cutoff]
    approaching = bool(recent and older and
                       sum(item["dist_km"] for item in recent) / len(recent) <
                       sum(item["dist_km"] for item in older) / len(older) - 2)
    count_15km_1h = sum(1 for item in recent if item["dist_km"] <= 15)
    count_30km_1h = sum(1 for item in recent if item["dist_km"] <= 30)
    result = {
        "lightning_count_30km": sum(item["dist_km"] <= 30 for item in nearby),
        "lightning_count_15km": sum(item["dist_km"] <= 15 for item in nearby),
        "lightning_count_15km_1h": count_15km_1h,
        "lightning_count_30km_1h": count_30km_1h,
        "lightning_nearest_km": round(nearest["dist_km"], 1),
        "lightning_nearest_bearing": round(nearest["bearing"]),
        "lightning_nearest_compass": _bearing_to_compass(nearest["bearing"]),
        "lightning_cloud_ground": sum(item["cloud_ground"] for item in nearby),
        "lightning_max_current_ka": round(max(item["current_ka"] for item in nearby), 1),
        "lightning_approaching": approaching,
        "lightning_has_activity": True,
    }
    set_cached(cache_key, result)
    return result


def _empty_lightning_result(radius_km: float = config.RADAR_SCAN_RADIUS_KM) -> dict:
    return {
        "lightning_count_30km": 0, "lightning_count_15km": 0,
        "lightning_count_15km_1h": 0, "lightning_count_30km_1h": 0,
        "lightning_nearest_km": radius_km, "lightning_nearest_bearing": None,
        "lightning_nearest_compass": None, "lightning_cloud_ground": 0,
        "lightning_max_current_ka": 0.0, "lightning_approaching": False,
        "lightning_has_activity": False,
    }
