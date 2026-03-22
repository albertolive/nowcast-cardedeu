"""
Cache amb TTL per a crides a l'API Meteocat.
Quotes separades per servei (mensuals, reset dia 1 a 00:00 UTC):
  - XEMA: 750 crides/mes
  - XDDE: 250 crides/mes
  - Predicció: 100 crides/mes
  - Referència: 2000 crides/mes
  - Quota (consum-actual): 300 crides/mes

Endpoint de consum: GET /quotes/v1/consum-actual
"""
import hashlib
import json
import logging
import os
import time

import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


def fetch_quota() -> dict:
    """
    Check current API quota consumption.
    Returns dict mapping plan name → {max, used, remaining}.
    """
    if not config.METEOCAT_API_KEY:
        return {}
    try:
        r = requests.get(
            f"{config.METEOCAT_BASE_URL}/quotes/v1/consum-actual",
            headers={"X-Api-Key": config.METEOCAT_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        result = {}
        for plan in data.get("plans", []):
            result[plan["nom"]] = {
                "max": plan["maxConsultes"],
                "used": plan["consultesRealitzades"],
                "remaining": plan["consultesRestants"],
            }
        return result
    except Exception as e:
        logger.warning(f"Could not fetch quota: {e}")
        return {}


def get_remaining(plan_name: str) -> int:
    """Get remaining calls for a specific plan. Returns -1 if unknown."""
    quota = fetch_quota()
    if plan_name in quota:
        return quota[plan_name]["remaining"]
    return -1


def _cache_path(key: str) -> str:
    """Return path for a cache entry."""
    os.makedirs(config.METEOCAT_CACHE_DIR, exist_ok=True)
    safe_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(config.METEOCAT_CACHE_DIR, f"{safe_key}.json")


def get_cached(cache_key: str, ttl_minutes: int):
    """
    Return cached value if it exists and is within TTL, else None.
    """
    path = _cache_path(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            entry = json.load(f)
        age_minutes = (time.time() - entry["timestamp"]) / 60
        if age_minutes <= ttl_minutes:
            logger.debug(f"Cache hit: {cache_key} (age={age_minutes:.0f}m)")
            return entry["data"]
        logger.debug(f"Cache expired: {cache_key} (age={age_minutes:.0f}m > {ttl_minutes}m)")
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def set_cached(cache_key: str, data):
    """Store a value in the cache with current timestamp."""
    path = _cache_path(cache_key)
    try:
        os.makedirs(config.METEOCAT_CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"timestamp": time.time(), "data": data}, f)
    except OSError as e:
        logger.debug(f"Cache write failed: {e}")
