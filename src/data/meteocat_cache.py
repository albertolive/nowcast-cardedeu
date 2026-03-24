"""
Cache amb TTL per a crides a l'API Meteocat.
Quotes separades per servei (mensuals, reset dia 1 a 00:00 UTC):
  - XEMA: 750 crides/mes
  - XDDE: 250 crides/mes
  - Predicció: 100 crides/mes
  - Referència: 2000 crides/mes
  - Quota (consum-actual): 300 crides/mes

Persistence: single JSON file (data/meteocat_cache.json) committed to git,
so cache survives across GitHub Actions runs. Same pattern as aemet_cache.py.

Endpoint de consum: GET /quotes/v1/consum-actual
"""
import json
import logging
import os
import time

import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(config.PROJECT_ROOT, "data", "meteocat_cache.json")
# Max entries before pruning old ones (keep cache file manageable)
_MAX_ENTRIES = 200


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


def _load_cache() -> dict:
    """Load entire cache from single JSON file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    """Write cache to disk. Prune old entries if over limit."""
    if len(cache) > _MAX_ENTRIES:
        # Keep most recent entries
        sorted_keys = sorted(cache.keys(),
                             key=lambda k: cache[k].get("timestamp", 0),
                             reverse=True)
        cache = {k: cache[k] for k in sorted_keys[:_MAX_ENTRIES]}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.debug(f"Cache write failed: {e}")


def get_cached(cache_key: str, ttl_minutes: int):
    """
    Return cached value if it exists and is within TTL, else None.
    """
    cache = _load_cache()
    entry = cache.get(cache_key)
    if not entry:
        return None
    try:
        age_minutes = (time.time() - entry["timestamp"]) / 60
        if age_minutes <= ttl_minutes:
            logger.debug(f"Cache hit: {cache_key} (age={age_minutes:.0f}m)")
            return entry["data"]
        logger.debug(f"Cache expired: {cache_key} (age={age_minutes:.0f}m > {ttl_minutes}m)")
    except (KeyError, TypeError):
        pass
    return None


def set_cached(cache_key: str, data):
    """Store a value in the cache with current timestamp."""
    cache = _load_cache()
    cache[cache_key] = {"timestamp": time.time(), "data": data}
    _save_cache(cache)
