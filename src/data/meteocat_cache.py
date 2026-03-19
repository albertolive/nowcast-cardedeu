"""
Cache amb TTL per a crides a l'API Meteocat.
Evita superar el límit de 750 crides/mes del pla gratuït.

Amb prediccions cada 10 min (144/dia), sense cache:
  - XDDE (4 crides/predicció): 17,280/mes
  - SMC (1 crida/predicció): 4,320/mes
  - XEMA (4 crides, rain gate): variable

Amb cache TTL de 30-60 min:
  - XDDE: ~384/mes (48/dia × 4 × ~2 per cache miss)
  - SMC: ~720/mes (24/dia)
  - XEMA: similar a sense cache (ja gated)
"""
import hashlib
import json
import logging
import os
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)


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
