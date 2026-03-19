"""
Cache persistent per a les dades d'AEMET.

AEMET's free API has undocumented rate limits (~25 requests/min shared).
At 10-min prediction intervals, consecutive runs can trigger 429 errors.

This module provides a file-based cache (data/aemet_cache.json) that
persists across GitHub Actions runs via git commits. Cache TTLs:
  - Radar: 10 min (updates every ~10 min)
  - Forecast: 60 min (updates every ~6-12h, but we check hourly)
"""
import json
import logging
import os
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(config.PROJECT_ROOT, "data", "aemet_cache.json")

# TTL in seconds
RADAR_TTL = 10 * 60     # 10 min — radar updates every ~10 min
FORECAST_TTL = 60 * 60  # 60 min — forecast updates every ~6-12h


def _load_cache() -> dict:
    """Load cache from disk."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    """Write cache to disk."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def get_cached(key: str, ttl: int) -> dict | None:
    """Return cached data if fresh, else None."""
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    cached_at = entry.get("cached_at", 0)
    age = _now_ts() - cached_at
    if age > ttl:
        logger.info(f"  Cache AEMET '{key}' expirat (edat: {age/60:.0f} min, TTL: {ttl/60:.0f} min)")
        return None
    logger.info(f"  Cache AEMET '{key}' HIT (edat: {age/60:.0f} min)")
    return entry.get("data")


def set_cached(key: str, data: dict) -> None:
    """Store data in cache."""
    cache = _load_cache()
    cache[key] = {
        "cached_at": _now_ts(),
        "data": data,
    }
    _save_cache(cache)
