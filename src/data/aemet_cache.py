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
# Radar TTL > run interval (10 min) so alternate runs reuse the cache instead
# of hitting the API every run, which is what triggers sustained 429s.
RADAR_TTL = 15 * 60     # 15 min — radar updates every ~10 min
FORECAST_TTL = 60 * 60  # 60 min — forecast updates every ~6-12h

# Stale fallback: when a fetch fails (429, timeout), an expired cache entry
# younger than this is still far better than reporting "no data".
RADAR_STALE_MAX_AGE = 45 * 60       # 45 min
FORECAST_STALE_MAX_AGE = 6 * 3600   # 6 h — forecast updates every ~6-12h


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


def get_stale(key: str, max_age: int) -> dict | None:
    """Return cached data even if expired, as long as it's younger than max_age.

    Fallback for when the live fetch fails (AEMET 429s can last hours):
    a slightly old radar image beats reporting the radar as unavailable.
    """
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    age = _now_ts() - entry.get("cached_at", 0)
    if age > max_age:
        return None
    logger.warning(
        f"  Cache AEMET '{key}' STALE fallback (edat: {age/60:.0f} min, màx: {max_age/60:.0f} min)"
    )
    return entry.get("data")


def set_cached(key: str, data: dict) -> None:
    """Store data in cache."""
    cache = _load_cache()
    cache[key] = {
        "cached_at": _now_ts(),
        "data": data,
    }
    _save_cache(cache)
