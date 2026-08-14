"""
Persistent TTL cache for Meteocat API calls.

The cache is committed by the GitHub Actions prediction job so it survives
between runners. Local writes are protected with an advisory lock and an
atomic replace: a killed process can leave an old cache, never half JSON.
"""
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager

import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(config.PROJECT_ROOT, "data", "meteocat_cache.json")
_MAX_ENTRIES = 200
RATE_LIMIT_KEY = "meteocat_rate_limit_cooldown"  # per-service key prefix
XEMA_RATE_LIMIT_KEY = "xema_rate_limit_cooldown"  # legacy alias, honored for old caches

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@contextmanager
def _cache_lock():
    """Serialize local read-modify-write operations for this cache file."""
    lock_path = f"{CACHE_FILE}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def fetch_quota() -> dict:
    """Check current API quota consumption."""
    if not config.METEOCAT_API_KEY:
        return {}
    from src.data._http import create_session
    if is_meteocat_rate_limited("quota"):
        return {}
    try:
        session = create_session(retry_429=False)
        r = session.get(
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
        if (getattr(getattr(e, "response", None), "status_code", None) == 429
                or getattr(locals().get("r", None), "status_code", None) == 429):
            mark_meteocat_rate_limited("quota")
        logger.warning("Could not fetch quota: %s", e)
        return {}


def get_remaining(plan_name: str) -> int:
    """Get remaining calls for a specific plan. Returns -1 if unknown."""
    quota = fetch_quota()
    return quota[plan_name]["remaining"] if plan_name in quota else -1


def _quarantine_corrupt_cache() -> None:
    """Move malformed JSON aside so it cannot be silently overwritten."""
    if not os.path.exists(CACHE_FILE):
        return
    quarantined = f"{CACHE_FILE}.corrupt.{os.getpid()}.{time.time_ns()}"
    try:
        os.replace(CACHE_FILE, quarantined)
        logger.error("Meteocat cache was malformed; quarantined as %s", quarantined)
    except OSError as exc:
        logger.error("Meteocat cache was malformed and could not be quarantined: %s", exc)


def _load_cache() -> dict:
    """Load cache safely; atomic writers mean readers never see partial JSON."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            raise ValueError("cache root is not an object")
        return value
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.error("Could not load Meteocat cache %s: %s", CACHE_FILE, exc)
        _quarantine_corrupt_cache()
        return {}


def _pruned_cache(cache: dict) -> dict:
    if len(cache) <= _MAX_ENTRIES:
        return cache
    sorted_keys = sorted(
        cache,
        key=lambda k: cache[k].get("timestamp", 0) if isinstance(cache[k], dict) else 0,
        reverse=True,
    )
    return {k: cache[k] for k in sorted_keys[:_MAX_ENTRIES]}


def _save_cache(cache: dict) -> None:
    """Atomically replace the cache file; caller owns any read-modify lock."""
    cache = _pruned_cache(cache)
    directory = os.path.dirname(CACHE_FILE) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(CACHE_FILE)}.", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, CACHE_FILE)
        temp_path = None
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Meteocat cache write failed: %s", exc)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _cache_entry(cache_key: str):
    return _load_cache().get(cache_key)


def _entry_age_minutes(entry) -> float | None:
    try:
        age = (time.time() - float(entry["timestamp"])) / 60
        return age if age >= 0 else 0.0
    except (KeyError, TypeError, ValueError):
        return None


def get_cached(cache_key: str, ttl_minutes: int):
    """Return cached value if it exists and is within TTL, else None."""
    entry = _cache_entry(cache_key)
    if not isinstance(entry, dict):
        return None
    age_minutes = _entry_age_minutes(entry)
    if age_minutes is not None and age_minutes <= ttl_minutes:
        logger.debug("Cache hit: %s (age=%.0fm)", cache_key, age_minutes)
        return entry.get("data")
    if age_minutes is not None:
        logger.debug("Cache expired: %s (age=%.0fm > %sm)", cache_key, age_minutes, ttl_minutes)
    return None


def get_stale_cached(cache_key: str, max_age_minutes: int):
    """Return a cached value within an explicit bounded stale-fallback age."""
    entry = _cache_entry(cache_key)
    if not isinstance(entry, dict):
        return None
    age_minutes = _entry_age_minutes(entry)
    if age_minutes is not None and age_minutes <= max_age_minutes:
        logger.debug("Stale fallback hit: %s (age=%.0fm)", cache_key, age_minutes)
        return entry.get("data")
    return None


def set_cached(cache_key: str, data) -> None:
    """Store a value using a locked read-modify-write and atomic replacement."""
    with _cache_lock():
        cache = _load_cache()
        cache[cache_key] = {"timestamp": time.time(), "data": data}
        _save_cache(cache)


def _service_rate_limit_key(service: str) -> str:
    return f"{RATE_LIMIT_KEY}_{service}"


def is_meteocat_rate_limited(service: str | None = None) -> bool:
    """Return whether the breaker for `service` is active.

    Per-service: a 429 on one endpoint (XDDE, SMC, XEMA, quota) only blocks that
    endpoint, never the others. XEMA also honors the legacy key written by older
    deployments, so pre-existing caches keep working.
    """
    if not service:
        return False
    ttl = getattr(config, "METEOCAT_429_COOLDOWN_MIN", 60)
    if get_cached(_service_rate_limit_key(service), ttl) is not None:
        return True
    if service == "xema" and get_cached(XEMA_RATE_LIMIT_KEY, ttl) is not None:
        return True
    return False


def mark_meteocat_rate_limited(service: str | None = None) -> None:
    """Persist a per-service breaker; a 429 on one endpoint does not block others."""
    if not service:
        return
    now = time.time()
    keys = [_service_rate_limit_key(service)]
    if service == "xema":
        keys.append(XEMA_RATE_LIMIT_KEY)
    with _cache_lock():
        cache = _load_cache()
        for key in keys:
            cache[key] = {"timestamp": now, "data": True}
        _save_cache(cache)
