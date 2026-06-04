"""
Tests del fallback de cache caducada (get_stale).

Motivació: el 2026-06-04 l'API d'AEMET va retornar 429 durant ~3h i el radar
constava com a "no disponible", deixant cegues les regles físiques mentre
una tempesta de 56 dBZ era a 7 km.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import aemet_cache


def _write_cache(monkeypatch, tmp_path, age_seconds):
    cache_file = tmp_path / "aemet_cache.json"
    monkeypatch.setattr(aemet_cache, "CACHE_FILE", str(cache_file))
    now = aemet_cache._now_ts()
    monkeypatch.setattr(aemet_cache, "_now_ts", lambda: now)
    cache_file.write_text(
        '{"radar": {"cached_at": %f, "data": {"aemet_radar_dbz": 40.0}}}'
        % (now - age_seconds)
    )


def test_get_stale_returns_expired_entry_within_max_age(monkeypatch, tmp_path):
    # 20 min: expirada per al TTL normal (15 min) però dins el màxim stale (45 min)
    _write_cache(monkeypatch, tmp_path, age_seconds=20 * 60)
    assert aemet_cache.get_cached("radar", aemet_cache.RADAR_TTL) is None
    stale = aemet_cache.get_stale("radar", aemet_cache.RADAR_STALE_MAX_AGE)
    assert stale == {"aemet_radar_dbz": 40.0}


def test_get_stale_rejects_entry_older_than_max_age(monkeypatch, tmp_path):
    _write_cache(monkeypatch, tmp_path, age_seconds=60 * 60)
    assert aemet_cache.get_stale("radar", aemet_cache.RADAR_STALE_MAX_AGE) is None


def test_get_stale_returns_none_when_no_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(aemet_cache, "CACHE_FILE", str(tmp_path / "missing.json"))
    assert aemet_cache.get_stale("radar", aemet_cache.RADAR_STALE_MAX_AGE) is None
