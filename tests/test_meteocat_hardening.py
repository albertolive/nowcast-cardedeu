"""Offline tests for Meteocat quota protection and cache durability."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _use_tmp_cache(monkeypatch, tmp_path):
    from src.data import meteocat_cache
    path = str(tmp_path / "meteocat_cache.json")
    monkeypatch.setattr(meteocat_cache, "CACHE_FILE", path)
    return meteocat_cache


def _429_response():
    import requests
    response = type("Response", (), {"status_code": 429})()
    error = requests.HTTPError("rate limited")
    error.response = response
    return response, error


class TestMeteocatCacheDurability:
    def test_atomic_cache_is_valid_after_concurrent_writes(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        def write(i):
            cache.set_cached(f"key_{i}", {"value": i})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(32)))
        with open(cache.CACHE_FILE, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert {payload[f"key_{i}"]["data"]["value"] for i in range(32)} == set(range(32))
        assert not list(tmp_path.glob("*.tmp"))

    def test_corrupt_cache_is_quarantined(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        with open(cache.CACHE_FILE, "w", encoding="utf-8") as fh:
            fh.write('{"broken":')
        assert cache._load_cache() == {}
        quarantined = list(tmp_path.glob("meteocat_cache.json.corrupt.*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text(encoding="utf-8") == '{"broken":'

    def test_breaker_is_per_service(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        cache.mark_meteocat_rate_limited("xema")
        assert cache.is_meteocat_rate_limited("xema")
        assert not cache.is_meteocat_rate_limited("smc")
        assert not cache.is_meteocat_rate_limited("xdde")

    def test_quota_429_only_blocks_quota(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import src.data._http as http
        import src.data.meteocat_cache as module
        import requests

        class FakeResponse:
            status_code = 429
            def raise_for_status(self):
                error = requests.HTTPError("rate limited")
                error.response = self
                raise error

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(http, "create_session", lambda **kwargs: FakeSession())
        assert module.fetch_quota() == {}
        assert cache.is_meteocat_rate_limited("quota")
        assert not cache.is_meteocat_rate_limited("smc")


class TestMeteocatTransportPolicy:
    def test_meteocat_session_does_not_retry_429(self):
        from src.data._http import create_session
        retries = create_session(retry_429=False).get_adapter("https://").max_retries
        assert 429 not in retries.status_forcelist
        assert {502, 503, 504}.issubset(set(retries.status_forcelist))

    def test_other_sessions_keep_429_retry_policy(self):
        from src.data._http import create_session
        assert 429 in create_session().get_adapter("https://").max_retries.status_forcelist


class TestMeteocatEndpointBreakers:
    def test_smc_429_only_blocks_smc(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import src.data.meteocat_prediccio as smc
        response, error = _429_response()
        calls = []
        def get(*args, **kwargs):
            calls.append(1)
            raise error
        monkeypatch.setattr(smc.SESSION, "get", get)
        assert smc.fetch_smc_hourly_df().empty
        assert len(calls) == 1
        assert cache.is_meteocat_rate_limited("smc")
        assert not cache.is_meteocat_rate_limited("xdde")
        assert smc.fetch_smc_hourly_df().empty
        assert len(calls) == 1

    def test_xdde_429_only_blocks_xdde(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import src.data.meteocat_xdde as xdde
        response, error = _429_response()
        calls = []
        def get(*args, **kwargs):
            calls.append(1)
            raise error
        monkeypatch.setattr(xdde.SESSION, "get", get)
        assert xdde._fetch_lightning_hour(date(2026, 3, 24), 12) == []
        assert len(calls) == 1
        assert cache.is_meteocat_rate_limited("xdde")
        assert not cache.is_meteocat_rate_limited("smc")
        assert xdde._fetch_lightning_hour(date(2026, 3, 24), 13) == []
        assert len(calls) == 1

    def test_xema_breaker_does_not_block_smc_and_xdde_http(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        cache.mark_meteocat_rate_limited("xema")
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import src.data.meteocat_prediccio as smc
        import src.data.meteocat_xdde as xdde
        calls = []
        class OkResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return []
        def record_http(*args, **kwargs):
            calls.append(args[0])
            return OkResponse()
        monkeypatch.setattr(smc.SESSION, "get", record_http)
        monkeypatch.setattr(xdde.SESSION, "get", record_http)
        assert isinstance(smc.fetch_municipal_hourly_forecast(), dict)
        assert smc.fetch_smc_hourly_df().empty
        assert xdde._fetch_lightning_hour(date(2026, 3, 24), 12) == []
        assert len(calls) >= 3, "SMC and XDDE must still run HTTP while only XEMA is limited"

    def test_smc_dataframe_cache_roundtrip_restores_datetime(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import src.data.meteocat_prediccio as smc
        key = f"smc_hourly_{datetime.now().strftime('%Y%m%d_%H')}"
        rows = [{"datetime": "2026-03-24T12:00:00", "smc_prob_precip_1h": 40.0,
                 "smc_precip_intensity": 0.2, "smc_prob_precip_6h": 60.0}]
        cache.set_cached(key, rows)
        monkeypatch.setattr(smc.SESSION, "get", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fresh cache should avoid HTTP")))
        result = smc.fetch_smc_hourly_df()
        assert len(result) == 1
        assert isinstance(result.iloc[0]["datetime"], pd.Timestamp)
        assert result.iloc[0]["smc_prob_precip_1h"] == 40.0

    def test_timeout_does_not_mark_breaker(self, monkeypatch, tmp_path):
        cache = _use_tmp_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(config, "METEOCAT_API_KEY", "test-key")
        import requests
        import src.data.meteocat_prediccio as smc
        monkeypatch.setattr(smc.SESSION, "get", lambda *a, **k: (_ for _ in ()).throw(
            requests.Timeout("offline test")))
        assert smc.fetch_smc_hourly_df().empty
        assert not cache.is_meteocat_rate_limited("smc")
