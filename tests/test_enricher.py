"""
Tests per al mòdul d'enriquiment AI (src/ai/enricher.py).
Cobreix: context de narrativa d'accuracy amb/sense pluja, i la cadena de
proveïdors llegida d'ai-gateway/models.json.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai import enricher


class TestBuildProviderChain:
    """La cadena ve del cascade 'general' d'ai-gateway, no de GitHub Models."""

    _FAKE_CONFIG = {
        "providers": {
            "openrouter": {"url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
            "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY"},
        },
        "cascades": {
            "general": [
                {"provider": "gemini", "model": "gemini-2.0-flash"},
                {"provider": "openrouter", "model": "openrouter/free"},
            ]
        },
    }

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def test_no_keys_configured_gives_empty_chain(self, monkeypatch):
        monkeypatch.setattr(enricher.requests, "get",
                            lambda *a, **k: self._FakeResponse(self._FAKE_CONFIG))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert enricher._build_provider_chain() == []

    def test_only_configured_providers_included(self, monkeypatch):
        monkeypatch.setattr(enricher.requests, "get",
                            lambda *a, **k: self._FakeResponse(self._FAKE_CONFIG))
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        chain = enricher._build_provider_chain()
        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"
        assert chain[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert chain[0]["key"] == "test-key"

    def test_fetch_failure_gives_empty_chain(self, monkeypatch):
        def _raise(*a, **k):
            raise ConnectionError("network down")
        monkeypatch.setattr(enricher.requests, "get", _raise)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert enricher._build_provider_chain() == []

    def test_no_github_models_reference(self):
        """Regressió: GitHub Models (retirat 30/07/2026) no ha de reaparèixer."""
        import inspect
        src = inspect.getsource(enricher)
        assert "models.inference.ai.azure.com" not in src
        assert "AI_GITHUB_TOKEN" not in src


class TestAccuracyNarrativeContext:
    """
    El context enviat a l'LLM ha d'indicar explícitament si ha plogut o no.
    Sense això, l'AI interpreta TP=0 + FN=0 com 'el model ha fallat en
    detectar pluja' quan en realitat no hi ha hagut pluja.
    """

    @staticmethod
    def _build_no_rain_flag(cm):
        """Reprodueix la lògica de generate_accuracy_narrative."""
        return (cm.get("tp", 0) + cm.get("fn", 0)) == 0

    @staticmethod
    def _build_context_fragment(no_rain):
        return (
            "NO — TP+FN=0, no hi ha hagut pluja real. Recall no es pot avaluar."
            if no_rain
            else "SÍ — hi ha hagut episodis de pluja."
        )

    def test_no_rain_detected(self):
        """TP=0 + FN=0 → no_rain=True."""
        assert self._build_no_rain_flag({"tp": 0, "fp": 15, "tn": 356, "fn": 0})

    def test_rain_detected(self):
        """TP+FN > 0 → no_rain=False."""
        assert not self._build_no_rain_flag({"tp": 5, "fp": 3, "tn": 350, "fn": 2})

    def test_only_fn_counts_as_rain(self):
        """FN>0 (pluja no detectada) vol dir que sí ha plogut."""
        assert not self._build_no_rain_flag({"tp": 0, "fp": 0, "tn": 350, "fn": 3})

    def test_context_no_rain_text(self):
        """Context sense pluja inclou 'no hi ha hagut pluja'."""
        fragment = self._build_context_fragment(no_rain=True)
        assert "no hi ha hagut pluja" in fragment

    def test_context_with_rain_text(self):
        """Context amb pluja inclou 'episodis de pluja'."""
        fragment = self._build_context_fragment(no_rain=False)
        assert "episodis de pluja" in fragment
