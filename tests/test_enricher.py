"""
Tests per al mòdul d'enriquiment AI (src/ai/enricher.py).
Cobreix el client del gateway hostatjat: sense token → None, retry sobre
errors transitoris, i no-insistència després d'una caiguda.
Les narratives (context accuracy) es testegen a part, són pures.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai import enricher


class _FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload if payload is not None else {
            "choices": [{"message": {"content": "Resposta IA"}}]
        }

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


@pytest.fixture(autouse=True)
def _reset_run_state(monkeypatch):
    """Cada test comença amb el run-state net i sense token real."""
    enricher._exhausted_run = False
    monkeypatch.setattr(enricher.config, "GATEWAY_TOKEN", "")
    yield
    enricher._exhausted_run = False


class TestGatewayClient:
    def test_no_token_returns_none_without_calling(self, monkeypatch):
        calls = []
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: calls.append(1))
        assert enricher._call_gateway([{"role": "user", "content": "x"}],
                                      0.3, 100) is None
        assert calls == []

    def test_success_parses_content(self, monkeypatch):
        monkeypatch.setattr(enricher.config, "GATEWAY_TOKEN", "tok")
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: _FakeResponse())
        out = enricher._call_gateway([{"role": "user", "content": "x"}], 0.3, 100)
        assert out == "Resposta IA"

    def test_transient_503_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(enricher.config, "GATEWAY_TOKEN", "tok")
        monkeypatch.setattr(enricher.config, "AI_MAX_RETRIES", 2)
        monkeypatch.setattr(enricher.config, "AI_RETRY_BASE_DELAY_MS", 0)
        responses = [_FakeResponse(status=503), _FakeResponse()]
        seen = []
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: seen.append(1) or responses.pop(0))
        out = enricher._call_gateway([{"role": "user", "content": "x"}], 0.3, 100)
        assert out == "Resposta IA"
        assert len(seen) == 2

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(enricher.config, "GATEWAY_TOKEN", "tok")
        monkeypatch.setattr(enricher.config, "AI_MAX_RETRIES", 1)
        monkeypatch.setattr(enricher.config, "AI_RETRY_BASE_DELAY_MS", 0)
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: _FakeResponse(status=503))
        assert enricher._call_gateway([{"role": "user", "content": "x"}], 0.3, 100) is None


class TestCallWithFallback:
    def test_no_token_skips_narrative(self):
        enricher._exhausted_run = False
        assert enricher._call_with_retry_and_fallback(
            [{"role": "user", "content": "x"}]) is None

    def test_gateway_down_disables_rest_of_run(self, monkeypatch):
        monkeypatch.setattr(enricher.config, "GATEWAY_TOKEN", "tok")
        enricher._exhausted_run = False
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: _FakeResponse(status=503))
        monkeypatch.setattr(enricher.config, "AI_MAX_RETRIES", 0)
        assert enricher._call_with_retry_and_fallback(
            [{"role": "user", "content": "x"}]) is None
        # Second call must short-circuit without hitting the network
        calls = []
        monkeypatch.setattr(enricher.requests, "post",
                            lambda *a, **k: calls.append(1))
        assert enricher._call_with_retry_and_fallback(
            [{"role": "user", "content": "x"}]) is None
        assert calls == []


class TestNoLegacyProviders:
    def test_no_direct_provider_references(self):
        """Regressió: claus per proveïdor no han de reaparèixer al client."""
        import inspect
        src = inspect.getsource(enricher)
        for legacy in ("models.inference.ai.azure.com",
                       "openrouter.ai/api/v1",
                       "OPENROUTER_API_KEY",
                       "_build_provider_chain"):
            assert legacy not in src, f"legacy reference found: {legacy}"


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
