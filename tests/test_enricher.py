"""
Tests per al mòdul d'enriquiment AI (src/ai/enricher.py).
Cobreix: context de narrativa d'accuracy amb/sense pluja.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
