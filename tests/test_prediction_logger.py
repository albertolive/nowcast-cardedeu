"""
Tests per al logger de prediccions (src/feedback/logger.py).
Cobreix: _sanitize_nans (recursiu), _NumpyEncoder (numpy types),
log_prediction (JSONL output), load_predictions_log (parse).
"""
import json
import math
import os
import pytest

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feedback.logger import (
    _sanitize_nans,
    _NumpyEncoder,
    log_prediction,
    load_predictions_log,
)


# ── _sanitize_nans ──

class TestSanitizeNans:
    def test_float_nan_becomes_none(self):
        assert _sanitize_nans(float("nan")) is None

    def test_float_inf_becomes_none(self):
        assert _sanitize_nans(float("inf")) is None

    def test_float_neg_inf_becomes_none(self):
        assert _sanitize_nans(float("-inf")) is None

    def test_normal_float_preserved(self):
        assert _sanitize_nans(3.14) == 3.14

    def test_zero_preserved(self):
        assert _sanitize_nans(0.0) == 0.0

    def test_nested_dict(self):
        obj = {"a": 1, "b": {"c": float("nan"), "d": 2.0}}
        result = _sanitize_nans(obj)
        assert result["a"] == 1
        assert result["b"]["c"] is None
        assert result["b"]["d"] == 2.0

    def test_nested_list(self):
        obj = [1.0, float("nan"), [float("inf"), 3.0]]
        result = _sanitize_nans(obj)
        assert result == [1.0, None, [None, 3.0]]

    def test_tuple_becomes_list(self):
        """Tuples es tracten com llistes."""
        obj = (1.0, float("nan"))
        result = _sanitize_nans(obj)
        assert result == [1.0, None]

    def test_dict_in_list(self):
        obj = [{"val": float("nan")}]
        result = _sanitize_nans(obj)
        assert result == [{"val": None}]

    def test_string_passthrough(self):
        assert _sanitize_nans("hello") == "hello"

    def test_int_passthrough(self):
        assert _sanitize_nans(42) == 42

    def test_none_passthrough(self):
        assert _sanitize_nans(None) is None

    def test_deeply_nested(self):
        obj = {"a": [{"b": [float("nan"), {"c": float("inf")}]}]}
        result = _sanitize_nans(obj)
        assert result == {"a": [{"b": [None, {"c": None}]}]}


# ── _NumpyEncoder ──

class TestNumpyEncoder:
    def _encode(self, obj):
        return json.loads(json.dumps(obj, cls=_NumpyEncoder))

    def test_numpy_bool_true(self):
        assert self._encode(np.bool_(True)) == 1

    def test_numpy_bool_false(self):
        assert self._encode(np.bool_(False)) == 0

    def test_numpy_int32(self):
        assert self._encode(np.int32(42)) == 42

    def test_numpy_int64(self):
        assert self._encode(np.int64(-7)) == -7

    def test_numpy_float64(self):
        assert self._encode(np.float64(3.14)) == pytest.approx(3.14)

    def test_numpy_float_nan(self):
        """NaN numpy dins de dict → _sanitize_nans + _NumpyEncoder → null."""
        obj = {"val": np.float64("nan")}
        result = json.loads(json.dumps(_sanitize_nans(obj), cls=_NumpyEncoder))
        assert result["val"] is None

    def test_numpy_float_inf(self):
        """Inf numpy dins de dict → _sanitize_nans + _NumpyEncoder → null."""
        obj = {"val": np.float64("inf")}
        result = json.loads(json.dumps(_sanitize_nans(obj), cls=_NumpyEncoder))
        assert result["val"] is None

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        assert self._encode(arr) == [1, 2, 3]

    def test_numpy_array_with_nan(self):
        """NaN dins d'un array: tolist() produeix float nan, necessita _sanitize_nans."""
        arr = np.array([1.0, np.nan, 3.0])
        result = arr.tolist()  # _NumpyEncoder fa tolist()
        # Sense _sanitize_nans, el NaN seria float('nan')
        # Amb _sanitize_nans (que log_prediction aplica), es converteix a None
        assert math.isnan(result[1])  # tolist() preserva NaN

    def test_unsupported_type_raises(self):
        """Tipus no suportat → TypeError (via super().default())."""
        with pytest.raises(TypeError):
            json.dumps(object(), cls=_NumpyEncoder)


# ── log_prediction + load_predictions_log ──

class TestLogPrediction:
    def test_round_trip(self, tmp_path, monkeypatch):
        """Escriure i rellegir una predicció."""
        log_path = str(tmp_path / "predictions.jsonl")
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG", log_path)

        result = {
            "timestamp": "2026-03-23T12:00:00",
            "probability": 0.72,
            "probability_pct": 72.0,
            "will_rain": True,
            "confidence": "Alta",
            "threshold": 0.40,
            "raw_probability": 0.68,
            "rain_gate_open": True,
            "conditions": {"temperature": 18.5},
            "radar": {"radar_dbz": 25.0},
        }
        log_prediction(result)

        entries = load_predictions_log()
        assert len(entries) == 1
        assert entries[0]["probability"] == 0.72
        assert entries[0]["will_rain"] is True
        assert entries[0]["verified"] is False
        assert entries[0]["actual_rain"] is None
        assert entries[0]["conditions"]["temperature"] == 18.5

    def test_multiple_entries_append(self, tmp_path, monkeypatch):
        """Múltiples prediccions s'afegeixen (append, no overwrite)."""
        log_path = str(tmp_path / "predictions.jsonl")
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG", log_path)

        for i in range(3):
            result = {
                "timestamp": f"2026-03-23T{12+i}:00:00",
                "probability": 0.50 + i * 0.1,
                "probability_pct": 50.0 + i * 10,
                "will_rain": False,
                "confidence": "Mitjana",
            }
            log_prediction(result)

        entries = load_predictions_log()
        assert len(entries) == 3

    def test_nan_in_features_serialized(self, tmp_path, monkeypatch):
        """NaN a features ha de serialitzar-se com a null."""
        log_path = str(tmp_path / "predictions.jsonl")
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG", log_path)

        result = {
            "timestamp": "2026-03-23T12:00:00",
            "probability": 0.50,
            "probability_pct": 50.0,
            "will_rain": False,
            "confidence": "Mitjana",
            "feature_vector": {"radar_dbz": float("nan"), "temp": 20.0},
        }
        log_prediction(result)

        entries = load_predictions_log()
        assert entries[0]["features"]["radar_dbz"] is None
        assert entries[0]["features"]["temp"] == 20.0

    def test_numpy_types_serialized(self, tmp_path, monkeypatch):
        """Tipus NumPy s'han de serialitzar correctament."""
        log_path = str(tmp_path / "predictions.jsonl")
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG", log_path)

        result = {
            "timestamp": "2026-03-23T12:00:00",
            "probability": np.float64(0.72),
            "probability_pct": np.float64(72.0),
            "will_rain": np.bool_(True),
            "confidence": "Alta",
            "feature_vector": {"val": np.int32(42)},
        }
        log_prediction(result)

        entries = load_predictions_log()
        assert entries[0]["probability"] == 0.72
        # np.bool_ → int in JSON, then parsed as int
        assert entries[0]["will_rain"] in (True, 1)

    def test_empty_log_returns_empty_list(self, tmp_path, monkeypatch):
        """Log inexistent → llista buida."""
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG",
                            str(tmp_path / "missing.jsonl"))
        assert load_predictions_log() == []

    def test_catalan_unicode_preserved(self, tmp_path, monkeypatch):
        """Caràcters catalans (ç, à, é) preservats."""
        log_path = str(tmp_path / "predictions.jsonl")
        monkeypatch.setattr("src.feedback.logger.PREDICTIONS_LOG", log_path)

        result = {
            "timestamp": "2026-03-23T12:00:00",
            "probability": 0.50,
            "probability_pct": 50.0,
            "will_rain": False,
            "confidence": "Mitjana",
            "wind_regime": {"name": "Garbí humit d'estació"},
        }
        log_prediction(result)

        with open(log_path, "r", encoding="utf-8") as f:
            line = f.readline()
        assert "Garbí" in line
        assert "d'estació" in line
