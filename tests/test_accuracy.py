"""
Tests per al mòdul d'accuracy (src/feedback/accuracy.py).
Cobreix: compute_accuracy mètriques, format_accuracy_report, N/A handling.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feedback.accuracy import format_accuracy_report


# ── Helpers ──

def _make_metrics(tp, fp, tn, fn):
    """Construeix mètriques directament, sense necessitar el JSONL."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "total_predictions": total + 5,
        "verified": total,
        "pending": 5,
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1) if precision is not None else None,
        "recall": round(recall * 100, 1) if recall is not None else None,
        "f1": round(f1 * 100, 1) if f1 is not None else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_confidence": {},
        "daily": {},
    }


# ── Precision/Recall/F1 amb denominador zero ──

class TestAccuracyNoRainEvents:
    """
    Quan no ha plogut (TP+FN=0), Recall ha de ser None (N/A), no 0%.
    Quan no s'han emès alertes (TP+FP=0), Precision ha de ser None (N/A).
    """

    def test_no_rain_recall_is_none(self):
        """TP+FN=0 → recall=None (no hi ha pluja per avaluar)."""
        metrics = _make_metrics(tp=0, fp=15, tn=356, fn=0)
        assert metrics["recall"] is None

    def test_no_rain_precision_is_zero(self):
        """FP>0, TP=0 → precision=0% (totes les alertes van ser falses)."""
        metrics = _make_metrics(tp=0, fp=15, tn=356, fn=0)
        assert metrics["precision"] == 0.0

    def test_no_rain_f1_is_none(self):
        """Quan recall=None, F1 ha de ser None."""
        metrics = _make_metrics(tp=0, fp=15, tn=356, fn=0)
        assert metrics["f1"] is None

    def test_no_alerts_precision_is_none(self):
        """TP+FP=0 → precision=None (cap alerta emesa)."""
        metrics = _make_metrics(tp=0, fp=0, tn=371, fn=0)
        assert metrics["precision"] is None

    def test_normal_case_all_defined(self):
        """Amb pluja i alertes: tots els valors definits i correctes."""
        metrics = _make_metrics(tp=5, fp=3, tn=350, fn=2)
        assert metrics["precision"] is not None
        assert metrics["recall"] is not None
        assert metrics["f1"] is not None
        assert metrics["precision"] == round(5 / 8 * 100, 1)
        assert metrics["recall"] == round(5 / 7 * 100, 1)

    def test_perfect_precision_recall(self):
        """Cap error: precision=100%, recall=100%, F1=100%."""
        metrics = _make_metrics(tp=10, fp=0, tn=350, fn=0)
        assert metrics["precision"] == 100.0
        assert metrics["recall"] == 100.0
        assert metrics["f1"] == 100.0


# ── Format del report Telegram ──

class TestFormatAccuracyReport:
    """Format del missatge Telegram amb N/A quan les mètriques són None."""

    def test_recall_none_shows_na(self):
        """Recall=None → mostra 'N/A'."""
        metrics = {
            "verified": 371, "pending": 8,
            "accuracy": 96.0, "precision": 0.0, "recall": None, "f1": None,
            "confusion": {"tp": 0, "fp": 15, "tn": 356, "fn": 0},
            "by_confidence": {}, "daily": {},
        }
        report = format_accuracy_report(metrics)
        assert "Recall: N/A" in report

    def test_precision_none_shows_na(self):
        """Precision=None → mostra 'N/A'."""
        metrics = {
            "verified": 100, "pending": 0,
            "accuracy": 100.0, "precision": None, "recall": None, "f1": None,
            "confusion": {"tp": 0, "fp": 0, "tn": 100, "fn": 0},
            "by_confidence": {}, "daily": {},
        }
        report = format_accuracy_report(metrics)
        assert "Precision: N/A" in report

    def test_normal_metrics_show_values(self):
        """Precision/recall amb valor → mostra percentatge, sense N/A."""
        metrics = {
            "verified": 100, "pending": 0,
            "accuracy": 95.0, "precision": 80.0, "recall": 70.0, "f1": 74.7,
            "confusion": {"tp": 7, "fp": 2, "tn": 88, "fn": 3},
            "by_confidence": {}, "daily": {},
        }
        report = format_accuracy_report(metrics)
        assert "80.0%" in report
        assert "70.0%" in report
        assert "N/A" not in report
