"""
Accuracy tracker — calcula mètriques de rendiment acumulades
a partir de les prediccions verificades.
"""
import logging
from collections import Counter
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.feedback.logger import load_predictions_log

logger = logging.getLogger(__name__)


def compute_accuracy(days: int = None) -> dict:
    """
    Calcula mètriques d'accuracy a partir del log de prediccions.
    Si `days` és None, calcula sobre totes les prediccions.
    Si `days` és un int, calcula sobre els últims N dies.

    Retorna:
      total, verified, accuracy, precision, recall, f1,
      confusion: {tp, fp, tn, fn},
      by_confidence: {level: {total, correct, accuracy}}
    """
    entries = load_predictions_log()
    verified = [e for e in entries if e.get("verified")]

    if days is not None:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        verified = [e for e in verified if e["timestamp"] >= cutoff]

    if not verified:
        return {
            "total_predictions": len(entries),
            "verified": 0,
            "message": "Cap predicció verificada encara.",
        }

    # Confusion matrix
    tp = sum(1 for e in verified if e["will_rain"] and e["actual_rain"])
    fp = sum(1 for e in verified if e["will_rain"] and not e["actual_rain"])
    tn = sum(1 for e in verified if not e["will_rain"] and not e["actual_rain"])
    fn = sum(1 for e in verified if not e["will_rain"] and e["actual_rain"])

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Accuracy per nivell de confiança
    by_confidence = {}
    for level in ["Molt Baixa", "Baixa", "Mitjana", "Alta", "Molt Alta"]:
        level_entries = [e for e in verified if e.get("confidence") == level]
        if level_entries:
            correct = sum(1 for e in level_entries if e["correct"])
            by_confidence[level] = {
                "total": len(level_entries),
                "correct": correct,
                "accuracy": round(correct / len(level_entries) * 100, 1),
            }

    # Accuracy per dia (últims 7 dies)
    daily = {}
    for e in verified:
        day = e["timestamp"][:10]
        if day not in daily:
            daily[day] = {"total": 0, "correct": 0}
        daily[day]["total"] += 1
        if e["correct"]:
            daily[day]["correct"] += 1

    for day in daily:
        daily[day]["accuracy"] = round(
            daily[day]["correct"] / daily[day]["total"] * 100, 1
        )

    return {
        "total_predictions": len(entries),
        "verified": total,
        "pending": len(entries) - total,
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1 * 100, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_confidence": by_confidence,
        "daily": dict(sorted(daily.items(), reverse=True)[:7]),
    }


def format_accuracy_report(metrics: dict) -> str:
    """Formata les mètriques en un missatge Telegram."""
    if metrics.get("verified", 0) == 0:
        return (
            "📊 <b>Nowcast Cardedeu — Informe de rendiment</b>\n\n"
            f"Prediccions registrades: {metrics.get('total_predictions', 0)}\n"
            "Cap predicció verificada encara. Espera almenys 75 min."
        )

    cm = metrics["confusion"]
    lines = [
        "📊 <b>Nowcast Cardedeu — Informe de rendiment</b>",
        "",
        f"📋 Prediccions verificades: <b>{metrics['verified']}</b>",
        f"⏳ Pendents: {metrics.get('pending', 0)}",
        "",
        f"🎯 <b>Accuracy: {metrics['accuracy']}%</b>",
        f"🔎 Precision: {metrics['precision']}% (de les alertes, quantes van ser pluja real)",
        f"📡 Recall: {metrics['recall']}% (de les pluges reals, quantes vam predir)",
        f"⚖️ F1: {metrics['f1']}%",
        "",
        "📊 <b>Matriu de confusió:</b>",
        f"  ✅ True Positive (pluja predita correcta): {cm['tp']}",
        f"  ❌ False Positive (alerta falsa): {cm['fp']}",
        f"  ✅ True Negative (sec predit correcte): {cm['tn']}",
        f"  ❌ False Negative (pluja no predita): {cm['fn']}",
    ]

    # Accuracy per confiança
    if metrics.get("by_confidence"):
        lines.append("")
        lines.append("📈 <b>Accuracy per confiança:</b>")
        for level, data in metrics["by_confidence"].items():
            bar = "█" * int(data["accuracy"] / 10) + "░" * (10 - int(data["accuracy"] / 10))
            lines.append(f"  {level}: {bar} {data['accuracy']}% ({data['total']})")

    # Últims dies
    if metrics.get("daily"):
        lines.append("")
        lines.append("📅 <b>Últims dies:</b>")
        for day, data in list(metrics["daily"].items())[:5]:
            lines.append(f"  {day}: {data['accuracy']}% ({data['correct']}/{data['total']})")

    return "\n".join(lines)
