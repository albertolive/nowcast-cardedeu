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

    # Fair accuracy: only count predictions outside uncertain zone (sec + probable)
    scorable = [e for e in verified if e.get("rain_category") in ("sec", "probable")]
    uncertain = [e for e in verified if e.get("rain_category") == "incert"
                 or e.get("uncertain")]

    # Confusion matrix (fair: only sec + probable predictions)
    tp = sum(1 for e in scorable if e.get("rain_category") == "probable" and e["actual_rain"])
    fp = sum(1 for e in scorable if e.get("rain_category") == "probable" and not e["actual_rain"])
    tn = sum(1 for e in scorable if e.get("rain_category") == "sec" and not e["actual_rain"])
    fn = sum(1 for e in scorable if e.get("rain_category") == "sec" and e["actual_rain"])

    # Legacy confusion matrix (all verified, using will_rain for backward compat)
    legacy_tp = sum(1 for e in verified if e["will_rain"] and e["actual_rain"])
    legacy_fp = sum(1 for e in verified if e["will_rain"] and not e["actual_rain"])
    legacy_tn = sum(1 for e in verified if not e["will_rain"] and not e["actual_rain"])
    legacy_fn = sum(1 for e in verified if not e["will_rain"] and e["actual_rain"])

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # Brier score: mesura qualitat probabilística (0=perfecte, 1=pitjor)
    brier_components = [e.get("brier_component") for e in verified
                        if e.get("brier_component") is not None]
    brier_score = sum(brier_components) / len(brier_components) if brier_components else None

    # Calibration bins: "quan diem X%, plou X% de les vegades?"
    calibration = {}
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for lo, hi in bins:
        bin_entries = [e for e in verified
                       if lo <= e["probability_pct"] < hi]
        if bin_entries:
            actual_rain_pct = sum(1 for e in bin_entries if e["actual_rain"]) / len(bin_entries) * 100
            calibration[f"{lo}-{hi}%"] = {
                "predicted_range": f"{lo}-{hi}%",
                "count": len(bin_entries),
                "actual_rain_pct": round(actual_rain_pct, 1),
                "well_calibrated": abs(actual_rain_pct - (lo + hi) / 2) < 15,
            }

    # Accuracy per nivell de confiança
    by_confidence = {}
    for level in ["Molt Baixa", "Baixa", "Mitjana", "Alta", "Molt Alta"]:
        level_entries = [e for e in verified if e.get("confidence") == level]
        if level_entries:
            correct = sum(1 for e in level_entries if e.get("correct") is True)
            by_confidence[level] = {
                "total": len(level_entries),
                "correct": correct,
                "accuracy": round(correct / len(level_entries) * 100, 1),
            }

    # Accuracy per dia (últims 7 dies) — fair: only scorable predictions
    daily = {}
    for e in verified:
        day = e["timestamp"][:10]
        if day not in daily:
            daily[day] = {"total": 0, "correct": 0, "uncertain": 0, "scorable": 0}
        daily[day]["total"] += 1
        if e.get("uncertain") or e.get("rain_category") == "incert":
            daily[day]["uncertain"] += 1
        else:
            daily[day]["scorable"] += 1
            if e.get("correct") is True:
                daily[day]["correct"] += 1

    for day in daily:
        s = daily[day]["scorable"]
        daily[day]["accuracy"] = round(
            daily[day]["correct"] / s * 100, 1
        ) if s > 0 else None

    return {
        "total_predictions": len(entries),
        "verified": len(verified),
        "scorable": total,
        "uncertain_count": len(uncertain),
        "pending": len(entries) - len(verified),
        "accuracy": round(accuracy * 100, 1),
        "precision": round(precision * 100, 1) if precision is not None else None,
        "recall": round(recall * 100, 1) if recall is not None else None,
        "f1": round(f1 * 100, 1) if f1 is not None else None,
        "brier_score": round(brier_score, 4) if brier_score is not None else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "legacy_confusion": {"tp": legacy_tp, "fp": legacy_fp, "tn": legacy_tn, "fn": legacy_fn},
        "calibration": calibration,
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
        f"📋 Verificades: <b>{metrics['verified']}</b> ({metrics.get('scorable', 0)} puntuables, {metrics.get('uncertain_count', 0)} incertes)",
        f"⏳ Pendents: {metrics.get('pending', 0)}",
        "",
        f"🎯 <b>Accuracy (justa): {metrics['accuracy']}%</b>",
        f"🔎 Precision: {metrics['precision']}%" if metrics['precision'] is not None else "🔎 Precision: N/A",
        f"📡 Recall: {metrics['recall']}%" if metrics['recall'] is not None else "📡 Recall: N/A",
        f"⚖️ F1: {metrics['f1']}%" if metrics['f1'] is not None else "⚖️ F1: N/A",
    ]

    # Brier score
    if metrics.get("brier_score") is not None:
        bs = metrics["brier_score"]
        # Brier: 0=perfecte, 0.25=climatologia, 1=pitjor. <0.1 és excel·lent.
        quality = "excel·lent" if bs < 0.10 else "bo" if bs < 0.20 else "acceptable" if bs < 0.25 else "millorable"
        lines.append(f"📐 Brier Score: {bs:.4f} ({quality})")

    lines.extend([
        "",
        "📊 <b>Matriu de confusió (justa: sec + probable):</b>",
        f"  ✅ Pluja encertada: {cm['tp']}",
        f"  ❌ Falsa alarma: {cm['fp']}",
        f"  ✅ Sec encertat: {cm['tn']}",
        f"  ❌ Pluja no prevista: {cm['fn']}",
    ])

    # Calibration
    if metrics.get("calibration"):
        lines.append("")
        lines.append("🌡️ <b>Calibratge (diem X%, plou X%?):</b>")
        for bin_label, data in metrics["calibration"].items():
            check = "✅" if data["well_calibrated"] else "⚠️"
            lines.append(
                f"  {check} {bin_label}: plou {data['actual_rain_pct']}% "
                f"({data['count']} pred.)"
            )

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
