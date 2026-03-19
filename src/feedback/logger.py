"""
Prediction logger — registra cada predicció per a verificació posterior.
Guarda un log append-only en format JSONL (una línia JSON per predicció).
"""
import json
import logging
import os

import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


PREDICTIONS_LOG = os.path.join(config.PROJECT_ROOT, "data", "predictions_log.jsonl")


def log_prediction(result: dict) -> None:
    """
    Afegeix una predicció al log JSONL.
    Inclou el vector de features complet per alimentar el feedback loop,
    més totes les dades de sensors per a diagnòstic futur.
    """
    entry = {
        "timestamp": result["timestamp"],
        "probability": result["probability"],
        "probability_pct": result["probability_pct"],
        "will_rain": result["will_rain"],
        "confidence": result["confidence"],
        "threshold": result.get("threshold"),
        "raw_probability": result.get("raw_probability"),
        "rain_gate_open": result.get("rain_gate_open"),
        # Condicions locals
        "conditions": result.get("conditions", {}),
        # Radar complet (RainViewer)
        "radar": result.get("radar", {}),
        # Radar AEMET
        "aemet": result.get("aemet", {}),
        # Sentinella XEMA
        "sentinel": result.get("sentinel", {}),
        # Ensemble
        "ensemble": result.get("ensemble", {}),
        # Règim de vent
        "wind_regime": result.get("wind_regime", {}),
        # Nivells de pressió
        "pressure_levels": result.get("pressure_levels", {}),
        # Bias forecast vs observació
        "bias": result.get("bias", {}),
        # Vector de features complet (per feedback loop)
        "features": result.get("feature_vector", {}),
        # Camps de verificació (es completen després)
        "verified": False,
        "actual_rain": None,
        "actual_rain_mm": None,
        "correct": None,
    }

    os.makedirs(os.path.dirname(PREDICTIONS_LOG), exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, cls=_NumpyEncoder) + "\n")

    logger.info(f"Predicció registrada al log ({PREDICTIONS_LOG})")


def load_predictions_log() -> list[dict]:
    """Carrega totes les prediccions del log."""
    if not os.path.exists(PREDICTIONS_LOG):
        return []
    entries = []
    with open(PREDICTIONS_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_predictions_log(entries: list[dict]) -> None:
    """Reescriu tot el log (usat després de verificar)."""
    os.makedirs(os.path.dirname(PREDICTIONS_LOG), exist_ok=True)
    with open(PREDICTIONS_LOG, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, cls=_NumpyEncoder) + "\n")
