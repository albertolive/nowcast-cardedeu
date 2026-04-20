---
applyTo: "src/data/**/*.py"
description: "Enforces graceful degradation, rain gate gating, and NaN-fallback patterns for all data source modules. Use when creating or editing any file under src/data/."
---

# Data Module Conventions

Every module in `src/data/` is an **independent API client** that must never crash the pipeline. Follow these patterns exactly.

## Required Structure

```python
"""Descripció en català del mòdul i la font de dades."""
import logging
import requests
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NowcastCardedeu/1.0"})
```

## Graceful Degradation (Mandatory)

Every public function wraps its logic in try/except and returns a dict with NaN (or safe default) values on failure:

```python
def fetch_something() -> dict:
    try:
        # ... API call and processing ...
        logger.info(f"Source: key_metric={value}")
        return result
    except Exception as e:
        logger.warning(f"Error obtenint source: {e}")
        return _empty_something_result()

def _empty_something_result() -> dict:
    return {
        "source_metric_a": np.nan,
        "source_metric_b": np.nan,
        "source_flag": False,
    }
```

Rules:
- Numeric values → `np.nan` (XGBoost handles natively)
- Boolean flags → `False` (safe default)
- Optional strings → `None`
- Never raise exceptions from public functions
- Log level: `info` for success, `warning` for degraded/failure, `error` for critical

## Rain Gate (Cost Optimization)

If the API is **rate-limited or requires an API key**, the module must support being gated:

1. Add a `_is_configured()` check for the API key
2. Return the empty result immediately if not configured
3. The rain gate decision lives in `src/model/predict.py` — the module itself just checks configuration

```python
def _is_configured() -> bool:
    return bool(config.YOUR_API_KEY)

def fetch_gated_data() -> dict:
    if not _is_configured():
        logger.warning("API_KEY no configurada per source")
        return _empty_result()
    # ... proceed with API call ...
```

Rain gate signals (checked in `src/model/predict.py`, not in the module):
- Radar echo detected (spatial scan: nearest echo < 30km, or `radar_has_echo`)
- Ensemble agreement ≥ 20% (`RAIN_GATE_ENSEMBLE_PROB = 0.2`)
- CAPE ≥ 800 J/kg (max of next 6h)
- AEMET storm prob ≥ 10%
- AEMET radar has echo

**Note:** AEMET modules are NOT rain-gated — they are called unconditionally (gated only by API key) because their output feeds INTO the rain gate decision. Only Meteocat modules (XEMA, XDDE, Predicció) are behind the rain gate.
- AEMET storm prob ≥ 10%
- AEMET radar has echo

## Config Integration

- All URLs, coordinates, thresholds, and API keys come from `config.py`
- Reference: `config.LATITUDE`, `config.LONGITUDE`, `config.YOUR_API_KEY`
- Never hardcode coordinates, URLs, or magic numbers

## Naming

- Public function: `fetch_*()` or `compute_*_features()`
- Empty result helper: `_empty_*_result()`
- Private helpers: `_parse_*()`, `_extract_*()`, etc.
- Dict keys: `source_metric_name` (snake_case, prefixed by source)
- All comments, docstrings, and log messages in **Catalan**

## Existing Modules (9 total)

| Module | API Key | Rain-gated | Notes |
|--------|---------|------------|-------|
| `open_meteo.py` | No | No | Surface + pressure levels (925/850/700/500/300hPa), CAPE/CIN backfill (Historical Forecast API, April 2021+), historical + forecast + NOAA ERDDAP SST (2015-present) |
| `meteocardedeu.py` | No | No | Real-time station + NOAA historical |
| `rainviewer.py` | No | No | Radar tiles → dBZ/mm/h, spatial scan 30km, storm tracking |
| `ensemble.py` | No | No | ECMWF + GFS + ICON + AROME agreement |
| `meteocat.py` | Yes | Yes | XEMA sentinel stations (Granollers, ETAP Cardedeu) |
| `meteocat_xdde.py` | Yes | Yes | Lightning data (XDDE) within 30km |
| `meteocat_prediccio.py` | Yes | Yes | SMC municipal forecast |
| `aemet.py` | Yes | No* | Expert storm/precip probability |
| `aemet_radar.py` | Yes | No* | Regional Barcelona radar |

\* AEMET modules require an API key but are NOT rain-gated — their output feeds into the rain gate decision.
