---
description: "Scaffold a new data source module in src/data/ with graceful degradation, rain gate support, config integration, and predict_now.py wiring. Use when adding a weather data API, sensor, or external data feed."
---

# Add Data Source

Create a new independent data module following the project's established patterns.

## Inputs

- **Source name** (e.g., "AEMET radar", "Meteo.cat XEMA"): `${input:sourceName}`
- **Needs API key?** (yes/no): `${input:needsApiKey}`
- **Should be rain-gated?** (yes = only queried when rain signals present): `${input:rainGated}`
- **Base URL**: `${input:baseUrl}`
- **Output features** (comma-separated key names): `${input:outputFeatures}`

## Steps

### 1. Add config constants

In `config.py`, add:
- The base URL as `SOURCE_BASE_URL`
- If API key needed: `SOURCE_API_KEY = os.environ.get("SOURCE_API_KEY", "")`
- Any source-specific thresholds as `UPPER_CASE` constants
- If rain-gated: a corresponding `RAIN_GATE_*` threshold if this source contributes a new gate signal

### 2. Create the data module

Create `src/data/source_name.py` following the exact pattern from `.github/instructions/data-modules.instructions.md`:

```python
"""Descripció en català: client per obtenir dades de [source]."""
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

Must include:
- `fetch_*()` public function returning a dict
- `_empty_*_result()` returning NaN/False defaults for every output key
- try/except wrapping all API calls with `logger.warning` on failure
- If API key needed: `_is_configured()` guard that checks config
- Catalan docstrings, comments, and log messages

### 3. Wire into predict_now.py

In `scripts/predict_now.py`:

1. Import the new fetch function
2. Call it in the appropriate section:
   - **Not rain-gated**: Call alongside other always-on sources (radar, ensemble)
   - **Rain-gated**: Call inside the `if rain_signals:` block, with empty dict fallback in else branch
3. Merge the returned dict into the features dict with `features.update(source_data)`

### 4. Register features (if features should feed into the model)

If the output features should be used by XGBoost:
- Add feature column names to `FEATURE_COLUMNS` in `src/features/engineering.py`
- Add a `_add_source_features(df)` function if derived features are needed (e.g., deltas, interactions)
- Call it from `build_features_from_hourly()` pipeline
- Note: if features are **real-time only** (no historical data), they go in the real-time pipeline only — XGBoost handles NaN for training rows that lack these columns

### 5. Update documentation

- Add the source to `.github/instructions/data-modules.instructions.md` module table
- Add the source to the architecture diagram in `.github/copilot-instructions.md`
- Add any new env var to the Secrets section and to `.github/workflows/nowcast.yml` env block

### 6. Run feature analysis

After retraining, verify the new features contribute:
```bash
python scripts/feature_analysis.py
```
If features show zero importance, prefer keeping the continuous source variable
over adding binary threshold features — XGBoost learns thresholds better from continuous values.

## Validation Checklist

- [ ] Module returns correct dict on success AND on failure (NaN fallback)
- [ ] `logger.info()` on success, `logger.warning()` on failure
- [ ] No hardcoded URLs, coordinates, or thresholds — all from `config.py`
- [ ] If rain-gated: called inside `if rain_signals:` in `predict_now.py`
- [ ] If API key needed: `_is_configured()` guard present
- [ ] All text in Catalan
- [ ] Feature keys use `source_metric` naming pattern (snake_case, prefixed)
- [ ] Run `python scripts/feature_analysis.py` after retrain to verify new features have non-zero gain
