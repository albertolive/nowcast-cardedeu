# Nowcast Cardedeu — Project Guidelines

Hyperlocal rain nowcasting system for Cardedeu (Vallès Oriental) using XGBoost to correct global NWP models with local measurements from MeteoCardedeu.net.

## Architecture

```
src/data/       → 9 independent API clients (graceful degradation: each returns empty dict on failure)
src/features/   → Feature engineering (54 train / 100 real-time) + wind regime detection
src/model/      → XGBoost training (TimeSeriesSplit CV + IsotonicRegression calibration) + prediction
src/notify/     → Telegram alerts with state machine (hysteresis: up=0.65, down=0.30)
src/feedback/   → JSONL prediction log, verification (60+ min later), feedback export for retraining
scripts/        → Entry points: download_history, build_dataset, train_model, predict_now, daily_summary, accuracy_report
config.py       → All constants, paths, thresholds, coordinates — single source of truth
```

**Key pattern — Rain Gate:** Expensive APIs (Meteocat XEMA, AEMET) are only queried when `rain_gate` signals are present (radar echo, ensemble agreement ≥ 25%, CAPE ≥ 800, lightning, or AEMET storm prob ≥ 10%). Always preserve this cost optimization.

**Key pattern — Graceful degradation:** Every `src/data/` module wraps API calls in try/except, logs warnings, and returns a dict with NaN values on failure. XGBoost handles NaN natively. Never let a single API failure crash the pipeline.

**Key pattern — Feature split:** 100 features defined in `FEATURE_COLUMNS` but only 54 exist in historical training data. The remaining 46 are real-time only (radar, sentinel, ensemble, AEMET, lightning, forecast bias). XGBoost handles NaN natively for training rows missing these columns. The feedback loop gradually adds real-time features to the training set as verified predictions accumulate.

**Key pattern — Wind regimes at 850hPa:** Wind classification (Llevantada, Garbí, Ponent, Tramuntana, Migjorn) uses the synoptic 850hPa wind, not the 10m surface wind which is distorted by local orography (Montseny). The raw binary regime flags have zero model importance — the interaction terms (`llevantada_strength`, `llevantada_moisture`, `garbi_strength`) carry the signal.

**Key pattern — Feature pruning:** Binary threshold features (e.g., `cape_high`, `cold_500_moderate`) tend to have zero importance because the continuous source variable is always more informative. Prefer continuous features; only add binary indicators if XGBoost can't learn the threshold from the continuous value (very rare).

## Build and Test

```bash
# Install
pip install -r requirements.txt   # Python 3.12

# Train pipeline (first time)
python scripts/download_history.py  # Fetch 12 years from Open-Meteo + NOAA + pressure levels
python scripts/build_dataset.py     # Engineer features → data/processed/training_dataset.parquet
python scripts/train_model.py       # XGBoost + calibration → models/

# Real-time prediction
python scripts/predict_now.py       # Fetch current data → predict → log → notify

# Reports
python scripts/daily_summary.py     # 3-slot hourly outlook → Telegram
python scripts/accuracy_report.py   # Weekly verified accuracy → Telegram

# Analysis
python scripts/feature_analysis.py  # Feature importance audit (gain, splits, zero-importance)
```

No formal test suite exists (`tests/` is empty). Validation relies on the feedback loop (verify predictions 60+ min later) and weekly accuracy reports.

## Conventions

- **Language:** Catalan throughout — variable names, comments, docstrings, log messages, user-facing text. Maintain this consistently.
- **Naming:** `snake_case` for functions/variables, `UPPER_CASE` for constants in `config.py`.
- **Logging:** Use `logging.getLogger(__name__)` with format `"%(asctime)s [%(levelname)s] %(message)s"`. Levels: `info` for normal flow, `warning` for degraded API, `error` for failures.
- **Timestamps:** ISO 8601 everywhere (`datetime.isoformat()`, `datetime.fromisoformat()`).
- **Telegram:** HTML parse mode with emoji, never Markdown. Catalan month abbreviations ('gen', 'feb', 'mar', etc.).
- **JSON serialization:** Use `_NumpyEncoder` (handles numpy bool_, integer, floating, ndarray).
- **New data sources:** Add as independent module in `src/data/`, follow existing pattern (try/except → logger.warning → return dict with NaN). Gate behind `rain_gate` if the API is rate-limited or expensive.
- **Features:** Register new features in `src/features/engineering.py`. Training features go in historical pipeline; real-time-only features (radar, sentinel, ensemble, lightning) are added only in `predict_now.py`.
- **Config:** All thresholds, paths, and coordinates live in `config.py` — never hardcode magic numbers in modules.
- **Feature design:** Prefer continuous features over binary indicators. Binary threshold features (e.g., `cape_high`, `cold_500_moderate`) consistently show zero importance because XGBoost can learn any threshold from the continuous source. Use interaction terms (regime × magnitude) for wind patterns. Run `python scripts/feature_analysis.py` after adding features to verify they contribute.

## CI/CD

GitHub Actions (`.github/workflows/nowcast.yml`):
- **predict**: Every 15 min (6–23h Barcelona) → `predict_now.py`
- **daily_summary**: 7:00 Barcelona → `daily_summary.py`
- **accuracy_report**: Monday 8:00 → `accuracy_report.py`
- **retrain**: Daily 2:00 → download + build + train + git commit model

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `METEOCAT_API_KEY`, `AEMET_API_KEY`
