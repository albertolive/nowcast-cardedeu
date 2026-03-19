# Nowcast Cardedeu — Project Guidelines

Hyperlocal rain nowcasting system for Cardedeu (Vallès Oriental) using XGBoost to correct global NWP models with local measurements from MeteoCardedeu.net.

## Architecture

```
src/data/       → 12 independent API clients (graceful degradation: each returns empty dict on failure)
src/features/   → Feature engineering (112 features: 68 with historical data, 44 real-time only) + wind regime detection
src/features/regime.py → Regime change detection (Llevantada onset, Garbí+instability, pressure drops, backing wind)
src/model/      → XGBoost training (TimeSeriesSplit CV + IsotonicRegression calibration) + prediction + ML-powered hourly forecast
src/notify/     → Telegram alerts with state machine (hysteresis: up=0.65, down=0.30) + regime alerts
src/feedback/   → JSONL prediction log, verification (60+ min later), feedback export for retraining
scripts/        → Entry points: download_history, build_dataset, train_model, predict_now, daily_summary, accuracy_report, backfill_lightning, backfill_ensemble, backfill_xema
config.py       → All constants, paths, thresholds, coordinates — single source of truth
```

**Key pattern — Rain Gate:** Expensive APIs (Meteocat XEMA, AEMET) are only queried when `rain_gate` signals are present (radar echo, ensemble agreement ≥ 25%, CAPE ≥ 800, lightning, or AEMET storm prob ≥ 10%). Always preserve this cost optimization.

**Key pattern — Graceful degradation:** Every `src/data/` module wraps API calls in try/except, logs warnings, and returns a dict with NaN values on failure. XGBoost handles NaN natively. Never let a single API failure crash the pipeline.

**Key pattern — Feature split:** 112 features defined in `FEATURE_COLUMNS` but 68 exist in historical training data (54 original + 6 ensemble + 6 sentinel + 2 Tramuntana interactions). The remaining 44 are real-time only (radar, AEMET, AEMET radar, SMC forecast, radar quadrants, echo bearing, forecast bias). Ensemble features are backfilled via `scripts/backfill_ensemble.py` (Open-Meteo Historical Forecast API, free, from Jan 2022). XEMA sentinel features are backfilled incrementally via `scripts/backfill_xema.py` (Meteocat API, 15 days/run). Lightning features (7) can be backfilled via `scripts/backfill_lightning.py`.

**Key pattern — Diagnostic logging:** Every prediction logs a full snapshot to `predictions_log.jsonl`: conditions, radar (RainViewer), AEMET (radar+forecast), sentinel (XEMA), ensemble, pressure_levels, wind_regime, bias, plus the complete 68-feature vector. This enables post-hoc analysis of missed predictions.

**Key pattern — Isotonic calibration:** Raw XGBoost probabilities are calibrated using IsotonicRegression fitted on out-of-fold predictions. This maps raw scores to true probabilities. The optimal F1 threshold (0.3513) is derived from the calibrated OOF predictions, not the default 0.5.

**Key pattern — ML-powered daily forecast:** The daily summary (7:00) runs `predict_hourly_forecast()` which applies the XGBoost model to each future hour using Open-Meteo forecast + pressure levels + SMC municipal forecast as input features. This replaces raw weather-code-based forecasts with actual ML predictions.

**Key pattern — Wind regimes at 850hPa:** Wind classification (Llevantada, Garbí, Ponent, Tramuntana, Migjorn) uses the synoptic 850hPa wind, not the 10m surface wind which is distorted by local orography (Montseny). The raw binary regime flags have zero model importance — the interaction terms (`llevantada_strength`, `llevantada_moisture`, `garbi_strength`, `tramuntana_strength`, `tramuntana_moisture`) carry the signal. Note: Tramuntana (N wind) accounts for 13.8% of rain events — it is NOT negligible despite being commonly labeled "dry".

**Key pattern — Feature pruning:** Binary threshold features (e.g., `cape_high`, `cold_500_moderate`) tend to have zero importance because the continuous source variable is always more informative. Prefer continuous features; only add binary indicators if XGBoost can't learn the threshold from the continuous value (very rare).

**Key pattern — Spatial radar:** RainViewer radar scans a 30km radius around Cardedeu (not just one pixel). Uses 850hPa wind direction to prioritize the upwind sector. Tracks storm movement across 6 frames (~1h) to estimate velocity and ETA. The `radar_nearest_echo_km` feature is far more informative than the point `radar_dbz`. Radar quadrant features (`radar_quadrant_max_dbz_N/E/S/W`, `radar_quadrant_coverage_N/E/S/W`) give the model directional awareness independent of wind regime. The nearest echo bearing is encoded cyclically (`radar_echo_bearing_sin`, `radar_echo_bearing_cos`).

**Key pattern — Regime change alerts:** The system alerts on atmospheric **transitions** (cause), not just rain probability (effect). Four types: Llevantada onset (E/SE wind + humidity ≥75%), Garbí + instability (SW wind + TT>44 or LI<-2), rapid pressure drop (≤-2 hPa in 3h), and backing wind with high humidity. Regime alerts have an independent 2h cooldown (`REGIME_COOLDOWN_MIN`) separate from rain alerts.

**Key pattern — Feedback loop:** Every prediction is logged to JSONL. 60+ min later, the system verifies against actual MeteoCardedeu station data. Verified predictions feed back into the training set on the next retrain cycle. This is how real-time-only features (radar, AEMET, SMC forecast) gradually enter the training data.

**Key pattern — Notification types:** Four distinct Telegram alerts: `rain_incoming` (prob crosses above 0.65), `rain_clearing` (prob drops below 0.30), `regime_change` (atmospheric setup shifts to historically rainy pattern), and `daily_summary` (morning 3-slot outlook at 7:00). State machine with hysteresis prevents notification spam.

**Key pattern — Daily forecast progressive disclosure:** The daily summary uses a dual-audience design. Top: outlook + ML time slots + next rain (general audience). Middle: compact conditions (temp + humidity + dewpoint, pressure with numeric 3h trend, wind + cloud cover). Bottom: "Detall tècnic" section with ensemble count, 850hPa wind/temp/RH, instability indices (TT/LI/VT), and smart radar summary. Radar display filters non-significant echoes (needs both <10km proximity AND >5% coverage to show).

**Key pattern — Slot datetime filtering:** When building hourly outlook slots (Matí/Tarda/Nit), always filter by explicit datetime ranges (not hour-only), to prevent tomorrow's cold morning hours from contaminating today's temperature ranges.

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
- **retrain**: Daily 3:00 Barcelona → download + build + train + git commit model

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `METEOCAT_API_KEY`, `AEMET_API_KEY`

**Meteocat API endpoints (all working):**
- XEMA (sentinel stations): `/xema/v1/variables/mesurades/{var}/{YYYY}/{MM}/{DD}`
- XDDE (lightning): `/xdde/v1/catalunya/{YYYY}/{MM}/{DD}/{HH}` (lowercase, hour required)
- Predicció (municipal forecast): `/pronostic/v1/municipalHoraria/080462`

**CI data persistence:** `predictions_log.jsonl`, `notification_state.json`, and `latest_prediction.json` are git-committed by each predict run. This gives permanent, queryable history of every prediction with full diagnostics. The `concurrency: predict-push` group prevents overlapping pushes.

**Note:** GitHub Actions free tier runs `*/15` cron but actual execution is ~hourly due to queue congestion. This is a known limitation — a VPS would give true 15-min resolution.
