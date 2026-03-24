# Nowcast Cardedeu — Project Guidelines

Hyperlocal rain nowcasting system for Cardedeu (Vallès Oriental) using XGBoost to correct global NWP models with local measurements from MeteoCardedeu.net.

## Architecture

```
src/data/       → 12 independent API clients (graceful degradation: each returns empty dict on failure)
src/features/   → Feature engineering (209 features: 163 with historical data, 46 real-time only) + wind regime detection
src/features/regime.py → Regime change detection (Llevantada onset, Garbí+instability, pressure drops, backing wind)
src/model/      → XGBoost training (TimeSeriesSplit CV + IsotonicRegression calibration) + prediction + ML-powered hourly forecast
src/ai/         → LLM narrative generation (GitHub Models gpt-4o-mini + OpenRouter free fallback) for daily summary and accuracy reports
src/notify/     → Telegram alerts with state machine (hysteresis: up=0.65, down=0.30) + regime alerts
src/feedback/   → JSONL prediction log, verification (60+ min later), feedback export for retraining
scripts/        → Entry points: download_history, build_dataset, train_model, predict_now, daily_summary, accuracy_report, backfill_lightning, backfill_ensemble, backfill_xema
config.py       → All constants, paths, thresholds, coordinates — single source of truth
```

**Key pattern — Rain Gate:** Quota-limited Meteocat APIs (XEMA, XDDE, Predicció) are only queried when `rain_gate` signals are present (radar echo, ensemble agreement ≥ 20%, CAPE ≥ 800, AEMET storm prob ≥ 10%, or AEMET radar echo). AEMET modules are called unconditionally (gated only by API key) because their output feeds INTO the rain gate decision. The gate logic lives in `src/model/predict.py`. Always preserve this cost optimization.

**Key pattern — Graceful degradation:** Every `src/data/` module wraps API calls in try/except, logs warnings, and returns a dict with NaN values on failure. XGBoost handles NaN natively. Never let a single API failure crash the pipeline.

**Key pattern — Feature split:** 209 features defined in `FEATURE_COLUMNS`. 163 have historical data (54 original + 6 ensemble + 6 sentinel + 2 Tramuntana interactions + 8 physics-based + 11 from 925/300hPa levels + 6 physics composites + 3 soil moisture + 1 CIN + 1 SST + 13 meteorological easy-wins: cloud layers, wet bulb, radiation breakdown, wind gusts, visibility, freezing level + 5 FP-killer features: nwp_rain_amount, nwp_rain_drying, nwp_rain_confirmed, afternoon_fp_risk, nwp_rain_dry_air + 23 Tier 1-3 features: 11 ERA5 surface (showers, ET0, soil temp, sunshine, 100m wind, snowfall), 8 upper-air (lifted index, GPH 850, RH 500, 700hPa wind), 4 derived composites (rain ending signal, cloud thickness, radiation conflict, moisture flux change) + 10 Tier 4 ERA5 atmospheric column: tcwv, tcwv_change_3h, tcwv_change_6h, boundary_layer_height, blh_change_3h, tcwv_blh_ratio, terrestrial_radiation, soil_moisture_28_to_100cm, soil_saturation_ratio, tcwv_monthly_anomaly + 7 Tier 5 blind-spot features: hours_since_sunrise, rh_700_change_3h, rh_700_change_6h, temp_850_change_3h, k_index, bulk_richardson + 9 Tier 6 NWP error detection & context: has_pressure_levels, rain_accum_24h, pressure_min_24h, cape_diurnal_weighted, nwp_rain_persistence_6h, nwp_rain_trend_3h, weather_code_change_3h, cloud_humidity_convergence, precip_trend_3h + 2 Tier 7 NWP decomposition: nwp_precip_severity (continuous WMO code intensity 0-5), cape (raw continuous CAPE, 44.5% coverage) + 2 Tier 8 storm tracking: radar_storm_velocity_ns (N-S movement), radar_storm_velocity_ew (E-W movement) + wind regime interactions: llevantada_strength, llevantada_moisture, garbi_strength, tramuntana_strength, tramuntana_moisture). The remaining 46 are real-time only (radar, AEMET, AEMET radar, SMC forecast, radar quadrants, echo bearing, forecast bias). The model trains on all 209 columns — 46 are NaN in historical data but XGBoost handles this natively. As feedback rows accumulate with real radar/lightning data, the model learns from them without code changes. Ensemble features are backfilled via `scripts/backfill_ensemble.py` (Open-Meteo Historical Forecast API, free, from Jan 2022). XEMA sentinel features are backfilled incrementally via `scripts/backfill_xema.py` (Meteocat API, 15 days/run). Lightning features (7) can be backfilled via `scripts/backfill_lightning.py`. CAPE/CIN are backfilled from the Historical Forecast API (from March 2021, ~45% coverage). SST is backfilled from NOAA ERDDAP OISST v2.1 (from 2015, ~98% coverage).

**Key pattern — Diagnostic logging:** Every prediction logs a full snapshot to `predictions_log.jsonl`: conditions, radar (RainViewer), AEMET (radar+forecast), sentinel (XEMA), ensemble, pressure_levels, wind_regime, bias, plus the complete 209-feature vector (all `FEATURE_COLUMNS`). This enables post-hoc analysis of missed predictions and ensures radar/lightning data feeds back into retraining.

**Key pattern — Isotonic calibration:** Raw XGBoost probabilities are calibrated using IsotonicRegression fitted on out-of-fold predictions. This maps raw scores to true probabilities. The optimal F1 threshold (~0.40) is derived from the calibrated OOF predictions, not the default 0.5.

**Key pattern — ML-powered daily forecast:** The daily summary (7:00) runs `predict_hourly_forecast()` which applies the XGBoost model to each future hour using Open-Meteo forecast + pressure levels + SMC municipal forecast as input features. This replaces raw weather-code-based forecasts with actual ML predictions.

**Key pattern — Wind regimes at 850hPa:** Wind classification (Llevantada, Garbí, Ponent, Tramuntana, Migjorn) uses the synoptic 850hPa wind, not the 10m surface wind which is distorted by local orography (Montseny). Surface wind and 850hPa wind only agree 26% of the time — never use surface wind as a fallback for 850hPa regime classification. XGBoost handles NaN natively for pre-2021 rows where 850hPa is unavailable. Direction ranges: Tramuntana 340°-60° (includes Gregal/NE), Llevantada 60°-150°, Migjorn 150°-190°, Garbí 190°-250°, Ponent 250°-340° — full 360° coverage with no gaps. The raw binary regime flags (`is_tramuntana`, `is_llevantada`, etc.) are NOT in FEATURE_COLUMNS — they have zero model importance. Only interaction terms enter the model: `llevantada_strength`, `llevantada_moisture`, `garbi_strength`, `garbi_moisture`, `tramuntana_strength`, `tramuntana_moisture`, `migjorn_strength`, `migjorn_moisture`, `ponent_strength`. Note: Tramuntana (N/NE wind) has a 4.8% rain rate — not negligible but far less than Llevantada (18.5%).

**Key pattern — Feature pruning:** Binary threshold features (e.g., `cape_high`, `cold_500_moderate`) tend to have zero importance because the continuous source variable is always more informative. Prefer continuous features; only add binary indicators if XGBoost can't learn the threshold from the continuous value (very rare). Feature pruning experiments (removing 61 zero-gain features) showed NO improvement — because 44 of those are real-time-only features (radar, lightning, AEMET, sentinel) that are NaN in historical data by design. As the feedback loop accumulates verified predictions with real radar/lightning values, these features will gain importance. Never prune real-time-only features.

**Key pattern — Spatial radar:** RainViewer radar scans a 30km radius around Cardedeu (not just one pixel). Uses 850hPa wind direction to prioritize the upwind sector. Tracks storm movement across 6 frames (~1h) to estimate velocity, direction, and ETA. Storm velocity is decomposed into N-S (`radar_storm_velocity_ns`, + = south) and E-W (`radar_storm_velocity_ew`, + = east) components so the model knows storm movement direction, not just speed. The `radar_nearest_echo_km` feature is far more informative than the point `radar_dbz`. Radar quadrant features (`radar_quadrant_max_dbz_N/E/S/W`, `radar_quadrant_coverage_N/E/S/W`) give the model directional awareness independent of wind regime. The nearest echo bearing is encoded cyclically (`radar_echo_bearing_sin`, `radar_echo_bearing_cos`). RainViewer tile coordinates for Cardedeu: zoom=8, tile=(129,95), pixel=(174,97).

**Key pattern — AEMET radar artifact filtering:** The AEMET radar regional/ba endpoint returns a pre-composited GIF image (480×530, ~45 palette colors) where geographic borders and coastlines use yellow (255,255,0) — the SAME color as 40 dBZ radar echoes. A color legend bar at y=506-517 also uses radar echo colors. The module in `src/data/aemet_radar.py` applies two-stage filtering: (1) morphological opening (3×3 cross erosion + dilation) to remove thin border lines (1-2px), (2) minimum connected-component cluster size filter (`AEMET_RADAR_MIN_ECHO_CLUSTER_PX=10` in config.py) to remove small junction artifacts. Real precipitation echoes are broad areas (>10px) that survive both filters. Without this filtering, yellow map borders near Cardedeu (~6km SSW) produce false positive radar echoes of 40 dBZ with ~4% coverage. Always preserve this artifact filtering pipeline when modifying AEMET radar image parsing.

**Key pattern — Regime change alerts:** The system alerts on atmospheric **transitions** (cause), not just rain probability (effect). Four types: Llevantada onset (E/SE wind + humidity ≥75%), Garbí + instability (SW wind + TT>44 or LI<-2), rapid pressure drop (≤-2 hPa in 3h), and backing wind with high humidity. Regime alerts have an independent 2h cooldown (`REGIME_COOLDOWN_MIN`) separate from rain alerts.

**Key pattern — Feedback loop:** Every prediction is logged to JSONL with all 209 `FEATURE_COLUMNS` (including radar/lightning/sentinel values). 60+ min later, the system verifies against actual MeteoCardedeu station data. Verified predictions feed back into the training set on the next retrain cycle. This is how real-time-only features (radar, AEMET, SMC forecast) gradually enter the training data. The model trains on all 209 features from day 1 (46 as NaN); as feedback rows accumulate, XGBoost learns from the newly populated columns automatically.

**Key pattern — Physics-based features:** Weather code decomposition (`wc_is_thunderstorm`, `wc_is_rain`, `wc_is_drizzle`) captures different precipitation physics. NWP error detection (`nwp_dry_conflict`, `nwp_wet_conflict`) flags when the NWP model disagrees with surface conditions. FP-killer features target false positives: `nwp_rain_amount` (continuous NWP rain mm, rank #4), `nwp_rain_drying` (NWP rain × humidity drop = drying signal), `nwp_rain_confirmed` (NWP rain × rain_accum_3h = already raining, rank #5), `afternoon_fp_risk` (afternoon × NWP rain × clear sky = convective FP), `nwp_rain_dry_air` (NWP rain × dew_point_depression = virga/evaporation, rank #9). `moisture_flux_850` and `moisture_flux_925` (wind×humidity at 850/925hPa) measure water transport at two levels; `theta_e_deficit` captures convective instability; `cape_change_3h` detects rapid destabilization. `inversion_925` (T925−T_sfc) detects boundary layer inversions that suppress convection. `deep_layer_shear` (850–300hPa wind difference) measures storm organization potential. `jet_speed_300` captures upper-level divergence from the jet stream. Physics composites combine multiple variables: `orographic_forcing` (wind⊥mountain×humidity), `frontal_passage` (pressure+wind+temp changes), `convective_composite` (instability×moisture×shear), `thermal_buildup` (diurnal heating), `low_level_convergence` (wind decel+humidity rise+pressure drop), `dry_intrusion_700` (850-700hPa humidity gap). Soil moisture (`soil_moisture_0_to_7cm`, `soil_moisture_7_to_28cm`, `soil_moisture_change_24h`) from ERA5 archive captures saturated soil amplifying precipitation. `convective_inhibition` (CIN) measures the energy barrier to convection — backfilled from Open-Meteo Historical Forecast API (April 2021+, ~44% coverage). `sst_med` (Mediterranean SST) captures sea surface temperature feeding moisture and convection — real-time from Open-Meteo Marine API, historical backfill from NOAA ERDDAP OISST v2.1 (2015-present, ~98% coverage). Tier 1 ERA5 features: `showers` (convective precipitation), `et0_fao_evapotranspiration` (FAO reference ET), `soil_temperature_0_to_7cm` + `soil_air_temp_diff` (soil-air coupling), `sunshine_duration` + `sunshine_accum_3h` (clear-sky proxy), `wind_speed_100m` + `boundary_layer_shear` + `wind_dir_shear_100m` (BL turbulence), `snowfall`. Tier 2 upper-air: `nwp_lifted_index` (direct instability from API), `gph_850` + `gph_850_change_3h` (synoptic pattern + tendencies), `rh_500` + `dry_intrusion_500` (mid-level dryness), `wind_700_speed` + `wind_700_dir` + `steering_onshore_700` (steering flow). Tier 3 derived: `rain_ending_signal` (rained recently but drying), `cloud_thickness_proxy` ((low+mid)/2 − high), `radiation_rain_conflict` (model predicts precip but radiation is high), `moisture_flux_change_3h` (moisture transport trend). Tier 4 atmospheric column (ERA5 100% coverage): `tcwv` + `tcwv_change_3h` + `tcwv_change_6h` (total column integrated water vapour — precipitable water, moisture loading), `boundary_layer_height` + `blh_change_3h` (PBL depth — convective mixing indicator), `tcwv_blh_ratio` (moisture per unit mixing depth — convective efficiency), `terrestrial_radiation` (longwave IR — nighttime cloud detection), `soil_moisture_28_to_100cm` + `soil_saturation_ratio` (deep soil saturation for runoff events), `tcwv_monthly_anomaly` (TCWV relative to ERA5 monthly climatology). Tier 5 blind-spot features: `hours_since_sunrise` (solar timing for convective initiation), `rh_700_change_3h` + `rh_700_change_6h` (mid-level drying trends — virga/entrainment detection), `temp_850_change_3h` (warm/cold advection proxy), `k_index` (moist layer depth — complements TT/VT), `bulk_richardson` (BRN = CAPE/shear² — storm mode discriminator).

**Key pattern — Vertical profile (5 pressure levels):** The model uses 5 standard pressure levels from Open-Meteo (925/850/700/500/300 hPa), available from March 2021. 925hPa: boundary layer (low-level jet, inversions, bulk moisture flux). 850hPa: synoptic flow (wind regime classification, moisture transport). 700hPa: dry air intrusion (capping layer). 500hPa: cold pool (lapse rate, VT/TT/LI indices). 300hPa: jet stream (dynamic trigger, deep-layer shear). Pre-2021 rows are NaN — XGBoost handles natively.

**Key pattern — NWP dominance (reduced):** ~70% of model gain comes from NWP-derived features (down from 84%→73%→70% across three rounds). Top features: `model_predicts_precip` (~30%), `nwp_precip_severity` (~21%), `weather_code` (~19%), `precipitation` (~9%). The continuous `nwp_precip_severity` (WMO code → 0-5 intensity scale) broke the binary `model_predicts_precip` dominance — the model now discriminates drizzle FPs (severity=1) from real rain (severity=2+). Combined `colsample_bytree=0.7 × colsample_bynode=0.7` forces diversity at both tree and split level. To further beat the NWP, independent observation data (radar, lightning) is needed — this requires the feedback loop to accumulate real-time data over months.

**Key pattern — NWP temporal consistency features:** These features detect when the NWP is wrong: `nwp_rain_persistence_6h` (persistent NWP rain = frontal = reliable; isolated = likely FP), `nwp_rain_trend_3h` (NWP ramping up = front approaching; backing off = residual FP), `weather_code_change_3h` (sudden WMO code jump = less reliable), `cloud_humidity_convergence` (clouds + humidity rising together = rain developing), `precip_trend_3h` (precipitation intensifying or weakening). These entered the top 15 features immediately.

**Key pattern — Hyperparameter tuning after feature expansion:** Current optimal hyperparams (tuned 2026-03-22 via 3-round grid search): `n_estimators=1200`, `max_depth=7`, `learning_rate=0.012`, `subsample=0.75`, `colsample_bytree=0.7`, `colsample_bynode=0.7`, `min_child_weight=6`, `gamma=0.15`, `reg_alpha=0.3`, `reg_lambda=2.0`, `early_stopping_rounds=96`. No `scale_pos_weight` — isotonic calibration + threshold search handles class imbalance better than direct upweighting. Key insight: combined `colsample_bytree=0.7 × colsample_bynode=0.7` provides dual-level diversity — each tree sees 70% of features AND each split point sees 70% of the tree's features. This broke `model_predicts_precip` dominance from 54%→30%. `max_depth=7` captures complex NWP×surface interactions. Stronger regularization (gamma, reg_alpha, reg_lambda all increased) prevents overfitting with deeper, more diverse trees. Cal F1 progression: 0.7033→0.7061→0.7070. `eval_metric=logloss` (not aucpr) — optimizes probability quality for isotonic calibration pipeline.

**Key pattern — Notification types:** Four distinct Telegram alerts: `rain_incoming` (prob crosses above 0.65), `rain_clearing` (prob drops below 0.30), `regime_change` (atmospheric setup shifts to historically rainy pattern), and `daily_summary` (morning 3-slot outlook at 7:00 with optional AI narrative). State machine with hysteresis prevents notification spam.

**Key pattern — Daily forecast progressive disclosure:** The daily summary uses a dual-audience design. Top: outlook + ML time slots + next rain (general audience). Middle: compact conditions (temp + humidity + dewpoint, pressure with numeric 3h trend, wind + cloud cover). Bottom: "Detall tècnic" section with ensemble count, 850hPa wind/temp/RH, instability indices (TT/LI/VT), and smart radar summary. Radar display filters non-significant echoes (needs both <10km proximity AND >5% coverage to show).

**Key pattern — Slot datetime filtering:** When building hourly outlook slots (Matí/Tarda/Nit), always filter by explicit datetime ranges (not hour-only), to prevent tomorrow's cold morning hours from contaminating today's temperature ranges.

**Key pattern — AI narrative enrichment:** The daily summary (7:00) and weekly accuracy report include an optional LLM-generated narrative paragraph in Catalan. Uses GitHub Models gpt-4o-mini (free via GITHUB_TOKEN in Actions) as primary, OpenRouter free models as fallback. The AI call is NEVER in the critical alert path (predict_now.py / rain alerts) — only in low-frequency scripts (1 call/day, 1 call/week). All AI calls are wrapped in try/except with graceful fallback to the existing template output. The enricher module is at `src/ai/enricher.py` following the dual-provider retry+fallback pattern from gencat-cultural-agenda.

## Build and Test

```bash
# Install
pip install -r requirements.txt   # Python 3.12

# Train pipeline (first time)
python scripts/download_history.py  # Fetch 12 years from Open-Meteo + NOAA + pressure levels + CAPE/CIN + SST
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
- **Features:** Register new features in `src/features/engineering.py` and add to `FEATURE_COLUMNS`. All 209 features are in the model — real-time-only features are NaN in historical data but present as columns. XGBoost learns from them as feedback data accumulates. CIN, SST, cloud layers, wet bulb, radiation, wind gusts, visibility, freezing level, FP-killer interactions, and Tier 1-3 features (ERA5 surface, upper-air, derived composites) all have historical backfill.
- **Config:** All thresholds, paths, and coordinates live in `config.py` — never hardcode magic numbers in modules.
- **AI narratives:** LLM calls live in `src/ai/enricher.py`. NEVER add AI calls to predict_now.py or the rain alert path. Only use in low-frequency scripts (daily_summary.py, accuracy_report.py). Always wrap in try/except with graceful fallback.
- **Feature design:** Prefer continuous features over binary indicators. Binary threshold features (e.g., `cape_high`, `cold_500_moderate`) consistently show zero importance because XGBoost can learn any threshold from the continuous source. Use interaction terms (regime × magnitude) for wind patterns. Run `python scripts/feature_analysis.py` after adding features to verify they contribute.

## CI/CD

GitHub Actions (`.github/workflows/nowcast.yml`):
- **predict**: Every 10 min (6–23h Barcelona) via cron-job.org → `predict_now.py`
- **daily_summary**: 7:00 Barcelona → `daily_summary.py`
- **accuracy_report**: Monday 8:00 → `accuracy_report.py`
- **retrain**: Daily 3:00 Barcelona → download + build + train + git commit model

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `METEOCAT_API_KEY`, `AEMET_API_KEY`
`AI_GITHUB_TOKEN` uses the automatic `GITHUB_TOKEN` (no extra secret needed). Optional: `AI_OPENROUTER_KEY` for fallback to OpenRouter free models.

**Meteocat API endpoints (all working):**
- XEMA (sentinel stations): `/xema/v1/variables/mesurades/{var}/{YYYY}/{MM}/{DD}`
- XDDE (lightning): `/xdde/v1/catalunya/{YYYY}/{MM}/{DD}/{HH}` (lowercase `catalunya`, hour required)
- Predicció (municipal forecast): `/pronostic/v1/municipalHoraria/080462`
- Quota check: `/quotes/v1/consum-actual`

**Meteocat API quotas (separate per service, monthly, reset 1st 00:00 UTC):**
- XEMA: 750 calls/month
- XDDE: 250 calls/month
- Predicció: 100 calls/month
- All Meteocat calls are behind the rain gate (only fire when rain signals detected)
- Backfill scripts check quota via `get_remaining()` before running

**CI data persistence:** `predictions_log.jsonl`, `notification_state.json`, and `latest_prediction.json` are git-committed by each predict run. This gives permanent, queryable history of every prediction with full diagnostics. The `concurrency: predict-push` group prevents overlapping pushes.

**Note:** Predictions are triggered every 10 min via cron-job.org → workflow_dispatch (not GitHub's native cron, which has ~hourly queue congestion).
