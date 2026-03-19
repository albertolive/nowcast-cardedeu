---
description: "Register a new feature in the engineering pipeline and wire it into training or real-time prediction. Use when adding a derived meteorological feature, interaction term, or new input signal to XGBoost."
---

# Add Feature

Add a new derived feature to the XGBoost prediction pipeline.

## Inputs

- **Feature name(s)** (snake_case, e.g., "fog_risk_index"): `${input:featureNames}`
- **Source columns** (existing columns needed to compute this): `${input:sourceColumns}`
- **Available in historical data?** (yes = in Open-Meteo/training, no = real-time only): `${input:hasHistorical}`
- **Description** (what it captures meteorologically): `${input:description}`

## Steps

### 1. Add the computation function

In `src/features/engineering.py`, add a private function following the existing pattern:

```python
def _add_yourfeature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Descripció en català del que captura aquesta feature."""
    df = df.copy()
    if "source_column" in df.columns:
        df["your_feature"] = ...  # Computation
        # Optional: derived/interaction features
        df["your_feature_change_3h"] = df["your_feature"].diff(3)
    return df
```

Rules:
- Always `df = df.copy()` first
- Guard with `if "col" in df.columns:` — missing columns stay NaN (XGBoost handles it)
- Use `pd.to_numeric(df[col], errors="coerce")` for safety
- Use `np.nan` for undefined values, never `None` or `0` for missing numerics
- Catalan docstring explaining the meteorological meaning

### 2. Wire into the pipeline

**If the feature has historical data** (available in Open-Meteo hourly training data):

In `build_features_from_hourly()`, add the call in the appropriate position:
```python
def build_features_from_hourly(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing calls ...
    df = _add_yourfeature_features(df)  # Add in logical order
    return df
```

**If the feature is real-time only** (radar, sentinel, ensemble, lightning, AEMET):

The feature columns will be NaN in training data. XGBoost handles this natively. Add the computation either:
- In the data module's `fetch_*()` return dict (if it's a raw measurement)
- In the `_add_*_features()` function (if it's derived from multiple sources)
- Wire it in `scripts/predict_now.py` where real-time features are merged

### 3. Register in FEATURE_COLUMNS

Add the new feature name(s) to the `FEATURE_COLUMNS` list at the bottom of `src/features/engineering.py`:

```python
FEATURE_COLUMNS = [
    # ... existing features ...
    # Your category
    "your_feature",
    "your_feature_change_3h",
]
```

Group with related features and add a comment for the category.

### 4. Verify feature alignment

The model uses `models/feature_names.json` to align features at prediction time. After adding features:
- **Retrain** (`python scripts/train_model.py`) to update `feature_names.json`
- New features in real-time that weren't in training → XGBoost treats as NaN (safe)
- Features in training that are missing in real-time → filled with NaN by `predict.py`

### 5. Config thresholds (if applicable)

If the feature uses thresholds (e.g., "moderate if > X"):
- Add `YOUR_FEATURE_THRESHOLD = value` to `config.py`
- Reference as `config.YOUR_FEATURE_THRESHOLD` in the computation
- Never hardcode thresholds in the feature function

## Feature Design Guidelines

- **Prefer continuous over binary**: Binary threshold features (`cape_high`, `cold_500_moderate`, `li_very_unstable`) consistently show zero importance because XGBoost learns thresholds from the continuous source variable (`cape`, `temp_500`, `li_index`). Only create binary indicators if XGBoost provably can't learn the threshold (very rare).
- **Interaction terms over raw flags**: Raw binary wind regime flags (`is_llevantada`, `is_tramuntana`) have zero importance. Interaction terms that combine regime × magnitude carry the signal: `llevantada_strength = is_llevantada × wind_speed`, `llevantada_moisture = is_llevantada × humidity`.
- **Cyclic encoding**: Use sin/cos for periodic features (hour, month): `hour_sin = sin(2π × hour/24)`
- **Temporal derivatives**: Add `_change_1h`, `_change_3h`, `_change_6h` for trending signals (pressure, humidity, VPD)
- **Prefixing**: All features from the same source share a prefix (e.g., `radar_*`, `sentinel_*`, `lightning_*`)
- **Wind regimes at 850hPa**: Wind classification must use 850hPa synoptic wind (not 10m surface wind distorted by Montseny orography). Fallback to 10m only when 850hPa is unavailable.
- **Always validate**: Run `python scripts/feature_analysis.py` after retrain. If a new feature shows zero gain/splits, reconsider it.

## Validation Checklist

- [ ] Function uses `df.copy()` and returns the modified df
- [ ] Column existence checked before computation
- [ ] Feature name(s) added to `FEATURE_COLUMNS`
- [ ] Thresholds in `config.py`, not hardcoded
- [ ] Catalan docstring explaining meteorological meaning
- [ ] After retrain, new features appear in `models/feature_names.json`
- [ ] `python scripts/feature_analysis.py` shows non-zero gain for the feature
- [ ] Prefer continuous features — avoid binary indicators that duplicate continuous source
