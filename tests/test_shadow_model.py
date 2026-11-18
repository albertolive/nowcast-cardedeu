import json
import logging
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import config
from scripts import train_model as cli
from src.feedback import logger as feedback_log
from src.model import predict, train


CUTOFF = pd.Timestamp("2026-11-01T00:00:00Z")


def _parquets():
    times = pd.date_range(CUTOFF - pd.Timedelta(days=49), CUTOFF + pd.Timedelta(days=2), freq="2h")
    wet = ((np.arange(len(times)) % 6) == 1).astype(int)
    history = pd.DataFrame({
        "datetime": times.tz_convert("Europe/Madrid").tz_localize(None),
        "precipitation": 1 - wet, "will_rain": wet,
    })
    feedback = pd.DataFrame({
        "datetime": times.tz_localize(None) + pd.Timedelta(minutes=1),
        "verified": True, "actual_rain": wet, "will_rain": wet,
        "precipitation": 1 - wet, "radar_dbz": np.where(wet, 50.0, 0.0),
    })
    return history.sample(frac=1, random_state=1), feedback.sample(frac=1, random_state=2)


def test_splits_use_issue_time_station_labels_and_a_time_embargo():
    history, feedback = _parquets()
    history["local_rain_xema"] = 2.
    feedback["local_rain_xema"] = 0.1
    legacy = feedback.iloc[[0]].assign(datetime=pd.Timestamp("2026-05-09T15:00:00"))
    feedback = pd.concat([feedback, legacy])
    no_vector = feedback.iloc[[2]].assign(datetime=pd.Timestamp("2026-10-25T00:03"),
                                          radar_dbz=None, precipitation=None, local_rain_xema=None, probability_pct=99)
    feedback = pd.concat([feedback, no_vector])
    boundaries = [CUTOFF - pd.Timedelta(days=28), CUTOFF - pd.Timedelta(days=14), CUTOFF]
    for boundary in boundaries:
        for minutes in (59, 60, 61):
            feedback = pd.concat([feedback, feedback.iloc[[0]].assign(
                datetime=(boundary - pd.Timedelta(minutes=minutes)).tz_localize(None))])
    early = history.loc[history["datetime"] == pd.Timestamp("2026-09-17T02:00:00")]
    assert len(early) == 1
    clash = feedback.iloc[[0]].assign(datetime=pd.Timestamp("2026-09-17T00:00"), radar_dbz=42)
    splits, metadata = train._shadow_splits(history, pd.concat([feedback, clash]), CUTOFF)
    for name, end in zip(("train", "calibration", "holdout"), boundaries):
        frame = splits[name]
        assert frame.index.is_monotonic_increasing
        assert frame.index.max() + pd.Timedelta(hours=1) < end
        assert end - pd.Timedelta(minutes=59) not in frame.index
        assert end - pd.Timedelta(minutes=60) not in frame.index
        assert end - pd.Timedelta(minutes=61) in frame.index
    assert splits["train"].loc[pd.Timestamp("2026-09-17T00:00Z"), "radar_dbz"] == 42
    assert splits["train"]["local_rain_xema"].dropna().eq(0.1).all()
    assert splits["train"].loc[pd.Timestamp("2026-09-18T00:00Z")].isna()["local_rain_xema"]
    assert metadata["feedback_before_utc_migration"] == 1
    assert pd.Timestamp("2026-05-09T15:00Z") not in splits["train"].index
    for name in ("calibration", "holdout"):
        frame = splits[name]
        assert frame["radar_dbz"].notna().all()
        assert set(frame["will_rain"]) == {0, 1}
        assert (frame["will_rain"] == 1 - frame["precipitation"]).all()
        assert metadata["splits"][name]["station_rows"] == len(frame)
    assert metadata["feedback_without_predictors"] == 1
    assert metadata["training_cutoff"] == "2026-11-01T00:00:00Z"


def test_bad_recent_coverage_and_one_class_are_rejected():
    history, feedback = _parquets()
    recent = feedback["datetime"] >= (CUTOFF - pd.Timedelta(days=14)).tz_localize(None)
    empty = feedback.copy()
    empty.loc[recent, ["radar_dbz", "precipitation"]] = np.nan
    with pytest.raises(ValueError, match="99 %"):
        train._shadow_splits(history, empty, CUTOFF)
    feedback.loc[recent, ["will_rain", "actual_rain"]] = 0
    with pytest.raises(ValueError, match="holdout: cobertura insuficient"):
        train._shadow_splits(history, feedback, CUTOFF)


def test_cli_round_trip_no_archive_no_live_writes_and_failed_run_keeps_candidate(tmp_path, monkeypatch):
    history, feedback = _parquets()
    processed = tmp_path / "processed"
    processed.mkdir()
    history.to_parquet(processed / "training_dataset.parquet", index=False)
    source = processed / "feedback_verified.parquet"
    feedback.to_parquet(source, index=False)
    original = source.read_bytes()
    models = tmp_path / "models"
    models.mkdir()
    live = models / "xgboost_nowcast.json"
    live.write_text("unchanged")
    pinned = models / "shadow"
    pinned.mkdir()
    (pinned / "metrics.json").write_text("field trial")
    monkeypatch.setattr(cli, "FEEDBACK_TRAINING_PATH", source)
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(train, "_new_classifier", lambda: xgb.XGBClassifier(
        n_estimators=10, max_depth=2, learning_rate=0.4, early_stopping_rounds=3, n_jobs=1, random_state=42))
    args = ["--shadow", "--before", "2026-11-01T00:00:00Z"]
    cli.main(args)
    shadow = models / "shadow-next"
    files = {f.name: f.read_bytes() for f in shadow.iterdir()}
    assert set(files) == {"xgboost_nowcast.json", "calibrator.json", "feature_names.json", "metrics.json"}
    model, names, calibrator, threshold = train.load_model(model_dir=shadow)
    p = calibrator.predict(model.predict_proba(feedback.reindex(columns=names))[:, 1])
    assert np.array_equal(p >= threshold, feedback["actual_rain"])
    metrics = json.loads(files["metrics.json"])
    assert metrics["model_id"] == model.nowcast_model_id
    assert metrics["holdout_f1"] == 1
    assert source.read_bytes() == original
    assert live.read_text() == "unchanged"
    assert set(models.iterdir()) == {shadow, live, pinned}
    assert (pinned / "metrics.json").read_text() == "field trial"

    (shadow / "calibrator.json").write_text('{"x":[0,1],"y":[0.2,0.3]}')
    with pytest.raises(ValueError, match="empremta"):
        train.load_model(model_dir=shadow)
    (shadow / "calibrator.json").write_bytes(files["calibrator.json"])
    (shadow / "metrics.json").write_text(json.dumps({**metrics, "optimal_threshold": threshold / 2}))
    with pytest.raises(ValueError, match="empremta"):
        train.load_model(model_dir=shadow)
    (shadow / "metrics.json").write_bytes(files["metrics.json"])

    cutoff_or_later = feedback.datetime >= (CUTOFF - pd.Timedelta(hours=1)).tz_localize(None)
    feedback.loc[cutoff_or_later, ["actual_rain", "will_rain"]] = 1 - feedback.loc[
        cutoff_or_later, ["actual_rain", "will_rain"]]
    feedback.to_parquet(source, index=False)
    cli.main(args)
    assert {f.name: f.read_bytes() for f in shadow.iterdir()} == files

    holdout = feedback.datetime >= (CUTOFF - pd.Timedelta(days=14)).tz_localize(None)
    feedback.loc[holdout, ["precipitation", "radar_dbz"]] = [1.0, 0.0]
    feedback.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="no distingeix la pluja"):
        cli.main(args)
    assert {f.name: f.read_bytes() for f in shadow.iterdir()} == files

    monkeypatch.setattr(train, "save_model", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        train.save_shadow_model(model, metrics, calibrator)
    assert {f.name: f.read_bytes() for f in shadow.iterdir()} == files
    assert live.read_text() == "unchanged"


def test_portable_calibrator_clips_and_interpolates_like_fitted_estimator():
    fitted = train.IsotonicRegression(out_of_bounds="clip").fit([0.1, 0.2, 0.4, 0.7, 0.9], [0, 1, 0, 1, 1])
    portable = train._ClippedIsotonic(fitted.X_thresholds_, fitted.y_thresholds_)
    probe = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(portable.predict(probe), fitted.predict(probe), atol=1e-7)
    with pytest.raises(ValueError, match="knots"):
        train._ClippedIsotonic([0.7, 0.1], [0, 1])


def test_published_candidate_is_the_one_the_field_trial_scores():
    model, names, calibrator, _ = train.load_model(model_dir=Path(config.MODELS_DIR) / "shadow")
    assert model.nowcast_model_id == config.WARNING_TRIAL_MODEL_ID
    inputs = pd.DataFrame([{"temperature_2m": 14., "relative_humidity_2m": 88.}], columns=names)
    probability, = calibrator.predict(model.predict_proba(inputs)[:, 1])
    assert 0 <= probability <= 1


def test_issued_shadow_is_optional_isolated_and_uses_same_floors(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "AEMET_API_KEY", "")
    monkeypatch.setattr(config, "METEOCAT_API_KEY", "")
    radar = {"radar_dbz": 12, "radar_has_echo": True, "radar_rain_rate": 0.2, "radar_approaching": False}
    sources = {}
    for name, data in {
        "fetch_series": pd.DataFrame(), "fetch_forecast": pd.DataFrame(), "fetch_pressure_levels": {},
        "fetch_pressure_levels_hourly": pd.DataFrame(), "fetch_sst_forecast": {}, "fetch_meteocat_radar": {},
        "fetch_ensemble_agreement": {}, "fetch_radar_at_cardedeu": radar, "fetch_latest": {},
        "fetch_sentinel_latest": {}, "compute_lightning_features": {},
    }.items():
        sources[name] = Mock(return_value=data)
        monkeypatch.setattr(predict, name, sources[name])
    monkeypatch.setattr(predict, "build_features_from_realtime", lambda *_: pd.DataFrame({"temperature_2m": [19]}))
    monkeypatch.setattr(predict, "compute_forecast_bias", lambda *_: {})
    monkeypatch.setattr(predict, "compute_sentinel_features", lambda *_: {})
    monkeypatch.setattr(predict, "compute_prediction_drivers", lambda *_: [])
    live = Mock()
    live.predict_proba.return_value = np.array([[0.95, 0.05]])
    candidate = Mock(nowcast_model_id="candidate-id")
    candidate.predict_proba.return_value = np.array([[0.9, 0.1]])
    calibrator = Mock()
    calibrator.predict.return_value = [0.2]

    def load(model_dir=None):
        if model_dir is None:
            return live, ["temperature_2m"], None, 0.5
        return candidate, ["radar_dbz", "temperature_2m", "rh_700"], calibrator, 0.01

    monkeypatch.setattr(predict, "load_model", load)
    baseline = predict.predict_now()
    assert "shadow_probability" not in baseline
    (tmp_path / "shadow").mkdir()
    result = predict.predict_now()
    assert result["shadow_probability"] == result["probability"] == 0.45
    assert result["will_rain"] is False
    X = candidate.predict_proba.call_args.args[0]
    assert list(X) == ["radar_dbz", "temperature_2m", "rh_700"]
    assert list(X.iloc[0, :2]) == [12, 19] and pd.isna(X.iloc[0, 2])

    monkeypatch.setattr(feedback_log, "PREDICTIONS_LOG", str(tmp_path / "issued.jsonl"))
    feedback_log.log_prediction(result)
    entry, = feedback_log.load_predictions_log()
    assert entry["shadow_probability"] == 0.45 and entry["shadow_model_id"] == "candidate-id"
    calibrator.predict.return_value = [0.99]
    disagreement = predict.predict_now()
    assert disagreement["shadow_probability"] == 0.99
    candidate.predict_proba.side_effect = ValueError("broken model")
    with caplog.at_level(logging.WARNING):
        broken = predict.predict_now()
    assert "shadow_probability" not in broken and "broken model" in caplog.text
    for snapshot in (result, disagreement, broken):
        for key in ("probability", "probability_pct", "raw_probability", "will_rain",
                    "rain_category", "threshold", "confidence", "physical_adjustments"):
            assert snapshot[key] == baseline[key]
    assert all(fetch.call_count == 4 for fetch in sources.values())


@pytest.mark.parametrize("invalid", [np.nan, -0.5])
def test_invalid_calibration_is_rejected_before_physical_floors(tmp_path, monkeypatch, caplog, invalid):
    (tmp_path / "shadow").mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    model = Mock(nowcast_model_id="candidate-id")
    model.predict_proba.return_value = np.array([[0.7, 0.3]])
    calibrator = Mock()
    calibrator.predict.return_value = [invalid]
    monkeypatch.setattr(predict, "load_model", lambda **_: (model, ["radar_dbz"], calibrator, 0.5))
    floor = Mock(return_value=(0.65, []))
    monkeypatch.setattr(predict, "_apply_physical_constraints", floor)
    with caplog.at_level(logging.WARNING):
        assert predict._shadow_score(pd.DataFrame({"radar_dbz": [42.0]}), {}) == {}
    floor.assert_not_called()
    assert "abans de l'ajust físic" in caplog.text
