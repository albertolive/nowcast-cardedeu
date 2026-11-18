from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_rain_warnings as ev


START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def t(minutes):
    return START + timedelta(minutes=minutes)


def day_csv(directory, day, reads):
    path = directory / f"{day}.csv"
    pd.DataFrame(reads, columns=["datetime", "PREC"]).to_csv(path, index=False)
    return day, path


def readings(first, last, tips=(), missing=()):
    return [(t(i).astimezone(ev.MADRID).replace(tzinfo=None), 0.2 if i in tips else 0)
            for i in range(first, last + 1, 5) if i not in missing]


def test_five_minute_gauge_intervals_and_gaps(tmp_path):
    local_day = date(2026, 11, 1)
    path = day_csv(tmp_path, local_day, readings(0, 420, tips=[75, 135, 250, 380],
                                                missing=[240, 245]))
    spans, episodes, files = ev.gauge_episodes([path, (local_day + timedelta(days=1), tmp_path / "absent")])
    assert spans == [(t(0), t(235)), (t(250), t(420))]
    assert files["2026-11-02"] is None
    assert episodes == [
        ev.Episode(t(70), t(75), t(135), 0.4, True, t(75)),
        ev.Episode(t(235), t(250), t(250), 0.2, False, t(250)),
        ev.Episode(t(375), t(380), t(380), 0.2, False, t(380)),
    ]


def test_midnight_and_dst_are_not_guessed(tmp_path):
    a = day_csv(tmp_path, date(2026, 11, 1), readings(1240, 1375, tips=[1370]))
    b = day_csv(tmp_path, date(2026, 11, 2), readings(1380, 1470, tips=[1380, 1385]))
    spans, episodes, _ = ev.gauge_episodes([a, b])
    assert len(spans) == 2
    # First increment of the second day is unknowable. No second storm.
    assert episodes == [ev.Episode(t(1365), t(1370), t(1385), 0.4, False, t(1370))]
    fold = day_csv(tmp_path, date(2026, 10, 25), [
        ("2026-10-25 01:55", 0), ("2026-10-25 02:00", 0.2), ("2026-10-25 02:00", 0.4),
        ("2026-10-25 03:00", 0), ("2026-10-25 03:05", 0)])
    spans, episodes, _ = ev.gauge_episodes([fold])
    assert len(spans) == 1 and not episodes
    assert spans[0][0] == datetime(2026, 10, 25, 2, tzinfo=timezone.utc)


class Model:
    def predict_proba(self, x):
        assert list(x.columns) == ["signal", "gated"] and x.gated.isna().all()
        return np.column_stack([1 - x.signal, x.signal])


class Calibrator:
    def predict(self, raw):
        return raw / 2


def snapshot(minute, **changes):
    return dict(dict(timestamp=ev.iso(t(minute)), feature_vector={"gated": None, "signal": 0.8},
                     probability=0.75, physical_adjustments=[], station_available=True,
                     station_raining_now=False), **changes)


def test_aligned_calibrated_replay_uses_only_recorded_floors_or_matching_shadow():
    rows = [snapshot(0), snapshot(10, physical_adjustments=["station"]),
            snapshot(20, shadow_model_id="other", shadow_probability=1.0),
            snapshot(30, shadow_model_id="same", shadow_probability=0.9),
            snapshot(40, station_available=False), snapshot(50, station_raining_now=None),
            snapshot(60, feature_vector={"signal": 0.8}),
            snapshot(70, feature_vector={"gated": None, "signal": None}),
            snapshot(80, physical_adjustments=["low floor"], probability=0.2),
            snapshot(90, physical_adjustments=None)]
    frame, counts = ev.aligned_scores(rows, (Model(), ["signal", "gated"], Calibrator(), 0.12), "same")
    assert list(frame.shadow) == [0.4, 0.75, 0.4, 0.9, 0.4]
    assert list(frame.index) == [t(i) for i in (0, 10, 20, 30, 80)]
    assert counts["station_unknown"] == 2 and counts["logged_shadow"] == 1
    assert counts["archived"] == 10 and counts["aligned"] == 5


def test_hidden_floor_on_already_wet_station_and_sensor_freshness_guards():
    wet = snapshot(0, station_raining_now=True, probability=0.9)
    assert ev.archived_floor(wet) == 0.75
    r = snapshot(0, radar=dict(frames_frozen=True, has_echo=True, dbz=39, storm_eta_min=3),
                 aemet_radar=dict(nearest_echo_km=4, max_dbz_20km=42, same_frame_streak=9))
    r["feature_vector"]["radar_storm_approaching"] = True
    assert ev.archived_floor(r) == 0
    r["radar"]["frames_frozen"] = False
    assert ev.archived_floor(r) == 0.65
    r["radar"].update(has_echo=False, max_dbz_20km=30)
    assert ev.archived_floor(r) == pytest.approx(0.38)
    r["feature_vector"].update(radar_storm_approaching=False, lightning_count_15km=12)
    assert ev.archived_floor(r) == 0  # Three-hour count is not a one-hour count.
    r["physical_adjustments"] = ["historical rule fired"]
    assert ev.archived_floor(r) == r["probability"]


def test_state_transitions_both_directions_share_cooldown_and_reset_only_after_sync():
    times = list(range(0, 150, 10)) + list(range(170, 261, 10))
    high = {70, 90, 110, 120, 130, 170, 180, 260}
    p = [0.65 if i in high else 0.30 for i in times]
    frame = pd.DataFrame(dict(issued=p, shadow=p, persistence=[v == 0.65 for v in p]),
                         index=[t(i) for i in times])
    spans, alerts = ev.replay(frame)
    assert spans == [(t(60), t(140)), (t(250), t(260))]
    for name in ev.NAMES:
        assert [when for when, _ in alerts[name]] == [t(70), t(130), t(260)]


def test_storm_warnings_are_unique_dry_and_advance_not_confirmation_or_false():
    spans = [(t(0), t(600))]
    episodes = [ev.Episode(t(115), t(120), t(130), 0.6, True, t(120)),
                ev.Episode(t(205), t(210), t(210), 0.2, True, t(210)),
                ev.Episode(t(565), t(570), t(570), 0.2, False, t(570))]
    alerts = dict(issued=[(t(70), False), (t(100), False), (t(165), False), (t(225), False)],
                  shadow=[(t(65), True), (t(118), False), (t(125), True), (t(275), True),
                          (t(563), False), (t(580), False)],
                  persistence=[(t(118), True)])
    r = ev.score(spans, episodes, spans, alerts, t(0), t(600))
    issued, shadow, base = [r["comparison"][n] for n in ev.NAMES]
    assert issued["storms"] == issued["advance"] == 2
    assert issued["alert_counts"] == {"advance": 2, "repeat": 1, "false": 1}
    assert issued["lead_minutes_median"] == 42.5 and issued["false_alerts_per_week"] == 18.667
    assert shadow["missed_advance"] == 2 and shadow["confirmation_only"] == 1
    assert shadow["alert_counts"] == dict(station_disagreement=2, timing_uncertain=1,
                                          confirmation=1, unscored_episode=1, unscored_window=1)
    assert base["advance"] == 0 and base["confirmation_only"] == base["missed_entirely"] == 1
    assert r["episodes"][2]["excluded"] == "station_gap_or_boundary"
    assert r["exposure_days"] == 0.375


def test_replay_abstains_across_archive_gap_and_requires_whole_episode():
    episodes = [ev.Episode(t(80), t(85), t(155), 1.4, True, t(85)),
                ev.Episode(t(200), t(205), t(207), 0.2, True, t(205))]
    events = {n: [(t(45), False)] for n in ev.NAMES}
    r = ev.score([(t(0), t(300))], episodes, [(t(0), t(140)), (t(190), t(290))],
                 events, t(0), t(300))
    assert [s["excluded"] for s in r["episodes"]] == ["archive_gap_or_warmup"] * 2
    assert r["comparison"]["issued"]["storms"] == r["comparison"]["issued"]["false_alerts"] == 0
    missing = ev.score([], [], [(t(0), t(300))], events, t(0), t(300))
    assert missing["comparison"]["issued"]["false_alerts_per_week"] is None
    assert missing["comparison"]["issued"]["alert_counts"] == {"unscored_window": 1}


def test_known_onset_counts_before_gap_but_not_when_gap_precedes_qualifying_rain(tmp_path):
    day = date(2026, 11, 1)
    sample = readings(0, 330, missing=[100, 105, 240, 245])
    for i, (stamp, _) in enumerate(sample):
        minute = (stamp.replace(tzinfo=ev.MADRID) - START).total_seconds() / 60
        if minute in (80, 85, 235, 250):
            sample[i] = stamp, 0.1
    spans, episodes, _ = ev.gauge_episodes([day_csv(tmp_path, day, sample)])
    alerts = dict(issued=[(t(77), False)], shadow=[(t(40), False)], persistence=[(t(78), True)])
    r = ev.score(spans, episodes, [(t(0), t(330))], alerts, t(0), t(330))
    assert r["comparison"]["shadow"]["storms"] == 0
    assert r["episodes"][0]["onset_excluded"] is None
    assert r["episodes"][1]["onset_excluded"] == "station_gap_or_boundary"
    assert r["episodes"][0]["qualified_at"] == ev.iso(t(85))
    assert r["onset_comparison"]["shadow"]["advance"] == 1
    assert r["onset_comparison"]["shadow"]["lead_minutes_median"] == 35
    assert r["onset_comparison"]["issued"]["timing_uncertain"] == 1
    assert r["onset_comparison"]["persistence"]["advance"] == 0
    assert r["alerts"]["shadow"] == [dict(time=ev.iso(t(40)), station_wet=False,
                                          kind="unscored_episode", onset_kind="advance")]


def test_cli_refuses_training_overlap_before_accessing_model_or_station(tmp_path, capsys):
    (tmp_path / "metrics.json").write_text(json.dumps({"training_cutoff": "2026-11-01T00:01:00Z"}))
    with pytest.raises(SystemExit) as exc:
        ev.main(["--start", "2026-11-01", "--end", "2026-11-02",
                 "--model-dir", str(tmp_path), "--fetch-station"])
    assert exc.value.code == 2 and "overlaps training_cutoff" in capsys.readouterr().err
