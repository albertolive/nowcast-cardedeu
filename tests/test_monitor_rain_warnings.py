import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pandas as pd
import pytest

import config
from scripts import evaluate_rain_warnings as ev, monitor_rain_warnings as trial
from src.data import meteocardedeu as station


START = datetime(2026, 11, 1, tzinfo=timezone.utc)


def t(minutes):
    return START + timedelta(minutes=minutes)


def snapshot(minute, **overrides):
    return dict(dict(timestamp=ev.iso(t(minute)), probability=0.1, station_available=True,
                     station_raining_now=False, shadow_model_id="trial", shadow_probability=0.2), **overrides)


def episode(minute, threshold_delay=0):
    lower = t(minute)
    onset = lower + timedelta(minutes=5)
    qualified = onset + timedelta(minutes=threshold_delay)
    return ev.Episode(lower, onset, qualified, 0.2, True, qualified,
                      ((lower, onset, 0.1), (qualified - timedelta(minutes=5), qualified, 0.1)))


def test_missing_or_different_shadows_cannot_activate_or_be_reconstructed():
    rows = [snapshot(-5), snapshot(0, shadow_model_id="replay", feature_vector={"rain": 1}),
            snapshot(4, shadow_probability=float("nan")), snapshot(10, station_available=False)]
    rows.extend(snapshot(i) for i in range(20, 91, 10))
    rows += [snapshot(92, shadow_model_id="new-weekly-model"),
             snapshot(94, probability=True), snapshot(96, shadow_probability=None)]
    rows.extend(snapshot(i) for i in range(100, 181, 10))
    frame, activation, last, counts = trial.issued_scores(rows, "trial", t(0), t(180))
    assert activation == t(10) and last == t(170)
    assert counts == dict(before_activation=2, unknown_station=1, matched=16, wrong_candidate=1,
                          missing_live_probability=1, missing_score=1)
    assert frame.loc[t(92)].isna().all()
    assert ev.replay(frame)[0] == [(t(80), t(90)), (t(160), t(170))]


def test_unactivated_archive_never_contacts_gauge_or_loads_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "WARNING_TRIAL_NOT_BEFORE", ev.iso(START))
    monkeypatch.setattr(trial.subprocess, "check_output",
                        lambda cmd, **kw: "0 abc" if "rev-list" in cmd else "abc")
    monkeypatch.setattr(ev, "iter_snapshots", lambda *a, **k: [snapshot(10, shadow_model_id=None)])
    monkeypatch.setattr(ev, "station_days", Mock(side_effect=AssertionError("premature gauge lookup")))
    out = tmp_path / "report.json"
    trial.main(["--fetch-station", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["status"] == "awaiting_activation" and r["first_issued_at"] is None
    assert r["observable_onsets"] == 0 and "comparison" not in r
    monkeypatch.setattr(trial.subprocess, "check_output",
                        lambda cmd, **kw: f"{int(t(4).timestamp())} abc" if "rev-list" in cmd else "abc")
    with pytest.raises(SystemExit) as exc:
        trial.main(["--fetch-station"])
    assert exc.value.code == 2


def test_ten_observable_onsets_exact_lead_and_frozen_exposure():
    rows = [snapshot(m, probability=0.8 if m in (120, 310) else 0.1,
                     shadow_probability=0.8 if m in (130, 300, 510) else 0.1)
            for m in range(0, 2401, 10)]
    frame, *_ = trial.issued_scores(rows, "trial", t(0), t(2401))
    episodes = [episode(30), episode(140), episode(318), episode(525, threshold_delay=55)]
    episodes.extend(episode(710 + 160 * n) for n in range(8))
    gauge = [(t(0), t(2500))]
    assert trial.trial_scores(gauge, [], frame, t(0), t(300))["observable_onsets"] == 0
    expected_end = episodes[10].qualified_at + timedelta(microseconds=1)
    original = trial.trial_scores(gauge, episodes[:-1], frame, t(0), expected_end)
    later = trial.trial_scores(gauge, episodes, frame, t(0), t(2400))
    assert original == later
    assert later["status"] == "review_ready" and later["observable_onsets"] == 10
    assert later["scored_until"] == ev.iso(expected_end)
    assert later["excluded_onsets"] == {"station_gap_or_boundary": 1}
    live, shadow, base = [later["comparison"][n] for n in ev.NAMES]
    assert live["advance"] == 1 and live["short_lead_only"] == 1
    assert shadow["advance"] == 2 and shadow["timing_uncertain"] == 1
    assert later["episodes"][1]["onset_outcomes"]["shadow"]["lead_minutes"] == 10
    assert base["advance"] == 0


def test_false_warning_denominator_gaps_partial_boundary_and_trace():
    trace = ev.Episode(t(100), t(105), t(105), 0.1, True, None, ((t(100), t(105), 0.1),))
    gauge = [(t(0), t(240)), (t(360), t(600))]
    watched = [(t(0), t(600))]
    alerts = dict(issued=[(t(48), False), (t(200), False), (t(240), False)],
                  shadow=[(t(370), False), (t(423), False), (t(430), True)],
                  persistence=[])
    seconds, result = trial.false_warnings(gauge, [trace, episode(420)], watched, alerts, t(0), t(600))
    assert seconds == 6 * 3600
    assert result["issued"]["false_warnings_per_week"] == 28
    assert result["issued"]["false_warning_times"] == [ev.iso(t(48))]
    assert result["shadow"]["false_warnings"] == 0
    assert result["shadow"]["uncertain_windows"] == [ev.iso(t(423))]
    assert trial.false_warnings([], [], watched, alerts, t(0), t(600))[1]["issued"]["false_warnings_per_week"] is None


def test_partial_local_gauge_day_is_refetched_and_missing_readings_stay_unknown(tmp_path, monkeypatch):
    day = START.date()
    stamps = pd.date_range(str(day), periods=288, freq="5min")
    full = pd.DataFrame({"datetime": stamps, "PREC": [float("nan")] + [0.] * 287})
    path = tmp_path / f"{day}.csv"
    full.iloc[:12].to_csv(path, index=False)
    fetch = Mock(return_value=full)
    monkeypatch.setattr(station, "fetch_daily", fetch)
    assert list(ev.station_days(t(0), t(60), tmp_path, True)) == [(day, path)]
    assert list(ev.station_days(t(0), t(60), tmp_path, True)) == [(day, path)]
    fetch.assert_called_once_with(day, preserve_missing=True)
    full.drop(index=[50, 51]).to_csv(path, index=False)
    list(ev.station_days(t(0), t(60), tmp_path, True))
    assert fetch.call_count == 2
    full[["datetime"]].to_csv(path, index=False)  # damaged: no PREC column
    list(ev.station_days(t(0), t(60), tmp_path, True))
    assert fetch.call_count == 3


def test_daily_archive_refuses_partial_pages_and_preserves_missing_intervals(monkeypatch):
    rows = [{"DATA": "2026-11-01", "HORA": f"10:{m:02}", "PREC": p}
            for m, p in ((0, 1), (5, None), (10, 1.2), (15, 1.4))]
    ok = Mock()
    ok.json.return_value = {"ok": True, "rows": rows, "meta": {"size": 500}}
    monkeypatch.setattr(station.SESSION, "get", Mock(return_value=ok))
    frame = station.fetch_daily(START.date(), preserve_missing=True)
    assert frame.PREC.iloc[:3].isna().all() and frame.PREC.iloc[3] == pytest.approx(0.2)
    ok.json.return_value["meta"]["size"] = len(rows)
    bad = Mock()
    bad.json.return_value = {"ok": False}
    monkeypatch.setattr(station.SESSION, "get", Mock(side_effect=[ok, bad]))
    assert station.fetch_daily(START.date(), preserve_missing=True).empty
