"""
Tests per a la màquina d'estats de notificacions (src/notify/state.py).
Cobreix: histèresi (up=0.65, down=0.30), cooldown, règim, update_state.
"""
import json
import os
import time
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.notify.state import (
    should_notify,
    should_notify_regime,
    update_state,
    load_state,
    save_state,
    DEFAULT_STATE,
    STATE_FILE,
)


# ── Helpers ──

def _clear_state(**overrides):
    """Crea estat 'clear' fresc amb cooldown expirat."""
    state = DEFAULT_STATE.copy()
    state["last_alert_time"] = 0  # cooldown expirat
    state.update(overrides)
    return state


def _rain_alert_state(**overrides):
    """Crea estat 'rain_alert' amb cooldown expirat."""
    state = DEFAULT_STATE.copy()
    state["current_state"] = "rain_alert"
    state["last_alert_time"] = 0
    state.update(overrides)
    return state


# ── Hysteresis: clear → rain_alert ──

class TestHysteresisUp:
    def test_triggers_at_threshold(self):
        """Prob exactament 0.65 ha de disparar rain_incoming."""
        assert should_notify(0.65, _clear_state()) == "rain_incoming"

    def test_triggers_above_threshold(self):
        assert should_notify(0.90, _clear_state()) == "rain_incoming"

    def test_no_trigger_just_below(self):
        """0.649 no ha d'activar — histèresi estricta."""
        assert should_notify(0.649, _clear_state()) is None

    def test_no_trigger_in_gap(self):
        """Prob entre 0.30 i 0.65, estat clear → no notificar."""
        assert should_notify(0.50, _clear_state()) is None


# ── Hysteresis: rain_alert → clear ──

class TestHysteresisDown:
    def test_clears_at_threshold(self):
        """Prob exactament 0.30 ha de disparar rain_clearing."""
        assert should_notify(0.30, _rain_alert_state()) == "rain_clearing"

    def test_clears_below_threshold(self):
        assert should_notify(0.10, _rain_alert_state()) == "rain_clearing"

    def test_no_clear_just_above(self):
        """0.301 no ha de fer clear — histèresi estricta."""
        assert should_notify(0.301, _rain_alert_state()) is None

    def test_no_clear_in_gap(self):
        """Prob 0.50, estat rain_alert → dins del gap, no notificar."""
        assert should_notify(0.50, _rain_alert_state()) is None

    def test_high_prob_in_rain_alert_no_retrigger(self):
        """Ja estem en rain_alert amb prob alta → no reenviar."""
        assert should_notify(0.90, _rain_alert_state()) is None


# ── Cooldown ──

class TestCooldown:
    def test_cooldown_blocks_notification(self):
        """Alerta recent dins del cooldown (30 min) → bloquejar."""
        state = _clear_state(last_alert_time=time.time() - 60)  # fa 1 min
        assert should_notify(0.90, state) is None

    def test_cooldown_expired_allows(self):
        """Cooldown expirat (>30 min) → permetre."""
        state = _clear_state(last_alert_time=time.time() - 1801)
        assert should_notify(0.90, state) == "rain_incoming"

    def test_cooldown_exact_boundary(self):
        """Exactament al límit del cooldown (30*60=1800s) → permetre."""
        state = _clear_state(last_alert_time=time.time() - 1800)
        # time_since_last = 1800 which is NOT < 1800, so cooldown is expired
        assert should_notify(0.90, state) == "rain_incoming"

    def test_cooldown_just_before_expiry(self):
        """1 segon abans del cooldown → encara bloquejat."""
        state = _clear_state(last_alert_time=time.time() - 1799)
        assert should_notify(0.90, state) is None


# ── Regime change alerts ──

class TestRegimeNotification:
    def test_no_change_returns_false(self):
        """Si no hi ha canvi de règim → no notificar."""
        assert should_notify_regime(None, _clear_state()) is False

    def test_new_regime_triggers(self):
        """Nou canvi de règim amb cooldown expirat → sí."""
        change = {"type": "llevantada_onset", "severity": "watch"}
        state = _clear_state(last_regime_alert_time=0, last_regime_alert_type=None)
        assert should_notify_regime(change, state) is True

    def test_regime_cooldown_blocks(self):
        """Cooldown de règim (2h) encara actiu → bloquejar."""
        change = {"type": "llevantada_onset", "severity": "watch"}
        state = _clear_state(last_regime_alert_time=time.time() - 60)
        assert should_notify_regime(change, state) is False

    def test_regime_cooldown_expired(self):
        """Cooldown de règim expirat (>2h) → permetre."""
        change = {"type": "garbi_inestable", "severity": "watch"}
        state = _clear_state(
            last_regime_alert_time=time.time() - 7201,
            last_regime_alert_type="llevantada_onset",
        )
        assert should_notify_regime(change, state) is True

    def test_same_regime_type_blocked(self):
        """Mateix tipus de règim ja alertat → bloquejar."""
        change = {"type": "llevantada_onset", "severity": "watch"}
        state = _clear_state(
            last_regime_alert_time=0,
            last_regime_alert_type="llevantada_onset",
        )
        assert should_notify_regime(change, state) is False

    def test_different_regime_type_allowed(self):
        """Diferent tipus de règim → permetre."""
        change = {"type": "garbi_inestable", "severity": "warning"}
        state = _clear_state(
            last_regime_alert_time=0,
            last_regime_alert_type="llevantada_onset",
        )
        assert should_notify_regime(change, state) is True


# ── update_state ──

class TestUpdateState:
    def test_rain_incoming_sets_rain_alert(self, tmp_path, monkeypatch):
        """rain_incoming ha de canviar l'estat a rain_alert."""
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "state.json"))
        state = _clear_state()
        result = update_state(state, "rain_incoming", 0.80)
        assert result["current_state"] == "rain_alert"
        assert result["last_alert_type"] == "rain_incoming"
        assert result["last_probability"] == 0.80

    def test_rain_clearing_sets_clear(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "state.json"))
        state = _rain_alert_state()
        result = update_state(state, "rain_clearing", 0.15)
        assert result["current_state"] == "clear"

    def test_regime_change_updates_regime_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "state.json"))
        state = _clear_state()
        result = update_state(state, "regime_change", 0.40, regime_alert_type="pressure_drop")
        assert result["last_regime_alert_type"] == "pressure_drop"
        assert result["last_regime_alert_time"] > 0

    def test_daily_summary_no_state_change(self, tmp_path, monkeypatch):
        """daily_summary no ha de canviar current_state."""
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "state.json"))
        state = _rain_alert_state()
        result = update_state(state, "daily_summary", 0.70)
        assert result["current_state"] == "rain_alert"  # no canvia

    def test_wind_regime_stored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "state.json"))
        state = _clear_state()
        regime = {"direction": 120, "name": "Llevantada"}
        result = update_state(state, "rain_incoming", 0.80, wind_regime=regime)
        assert result["last_wind_regime"] == regime


# ── State persistence ──

class TestStatePersistence:
    def test_load_missing_file_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.notify.state.STATE_FILE", str(tmp_path / "nonexistent.json"))
        state = load_state()
        assert state["current_state"] == "clear"
        assert state["last_alert_time"] == 0

    def test_save_and_reload(self, tmp_path, monkeypatch):
        path = str(tmp_path / "state.json")
        monkeypatch.setattr("src.notify.state.STATE_FILE", path)
        state = _rain_alert_state()
        state["last_probability"] = 0.75
        save_state(state)
        loaded = load_state()
        assert loaded["current_state"] == "rain_alert"
        assert loaded["last_probability"] == 0.75

    def test_corrupt_file_returns_default(self, tmp_path, monkeypatch):
        """Fitxer JSON corrupte → retorna DEFAULT_STATE."""
        path = str(tmp_path / "state.json")
        monkeypatch.setattr("src.notify.state.STATE_FILE", path)
        with open(path, "w") as f:
            f.write("{invalid json{{{")
        state = load_state()
        assert state["current_state"] == "clear"

    def test_missing_fields_backfilled(self, tmp_path, monkeypatch):
        """Fitxer amb camps incomplets → es completen amb defaults."""
        path = str(tmp_path / "state.json")
        monkeypatch.setattr("src.notify.state.STATE_FILE", path)
        with open(path, "w") as f:
            json.dump({"current_state": "rain_alert"}, f)
        state = load_state()
        assert state["current_state"] == "rain_alert"
        assert state["last_alert_time"] == 0
        assert state["last_regime_alert_type"] is None
