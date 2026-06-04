"""
Tests per a les regles físiques de _apply_physical_constraints.

Cas real que motiva la regla 2b (2026-06-04): eco de 56 dBZ a 7.2 km amb
vector de moviment 0.0 km/h, AEMET caigut per 429 durant 3h, i la predicció
es va quedar al 10% fins que va ploure sobre l'estació.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.predict import _apply_physical_constraints

AEMET_DOWN = {
    "aemet_radar_available": False,
    "aemet_radar_nearest_echo_km": None,
    "aemet_radar_max_dbz_20km": 0,
}
NO_SENTINEL = {"sentinel_raining": 0}


def _radar(**overrides):
    base = {
        "radar_dbz": 0.0,
        "radar_has_echo": False,
        "radar_nearest_echo_km": None,
        "radar_max_dbz_20km": 0.0,
        "radar_storm_approaching": 0,
        "radar_storm_eta_min": None,
        "radar_upwind_nearest_echo_km": 30.0,
    }
    base.update(overrides)
    return base


class TestStrongEchoNearby:
    def test_strong_upwind_echo_floors_at_55(self):
        # Cas real 2026-06-04 19:21: 56.5 dBZ a 7.2km a sobrevent, pred 10.3%
        radar = _radar(
            radar_nearest_echo_km=7.2,
            radar_max_dbz_20km=56.5,
            radar_upwind_nearest_echo_km=7.2,
        )
        prob, adj = _apply_physical_constraints(
            0.103, radar, NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.55)
        assert any("sobrevent" in a for a in adj)

    def test_strong_echo_not_upwind_floors_at_45(self):
        radar = _radar(radar_nearest_echo_km=7.2, radar_max_dbz_20km=56.5)
        prob, adj = _apply_physical_constraints(
            0.103, radar, NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.45)
        assert adj

    def test_weak_echo_nearby_does_not_trigger(self):
        # 30 dBZ a 7km: pluja feble a prop, no xàfec torrencial
        radar = _radar(radar_nearest_echo_km=7.2, radar_max_dbz_20km=30.0)
        prob, adj = _apply_physical_constraints(
            0.103, radar, NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.103)
        assert adj == []

    def test_strong_echo_far_does_not_trigger(self):
        radar = _radar(radar_nearest_echo_km=18.0, radar_max_dbz_20km=50.0)
        prob, adj = _apply_physical_constraints(
            0.103, radar, NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.103)

    def test_clear_sky_unaffected(self):
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.08)
        assert adj == []

    def test_consensus_present_but_radar_floor_higher_wins(self):
        # Si el radar ja ha pujat per sobre de 0.35, el consens no afegeix res
        radar = _radar(
            radar_nearest_echo_km=7.2,
            radar_max_dbz_20km=56.5,
            radar_upwind_nearest_echo_km=7.2,
        )
        prob, adj = _apply_physical_constraints(
            0.103, radar, NO_SENTINEL, AEMET_DOWN, None, None,
            aemet_forecast={"aemet_prob_precip": 100},
            ensemble={"ensemble_rain_agreement": 1.0, "ensemble_min_precip": 16.4},
        )
        assert prob == pytest.approx(0.55)
        assert not any("Consens" in a for a in adj)

    def test_floor_does_not_lower_higher_probability(self):
        radar = _radar(
            radar_nearest_echo_km=7.2,
            radar_max_dbz_20km=56.5,
            radar_upwind_nearest_echo_km=7.2,
        )
        prob, adj = _apply_physical_constraints(
            0.90, radar, NO_SENTINEL, AEMET_DOWN, None, None
        )
        assert prob == pytest.approx(0.90)
        assert adj == []


class TestLightningRules:
    def test_active_cell_within_15km_floors_at_50(self):
        prob, adj = _apply_physical_constraints(
            0.10, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            lightning={"lightning_count_15km_1h": 3, "lightning_count_30km_1h": 8,
                       "lightning_nearest_km": 9.0},
        )
        assert prob == pytest.approx(0.50)
        assert any("llamps" in a for a in adj)

    def test_activity_30km_with_nearby_strikes_floors_at_40(self):
        prob, adj = _apply_physical_constraints(
            0.10, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            lightning={"lightning_count_15km_1h": 0, "lightning_count_30km_1h": 6,
                       "lightning_nearest_km": 18.0},
        )
        assert prob == pytest.approx(0.40)

    def test_old_storm_3h_window_does_not_trigger(self):
        # Tempesta de fa 2h: comptadors de 3h alts però 1h a zero
        prob, adj = _apply_physical_constraints(
            0.10, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            lightning={"lightning_count_15km": 12, "lightning_count_30km": 30,
                       "lightning_count_15km_1h": 0, "lightning_count_30km_1h": 0,
                       "lightning_nearest_km": 8.0},
        )
        assert prob == pytest.approx(0.10)
        assert adj == []

    def test_distant_strikes_do_not_trigger(self):
        prob, adj = _apply_physical_constraints(
            0.10, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            lightning={"lightning_count_15km_1h": 0, "lightning_count_30km_1h": 6,
                       "lightning_nearest_km": 28.0},
        )
        assert prob == pytest.approx(0.10)


class TestSynopticConsensus:
    def test_strong_consensus_floors_at_35(self):
        # Cas real 2026-06-04 tarda: AEMET 100%, 4/4 models, min 16.4mm, pred 8%
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            aemet_forecast={"aemet_prob_precip": 100},
            ensemble={"ensemble_rain_agreement": 1.0, "ensemble_min_precip": 16.4},
        )
        assert prob == pytest.approx(0.35)
        assert any("Consens" in a for a in adj)

    def test_partial_agreement_does_not_trigger(self):
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            aemet_forecast={"aemet_prob_precip": 100},
            ensemble={"ensemble_rain_agreement": 0.75, "ensemble_min_precip": 16.4},
        )
        assert prob == pytest.approx(0.08)
        assert adj == []

    def test_low_amounts_do_not_trigger(self):
        # Tots d'acord però quantitats simbòliques: pot ser plugim irrellevant
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            aemet_forecast={"aemet_prob_precip": 90},
            ensemble={"ensemble_rain_agreement": 1.0, "ensemble_min_precip": 1.2},
        )
        assert prob == pytest.approx(0.08)

    def test_nan_aemet_prob_does_not_trigger(self):
        import numpy as np
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
            aemet_forecast={"aemet_prob_precip": np.nan},
            ensemble={"ensemble_rain_agreement": 1.0, "ensemble_min_precip": 16.4},
        )
        assert prob == pytest.approx(0.08)

    def test_missing_dicts_do_not_trigger(self):
        prob, adj = _apply_physical_constraints(
            0.08, _radar(), NO_SENTINEL, AEMET_DOWN, None, None,
        )
        assert prob == pytest.approx(0.08)
        assert adj == []
