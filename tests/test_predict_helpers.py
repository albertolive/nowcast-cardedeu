"""
Test suite: predict.py helpers.
Covers _compute_station_raining_now edge cases (PINT signal, PREC fallback,
station offline, malformed sensor values).
"""
import pandas as pd
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.predict import _compute_station_raining_now


def _station_df(prec_values):
    """Build a station_df with the given PREC sequence (latest at the end)."""
    return pd.DataFrame({"PREC": prec_values})


class TestComputeStationRainingNow:
    def test_pint_positive_returns_true(self):
        assert _compute_station_raining_now({"PINT": "0.4"}, None) is True

    def test_pint_zero_with_no_recent_prec_returns_false(self):
        df = _station_df([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert _compute_station_raining_now({"PINT": "0.0"}, df) is False

    def test_pint_zero_but_recent_prec_returns_true(self):
        df = _station_df([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
        assert _compute_station_raining_now({"PINT": "0"}, df) is True

    def test_pint_missing_falls_back_to_prec(self):
        df = _station_df([0.0, 0.0, 0.0, 0.0, 0.2, 0.0])
        assert _compute_station_raining_now({}, df) is True

    def test_no_current_no_station_returns_false(self):
        assert _compute_station_raining_now(None, None) is False

    def test_empty_station_df_returns_false(self):
        assert _compute_station_raining_now(None, pd.DataFrame()) is False

    def test_station_df_without_prec_column_returns_false(self):
        df = pd.DataFrame({"TEMP": [10, 11, 12]})
        assert _compute_station_raining_now(None, df) is False

    def test_malformed_pint_does_not_raise(self):
        df = _station_df([0.0] * 6)
        assert _compute_station_raining_now({"PINT": "n/a"}, df) is False

    def test_malformed_prec_does_not_raise(self):
        df = pd.DataFrame({"PREC": ["x", "y", "z"]})
        assert _compute_station_raining_now({}, df) is False

    def test_only_old_rain_outside_recent_window_returns_false(self):
        # PREC tail(6) sums the last 6 rows; rain 10 rows back is ignored.
        df = _station_df([0.5] + [0.0] * 10)
        assert _compute_station_raining_now({}, df) is False

    def test_prec_exactly_at_window_boundary_counted(self):
        # Sixth-from-last row counts (tail(6)).
        df = _station_df([0.0] * 4 + [0.3] + [0.0] * 5)
        assert _compute_station_raining_now({}, df.tail(6).reset_index(drop=True)) is True

    def test_negative_pint_treated_as_dry(self):
        # PINT can't physically be negative; any non-positive is dry.
        df = _station_df([0.0] * 6)
        assert _compute_station_raining_now({"PINT": "-0.1"}, df) is False
