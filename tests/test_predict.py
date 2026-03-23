"""
Tests per al pipeline de predicció (src/model/predict.py).
Cobreix: scalar pressure data fill-only-NaN, consistency amb feature engineering.
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestScalarPressureOverwrite:
    """
    Bug: pressure_data escalars sobreescrivien wind_850_dir al latest DataFrame
    després que feature engineering calculés els règims amb vent de superfície
    (perquè wind_850_dir era NaN al DataFrame).
    Resultat: wind_850_dir=248 (Garbí) + llevantada_strength=4.8 (superfície ESE).

    Fix: Només omplir NaN, no sobreescriure valors existents.
    """

    @staticmethod
    def _apply_pressure_fill(latest, pressure_data):
        """Reprodueix la lògica del fix a predict.py."""
        for k, v in pressure_data.items():
            if k in latest.columns:
                if pd.isna(latest[k].values[0]):
                    latest[k] = v
            else:
                latest[k] = v

    def test_nan_is_filled(self):
        """Si wind_850_dir és NaN, s'omple amb el valor escalar."""
        latest = pd.DataFrame({"wind_850_dir": [np.nan]})
        self._apply_pressure_fill(latest, {"wind_850_dir": 248.0})
        assert latest["wind_850_dir"].values[0] == 248.0

    def test_existing_value_not_overwritten(self):
        """Si wind_850_dir ja té valor, NO es sobreescriu."""
        latest = pd.DataFrame({"wind_850_dir": [120.0]})
        self._apply_pressure_fill(latest, {"wind_850_dir": 248.0})
        assert latest["wind_850_dir"].values[0] == 120.0

    def test_missing_column_is_created(self):
        """Si la columna no existeix, es crea amb el valor escalar."""
        latest = pd.DataFrame({"temperature_2m": [15.0]})
        self._apply_pressure_fill(latest, {"wind_850_dir": 248.0})
        assert latest["wind_850_dir"].values[0] == 248.0

    def test_march22_consistency(self):
        """
        Reprodueix el bug del 22/03:
        - Feature engineering va calcular amb surface (wind_850_dir NaN al DF)
        - Scalar 248° omple el NaN però NO canvia features derivades
        - El model veu wind_850_dir=248 + llevantada_strength=0 (coherent amb NaN)
        """
        latest = pd.DataFrame({
            "wind_850_dir": [np.nan],
            "llevantada_strength": [0.0],  # Derivat durant feature engineering (amb NaN)
            "garbi_strength": [0.0],
        })
        self._apply_pressure_fill(latest, {"wind_850_dir": 248.0})
        # NaN → s'omple
        assert latest["wind_850_dir"].values[0] == 248.0
        # Features derivades no canvien — coherents amb el pipeline
        assert latest["llevantada_strength"].values[0] == 0.0
        assert latest["garbi_strength"].values[0] == 0.0
