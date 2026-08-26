"""
Tests per al target d'entrenament a prepare_training_data.

Incident 2026-08-26: build_target_column es reaplicava sobre el df fusionat
(base + feedback) i sobreescriu will_rain amb un pseudo-label derivat de la
columna 'precipitation' (valor NWP). Resultat: 116 dels 222 events de pluja
verificats reetiquetats com a secs i 46 moments secs com a plujosos — el
feedback loop entrenava contra etiquetes NWP, no observades.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.train import prepare_training_data


def _df(rows):
    return pd.DataFrame(rows)


def test_verified_labels_survive_merge():
    """will_rain preexistent NO es pot recalcular des de 'precipitation'."""
    df = _df({
        "precipitation": [0.0, 5.0, 0.0, 0.0, 0.0],
        "pressure_msl": [1015.0] * 5,
        # fila índex 3: va ploure de veritat encara que el NWP digui sec;
        # fila índex 1: el NWP diu pluja però no en va caure res
        "will_rain": [0, 0, 0, 1, 0],
    })
    X, y = prepare_training_data(df)
    assert list(y) == [0, 0, 0, 1, 0]


def test_target_still_derived_when_absent():
    """Sense columna will_rain es deriva de precipitation (comportament previ)."""
    df = _df({
        "precipitation": [0.0, 0.0, 3.0, 0.0, 0.0],
        "pressure_msl": [1015.0] * 5,
    })
    X, y = prepare_training_data(df)
    # horitzó 1: plou si la fila següent acumula >= RAIN_THRESHOLD_MM
    import config
    thr = config.RAIN_THRESHOLD_MM
    nxt = df["precipitation"].shift(-1)
    expected = [int(bool(v >= thr)) for v in nxt.fillna(0.0)]
    assert list(y) == expected


def test_partial_will_rain_raises_instead_of_corrupting():
    """NaN parcials a will_rain: error explícit, mai inventar etiquetes."""
    df = _df({
        "precipitation": [0.0, 1.0, 0.0],
        "pressure_msl": [1015.0] * 3,
        "will_rain": [0, np.nan, 1],
    })
    with pytest.raises(ValueError, match="will_rain"):
        prepare_training_data(df)
