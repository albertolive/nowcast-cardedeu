"""
Client per a l'API Ensemble d'Open-Meteo.
Compara ECMWF, GFS, ICON i AROME per mesurar el grau d'acord entre models.
AROME (Meteo-France) aporta resolució de 2.5km — clau per convecció local.
Cap API key necessària.
"""
import logging
from datetime import datetime

import numpy as np
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "NowcastCardedeu/1.0 (research)"})

ENSEMBLE_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global", "meteofrance_arome_france0025"]


def fetch_ensemble_agreement() -> dict:
    """
    Consulta ECMWF, GFS, ICON i AROME per a les properes 6h i calcula
    el grau d'acord entre models per a precipitació i temperatura.
    AROME (2.5km) resol convecció local molt millor que ECMWF (9km).

    Retorna dict amb:
      - ensemble_rain_agreement: fracció de models que prediuen pluja (0-1)
      - ensemble_precip_spread: diferència max-min de precipitació entre models
      - ensemble_temp_spread: desviació estàndard de temperatura entre models
      - ensemble_max_precip: precipitació màxima predita per qualsevol model
      - ensemble_min_precip: precipitació mínima predita per qualsevol model
      - ensemble_models_rain: quants models prediuen pluja >0.1mm en 6h
    """
    try:
        model_precip_6h = []
        model_temps = []

        for model in ENSEMBLE_MODELS:
            params = {
                "latitude": config.LATITUDE,
                "longitude": config.LONGITUDE,
                "hourly": "precipitation,temperature_2m",
                "timezone": "Europe/Madrid",
                "forecast_hours": 6,
                "models": model,
            }
            r = SESSION.get(ENSEMBLE_URL, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            hourly = data.get("hourly", {})
            precip_vals = hourly.get("precipitation", [])
            temp_vals = hourly.get("temperature_2m", [])

            # Precipitació total en les properes 6h
            precip_sum = sum(v for v in precip_vals if v is not None)
            model_precip_6h.append(precip_sum)

            # Temperatura actual (primera hora)
            if temp_vals and temp_vals[0] is not None:
                model_temps.append(temp_vals[0])

        n_models = len(model_precip_6h)
        rain_models = sum(1 for p in model_precip_6h if p >= 0.1)

        result = {
            "ensemble_rain_agreement": rain_models / n_models if n_models > 0 else 0.0,
            "ensemble_precip_spread": max(model_precip_6h) - min(model_precip_6h) if model_precip_6h else 0.0,
            "ensemble_temp_spread": float(np.std(model_temps)) if len(model_temps) >= 2 else 0.0,
            "ensemble_max_precip": max(model_precip_6h) if model_precip_6h else 0.0,
            "ensemble_min_precip": min(model_precip_6h) if model_precip_6h else 0.0,
            "ensemble_models_rain": rain_models,
        }

        logger.info(
            f"Ensemble: {rain_models}/{n_models} models pluja, "
            f"spread={result['ensemble_precip_spread']:.1f}mm, "
            f"temp_std={result['ensemble_temp_spread']:.1f}°C"
        )
        return result

    except Exception as e:
        logger.warning(f"Error obtenint ensemble: {e}")
        return {
            "ensemble_rain_agreement": np.nan,
            "ensemble_precip_spread": np.nan,
            "ensemble_temp_spread": np.nan,
            "ensemble_max_precip": np.nan,
            "ensemble_min_precip": np.nan,
            "ensemble_models_rain": np.nan,
        }


def compute_forecast_bias(station_temp: float, station_hum: float,
                          forecast_df=None) -> dict:
    """
    Calcula el biaix entre la previsió d'Open-Meteo i les condicions
    reals mesurades a l'estació de Cardedeu en aquest moment.

    Retorna dict amb:
      - forecast_temp_bias: forecast - observat (°C)
      - forecast_humidity_bias: forecast - observat (%)
      - forecast_pressure_bias: forecast - observat (hPa)
    """
    try:
        if forecast_df is not None and not forecast_df.empty:
            now = datetime.now()
            forecast_df = forecast_df.copy()
            forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
            # Trobar la fila del forecast més propera a ara
            idx = (forecast_df["datetime"] - pd.Timestamp(now)).abs().idxmin()
            row = forecast_df.loc[idx]

            forecast_temp = row.get("temperature_2m")
            forecast_hum = row.get("relative_humidity_2m")

            result = {
                "forecast_temp_bias": (float(forecast_temp) - station_temp
                                       if forecast_temp is not None and station_temp else np.nan),
                "forecast_humidity_bias": (float(forecast_hum) - station_hum
                                           if forecast_hum is not None and station_hum else np.nan),
            }
        else:
            result = {
                "forecast_temp_bias": np.nan,
                "forecast_humidity_bias": np.nan,
            }

        if not np.isnan(result.get("forecast_temp_bias", np.nan)):
            logger.info(
                f"Bias: temp={result['forecast_temp_bias']:+.1f}°C, "
                f"hum={result['forecast_humidity_bias']:+.0f}%"
            )
        return result

    except Exception as e:
        logger.warning(f"Error calculant bias: {e}")
        return {
            "forecast_temp_bias": np.nan,
            "forecast_humidity_bias": np.nan,
        }


# Needed for compute_forecast_bias when forecast_df is a DataFrame
import pandas as pd
