"""
Client per a la Predicció Municipal horària del Meteocat (SMC).
Obté la predicció horària a 72h per al municipi de Cardedeu.
Complement o alternativa a la previsió d'AEMET, amb resolució local
específica per a Catalunya.
Documentació: https://apidocs.meteocat.gencat.cat/documentacio/prediccio/
"""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import requests

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session

logger = logging.getLogger(__name__)

SESSION = create_session()


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def fetch_municipal_hourly_forecast() -> dict:
    """
    Obté la predicció horària a 72h per Cardedeu del Meteocat (SMC).
    Endpoint: /pronostic/v1/municipalHoraria/{codiMunicipi}

    El Meteocat (SMC) és l'autoritat meteorològica de Catalunya i
    la seva predicció municipal és específica per l'orografia catalana.

    Retorna dict amb:
      - smc_prob_precip_1h: probabilitat de precipitació a l'hora actual (0-100)
      - smc_prob_precip_6h: màxima prob. precipitació properes 6h (0-100)
      - smc_precip_intensity: intensitat de precipitació esperada (codi WMO)
      - smc_temp_forecast: temperatura prevista per l'hora actual (°C)
      - smc_weather_symbol: símbol (codi d'estat del cel)
    """
    result = _empty_forecast()

    if not _is_configured():
        logger.info("Meteocat Predicció no configurat (sense METEOCAT_API_KEY)")
        return result

    try:
        from src.data.meteocat_cache import get_cached, set_cached
        cache_key = f"smc_forecast_{datetime.now().strftime('%Y%m%d_%H')}"
        cached = get_cached(cache_key, config.METEOCAT_CACHE_TTL_SMC)
        if cached is not None:
            logger.info("SMC forecast: using cached response")
            return cached

        url = (
            f"{config.METEOCAT_BASE_URL}/pronostic/v1/"
            f"municipalHoraria/{config.METEOCAT_MUNICIPALITY_CODE}"
        )
        r = SESSION.get(url, headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            raise ValueError("Resposta buida del Meteocat Predicció")

        now = datetime.now()
        current_hour = now.hour

        # La resposta conté la predicció horària.
        # Extreure precipitació, temperatura i símbol per les properes 6h.
        hourly_forecasts = _extract_hourly(data)

        if not hourly_forecasts:
            logger.warning("No s'han pogut extreure dades horàries del Meteocat")
            return result

        # Cercar l'hora actual o la més propera
        current = None
        today = now.date()
        for hf in hourly_forecasts:
            if hf.get("date") == today and hf.get("hour") == current_hour:
                current = hf
                break

        if current is None and hourly_forecasts:
            current = hourly_forecasts[0]

        # Probabilitat actual
        result["smc_prob_precip_1h"] = current.get("prob_precip", 0)
        result["smc_temp_forecast"] = current.get("temp")
        result["smc_weather_symbol"] = current.get("symbol")

        # Màxima probabilitat a 6h
        relevant = [hf for hf in hourly_forecasts
                     if hf.get("date") == today
                     and hf.get("hour", -1) >= current_hour
                     and hf.get("hour", -1) <= current_hour + 6]
        if relevant:
            result["smc_prob_precip_6h"] = max(
                hf.get("prob_precip", 0) for hf in relevant
            )
            # Intensitat de precipitació
            intensities = [hf.get("precip_intensity", 0) for hf in relevant]
            result["smc_precip_intensity"] = max(intensities) if intensities else 0

        logger.info(
            f"SMC Predicció: probPrecip1h={result['smc_prob_precip_1h']}%, "
            f"probPrecip6h={result['smc_prob_precip_6h']}%, "
            f"temp={result['smc_temp_forecast']}°C"
        )
        set_cached(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Error obtenint Meteocat Predicció: {e}")
        return result


def _extract_hourly(data: dict) -> list[dict]:
    """
    Extreu les previsions horàries de la resposta del Meteocat.
    Format real de l'API:
    {
      "codiMunicipi": "080462",
      "dies": [{
        "data": "2026-03-19Z",
        "variables": {
          "temp": {"unitat": "°C", "valors": [{"valor": "9.6", "data": "2026-03-19T00:00Z"}, ...]},
          "precipitacio": {"unitat": "mm", "valors": [...]},
          "estatCel": {"valors": [{"valor": "20", "data": "..."}]},
          "humitat": {"valors": [...]},
          ...
        }
      }]
    }
    """
    hourly = []
    dies = data.get("dies", [])

    for dia in dies:
        variables = dia.get("variables", {})
        if not isinstance(variables, dict):
            continue

        # Build lookup tables: hour → value for each variable
        temp_by_dt = {}
        precip_by_dt = {}
        symbol_by_dt = {}
        humidity_by_dt = {}

        for var_name, var_data in variables.items():
            valors = var_data.get("valors", [])
            for v in valors:
                dt_str = v.get("data", "")
                val = v.get("valor")
                if not dt_str or val is None:
                    continue
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    hour = dt.hour
                    # Use (date, hour) as key to handle multi-day
                    key = (dt.date(), hour)
                except (ValueError, AttributeError):
                    continue

                if var_name == "temp":
                    temp_by_dt[key] = float(val)
                elif var_name == "precipitacio":
                    precip_by_dt[key] = float(val)
                elif var_name == "estatCel":
                    symbol_by_dt[key] = val
                elif var_name == "humitat":
                    humidity_by_dt[key] = float(val)

        # Merge all variables by (date, hour) key
        all_keys = set()
        all_keys.update(temp_by_dt.keys(), symbol_by_dt.keys(), humidity_by_dt.keys())
        if precip_by_dt:
            all_keys.update(precip_by_dt.keys())

        for key in sorted(all_keys):
            date_part, hour = key
            precip_mm = precip_by_dt.get(key, 0)
            hourly.append({
                "date": date_part,
                "hour": hour,
                "prob_precip": min(precip_mm * 100, 100) if precip_mm > 0 else 0,
                "precip_intensity": precip_mm,
                "temp": temp_by_dt.get(key),
                "symbol": symbol_by_dt.get(key),
            })

    return hourly


def _empty_forecast() -> dict:
    return {
        "smc_prob_precip_1h": np.nan,
        "smc_prob_precip_6h": np.nan,
        "smc_precip_intensity": np.nan,
        "smc_temp_forecast": np.nan,
        "smc_weather_symbol": np.nan,
    }


def fetch_smc_hourly_df() -> "pd.DataFrame":
    """
    Obté la predicció horària municipal del SMC com a DataFrame.
    Columnes: datetime, smc_prob_precip_1h, smc_precip_intensity.
    Per injectar a build_features_from_forecast().
    """
    import pandas as pd

    if not _is_configured():
        return pd.DataFrame()

    try:
        url = (
            f"{config.METEOCAT_BASE_URL}/pronostic/v1/"
            f"municipalHoraria/{config.METEOCAT_MUNICIPALITY_CODE}"
        )
        r = SESSION.get(url, headers=_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()

        if not data:
            return pd.DataFrame()

        hourly_forecasts = _extract_hourly(data)
        if not hourly_forecasts:
            return pd.DataFrame()

        rows = []
        for hf in hourly_forecasts:
            hour = hf.get("hour")
            date_part = hf.get("date")
            if hour is None or date_part is None:
                continue
            dt = datetime(date_part.year, date_part.month, date_part.day, hour)
            rows.append({
                "datetime": dt,
                "smc_prob_precip_1h": hf.get("prob_precip", np.nan),
                "smc_precip_intensity": hf.get("precip_intensity", np.nan),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Compute smc_prob_precip_6h as rolling max of next 6 hours
        df = df.sort_values("datetime").reset_index(drop=True)
        df["smc_prob_precip_6h"] = (
            df["smc_prob_precip_1h"]
            .rolling(6, min_periods=1)
            .max()
            .shift(-5)
            .fillna(df["smc_prob_precip_1h"])
        )

        logger.info(f"SMC hourly forecast: {len(df)} hores obtingudes")
        return df

    except Exception as e:
        logger.warning(f"Error obtenint SMC hourly forecast: {e}")
        return pd.DataFrame()
