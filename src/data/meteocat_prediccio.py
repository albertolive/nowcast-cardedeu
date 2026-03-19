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

logger = logging.getLogger(__name__)

SESSION = requests.Session()


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def fetch_municipal_hourly_forecast() -> dict:
    """
    Obté la predicció horària a 72h per Cardedeu del Meteocat (SMC).
    Endpoint: /pronostic/v1/municipal/{codiMunicipi}/horaria

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
        url = (
            f"{config.METEOCAT_BASE_URL}/pronostic/v1/municipal/"
            f"{config.METEOCAT_MUNICIPALITY_CODE}/horaria"
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
        for hf in hourly_forecasts:
            if hf.get("hour") == current_hour:
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
                     if hf.get("hour", -1) >= current_hour
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
        return result

    except Exception as e:
        logger.warning(f"Error obtenint Meteocat Predicció: {e}")
        return result


def _extract_hourly(data: dict) -> list[dict]:
    """
    Extreu les previsions horàries de la resposta del Meteocat.
    L'estructura pot variar, intentem diversos formats possibles.
    """
    hourly = []

    # Format possible 1: la predicció conté "dies" → cada dia "hores"
    dies = data.get("dies", data.get("prediccioDies", []))
    if isinstance(dies, list):
        for dia in dies:
            hores = dia.get("variables", dia.get("hores", []))
            if isinstance(hores, list):
                for h in hores:
                    hourly.append({
                        "hour": h.get("hora", h.get("h")),
                        "prob_precip": h.get("probabilitatPrecipitacio",
                                             h.get("probPrecip", 0)),
                        "precip_intensity": h.get("quantitatPrecipitacio",
                                                  h.get("precipitacio", 0)),
                        "temp": h.get("temperatura", h.get("temp")),
                        "symbol": h.get("simbol", h.get("estatCel")),
                    })

    # Format possible 2: predicció directa amb arrays per variable
    if not hourly:
        variables = data.get("variables", [])
        if isinstance(variables, list):
            temp_data = {}
            precip_data = {}
            for var in variables:
                codi = var.get("codi", var.get("nom", ""))
                valors = var.get("valors", [])
                if "temperatura" in str(codi).lower() or codi == "temp":
                    for v in valors:
                        h = v.get("hora", v.get("h"))
                        if h is not None:
                            temp_data[h] = v.get("valor")
                elif "precipitacio" in str(codi).lower() or "precip" in str(codi).lower():
                    for v in valors:
                        h = v.get("hora", v.get("h"))
                        if h is not None:
                            precip_data[h] = v.get("valor", 0)

            all_hours = set(list(temp_data.keys()) + list(precip_data.keys()))
            for h in sorted(all_hours):
                hourly.append({
                    "hour": h,
                    "prob_precip": precip_data.get(h, 0),
                    "precip_intensity": 0,
                    "temp": temp_data.get(h),
                    "symbol": None,
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
