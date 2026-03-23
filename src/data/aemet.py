"""
Client per a l'API d'AEMET OpenData.
Obté previsions horàries amb probabilitat de precipitació i tempesta
per al municipi de Cardedeu (08052).
Requereix API key gratuïta: https://opendata.aemet.es/centrodedescargas/altaUsuario
"""
import logging
from datetime import datetime

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session
from src.data.aemet_cache import get_cached, set_cached, FORECAST_TTL

logger = logging.getLogger(__name__)

SESSION = create_session({"api_key": config.AEMET_API_KEY})


def _aemet_fetch(endpoint: str) -> dict | list | None:
    """
    Fetch AEMET amb el patró de 2 passos: primer obtenim la URL de dades,
    després descarreguem les dades reals.
    """
    if not config.AEMET_API_KEY:
        logger.warning("AEMET_API_KEY no configurada")
        return None

    r = SESSION.get(f"{config.AEMET_BASE_URL}{endpoint}", timeout=15)
    r.raise_for_status()
    meta = r.json()

    if meta.get("estado") != 200:
        logger.warning(f"AEMET error: {meta.get('descripcion', 'unknown')}")
        return None

    datos_url = meta.get("datos")
    if not datos_url:
        return None

    r2 = SESSION.get(datos_url, timeout=15)
    r2.raise_for_status()
    return r2.json()


def fetch_hourly_forecast() -> dict:
    """
    Obté la previsió horària d'AEMET per Cardedeu.
    Extreu probPrecipitacion i probTormenta per a les properes hores.

    Retorna dict amb:
      - aemet_prob_precip: màxima prob. de precipitació properes 6h (0-100)
      - aemet_prob_storm: màxima prob. de tempesta properes 6h (0-100)
      - aemet_precip_today: prob. de precipitació avui (0-100)
    """
    try:
        # Check cache first (AEMET forecast updates every ~6-12h)
        cached = get_cached("forecast", FORECAST_TTL)
        if cached is not None:
            return cached

        data = _aemet_fetch(
            f"/prediccion/especifica/municipio/horaria/{config.AEMET_MUNICIPALITY_CODE}"
        )

        if not data or not isinstance(data, list):
            raise ValueError("Resposta AEMET buida o inesperada")

        # L'API retorna una llista amb 1 element que conté la predicció
        pred = data[0].get("prediccion", {})
        dias = pred.get("dia", [])

        if not dias:
            raise ValueError("Cap dia a la predicció AEMET")

        now = datetime.now()
        current_hour = now.hour

        max_prob_precip = 0
        max_prob_storm = 0
        precip_today = 0

        # Mirar avui i demà (per cobrir les properes 6h)
        for dia in dias[:2]:
            # probPrecipitacion: llista de {periodo, valor}
            for pp in dia.get("probPrecipitacion", []):
                periodo = pp.get("periodo", "")
                valor = int(pp.get("value", 0) or 0)

                # Parsejar el període (format "0006", "0612", "1218", "1824" o "00-24")
                if len(periodo) == 4:
                    h_start = int(periodo[:2])
                    h_end = int(periodo[2:])
                    # Es rellevant si cobreix les properes 6h?
                    if h_start <= current_hour + 6 and h_end > current_hour:
                        max_prob_precip = max(max_prob_precip, valor)
                elif periodo == "" or periodo == "00-24":
                    precip_today = max(precip_today, valor)

            # probTormenta: llista de {periodo, valor}
            for pt in dia.get("probTormenta", []):
                periodo = pt.get("periodo", "")
                valor = int(pt.get("value", 0) or 0)

                if len(periodo) == 4:
                    h_start = int(periodo[:2])
                    h_end = int(periodo[2:])
                    if h_start <= current_hour + 6 and h_end > current_hour:
                        max_prob_storm = max(max_prob_storm, valor)

        result = {
            "aemet_prob_precip": max_prob_precip,
            "aemet_prob_storm": max_prob_storm,
            "aemet_precip_today": precip_today,
        }

        logger.info(
            f"AEMET: probPrecip={max_prob_precip}%, "
            f"probTormenta={max_prob_storm}%, "
            f"precipAvui={precip_today}%"
        )
        set_cached("forecast", result)
        return result

    except Exception as e:
        logger.warning(f"Error obtenint AEMET: {e}")
        return {
            "aemet_prob_precip": np.nan,
            "aemet_prob_storm": np.nan,
            "aemet_precip_today": np.nan,
        }
