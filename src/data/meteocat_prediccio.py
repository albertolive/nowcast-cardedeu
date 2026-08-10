"""
Client per a la Predicció Municipal horària del Meteocat (SMC).
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data._http import create_session

logger = logging.getLogger(__name__)
SESSION = create_session(retry_429=False)


def _headers() -> dict:
    return {"X-Api-Key": config.METEOCAT_API_KEY}


def _is_configured() -> bool:
    return bool(config.METEOCAT_API_KEY)


def _rate_limited() -> bool:
    from src.data.meteocat_cache import is_meteocat_rate_limited
    return is_meteocat_rate_limited("smc")


def _mark_rate_limited() -> None:
    from src.data.meteocat_cache import mark_meteocat_rate_limited
    mark_meteocat_rate_limited("smc")


def _is_429(exc, response=None) -> bool:
    return (getattr(getattr(exc, "response", None), "status_code", None) == 429
            or getattr(response, "status_code", None) == 429)


def fetch_municipal_hourly_forecast() -> dict:
    """Fetch the 72h municipal forecast, with a shared breaker and TTL cache."""
    result = _empty_forecast()
    if not _is_configured():
        logger.info("Meteocat Predicció no configurat (sense METEOCAT_API_KEY)")
        return result

    from src.data.meteocat_cache import get_cached, get_stale_cached, set_cached
    cache_key = f"smc_forecast_{datetime.now().strftime('%Y%m%d_%H')}"
    cached = get_cached(cache_key, config.METEOCAT_CACHE_TTL_SMC)
    if cached is not None:
        logger.info("SMC forecast: using cached response")
        return cached

    if _rate_limited():
        stale = get_stale_cached(cache_key, config.METEOCAT_SMC_STALE_MAX_MIN)
        if stale is not None:
            logger.warning("SMC cooldown active — using bounded stale forecast")
            return stale
        logger.warning("SMC cooldown active — skipping request")
        return result

    url = (f"{config.METEOCAT_BASE_URL}/pronostic/v1/"
           f"municipalHoraria/{config.METEOCAT_MUNICIPALITY_CODE}")
    response = None
    try:
        response = SESSION.get(url, headers=_headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            set_cached(cache_key, result)
            return result
        hourly_forecasts = _extract_hourly(data)
        if not hourly_forecasts:
            logger.warning("No s'han pogut extreure dades horàries del Meteocat")
            set_cached(cache_key, result)
            return result
        now = datetime.now()
        current = next((hf for hf in hourly_forecasts
                        if hf.get("date") == now.date() and hf.get("hour") == now.hour),
                       hourly_forecasts[0])
        result["smc_prob_precip_1h"] = current.get("prob_precip", 0)
        result["smc_temp_forecast"] = current.get("temp")
        result["smc_weather_symbol"] = current.get("symbol")
        target_end = now + timedelta(hours=6)
        relevant = []
        for hf in hourly_forecasts:
            hf_date, hf_hour = hf.get("date"), hf.get("hour", -1)
            if hf_date is None or hf_hour < 0:
                continue
            hf_dt = datetime.combine(hf_date, datetime.min.time()).replace(hour=hf_hour)
            if now <= hf_dt <= target_end:
                relevant.append(hf)
        if relevant:
            result["smc_prob_precip_6h"] = max(hf.get("prob_precip", 0) for hf in relevant)
            result["smc_precip_intensity"] = max((hf.get("precip_intensity", 0) for hf in relevant), default=0)
        set_cached(cache_key, result)
        return result
    except Exception as exc:
        if _is_429(exc, response):
            _mark_rate_limited()
        stale = get_stale_cached(cache_key, config.METEOCAT_SMC_STALE_MAX_MIN)
        if stale is not None:
            logger.warning("SMC request failed — using bounded stale forecast: %s", exc)
            return stale
        logger.warning("Error obtenint Meteocat Predicció: %s", exc)
        return result


def _extract_hourly(data: dict) -> list[dict]:
    """Extreu les previsions horàries del format real de l'API."""
    hourly = []
    for dia in data.get("dies", []):
        variables = dia.get("variables", {})
        if not isinstance(variables, dict):
            continue
        temp_by_dt, precip_by_dt, symbol_by_dt, humidity_by_dt = {}, {}, {}, {}
        for var_name, var_data in variables.items():
            for value in var_data.get("valors", []):
                dt_str, val = value.get("data", ""), value.get("valor")
                if not dt_str or val is None:
                    continue
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    key = (dt.date(), dt.hour)
                except (ValueError, AttributeError):
                    continue
                if var_name == "temp": temp_by_dt[key] = float(val)
                elif var_name == "precipitacio": precip_by_dt[key] = float(val)
                elif var_name == "estatCel": symbol_by_dt[key] = val
                elif var_name == "humitat": humidity_by_dt[key] = float(val)
        all_keys = set(temp_by_dt) | set(precip_by_dt) | set(symbol_by_dt) | set(humidity_by_dt)
        for date_part, hour in sorted(all_keys):
            precip_mm = precip_by_dt.get((date_part, hour), 0)
            hourly.append({
                "date": date_part, "hour": hour,
                "prob_precip": min(precip_mm * 100, 100) if precip_mm > 0 else 0,
                "precip_intensity": precip_mm,
                "temp": temp_by_dt.get((date_part, hour)),
                "symbol": symbol_by_dt.get((date_part, hour)),
            })
    return hourly


def _empty_forecast() -> dict:
    return {
        "smc_prob_precip_1h": np.nan, "smc_prob_precip_6h": np.nan,
        "smc_precip_intensity": np.nan, "smc_temp_forecast": np.nan,
        "smc_weather_symbol": np.nan,
    }


def _hourly_to_dataframe(hourly_forecasts):
    import pandas as pd
    rows = []
    for hf in hourly_forecasts:
        hour, date_part = hf.get("hour"), hf.get("date")
        if hour is None or date_part is None:
            continue
        rows.append({
            "datetime": datetime(date_part.year, date_part.month, date_part.day, hour).isoformat(),
            "smc_prob_precip_1h": hf.get("prob_precip", np.nan),
            "smc_precip_intensity": hf.get("precip_intensity", np.nan),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["smc_prob_precip_6h"] = (df["smc_prob_precip_1h"].rolling(6, min_periods=1)
                                 .max().shift(-5).fillna(df["smc_prob_precip_1h"]))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _restore_hourly_dataframe(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_smc_hourly_df() -> "pd.DataFrame":
    """Fetch SMC hourly data as a cached DataFrame; never raises on API failure."""
    import pandas as pd
    if not _is_configured():
        return pd.DataFrame()
    from src.data.meteocat_cache import get_cached, get_stale_cached, set_cached
    cache_key = f"smc_hourly_{datetime.now().strftime('%Y%m%d_%H')}"
    cached = get_cached(cache_key, config.METEOCAT_CACHE_TTL_SMC)
    if cached is not None:
        return _restore_hourly_dataframe(cached)
    stale = lambda: get_stale_cached(cache_key, config.METEOCAT_SMC_STALE_MAX_MIN)
    if _rate_limited():
        old = stale()
        return _restore_hourly_dataframe(old) if old is not None else pd.DataFrame()
    url = (f"{config.METEOCAT_BASE_URL}/pronostic/v1/"
           f"municipalHoraria/{config.METEOCAT_MUNICIPALITY_CODE}")
    response = None
    try:
        response = SESSION.get(url, headers=_headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
        if not data:
            set_cached(cache_key, [])
            return pd.DataFrame()
        df = _hourly_to_dataframe(_extract_hourly(data))
        if df.empty:
            set_cached(cache_key, [])
            return df
        set_cached(cache_key, df.assign(datetime=df["datetime"].astype(str)).to_dict("records"))
        return df
    except Exception as exc:
        if _is_429(exc, response):
            _mark_rate_limited()
        old = stale()
        if old is not None:
            return _restore_hourly_dataframe(old)
        logger.warning("Error obtenint SMC hourly forecast: %s", exc)
        return pd.DataFrame()
