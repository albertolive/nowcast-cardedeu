"""
Sessió HTTP compartida amb reintents automàtics per errors transitoris.
Totes les crides a APIs externes han d'usar create_session() per obtenir
un requests.Session amb retry integrat a nivell de transport.

Meteocat callers pass retry_429=False so the application-level circuit
breaker sees the first response. Other API callers retain the historical
transport retry behavior by default.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session(api_key_header: dict | None = None, *, retry_429: bool = True) -> requests.Session:
    """
    Crea un requests.Session amb retry automàtic per errors transitoris.
    Gestiona: ConnectionError, Timeout, RemoteDisconnected, 502/503/504.
    """
    status_forcelist = [502, 503, 504]
    if retry_429:
        status_forcelist.insert(0, 429)
    retry = Retry(
        total=3,
        backoff_factor=1,           # 1s, 2s, 4s entre reintents
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],    # Només GET (tots els clients són lectura)
        raise_on_status=False,      # Deixar que requests.raise_for_status() gestioni
        respect_retry_after_header=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "NowcastCardedeu/1.0"})

    if api_key_header:
        session.headers.update(api_key_header)

    return session
