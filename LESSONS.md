# LESSONS

Gotchas no obvis del projecte. Llegir abans de tocar el pipeline o diagnosticar incidents.

## Dades velles disfressades de fresques (el mode de fallada recurrent)

Cada incident greu del projecte ha estat una variant del mateix patró: una font serveix dades velles com si fossin d'ara, i el sistema se les creu.

- **2026-03-23**: la cache de radar AEMET servia ecos vells → falsos positius.
- **2026-05-17**: el snapshot de Vercel servia prediccions de dies enrere com a actuals (el deploy només es fa amb canvis de frontend; el fallback a raw.githubusercontent del dashboard ho mitiga).
- **2026-06-04**: API AEMET en 429 durant ~3h → radar "no disponible" i regles físiques cegues durant una tempesta. Predicció al 10% plovent a bots i barrals.
- **2026-06-04/05**: RainViewer congelat >12h — frames amb **timestamps nous però contingut MD5 idèntic**. Un eco fantasma de 56 dBZ va mantenir la predicció clavada al 55% tota la nit.

**Regla pràctica**: si els valors de radar no canvien gens entre runs (mateix `nearest_echo_km`, mateix `max_dbz` exacte durant >30 min), sospita de la font, no de la meteorologia. La pluja real fluctua sempre. Comprova hashes dels tiles o l'edat de la cache abans de culpar el model.

Mitigacions existents: `radar_frames_frozen` (rainviewer.py, desactiva les regles físiques de RainViewer), `get_stale()` amb límit d'edat (aemet_cache.py), watchdog amb alerta de "cegues totals" quan fallen les dues fonts de radar alhora.

## El model ML ignora radar i llamps

El 85.8% del guany està en 4 features NWP d'Open-Meteo; totes les features de radar (31) i llamps (7) tenen guany **zero** perquè els 7 anys d'entrenament tenien NaN allà. Les regles físiques de `_apply_physical_constraints()` (predict.py) existeixen per suplir-ho fins que el feedback loop acumuli prou dades verificades per reentrenar. No esperis que el model "vegi" una tempesta al radar: no la veu.

## L'API d'AEMET falla quan més la necessites

OpenData d'AEMET és infraestructura compartida que llença 429 globals sota càrrega, exactament durant episodis de mal temps (tothom la consulta alhora). No és culpa del nostre volum (~10 req/h). Defenses: retry curt a `_http.py`, cache stale a `aemet_cache.py`. No afegir més retries: no arreglen una saturació de hores.

## PREC de MeteoCardedeu és total acumulat diari

PREC és el total des de mitjanit, NO pluja per minut. `fetch_series()` el converteix a increments amb `.diff().clip(lower=0)`. Si es toca, recordar que el 2026-03-26 aquest bug va invalidar 52 verificacions (mostrava 17-19mm quan n'havien caigut 0.4). XEMA KX (Meteocat) sí que és per interval.

## Vercel no es redesplega amb cada predicció

`deploy-dashboard.yml` només desplega amb canvis de frontend (expressament: límit de 100 deploys/dia). El JSON local de Vercel pot tenir setmanes; el dashboard té un fallback a raw.githubusercontent quan el local té >30 min (`app.js`). Si el web mostra valors vells, el primer sospitós és aquest fallback fallant (adblocker, rate limit de GitHub), no el pipeline.
