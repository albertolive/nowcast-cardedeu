# LESSONS

Gotchas no obvis del projecte. Llegir abans de tocar el pipeline o diagnosticar incidents.

## Dades velles disfressades de fresques (el mode de fallada recurrent)

Cada incident greu del projecte ha estat una variant del mateix patró: una font serveix dades velles com si fossin d'ara, i el sistema se les creu.

- **2026-03-23**: la cache de radar AEMET servia ecos vells → falsos positius.
- **2026-05-17**: el snapshot de Vercel servia prediccions de dies enrere com a actuals (el deploy només es fa amb canvis de frontend; el fallback a raw.githubusercontent del dashboard ho mitiga).
- **2026-06-04**: API AEMET en 429 durant ~3h → radar "no disponible" i regles físiques cegues durant una tempesta. Predicció al 10% plovent a bots i barrals.
- **2026-06-04/05**: RainViewer congelat >12h — frames amb **timestamps nous però contingut MD5 idèntic**. Un eco fantasma de 56 dBZ va mantenir la predicció clavada al 55% tota la nit.
- **2026-08-14**: RainViewer congelat mantenia `radar_nearest_echo_km` a ~7 km (eco espacial fantasma) i obria el **rain gate 24/7** encara que `radar_has_echo` fos fals → quotes XDDE (250/mes) i Predicció (100/mes) esgotades a mitjan mes. Fix: el gate ignora l'eco espacial quan `radar_frames_frozen` és cert. La porta encara s'obre per ensemble, AEMET i CAPE, així que el consum d'estiu continua si aquests llindars són baixos.

**Regla pràctica**: si els valors de radar no canvien gens entre runs (mateix `nearest_echo_km`, mateix `max_dbz` exacte durant >30 min), sospita de la font, no de la meteorologia. La pluja real fluctua sempre. Comprova hashes dels tiles o l'edat de la cache abans de culpar el model.

Mitigacions existents: `radar_frames_frozen` (rainviewer.py — desactiva les regles físiques de RainViewer i, des de l'agost 2026, també exclou l'eco espacial del rain gate a predict.py), `get_stale()` amb límit d'edat (aemet_cache.py), watchdog amb alerta de "cegues totals" quan fallen les dues fonts de radar alhora.

## El model ML ignora radar i llamps

El 85.8% del guany està en 4 features NWP d'Open-Meteo; totes les features de radar (31) i llamps (7) tenen guany **zero** perquè els 7 anys d'entrenament tenien NaN allà. Les regles físiques de `_apply_physical_constraints()` (predict.py) existeixen per suplir-ho fins que el feedback loop acumuli prou dades verificades per reentrenar. No esperis que el model "vegi" una tempesta al radar: no la veu.

## L'API d'AEMET falla quan més la necessites

OpenData d'AEMET és infraestructura compartida que llença 429 globals sota càrrega, exactament durant episodis de mal temps (tothom la consulta alhora). No és culpa del nostre volum (~10 req/h). Defenses: retry curt a `_http.py`, cache stale a `aemet_cache.py`. No afegir més retries: no arreglen una saturació de hores.

## PREC de MeteoCardedeu és total acumulat diari

PREC és el total des de mitjanit, NO pluja per minut. `fetch_series()` el converteix a increments amb `.diff().clip(lower=0)`. Si es toca, recordar que el 2026-03-26 aquest bug va invalidar 52 verificacions (mostrava 17-19mm quan n'havien caigut 0.4). XEMA KX (Meteocat) sí que és per interval.

## La quota XEMA es buida amb l'endpoint per variable

L'endpoint `/variables/mesurades/{var}/{y}/{m}/{d}` retorna **totes les ~200 estacions** per a una sola variable. Amb 3 variables (32/33/35) això són 3 crides per cicle; amb el job cada 10 min i cache de 30 min es cremaven ~144 crides/dia sobre una quota de 750/mes. L'SMC ho va detectar (agost 2026) i va recordar que les estacions publiquen cada 30 min: re-descarregar cada 10 min exporta les mateixes dades.

Fix: `/estacions/mesurades/{estacio}/{y}/{m}/{d}` retorna totes les variables d'una estació en **1 crida** → 2 crides/cicle (YM + KX). Mantenir el TTL de 60 min (== cadència de publicació) i el tallafoc de 429, i no tornar a l'endpoint per variable fora dels tests.

El tallafoc de 429 és **per servei** (xdde/smc/xema/quota), no global. Abans era compartit i un 429 de llamps (XDDE) bloquejava també el XEMA, deixant la sentinella buida en plena finestra de pluja. Si una font surt nul·la amb rain gate obert, mira primer si un altre servei ha marcat el seu propi 429, no si el XEMA en va rebre un.

## Vercel no es redesplega amb cada predicció

`deploy-dashboard.yml` només desplega amb canvis de frontend (expressament: límit de 100 deploys/dia). El JSON local de Vercel pot tenir setmanes; el dashboard té un fallback a raw.githubusercontent quan el local té >30 min (`app.js`). Si el web mostra valors vells, el primer sospitós és aquest fallback fallant (adblocker, rate limit de GitHub), no el pipeline.
