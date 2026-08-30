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

## El radar AEMET no publica l'edat del frame

El radar C-banda de Barcelona serveix la imatge sense timestamp de captura ni
hash. Fins a l'agost 2026 el pipeline no podia distingir un frame fresc d'un
de vell servit com a nou, i la sortida només guardava `cached_at` (l'hora de
la nostra descàrrega), que no diu res sobre la imatge mateixa.

- **2026-08-26**: frames amb ecos de 40 dBZ intermitents (l'"eco més proper"
  saltava WNW→SSW→ENE en 40 min) amb 0.0 mm a terra a tot el Vallès. La regla
  5 va disparar el floor del 55% en ràfegues i la predicció va oscil·lar
  55% ↔ 0.8% quatre vegades en una hora. Totes les verificacions van donar
  fals positiu (però al bloc *incert*, doncs sense cost d'accuracy).
- **Cadència en parelles**: amb `RADAR_TTL` de 15 min i runs cada 10 min,
  cada descàrrega nova arriba aproximadament cada 20 min i serveix **exactament
  2 runs consecutius**. Si un valor de radar apareix duplicat en parelles és
  la cache funcionant normal; si el valor canvia entre descàrregues però no
  quadra amb la realitat, sospita de la font.

Mitigacions (des de l'agost 2026): cada descàrrega registra md5, mida i hora
(`aemet_radar_frame_md5` etc., comitades a git dins `aemet_cache.json`) i un
comptador de frames idèntics consecutius (`aemet_radar_same_frame_streak`).
Amb ≥3 descàrregues idèntiques (~1h sense cap canvi a la imatge), la regla 5
s'inhabilita igual que les regles RainViewer amb frames congelats. El md5
històric permet diagnosticar episodis després del fet amb `git log -p` —
l'episodi del 26 d'agost va canviar de md5 a cada descàrrega, així que el
guard NO l'hauria aturat: el que hauria permès tancar-lo era precisament
aquest registre. `physical_adjustments` també es persisteix ara al JSONL
(abans mai: diagnòstic impossible després del fet).

## El retrain sobreescribia les etiquetes verificades amb pseudo-labels NWP

`prepare_training_data` (train.py) reaplicava `build_target_column` sobre el
dataframe fusionat (dataset històric + prediccions verificades del feedback).
Aquesta funció recalcula `will_rain` des de la columna `precipitation`, que a
les files de feedback conté la pluja **prevista pel NWP**, no l'observada.

Mesurat el 2026-08-26 sobre 18.664 files verificades: **116 dels 222 events
de pluja reetiquetats com a secs** i 46 moments secs com a plujosos. El
feedback loop portava mesos entrenant contra etiquetes del propi Open-Meteo:
ensenyava al model a desconfiar exactament dels patrons locals (radar,
sentinella) que havia d'aprendre, i explica que les features de radar
seguissin amb guany zero després de mesos de dades acumulades.

Fix: el target només es deriva si la columna `will_rain` no existeix; si hi
és parcialment NaN, error explícit en lloc d'inventar etiquetes. Test de
regressió: `tests/test_train_target.py`.

Pendent conegut (no confondre amb aquest bug): el merge concat base+feedback
**sense reordenar cronològicament** — el bloc de feedback (mar–ago) queda
després del dataset base que acaba l'agost, així que els folds temporals de
la CV barregen períodes duplicats i infla `cv_f1_std`. No afecta el model
final (entrena sobre tot), però sí la interpretació de les mètriques CV.

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

## VM GCP gratuït mort i Actions cremava quota (2026-08-30)

- **VM `nowcast-bot` últim push `2026-04-21`** - OCI A1 reclamat o `GIT_TOKEN` expirat 90d, 4 mesos sense pushes, `CF Worker` via `workflow_dispatch` era l'únic productor (`github-actions[bot]` cada 10min). Deshabilitar `predict` (`if:false`) va deixar `RAW` `34min stale` i `Vercel` amb fallback també stale.
- **Fix:** `gcloud compute instances create nowcast-vm --project=esdeveniments --zone=us-east1-b --machine-type=e2-micro` `34.139.5.189` sempre-free (`us-*` només, `europe-southwest1-a` `~$8/mo`). `deploy/oci/.env` + arrel `.env` (`chmod 600`) amb 10 vars (`GIT_TOKEN` via `gh auth token` reutilitzat, `TELEGRAM -1003766942798`, `METEOCAT fTVz`, `AEMET eyJ`, `GATEWAY_TOKEN a3a2` de `ai-gateway/.env`). `docker-entrypoint.sh` `/tmp/repo` ja existent causava `exit 128` al `restart` - `down && up -d` va recrear.
- **Billing quota:** nou projecte `nowcast-cardedeu-20260830` `654620029508` va fallar `billing quota exceeded 5/5` `016A84...` - esborrat, reutilitzat `esdeveniments` `381787440315` `0/24`. `e2-micro` `30GB pd-standard` `+swap 2GB` triga `~6min` a build.

## .env no guardat = pèrdua de secrets

`gh api .../actions/secrets` només llista noms, valors irrecuperables. `find .env` buit al Mac. Cal guardar `nowcast-cardedeu/.env` + `deploy/oci/.env` (`chmod 600`, gitignored `.gitignore:5`) per a futur. `quota-guard.yml` diari `80%` evita repetir `16k->3000`.
