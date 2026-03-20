# 🌧️ Nowcast Cardedeu

Sistema de predicció de pluja hiperlocal per a Cardedeu (Vallès Oriental) basat en Machine Learning.

Utilitza dades reals de l'estació [MeteoCardedeu.net](https://meteocardedeu.net) combinades amb models meteorològics globals (Open-Meteo), acord entre múltiples models (ECMWF, GFS, ICON, AROME), radar de precipitació en temps real (RainViewer), estacions sentinella del SMC (Meteocat XEMA), descàrregues elèctriques (Meteocat XDDE), radar C-banda Barcelona (AEMET), predicció municipal del SMC, probabilitats de tempesta calibrades per experts (AEMET), classificació de règims eòlics catalans (Llevantada, Garbí, Ponent), índexs d'inestabilitat (VT, TT, Lifted Index), cisalla de vent i llindars d'aire fred per aprendre els patrons del microclima local i predir si plourà en els propers 60 minuts.

## Com funciona

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  MeteoCardedeu   │  │ Ensemble 4 NWP  │  │   RainViewer     │
│  (dades reals)   │  │ECMWF+GFS+ICON  │  │  (radar precip)  │
└────────┬─────────┘  │   +AROME 2.5km  │  └────────┬─────────┘
         │            └────────┬─────────┘           │
┌──────────────────┐  │  ┌──────────────────┐  │
│  Meteocat XEMA   │  │  │     AEMET       │  │
│ (si rain gate    │  │  │  probTormenta   │  │
│  està obert)     │  │  │  probPrecip     │  │
└────────┬─────────┘  │  └────────┬─────────┘  │
         │                     │         │                      │
┌──────────────────┐  │  ┌──────────────────┐  │
│ Meteocat XDDE   │  │  │  AEMET Radar    │  │
│ (llamps)        │  │  │  (C-banda BCN)  │  │
└────────┬─────────┘  │  └────────┬─────────┘  │
         │                     │         │                      │
┌──────────────────┐  │  │                              │
│ SMC Predicció   │  │  │                              │
│ (municipal)     │  │  │                              │
└────────┬─────────┘  │  │                              │
         │                     │         │                      │
         ▼                     ▼         ▼                      ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                Feature Engineering (120 features)                    │
    │  Tendències · Ensemble · 850hPa · Radar · Sentinella · Llamps · Vent │
    └──────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │     XGBoost       │
                        │  + Isotònic Cal.  │
                        └────────┬──────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  Probabilitat de │──→ 🔔 Telegram
                       │  pluja (0-100%)  │
                       └──────────────────┘
                                 │
                    Resum diari / Accuracy
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  Narrativa IA    │──→ 💬 Català fluid
                       │ (GitHub Models)  │     (1 crida/dia)
                       └──────────────────┘
```

## Filosofia

El model **no intenta predir el temps des de zero**. El que fa és:
1. Rebre el que diuen els models globals (Open-Meteo)
2. Comparar-ho amb les condicions reals mesurades a Cardedeu
3. **Corregir els errors dels models** basant-se en patrons apresos de +10 anys d'històric

Per exemple, aprèn coses com:
- "Quan el model diu pluja però el vent de Cardedeu és sec del Montseny → no plourà"
- "Quan la pressió baixa ràpidament + humitat >85% + vent del SE (Llevantada) → plou sempre aquí"

## Setup

### 1. Instal·lar dependències
```bash
pip install -r requirements.txt
```

### 2. Entrenar el model (primera vegada)
```bash
# Pas 1: Descarregar 12 anys d'històric
python scripts/download_history.py

# Pas 2: Construir el dataset d'entrenament
python scripts/build_dataset.py

# Pas 3: Entrenar XGBoost
python scripts/train_model.py
```

### 3. Fer una predicció
```bash
python scripts/predict_now.py
```

### 4. Configurar API Meteocat (recomanat)
Per obtenir dades de les estacions sentinella del SMC:
1. Demana una clau API gratuïta a [apidocs.meteocat.gencat.cat](https://apidocs.meteocat.gencat.cat)
2. Configura la variable d'entorn:
```bash
export METEOCAT_API_KEY="la_teva_clau"
```
> Sense clau, el sistema funciona igualment. Meteocat només es consulta quan el **rain gate** detecta senyals de pluja (estalvi d'API).

### 5. Configurar API AEMET (recomanat)
Per obtenir probabilitats de tempesta calibrades per experts:
1. Registra't gratuïtament a [opendata.aemet.es](https://opendata.aemet.es/centrodedescargas/altaUsuario)
2. Configura la variable d'entorn:
```bash
export AEMET_API_KEY="la_teva_clau"
```

### 6. Configurar alertes Telegram (opcional)
1. Crea un bot amb [@BotFather](https://t.me/BotFather)
2. Configura les variables d'entorn:
```bash
export TELEGRAM_BOT_TOKEN="el_teu_token"
export TELEGRAM_CHAT_ID="el_teu_chat_id"
```

### 7. GitHub Actions (automatització)
El workflow `.github/workflows/nowcast.yml`:
- **Prediccions** cada 10 minuts (6h-23h) amb notificacions intel·ligents
- **Resum diari** a les 7:00 via Telegram
- **Informe d'accuracy** setmanal (dilluns 8:00) via Telegram
- **Re-entrenament** automàtic diari a les 3:00 (amb feedback loop + calibratge isotònic)
- Execució manual amb selector d'acció (predict / daily_summary / accuracy_report / retrain)
- Configura els secrets al repositori:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
  - `METEOCAT_API_KEY`
  - `AEMET_API_KEY`
  - `AI_GITHUB_TOKEN` usa automàticament `GITHUB_TOKEN` (gratuït, sense configuració extra)
  - `AI_OPENROUTER_KEY` (opcional, fallback a models gratuïts d'OpenRouter)

## Estructura

```
nowcast-cardedeu/
├── config.py                 # Configuració central (URLs, coordenades, llindars)
├── src/
│   ├── data/
│   │   ├── meteocardedeu.py  # API meteocardedeu.net (sèries minut a minut + NOAA)
│   │   ├── open_meteo.py     # API Open-Meteo (històric + forecast + pressure levels)
│   │   ├── ensemble.py       # Acord entre ECMWF/GFS/ICON/AROME + forecast bias
│   │   ├── rainviewer.py     # API RainViewer (radar precipitació + màscara clutter)
│   │   ├── aemet.py          # API AEMET OpenData (probTormenta/probPrecip)
│   │   ├── aemet_radar.py    # API AEMET Radar C-banda Barcelona
│   │   ├── meteocat.py       # API Meteocat XEMA (sentinella, gated by rain gate)
│   │   ├── meteocat_xdde.py  # API Meteocat XDDE (descàrregues elèctriques)
│   │   └── meteocat_prediccio.py # API Meteocat Predicció (forecast municipal)
│   ├── features/
│   │   ├── engineering.py    # Feature engineering (120 features, 76 historical)
│   │   └── regime.py         # Detecció de canvis de règim atmosfèric (Llevantada, Garbí, pressió)
│   ├── model/
│   │   ├── train.py          # Pipeline d'entrenament (XGBoost + TimeSeriesSplit)
│   │   └── predict.py        # Predicció en temps real (fusió 6 fonts + rain gate)
│   ├── ai/
│   │   └── enricher.py       # Narratives IA en català (GitHub Models + OpenRouter fallback)
│   ├── notify/
│   │   ├── telegram.py       # Missatges Telegram (3 tipus: alerta, clearing, resum)
│   │   └── state.py          # Màquina d'estats per notificacions (histèresi + cooldown)
│   └── feedback/
│       ├── logger.py         # Log JSONL de cada predicció
│       ├── verify.py         # Verificació automàtica (predicció vs realitat)
│       ├── accuracy.py       # Mètriques d'accuracy acumulades
│       └── export.py         # Exporta verificacions per reentrenar
├── scripts/
│   ├── download_history.py   # Descarregar 12+ anys d'històric
│   ├── build_dataset.py      # Construir dataset d'entrenament
│   ├── train_model.py        # Entrenar model (amb feedback loop + calibratge isotònic)
│   ├── predict_now.py        # Predicció + log + verificació (GitHub Actions)
│   ├── daily_summary.py      # Resum diari ML-powered (7:00) amb previsió per franges
│   ├── accuracy_report.py    # Informe setmanal d'accuracy (dilluns 8:00)
│   └── backfill_lightning.py  # Backfill històric de llamps XDDE al dataset
├── models/                   # Model entrenat (git tracked)
├── data/                     # Dades + logs de prediccions
├── requirements.txt          # Dependències Python
└── .github/workflows/        # Automatització cada 10 min
```

## Features del model

El model defineix **120 features** per predicció en temps real. El model s'entrena amb les **120 features completes** (76 amb dades històriques, 44 com a NaN per radar/llamps/AEMET). El **feedback loop** acumula gradualment les 44 features en temps real (radar, llamps, sentinella) a cada predicció verificada, permetent que el model aprengui d'observacions independents amb cada re-entrenament.

**Ensemble backfill**: Des de gener 2022, dades de 4 models NWP (ECMWF, GFS, ICON, AROME) descarregades via `scripts/backfill_ensemble.py`.
**XEMA sentinel backfill**: Dades de Granollers (YM) + ETAP Cardedeu (KX) via `scripts/backfill_xema.py` (incremental, 15 dies/execució per respectar el límit API).

| Categoria | Features | Per què? |
|-----------|----------|----------|
| Temporals | Hora, mes (codificació cíclica) | Patrons estacionals i diaris |
| Pressió | Valor + tendència 1h/3h/6h + acceleració | Indicador principal d'inestabilitat |
| Humitat | Valor + punt rosada + depressió + tendència + VPD + vpd_change_3h | Saturació = pluja imminent. VPD=0 → aire saturat |
| Vent | Components U/V + canvis + marinada | Marinada del mar = aire sec |
| Règims eòlics | Tramuntana, Llevantada, Migjorn, Garbí, Ponent (850hPa) + garbi_strength | Llevantada (E/SE) = pluja #1 a Cardedeu |
| Nivells pressió | Vent/T/RH a 850hPa, T/RH a 700hPa, T a 500hPa | Flux sinòptic real a 3 nivells |
| 🆕 Índexs inestabilitat | VT, TT, LI, li_unstable, li_very_unstable | Skew-T: detecció de convecció severa |
| 🆕 Cisalla de vent | wind_shear_speed, wind_shear_dir | Tempestes organitzades (supercèl·lules) |
| 🆕 Aire fred 500hPa | cold_500_moderate (<-17°C), cold_500_strong (<-24°C) | "Petita bomba" mediterrània (alexmeteo) |
| Pluja recent | Acumulat 3h/6h + ha plogut? | Context de fronts actius |
| Models NWP | CAPE, núvols, weather code, descomposició WC (thunderstorm/rain/drizzle) | Què diuen els models globals |
| 🆕 Detecció d'error NWP | nwp_dry_conflict, nwp_wet_conflict | Quan el NWP es contradiu amb les condicions reals |
| 🆕 Física convecció | moisture_flux_850, theta_e_deficit, cape_change_3h | Transport d'humitat, inestabilitat convectiva, destabilització ràpida |
| Radiació | Solar W/m² | Indicador indirecte de núvols |
| Radar | Intensitat, dBZ, mm/h, eco, tendència, aprox. | Precipitació real en temps real |
| Sentinella | Temp/hum Granollers + diffs amb Cardedeu + precip | Gradient territorial = front actiu |
| Ensemble | Acord ECMWF/GFS/ICON/AROME, spread precip, models pluja | Desacord = incertesa = zona ambigua |
| Bias | Forecast-observat temp/hum en temps real | Model biased = atmosfera impredictible |
| AEMET | probPrecipitació, probTormenta (experts) | Tempestes convectives mediterrànies |
| 🆕 Llamps (XDDE) | Count 30km/15km, distància, approaching, cloud-ground, corrent màxim | Activitat convectiva directa |
| 🆕 Radar AEMET | dBZ, eco, distància eco, cobertura 20km | Radar C-banda Barcelona (alta resolució) |
| 🆕 SMC Predicció | prob_precip_1h, prob_precip_6h, intensitat | Previsió municipal calibrada per Cardedeu |
| 🆕 Radar quadrants | max_dbz i cobertura per N/E/S/W | Consciència direccional: d'on ve la pluja |
| 🆕 Echo bearing | sin/cos del rumb de l'eco més proper | Direcció de la pluja codificada cíclicament |
| ❄️ Tramuntana | tramuntana_strength, tramuntana_moisture | Vent polar fred del nord, supressor de pluja (5.8% rain rate) |

## Fonts de dades

| Font | Tipus | Freqüència | Clau API |
|------|-------|------------|----------|
| [MeteoCardedeu.net](https://meteocardedeu.net) | Estació local (T, H, P, vent, pluja) | Cada minut, des de 2012 | No |
| [Open-Meteo](https://open-meteo.com) | Models NWP (GFS, ECMWF) - històric + forecast | Horària | No |
| [Open-Meteo Ensemble](https://open-meteo.com) | Acord ECMWF vs GFS vs ICON vs AROME | Cada predicció | No |
| [RainViewer](https://www.rainviewer.com/api.html) | Radar de precipitació compost (mosaic global) | Cada ~10 min | No |
| [AEMET OpenData](https://opendata.aemet.es) | probTormenta + probPrecipitació calibrades | Cada 6h | Sí (gratuïta) |
| [Meteocat XEMA](https://apidocs.meteocat.gencat.cat) | Estacions sentinella SMC (Granollers, ETAP Cardedeu) | Cada 30 min | Sí (gratuïta) |
| [Meteocat XDDE](https://apidocs.meteocat.gencat.cat) | Descàrregues elèctriques (llamps) a Catalunya | Temps real | Sí (gratuïta) |
| [AEMET Radar](https://opendata.aemet.es) | Radar C-banda regional Barcelona | Cada ~10 min | Sí (gratuïta) |
| [SMC Predicció](https://apidocs.meteocat.gencat.cat) | Previsió municipal horària (prob. precip) | Cada 6h | Sí (gratuïta) |
| [GitHub Models](https://github.com/marketplace/models) | IA narrativa (gpt-4o-mini) per resums diaris | 1 crida/dia | No (GITHUB_TOKEN) |

### Radar RainViewer
El sistema descarrega tiles de radar (zoom 8, tile 134/94) i fa dues coses:

1. **Detecció puntual**: Extreu la intensitat al píxel exacte de Cardedeu (dBZ, mm/h).
2. **Escaneig espacial (30 km)**: Analitza tots els píxels en un radi de 30 km, detectant ecos de pluja, la seva distància, direcció (punt cardinal), i cobertura. Amb el vent a 850 hPa, prioriza el sector de sobrevent (d'on esperem la pluja).
3. **Tracking de tempesta**: Compara el centroide dels ecos entre 6 frames (~1h) per estimar la velocitat de les cel·les de pluja, si s'acosten i l'ETA a Cardedeu.

Converteix intensitat PNG → dBZ → mm/h (Marshall-Palmer). Cada frame ≈ 10 minuts.

### Estacions sentinella Meteocat
Utilitza l'estació de **Granollers (YM)** com a sentinella: si plou a Granollers (7 km al SO), és probable que arribi a Cardedeu en pocs minuts. També consulta el **pluviòmetre ETAP Cardedeu (KX)** a 1.5 km del centre. Les features de diferencial (temperatura, humitat) entre Granollers i Cardedeu detecten fronts que travessen la zona.

### Coordenades
- **Cardedeu**: 41.633°N, 2.364°E, 190m alt
- **Granollers (sentinella)**: 41.608°N, 2.288°E
- **ETAP Cardedeu (pluviòmetre)**: ~41.63°N, ~2.36°E

### Rain gate (estalvi d'API)

Meteocat té quotes mensuals separades per servei (reset dia 1 a 00:00 UTC):

| Servei | Quota | Endpoint |
|--------|-------|----------|
| XEMA (estacions) | 750/mes | `/xema/v1/variables/mesurades/{var}/{YYYY}/{MM}/{DD}` |
| XDDE (llamps) | 250/mes | `/xdde/v1/catalunya/{YYYY}/{MM}/{DD}/{HH}` |
| Predicció | 100/mes | `/pronostic/v1/municipalHoraria/080462` |
| Consum actual | 300/mes | `/quotes/v1/consum-actual` |

**Totes** les crides Meteocat (XDDE, Predicció, XEMA) estan darrere d'un **rain gate** que només les activa quan almenys un senyal independent indica risc de pluja:

| Senyal | Llindar | Font |
|--------|---------|------|
| Ensemble rain agreement | ≥ 25% dels models | ECMWF + GFS + ICON + AROME |
| Radar echo | Qualsevol eco detectat | RainViewer |
| Radar AEMET | Qualsevol eco detectat | AEMET radar Barcelona |
| AEMET prob. tempesta | ≥ 10% | AEMET OpenData |
| CAPE (energia convectiva) | ≥ 800 J/kg | Open-Meteo GFS |

Quan el gate s'obre, es consulten les fonts Meteocat: llamps (XDDE), predicció municipal (SMC) i estacions sentinella (XEMA).

Amb ~8 dies de pluja/mes a Cardedeu i cache TTL, el consum real queda dins de les quotes. Els scripts de backfill comproven la quota via `get_remaining()` abans d'executar-se.

### Règims eòlics catalans

Cardedeu se situa al peu de la Serralada Prelitoral, a la confluència d'aire mediterrani i continental. La direcció del vent és un predictor clau de pluja:

| Règim | Direcció | Efecte a Cardedeu | Feature |
|-------|----------|-------------------|---------|
| 🌊 **Llevantada** | E/SE (60°-150°) | Humitat mediterrània contra les muntanyes → pluja #1 (14.1%) | `is_llevantada`, `llevantada_strength`, `llevantada_moisture` |
| 🌀 **Garbí/Xaloc** | SW (190°-250°) | Aire càlid inestable → tempestes convectives (10.9%) | `is_garbi`, `garbi_strength` |
| ☀️ **Migjorn** | S (150°-190°) | Aire africà càlid, segon en pluja (14.8%) | `is_migjorn` |
| 🏔️ **Ponent/Mestral** | W/NW (260°-340°) | Aire sec continental (Foehn) → supressor de pluja (5.7%) | `is_ponent` |
| ❄️ **Tramuntana** | N (340°-30°) | Vent polar fred del Montseny → supressor de pluja (5.8%) | `is_tramuntana`, `tramuntana_strength`, `tramuntana_moisture` |
| 🔄 **Backing wind** | Gir antihorari | Aproximació de front càlid o baixa → pluja imminent | `wind_dir_change_3h` (negatiu) |

La **Llevantada** és el patró més important: quan el vent bufa de l'est amb humitat alta, la pluja a Cardedeu és quasi segura. El model captura aquesta interacció amb `llevantada_moisture` = is_llevantada × humitat relativa.

### Índexs d'inestabilitat (Skew-T)

A més del vent a 850hPa, el sistema obté dades de temperatura a 850hPa i 500hPa per calcular índexs clàssics de radiosondatge ([ref: Anàlisis Skew-T](https://alexmeteo.com/2025/07/11/analisis-dun-radiosondatge-diagrames-termodinamics-skew-t/)):

| Índex | Fórmula | Significat |
|-------|---------|------------|
| **VT** (Vertical Totals) | T850 − T500 | Gradient tèrmic vertical. >26: inestable, >30: tempestes, >34: forta inestabilitat |
| **TT** (Total Totals) | VT + (Td850 − T500) | Combina gradient + humitat. >44: tronades possibles, >50: tempestes probables, >55: severes |
| **LI** (Lifted Index) | T_ambient_500 − T_parcel_500 | Estabilitat convectiva. <0: inestable, <-2: tempestes, <-6: severes |

### Cisalla de vent i aire fred (alexmeteo.com)

Inspirats per l'anàlisi del blog [alexmeteo.com](https://alexmeteo.com), el sistema inclou indicadors derivats d'articles tècnics sobre meteorologia mediterrània:

| Feature | Descripció | Referència |
|---------|-----------|------------|
| `wind_shear_speed` | Diferència de velocitat de vent entre 850hPa i superfície | "Ingredients per formar Tempestes" — cisalla necessària per organitzar tempestes |
| `wind_shear_dir` | Diferència de direcció entre 850hPa i superfície | Cisalla direccional indica rotació → supercel·les |
| `cold_500_moderate` | T500 < -17°C | "Canvi radical de temps" — -17°C a 500hPa = "petita bomba" a l'estiu |
| `cold_500_strong` | T500 < -24°C | "Quines situacions sinòptiques" — -24°C a 500hPa = bomba convectiva a la primavera |
| `li_unstable` | LI < -2 | Lifted Index negatiu = conveció probable |
| `li_very_unstable` | LI < -6 | LI molt negatiu = tempestes severes |
| `garbi_strength` | is_garbi × velocitat sinòptica | "Anuncia borrasques amb fortes precipitacions" |
| `rh_700` | Humitat relativa a 700hPa | "Baixa mediterrània" — aire sec a 700hPa inhibeix conveció |
| `temp_700` | Temperatura a 700hPa | Perfil tèrmic vertical complet |

### AROME: resolució 2.5km

El model AROME de Meteo-France és el 4t model de l'ensemble, amb resolució de 2.5km (vs 9km d'ECMWF). Això li permet resoldre cel·les convectives individuals i efectes orogràfics a la Serralada Prelitoral que els models globals no veuen.

## Rendiment del model

| Mètrica | Valor |
|---------|-------|
| AUC-ROC (CV) | 0.9547 ± 0.008 |
| F1-Score (CV) | 0.6731 ± 0.032 |
| F1-Score OOF (calibrat) | 0.6931 |
| AUC-ROC (final) | 0.9668 |
| Llindar òptim (calibrat) | 0.3742 |
| Mostres d'entrenament | 98,310 |
| Features (training) | 120 |
| Features (total) | 120 |
| Classe positiva (pluja) | ~9.3% |
| Cross-validation | TimeSeriesSplit (5 folds) |
| Calibratge | Isotonic Regression (OOF) |

> El model utilitza `scale_pos_weight=9.71` per compensar el desequilibri de classes, `eval_metric="aucpr"` per optimitzar la detecció de pluja, i **calibratge isotònic** sobre prediccions out-of-fold per obtenir probabilitats fiables.

## Feedback loop (auto-aprenentatge)

El sistema verifica automàticament les seves pròpies prediccions i aprèn dels errors:

```
┌─────────────────┐    +60 min     ┌─────────────────┐    diari      ┌─────────────────┐
│   Predicció     │──────────▶│  Verificació    │────────────▶│   Re-entrena   │
│  cada 10 min   │             │ va ploure?     │              │  amb feedback   │
└────────┬────────┘             └────────┬────────┘              └────────┬────────┘
         │                       │                              │
         ▼                       ▼                              │
  predictions_log.jsonl   ✓/✗ correct?                         │
                                │                              │
                                ▼                              │
                         Informe setmanal  ◄─────────────────┘
                         (accuracy %, F1)
                         via Telegram 📊
```

### Com funciona

1. **Log**: Cada predicció es registra a `predictions_log.jsonl` amb un snapshot complet: probabilitat, condicions, radar, AEMET, sentinella, ensemble, nivells de pressió, règim de vent, bias, i les 68 features del model
2. **Verificació**: 60-75 min després, el sistema consulta l'estació per veure si realment va ploure
3. **Classificació**: Cada predicció es marca com TP, FP, TN, o FN
4. **Informe**: Cada dilluns a les 8:00, reps un report amb accuracy, precisión, recall, F1, i tendència
5. **Re-entrenament**: El retrain diari incorpora les prediccions verificades com a dades noves, permetent al model aprendre dels seus errors recents

### Mètriques que rebràs

| Mètrica | Significat |
|---------|-----------|
| Accuracy | % de prediccions correctes (pluja i sec) |
| Precision | De les alertes, quantes van ser pluja real |
| Recall | De les pluges reals, quantes vam predir |
| F1 | Balanç entre precision i recall |
| Per confiança | Accuracy desglossada per nivell (Molt Baixa → Molt Alta) |
| Per dia | Evolució diària de l'accuracy |

## Sistema de notificacions

Les notificacions són basades en **transicions d'estat**, no en cada predicció. Això maximitza el senyal i minimitza el soroll.

### Tipus de notificació

| Tipus | Quan s'envia | Missatge |
|-------|-------------|----------|
| 🌧️ **Pluja imminent** | Probabilitat puja per sobre del **65%** | "⚠️ ALERTA: Pluja imminent en els propers 60 min!" |
| ☀️ **Pluja s'allunya** | Probabilitat baixa per sota del **30%** | "✅ La pluja s'allunya!" |
| 🌊 **Canvi de règim** | Vent gira a Llevantada/Garbí + condicions favorables | "🌊 Llevantada: entrada d'humitat mediterrània" |
| 📋 **Previsió diària** | Cada dia a les **7:00** | Outlook per franges (matí/tarda/nit) amb règim eòlic |

### Alertes de canvi de règim atmosfèric

El sistema detecta canvis en la configuració atmosfèrica que històricament produeixen pluja a Cardedeu:

| Règim | Detecció | Significat |
|-------|---------|-----------|
| 🌊 **Llevantada humida** | Vent gira a E/SE (850hPa) + HR ≥75% | Humitat mediterrània contra la Serralada → pluja #1 |
| 🌀 **Garbí inestable** | Vent SW + TT>44 o LI<-2 | Configuració de tempestes convectives |
| 📉 **Caiguda de pressió** | ≥2 hPa/3h en règim humit | Approximació de front o baixa |
| 🔄 **Backing wind** | Gir antihorari >20° en 3h + HR ≥70% | Front càlid o baixa en aproximació |

Aquestes alertes donen **hores** de lead time — alerten sobre la **causa** (configuració atmosfèrica), no l'**efecte** (pluja al terra).

- **Cooldown**: 2 hores entre alertes de règim
- **No repeteix**: El mateix tipus de règim no s'alerta dues vegades seguides

### Radar espacial (30 km)

El sistema no només mira el píxel de Cardedeu — escaneja un radi de **30 km** per detectar ecos de pluja que s'acosten:

- **Eco més proper**: Distància i direcció (ex: "eco a 18 km NNE")
- **Tracking de tempesta**: Velocitat i ETA (ex: "25 km/h, arriba en ~40 min")
- **Sector de sobrevent**: Analitza els ecos en la direcció d'on ve el vent (850hPa)
- **Cobertura**: Fracció de la zona amb ecos de radar

### Previsió diària millorada

El resum diari (7:00) està dissenyat per doble audiència — públic general i entusiastes de la meteorologia (progressive disclosure):

**Part superior (tothom):**
- Outlook del dia (☀️/🌥️/🌧️) + probabilitat actual
- 🆕 Narrativa IA en català (paràgraf fluid generat per GitHub Models gpt-4o-mini)
- Previsió ML per franges: Matí (7-13h), Tarda (13-19h), Nit (19-1h) amb rang de temperatura
- Propera pluja prevista (48h)

**Condicions actuals (compacte):**
- Temperatura + humitat + punt de rosada en línia compacta
- Pressió amb tendència numèrica (ex: `↑(+0.8/3h)`)
- Vent + cobertura de núvols

**Detall tècnic (entusiastes meteo):**
- Ensemble: quants models prediuen pluja
- Vent sinòptic a 850hPa amb direcció, velocitat, T850 i humitat relativa 850/700hPa
- Índexs d'inestabilitat (TT, LI, VT)
- Resum intel·ligent del radar (filtra ecos no significatius)

### Narrativa IA (GitHub Models)

El resum diari i l'informe setmanal d'accuracy inclouen un paràgraf generat per IA en català que interpreta les dades meteorològiques de manera natural:

```
💬 Avui a Cardedeu, el dia es presenta amb temperatures suaus que oscil·len
   entre els 10 i els 17 graus. Al matí, les probabilitats de pluja són mínimes,
   però a mesura que avanci la tarda, el risc de ruixats augmentarà una mica.
   El vent bufa lleugerament del nord-est, i la pressió ha baixat lleugerament,
   cosa que pot indicar canvis en el temps.
```

**Arquitectura (dual-provider, patró gencat-cultural-agenda):**
1. **GitHub Models gpt-4o-mini** (primari) — gratuït via `GITHUB_TOKEN` automàtic a GitHub Actions
2. **OpenRouter models gratuïts** (fallback) — gpt-oss-120b, llama-3.3, gemma-3, etc.

**Principis de disseny:**
- **Mai al camí crític**: Les crides IA NO estan a `predict_now.py` ni a les alertes de pluja. Només als scripts de baixa freqüència (1 crida/dia + 1 crida/setmana)
- **Fallback graciós**: Si la IA falla, s'envia el missatge template existent sense narrativa
- **Zero dependències noves**: Usa `requests` (ja inclòs) per cridar l'API compatible amb OpenAI
- **Zero cost**: `GITHUB_TOKEN` és automàtic i gratuït; OpenRouter free tier com a fallback

### Disseny anti-spam

```
           65%  ┌──────────────────┐
     clear ─────┤  rain_alert     │
                │  (🌧️ notifica)   │
                └─────┬────────────┘
           30%  │
     clear ◄────┘  (☀️ notifica)

     Zona 30-65% = histèresi (sense notificació)
```

- **Histèresi**: El gap de 35 punts entre llindars evita flip-flop quan la probabilitat oscil·la
- **Cooldown**: Mínim 30 minuts entre alertes consecutives
- **Persistència**: L'estat es manté entre execucions via git commits automàtics
- **Resultat**: 2-5 missatges en dies de pluja, 0-1 en dies clars

## Nivells de confiança

| Probabilitat | Nivell | Acció |
|-------------|--------|-------|
| < 20% | Molt Baixa | No es notifica |
| 20-40% | Baixa | No es notifica |
| 40-65% | Moderada | No es notifica |
| 65-85% | Alta | 🔔 Alerta Telegram |
| > 85% | Molt Alta | 🔔 Alerta Telegram |
