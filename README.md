# 🌧️ Nowcast Cardedeu

Sistema de predicció de pluja hiperlocal per a Cardedeu (Vallès Oriental) basat en Machine Learning.

Utilitza dades reals de l'estació [MeteoCardedeu.net](https://meteocardedeu.net) combinades amb models meteorològics globals (Open-Meteo), radar de precipitació en temps real (RainViewer) i estacions sentinella del Servei Meteorològic de Catalunya (Meteocat XEMA) per aprendre els patrons del microclima local i predir si plourà en els propers 60 minuts amb més precisió que els models estàndard.

## Com funciona

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  MeteoCardedeu   │  │   Open-Meteo     │  │   RainViewer     │  │  Meteocat XEMA   │
│  (dades reals)   │  │  (models NWP)    │  │  (radar precip)  │  │  (sentinelles)   │
│  T, H, P, Vent   │  │  GFS, ECMWF...   │  │  dBZ, mm/h       │  │  Granollers, etc │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │                     │
         ▼                     ▼                      ▼                     ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                       Feature Engineering                               │
    │   Tendències · Derivades · Context · Radar · Diferencial sentinella     │
    └──────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │     XGBoost       │
                        │  (corrector local)│
                        └────────┬──────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  Probabilitat de │──→ 🔔 Telegram
                       │  pluja (0-100%)  │
                       └──────────────────┘
```

## Filosofia

El model **no intenta predir el temps des de zero**. El que fa és:
1. Rebre el que diuen els models globals (Open-Meteo)
2. Comparar-ho amb les condicions reals mesurades a Cardedeu
3. **Corregir els errors dels models** basant-se en patrons apresos de +10 anys d'històric

Per exemple, aprèn coses com:
- "Quan el model diu pluja però el vent de Cardedeu és sec del Montseny → no plourà"
- "Quan la pressió baixa ràpidament + humitat >85% + vent del SE → plou sempre aquí"

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
> Sense clau, el sistema funciona igualment però sense les features de radar sentinella.

### 5. Configurar alertes Telegram (opcional)
1. Crea un bot amb [@BotFather](https://t.me/BotFather)
2. Configura les variables d'entorn:
```bash
export TELEGRAM_BOT_TOKEN="el_teu_token"
export TELEGRAM_CHAT_ID="el_teu_chat_id"
```

### 6. GitHub Actions (automatització)
El workflow `.github/workflows/nowcast.yml`:
- Executa prediccions cada 15 minuts (6h-23h)
- Re-entrena el model automàticament cada diumenge a les 3:00
- Configura els secrets al repositori:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
  - `METEOCAT_API_KEY`

## Estructura

```
nowcast-cardedeu/
├── config.py                 # Configuració central (URLs, coordenades, llindars)
├── src/
│   ├── data/
│   │   ├── meteocardedeu.py  # API meteocardedeu.net (sèries minut a minut + NOAA)
│   │   ├── open_meteo.py     # API Open-Meteo (històric + forecast)
│   │   ├── rainviewer.py     # API RainViewer (radar precipitació temps real)
│   │   └── meteocat.py       # API Meteocat XEMA (estacions sentinella SMC)
│   ├── features/
│   │   └── engineering.py    # Feature engineering (48 features)
│   ├── model/
│   │   ├── train.py          # Pipeline d'entrenament (XGBoost + TimeSeriesSplit)
│   │   └── predict.py        # Predicció en temps real (fusió 4 fonts)
│   └── notify/
│       └── telegram.py       # Alertes Telegram (amb info radar + sentinella)
├── scripts/
│   ├── download_history.py   # Descarregar 12+ anys d'històric
│   ├── build_dataset.py      # Construir dataset d'entrenament
│   ├── train_model.py        # Entrenar model
│   └── predict_now.py        # Predicció (GitHub Actions)
├── models/                   # Model entrenat (git tracked)
├── data/                     # Dades (raw no tracked, processed sí)
├── requirements.txt          # Dependències Python
└── .github/workflows/        # Automatització cada 15 min
```

## Features del model

El model utilitza **48 features** organitzades en categories:

| Categoria | Features | Per què? |
|-----------|----------|----------|
| Temporals | Hora, mes (codificació cíclica) | Patrons estacionals i diaris |
| Pressió | Valor + tendència 1h/3h/6h + acceleració | Indicador principal d'inestabilitat |
| Humitat | Valor + punt rosada + depressió + tendència | Saturació = pluja imminent |
| Vent | Components U/V + canvis + marinada | Marinada del mar = aire sec |
| Pluja recent | Acumulat 3h/6h + ha plogut? | Context de fronts actius |
| Models NWP | CAPE, núvols, weather code | Què diuen els models globals |
| Radiació | Solar W/m² | Indicador indirecte de núvols |
| 🆕 Radar | Intensitat, dBZ, mm/h, eco, tendència, aprox. | Precipitació real en temps real |
| 🆕 Sentinella | Temp/hum Granollers + diffs amb Cardedeu + precip | Gradient territorial = front actiu |

## Fonts de dades

| Font | Tipus | Freqüència | Clau API |
|------|-------|------------|----------|
| [MeteoCardedeu.net](https://meteocardedeu.net) | Estació local (T, H, P, vent, pluja) | Cada minut, des de 2012 | No |
| [Open-Meteo](https://open-meteo.com) | Models NWP (GFS, ECMWF) - històric + forecast | Horària | No |
| [RainViewer](https://www.rainviewer.com/api.html) | Radar de precipitació compost (mosaic global) | Cada ~10 min | No |
| [Meteocat XEMA](https://apidocs.meteocat.gencat.cat) | Estacions sentinella SMC (Granollers, ETAP Cardedeu) | Cada 30 min | Sí (gratuïta) |

### Radar RainViewer
El sistema descarrega tiles de radar (zoom 8, tile 134/94) i extreu la intensitat al píxel exacte de Cardedeu. Converteix la intensitat del PNG a dBZ i mm/h (fórmula Marshall-Palmer). Analitza els últims 6 frames (~1h) per detectar si la precipitació s'aproxima.

### Estacions sentinella Meteocat
Utilitza l'estació de **Granollers (YM)** com a sentinella: si plou a Granollers (7 km al SO), és probable que arribi a Cardedeu en pocs minuts. També consulta el **pluviòmetre ETAP Cardedeu (KX)** a 1.5 km del centre. Les features de diferencial (temperatura, humitat) entre Granollers i Cardedeu detecten fronts que travessen la zona.

### Coordenades
- **Cardedeu**: 41.633°N, 2.364°E, 190m alt
- **Granollers (sentinella)**: 41.608°N, 2.288°E
- **ETAP Cardedeu (pluviòmetre)**: ~41.63°N, ~2.36°E

## Rendiment del model

| Mètrica | Valor |
|---------|-------|
| AUC-ROC | 0.9501 ± 0.0079 |
| F1-Score | 0.6653 ± 0.0381 |
| Mostres d'entrenament | 98,208 |
| Features | 48 |
| Classe positiva (pluja) | ~9.3% |
| Cross-validation | TimeSeriesSplit (5 folds) |

> El model utilitza `scale_pos_weight=9.7` per compensar el desequilibri de classes i `eval_metric="aucpr"` per optimitzar la detecció de pluja.

## Nivells de confiança

| Probabilitat | Nivell | Acció |
|-------------|--------|-------|
| < 20% | Molt Baixa | No es notifica |
| 20-40% | Baixa | No es notifica |
| 40-65% | Moderada | No es notifica |
| 65-85% | Alta | 🔔 Alerta Telegram |
| > 85% | Molt Alta | 🔔 Alerta Telegram |

## API budget Meteocat

> ⚠️ **Important**: El pla gratuït de Meteocat permet 750 crides XEMA/mes. El sistema fa ~3 crides per predicció (temperatura, humitat, precipitació). Amb prediccions cada 15 min durant 17h/dia, això supera el límit. Opcions:
> - **Reduir freqüència**: Consultar Meteocat només quan la probabilitat base (sense sentinella) supera un llindar
> - **Cache**: Reutilitzar dades de Meteocat durant 30 min (la seva freqüència d'actualització)
> - **Pla de pagament**: Si cal més crides
