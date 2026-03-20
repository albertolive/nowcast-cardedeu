"""
Configuració central del projecte Nowcast Cardedeu.
Totes les constants i paràmetres en un sol lloc.
"""
import os

# ── Coordenades de l'estació MeteoCardedeu Poble Sec ──
LATITUDE = 41.63282
LONGITUDE = 2.364255
ALTITUDE = 190  # metres

# ── URLs de l'API de meteocardedeu.net ──
BASE_URL = "https://meteocardedeu.net"
YEAR = "2026"
SLUG = "cardedeu_poble_sec"

LATEST_URL = f"{BASE_URL}/{YEAR}/data/{SLUG}/latest.json"
SERIES_URL = f"{BASE_URL}/{YEAR}/api/graphs-series.php"
HISTORY_LIST_URL = f"{BASE_URL}/{YEAR}/api/history/historics_list.php"
HISTORY_FILE_URL = f"{BASE_URL}/{YEAR}/api/history/historics_file.php"

SERIES_VARS = "TEMP,HUM,VEL,DIR,PREC,BAR,SUN,UVI,PINT"

# ── Open-Meteo (sense API key) ──
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"

# Coordenades per SST (punt de mar proper — costa Maresme)
SEA_LATITUDE = 41.4
SEA_LONGITUDE = 2.5

# Variables horàries que demanem a Open-Meteo (forecast i històric)
OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cape",                    # Convective Available Potential Energy
    "shortwave_radiation",
    "weather_code",
    "vapour_pressure_deficit", # VPD (kPa) — indicador directe de saturació
    "convective_inhibition",   # CIN — supressió de convecció (J/kg)
    "soil_moisture_0_to_7cm",  # Humitat del sòl superficial (m³/m³)
    "soil_moisture_7_to_28cm", # Humitat del sòl profund (m³/m³)
]

# Variables del model per al forecast
OPEN_METEO_FORECAST_MODELS = [
    "best_match",  # Barreja automàtica del millor model
]

# ── Paràmetres del model ML ──
RAIN_THRESHOLD_MM = 0.2          # mm en 60 min per considerar "pluja"
PREDICTION_HORIZON_MIN = 60      # Predicció a 60 minuts vista

# ── Notificacions: histèresi per evitar flip-flop ──
ALERT_PROBABILITY_THRESHOLD = 0.65  # Compat: usat pel model per marcar will_rain
ALERT_THRESHOLD_UP = 0.65           # clear → rain_alert (probabilitat puja)
ALERT_THRESHOLD_DOWN = 0.30         # rain_alert → clear (probabilitat baixa)
NOTIFICATION_COOLDOWN_MIN = 30      # Minuts mínims entre alertes

# ── RainViewer (radar, sense API key) ──
RAINVIEWER_API_URL = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_TILE_BASE = "https://tilecache.rainviewer.com"
# Tile zoom=8 → cada tile ~1.5km. Cardedeu cau al tile x=134, y=94
RAINVIEWER_TILE_ZOOM = 8
RAINVIEWER_TILE_X = 134
RAINVIEWER_TILE_Y = 94
# Píxel dins del tile on cau Cardedeu (256x256)
RAINVIEWER_PIXEL_X = 88
RAINVIEWER_PIXEL_Y = 125

# ── Meteocat XEMA API ──
METEOCAT_API_KEY = os.environ.get("METEOCAT_API_KEY", "")
METEOCAT_BASE_URL = "https://api.meteo.cat"
# Estació sentinella: Granollers (YM) - 7km SO de Cardedeu
# Les tempestes solen venir de l'O/SO → Granollers les rep primer
SENTINEL_STATION_CODE = "YM"
SENTINEL_STATION_NAME = "Granollers"
# Estació pluviomètrica local: ETAP Cardedeu (KX) - 1.5km
LOCAL_RAIN_STATION_CODE = "KX"
# Variables XEMA: 32=Temp, 33=Humitat, 35=Precipitació
XEMA_VAR_TEMP = 32
XEMA_VAR_HUMIDITY = 33
XEMA_VAR_PRECIP = 35

# ── AEMET OpenData API ──
AEMET_API_KEY = os.environ.get("AEMET_API_KEY", "")
AEMET_MUNICIPALITY_CODE = "08052"  # Cardedeu

# ── AEMET Radar regional Barcelona ──
# Bounds geogràfics estimats per l'àrea del radar regional de Barcelona
# Cobreix Catalunya i voltants (~38°N-43°N, ~-1°E-4°E)
AEMET_RADAR_BOUNDS = {
    "lat_min": 38.5,
    "lat_max": 43.0,
    "lon_min": -1.0,
    "lon_max": 4.5,
}

# ── Meteocat Predicció Municipal ──
METEOCAT_MUNICIPALITY_CODE = "080462"  # Cardedeu (codi Meteocat via /referencia/v1/municipis)

# ── Rain gate: llindars per decidir si consultar fonts cares ──
RAIN_GATE_ENSEMBLE_PROB = 0.2   # Fracció de models amb pluja (≥1 de 4 = 0.25)
RAIN_GATE_CAPE_THRESHOLD = 800  # J/kg
RAIN_GATE_AEMET_STORM = 10      # %
RAIN_GATE_RADAR_NEARBY_KM = 30  # Obrir rain gate si ecos de radar dins d'aquest radi
RAIN_GATE_LIGHTNING_NEARBY_KM = 30  # Obrir rain gate si llamps dins d'aquest radi

# ── Radar Spatial Scanning ──
# A zoom 8, cada píxel ≈ 0.457 km a la latitud de Cardedeu (41.63°)
RADAR_SCAN_RADIUS_KM = 30       # km al voltant de Cardedeu per escanejar ecos
RADAR_PIXEL_SIZE_KM = 0.457     # km per píxel (zoom 8, lat ~41.6°)

# ── Detecció de canvi de règim eòlic ──
REGIME_COOLDOWN_MIN = 120        # Minuts mínims entre alertes de canvi de règim
REGIME_HUMIDITY_THRESHOLD = 75   # % RH per alerta de Llevantada humida
REGIME_PRESSURE_DROP_3H = -2.0   # hPa caiguda en 3h per alerta de pressió

# ── Telegram ──
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── IA (GitHub Models + OpenRouter fallback) ──
# Només usat al resum diari (1 crida/dia) i informe d'accuracy (1 crida/setmana).
# Mai al camí crític d'alertes de pluja.
# Primari: GitHub Models (gpt-4o-mini via GITHUB_TOKEN, gratuït a GitHub Actions).
# Fallback: OpenRouter (models gratuïts, requereix AI_API_KEY opcional).
AI_GITHUB_TOKEN = os.environ.get("AI_GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
AI_GITHUB_MODEL = os.environ.get("AI_GITHUB_MODEL", "gpt-4o-mini")
AI_GITHUB_BASE_URL = "https://models.inference.ai.azure.com/chat/completions"
AI_OPENROUTER_KEY = os.environ.get("AI_OPENROUTER_KEY", "")  # Opcional, fallback
AI_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
AI_MAX_RETRIES = int(os.environ.get("AI_MAX_RETRIES", "2"))
AI_RETRY_BASE_DELAY_MS = int(os.environ.get("AI_RETRY_BASE_DELAY_MS", "5000"))

# ── Meteocat API rate limiting ──
# Separate quotas per service: XDDE 250/month, Predicció 100/month, XEMA 750/month.
# ALL Meteocat calls are behind the rain gate (only fire when rain is likely).
# Typical rain days in Cardedeu: ~8/month, ~6 hours rain/event.
# Budget breakdown (rain gate + caching):
#   XDDE:       ~8 days × 6h × 4 calls = ~192/month (limit 250)
#   Predicció:  ~8 days × 6h / 3h TTL  = ~16/month  (limit 100)
#   XEMA:       ~8 days × 6h / 0.5h TTL = ~96/month (limit 750) + backfill ~90
#   Backfill XEMA: --max-days 3/retrain × 3 vars = ~270/month
METEOCAT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meteocat_cache")
METEOCAT_CACHE_TTL_XDDE = 120     # minutes — current hour cache; past hours cached 24h automatically
METEOCAT_CACHE_TTL_SMC = 180      # minutes — municipal forecast updates every 6h, 3h cache is safe
METEOCAT_CACHE_TTL_XEMA = 30      # minutes — sentinel data (already gated by rain gate)

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_nowcast.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
CALIBRATOR_PATH = os.path.join(MODELS_DIR, "calibrator.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Històric: anys a descarregar ──
HISTORY_YEARS = list(range(2015, 2027))  # 2015-2026 (12 anys)
