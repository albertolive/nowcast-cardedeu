"""
Configuració central del projecte Nowcast Cardedeu.
Totes les constants i paràmetres en un sol lloc.
"""
import os
from datetime import datetime as _dt

# ── Coordenades ──
# Estació MeteoCardedeu Poble Sec (font de dades locals)
LATITUDE = 41.63282
LONGITUDE = 2.364255
ALTITUDE = 190  # metres
# Centre del municipi de Cardedeu (radar, àrea de predicció)
TOWN_CENTER_LAT = 41.6385
TOWN_CENTER_LON = 2.3558
MUNICIPALITY_AREA_KM2 = 12.8     # km²
MUNICIPALITY_NS_KM = 6.5         # extensió nord-sud
MUNICIPALITY_EW_KM = 3.7         # extensió est-oest

# ── URLs de l'API de meteocardedeu.net ──
BASE_URL = "https://meteocardedeu.net"
YEAR = str(_dt.now().year)
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

# Models d'ensemble per acord NWP
ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global", "meteofrance_arome_france0025"]
ENSEMBLE_RAIN_THRESHOLD_MM = 0.1  # mm en 6h per considerar que un model prediu pluja

# URL NOAA OISST v2.1 per SST històric
NOAA_ERDDAP_SST_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv"

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
    "cloud_cover_low",              # Núvols baixos (< 2km) — boira, estrat, nimboestrat
    "cloud_cover_mid",              # Núvols mitjos (2-6km) — altoestrat, altocúmul
    "cloud_cover_high",             # Núvols alts (> 6km) — cirrus, cirroestrat
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cape",                    # Convective Available Potential Energy
    "shortwave_radiation",
    "direct_radiation",             # Radiació solar directa (W/m²)
    "diffuse_radiation",            # Radiació solar difusa (W/m²) — alta = núvols gruixuts
    "weather_code",
    "vapour_pressure_deficit", # VPD (kPa) — indicador directe de saturació
    "convective_inhibition",   # CIN — supressió de convecció (J/kg)
    "wet_bulb_temperature_2m",      # Temp bulb humit — fog/precip type indicator
    "soil_moisture_0_to_7cm",  # Humitat del sòl superficial (m³/m³) — només Archive/ERA5, NaN a forecast
    "soil_moisture_7_to_28cm", # Humitat del sòl profund (m³/m³) — només Archive/ERA5, NaN a forecast
    # Tier 1 — noves variables ERA5 (100% cobertura des de 2015)
    "showers",                      # Pluja convectiva separada de l'estratiforme
    "et0_fao_evapotranspiration",    # Evapotranspiració de referència FAO (mm)
    "soil_temperature_0_to_7cm",     # Temperatura del sòl — motor tèrmic convectiu
    "sunshine_duration",             # Durada de sol (s) — proxy calentament solar
    "wind_speed_100m",               # Vent a 100m — detecció low-level jet
    "wind_direction_100m",           # Direcció del vent a 100m
    "snowfall",                      # Neu — tipus de precipitació
    # Tier 4 — ERA5 surface expansion (100% coverage 2015+)
    "total_column_integrated_water_vapour",  # TCWV (kg/m²) — precipitable water
    "boundary_layer_height",                 # PBL depth (m) — convective mixing
    "terrestrial_radiation",                 # Longwave IR (W/m²) — night cloud detect
    "soil_moisture_28_to_100cm",             # Deep soil moisture (m³/m³)
]

# Variables del model per al forecast
OPEN_METEO_FORECAST_MODELS = [
    "best_match",  # Barreja automàtica del millor model
]

# ── Paràmetres del model ML ──
RAIN_THRESHOLD_MM = 0.2          # mm en 60 min per considerar "pluja"
PREDICTION_HORIZON_MIN = 60      # Predicció a 60 minuts vista

# ── Visualització: llindars per a la categoria de pluja mostrada ──
# Separat del llindar intern F1-optimal (0.36) que alimenta el feedback loop.
# L'usuari veu categories honestes: sec / incert / probable.
DISPLAY_THRESHOLD_RAIN = 0.65    # >= 65%: "Pluja probable" (alta confiança)
DISPLAY_THRESHOLD_UNCERTAIN = 0.30  # 30-65%: zona incerta (mostra %), <30%: sec

# ── Notificacions: histèresi per evitar flip-flop ──
ALERT_PROBABILITY_THRESHOLD = 0.65  # Compat: usat pel model per marcar will_rain
ALERT_THRESHOLD_UP = 0.65           # clear → rain_alert (probabilitat puja)
ALERT_THRESHOLD_DOWN = 0.30         # rain_alert → clear (probabilitat baixa)
NOTIFICATION_COOLDOWN_MIN = 30      # Minuts mínims entre alertes

# ── RainViewer (radar, sense API key) ──
RAINVIEWER_API_URL = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_TILE_BASE = "https://tilecache.rainviewer.com"
# Tile zoom=8 → cada tile ~1.5°. Centre Cardedeu (41.639°N, 2.356°E) cau al tile x=129, y=95
RAINVIEWER_TILE_ZOOM = 8
RAINVIEWER_TILE_X = 129
RAINVIEWER_TILE_Y = 95
# Píxel dins del tile on cau el centre de Cardedeu (256x256)
RAINVIEWER_PIXEL_X = 172
RAINVIEWER_PIXEL_Y = 96

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
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"
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
# A zoom 8, cada píxel ≈ 0.457 km a la latitud de Cardedeu (41.64°)
# 40km: encaixa dins del tile (96px nord > 87px radi), mínima retallada a l'est (84px vs 87px)
# Guanyem ~10km de detecció anticipada vs 30km → 12-20 min extra a 30-50 km/h
RADAR_SCAN_RADIUS_KM = 40       # km al voltant de Cardedeu per escanejar ecos
RADAR_PIXEL_SIZE_KM = 0.457     # km per píxel (zoom 8, lat ~41.6°)
RADAR_MIN_DBZ = 10              # dBZ mínim per considerar un eco com a pluja real (filtra soroll/AP)
AEMET_RADAR_MIN_ECHO_CLUSTER_PX = 10  # Píxels mínims per considerar un eco real (filtra fronteres/costes del mapa)

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
METEOCAT_CACHE_TTL_SMC = 90       # minutes — municipal forecast updates every 6h, 90min for fresher data
METEOCAT_CACHE_TTL_XEMA = 30      # minutes — sentinel data (already gated by rain gate)
METEOCAT_CACHE_TTL_XEMA_EMPTY = 60  # minutes — cache empty XEMA responses longer to save quota

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
HISTORY_YEARS = list(range(2015, _dt.now().year + 1))

# ── Grups de features per a explicabilitat (Catalan labels) ──
# Mapa: nom_feature → (grup, icona). Les features del mateix grup se sumen.
FEATURE_GROUP_MAP = {
    # Models globals (NWP)
    "model_predicts_precip": ("Models globals", "🌐"),
    "nwp_precip_severity": ("Models globals", "🌐"),
    "weather_code": ("Models globals", "🌐"),
    "precipitation": ("Models globals", "🌐"),
    "wc_is_thunderstorm": ("Models globals", "🌐"),
    "wc_is_rain": ("Models globals", "🌐"),
    "wc_is_drizzle": ("Models globals", "🌐"),
    "nwp_rain_amount": ("Models globals", "🌐"),
    "nwp_rain_drying": ("Models globals", "🌐"),
    "nwp_rain_confirmed": ("Pluja confirmada", "🌧️"),
    "nwp_rain_dry_air": ("Models globals", "🌐"),
    "afternoon_fp_risk": ("Models globals", "🌐"),
    "nwp_dry_conflict": ("Correcció local", "🔧"),
    "nwp_wet_conflict": ("Correcció local", "🔧"),
    "nwp_rain_persistence_6h": ("Consistència NWP", "📊"),
    "nwp_rain_trend_3h": ("Consistència NWP", "📊"),
    "weather_code_change_3h": ("Consistència NWP", "📊"),
    "precip_trend_3h": ("Consistència NWP", "📊"),
    "showers": ("Models globals", "🌐"),
    "nwp_showers_fraction": ("Models globals", "🌐"),
    "snowfall": ("Models globals", "🌐"),
    "nwp_lifted_index": ("Models globals", "🌐"),
    # Pluja recent / acumulada
    "rain_accum_3h": ("Pluja confirmada", "🌧️"),
    "rain_accum_6h": ("Pluja confirmada", "🌧️"),
    "rain_accum_24h": ("Pluja confirmada", "🌧️"),
    "rained_last_3h": ("Pluja confirmada", "🌧️"),
    "rain_ending_signal": ("Pluja confirmada", "🌧️"),
    # Radar
    "radar_dbz": ("Radar", "📡"),
    "radar_rain_rate": ("Radar", "📡"),
    "radar_has_echo": ("Radar", "📡"),
    "radar_frames_with_echo": ("Radar", "📡"),
    "radar_approaching": ("Radar", "📡"),
    "radar_max_intensity_1h": ("Radar", "📡"),
    "radar_nearest_echo_km": ("Radar", "📡"),
    "radar_max_dbz_20km": ("Radar", "📡"),
    "radar_coverage_20km": ("Radar", "📡"),
    "radar_upwind_nearest_echo_km": ("Radar", "📡"),
    "radar_upwind_max_dbz": ("Radar", "📡"),
    "radar_storm_velocity_kmh": ("Radar", "📡"),
    "radar_storm_velocity_ns": ("Radar", "📡"),
    "radar_storm_velocity_ew": ("Radar", "📡"),
    "radar_storm_approaching": ("Radar", "📡"),
    "radar_quadrant_max_dbz_N": ("Radar", "📡"),
    "radar_quadrant_max_dbz_E": ("Radar", "📡"),
    "radar_quadrant_max_dbz_S": ("Radar", "📡"),
    "radar_quadrant_max_dbz_W": ("Radar", "📡"),
    "radar_quadrant_coverage_N": ("Radar", "📡"),
    "radar_quadrant_coverage_E": ("Radar", "📡"),
    "radar_quadrant_coverage_S": ("Radar", "📡"),
    "radar_quadrant_coverage_W": ("Radar", "📡"),
    "radar_echo_bearing_sin": ("Radar", "📡"),
    "radar_echo_bearing_cos": ("Radar", "📡"),
    # Radar AEMET
    "aemet_radar_dbz": ("Radar", "📡"),
    "aemet_radar_has_echo": ("Radar", "📡"),
    "aemet_radar_nearest_echo_km": ("Radar", "📡"),
    "aemet_radar_max_dbz_20km": ("Radar", "📡"),
    "aemet_radar_coverage_20km": ("Radar", "📡"),
    "aemet_radar_echoes_found": ("Radar", "📡"),
    # Humitat i punt de rosada
    "relative_humidity_2m": ("Humitat", "💧"),
    "dew_point": ("Humitat", "💧"),
    "dew_point_depression": ("Humitat", "💧"),
    "humidity_change_1h": ("Humitat", "💧"),
    "humidity_change_3h": ("Humitat", "💧"),
    "vapour_pressure_deficit": ("Humitat", "💧"),
    "vpd_change_3h": ("Humitat", "💧"),
    "wet_bulb_temperature_2m": ("Humitat", "💧"),
    "wet_bulb_depression": ("Humitat", "💧"),
    "cloud_humidity_convergence": ("Humitat", "💧"),
    "moisture_flux_850": ("Humitat", "💧"),
    "moisture_flux_925": ("Humitat", "💧"),
    "moisture_flux_change_3h": ("Humitat", "💧"),
    "rh_850": ("Humitat", "💧"),
    "rh_925": ("Humitat", "💧"),
    "rh_700": ("Humitat", "💧"),
    "rh_700_change_3h": ("Humitat", "💧"),
    "rh_700_change_6h": ("Humitat", "💧"),
    "rh_500": ("Humitat", "💧"),
    "dry_intrusion_500": ("Humitat", "💧"),
    "dry_intrusion_700": ("Humitat", "💧"),
    "tcwv": ("Aigua precipitable", "🌊"),
    "tcwv_change_3h": ("Aigua precipitable", "🌊"),
    "tcwv_change_6h": ("Aigua precipitable", "🌊"),
    "tcwv_blh_ratio": ("Aigua precipitable", "🌊"),
    "tcwv_monthly_anomaly": ("Aigua precipitable", "🌊"),
    # Inestabilitat
    "vt_index": ("Inestabilitat", "⚡"),
    "tt_index": ("Inestabilitat", "⚡"),
    "li_index": ("Inestabilitat", "⚡"),
    "li_unstable": ("Inestabilitat", "⚡"),
    "k_index": ("Inestabilitat", "⚡"),
    "cape": ("Inestabilitat", "⚡"),
    "cape_change_3h": ("Inestabilitat", "⚡"),
    "cape_diurnal_weighted": ("Inestabilitat", "⚡"),
    "convective_inhibition": ("Inestabilitat", "⚡"),
    "theta_e_deficit": ("Inestabilitat", "⚡"),
    "convective_composite": ("Inestabilitat", "⚡"),
    "bulk_richardson": ("Inestabilitat", "⚡"),
    "thermal_buildup": ("Inestabilitat", "⚡"),
    "inversion_925": ("Inestabilitat", "⚡"),
    # Pressió i tendència
    "pressure_msl": ("Pressió", "📊"),
    "pressure_change_1h": ("Pressió", "📊"),
    "pressure_change_3h": ("Pressió", "📊"),
    "pressure_change_6h": ("Pressió", "📊"),
    "pressure_accel_3h": ("Pressió", "📊"),
    "pressure_min_24h": ("Pressió", "📊"),
    "frontal_passage": ("Pressió", "📊"),
    "gph_850": ("Pressió", "📊"),
    "gph_850_change_3h": ("Pressió", "📊"),
    "gph_300": ("Pressió", "📊"),
    # Règim de vent
    "llevantada_strength": ("Règim de vent", "🌀"),
    "llevantada_moisture": ("Règim de vent", "🌀"),
    "garbi_strength": ("Règim de vent", "🌀"),
    "tramuntana_strength": ("Règim de vent", "🌀"),
    "tramuntana_moisture": ("Règim de vent", "🌀"),
    "wind_dir_change_3h": ("Règim de vent", "🌀"),
    "wind_850_speed": ("Règim de vent", "🌀"),
    "wind_850_dir": ("Règim de vent", "🌀"),
    "wind_925_speed": ("Règim de vent", "🌀"),
    "wind_925_dir": ("Règim de vent", "🌀"),
    "wind_700_speed": ("Règim de vent", "🌀"),
    "wind_700_dir": ("Règim de vent", "🌀"),
    "steering_onshore_700": ("Règim de vent", "🌀"),
    "deep_layer_shear": ("Règim de vent", "🌀"),
    "wind_shear_speed": ("Règim de vent", "🌀"),
    "wind_shear_dir": ("Règim de vent", "🌀"),
    "orographic_forcing": ("Règim de vent", "🌀"),
    "low_level_convergence": ("Règim de vent", "🌀"),
    # Vent de superfície
    "wind_speed_10m": ("Vent", "💨"),
    "wind_u": ("Vent", "💨"),
    "wind_v": ("Vent", "💨"),
    "wind_speed_change_1h": ("Vent", "💨"),
    "wind_speed_change_3h": ("Vent", "💨"),
    "is_sea_breeze": ("Vent", "💨"),
    "wind_gusts_10m": ("Vent", "💨"),
    "gust_factor": ("Vent", "💨"),
    "wind_speed_100m": ("Vent", "💨"),
    "boundary_layer_shear": ("Vent", "💨"),
    "wind_dir_shear_100m": ("Vent", "💨"),
    "wind_300_speed": ("Vent", "💨"),
    "wind_300_dir": ("Vent", "💨"),
    "jet_speed_300": ("Vent", "💨"),
    # Núvols
    "cloud_cover": ("Núvols", "☁️"),
    "cloud_change_1h": ("Núvols", "☁️"),
    "cloud_change_3h": ("Núvols", "☁️"),
    "is_overcast": ("Núvols", "☁️"),
    "cloud_cover_low": ("Núvols", "☁️"),
    "cloud_cover_mid": ("Núvols", "☁️"),
    "cloud_cover_high": ("Núvols", "☁️"),
    "cloud_low_fraction": ("Núvols", "☁️"),
    "cloud_thickness_proxy": ("Núvols", "☁️"),
    # Temperatura
    "temperature_2m": ("Temperatura", "🌡️"),
    "temp_925": ("Temperatura", "🌡️"),
    "temp_850": ("Temperatura", "🌡️"),
    "temp_850_change_3h": ("Temperatura", "🌡️"),
    "temp_500": ("Temperatura", "🌡️"),
    "temp_700": ("Temperatura", "🌡️"),
    "soil_temperature_0_to_7cm": ("Temperatura", "🌡️"),
    "soil_air_temp_diff": ("Temperatura", "🌡️"),
    "sst_med": ("Temperatura", "🌡️"),
    "freezing_level_height": ("Temperatura", "🌡️"),
    # Hora del dia / temporalitat
    "hour_sin": ("Hora del dia", "🕐"),
    "hour_cos": ("Hora del dia", "🕐"),
    "month_sin": ("Hora del dia", "🕐"),
    "month_cos": ("Hora del dia", "🕐"),
    "hours_since_sunrise": ("Hora del dia", "🕐"),
    # Radiació solar
    "shortwave_radiation": ("Radiació solar", "☀️"),
    "direct_radiation": ("Radiació solar", "☀️"),
    "diffuse_radiation": ("Radiació solar", "☀️"),
    "diffuse_fraction": ("Radiació solar", "☀️"),
    "radiation_rain_conflict": ("Radiació solar", "☀️"),
    "sunshine_duration": ("Radiació solar", "☀️"),
    "sunshine_accum_3h": ("Radiació solar", "☀️"),
    "terrestrial_radiation": ("Radiació solar", "☀️"),
    # Terra
    "soil_moisture_0_to_7cm": ("Terra", "🌱"),
    "soil_moisture_7_to_28cm": ("Terra", "🌱"),
    "soil_moisture_28_to_100cm": ("Terra", "🌱"),
    "soil_moisture_change_24h": ("Terra", "🌱"),
    "soil_saturation_ratio": ("Terra", "🌱"),
    "et0_fao_evapotranspiration": ("Terra", "🌱"),
    # Capa límit
    "boundary_layer_height": ("Capa límit", "🏔️"),
    "blh_change_3h": ("Capa límit", "🏔️"),
    # Visibilitat
    "visibility": ("Visibilitat", "👁️"),
    # Llamps
    "lightning_count_30km": ("Llamps", "⚡"),
    "lightning_count_15km": ("Llamps", "⚡"),
    "lightning_nearest_km": ("Llamps", "⚡"),
    "lightning_cloud_ground": ("Llamps", "⚡"),
    "lightning_max_current_ka": ("Llamps", "⚡"),
    "lightning_approaching": ("Llamps", "⚡"),
    "lightning_has_activity": ("Llamps", "⚡"),
    # Sentinella (Granollers)
    "sentinel_temp_diff": ("Sentinella", "📍"),
    "sentinel_humidity_diff": ("Sentinella", "📍"),
    "sentinel_precip": ("Sentinella", "📍"),
    "sentinel_raining": ("Sentinella", "📍"),
    "local_rain_xema": ("Sentinella", "📍"),
    "local_rain_xema_3h": ("Sentinella", "📍"),
    # Previsió AEMET/SMC
    "aemet_prob_precip": ("Previsió oficial", "🏛️"),
    "aemet_prob_storm": ("Previsió oficial", "🏛️"),
    "aemet_precip_today": ("Previsió oficial", "🏛️"),
    "smc_prob_precip_1h": ("Previsió oficial", "🏛️"),
    "smc_prob_precip_6h": ("Previsió oficial", "🏛️"),
    "smc_precip_intensity": ("Previsió oficial", "🏛️"),
    # Ensemble NWP (acord)
    "ensemble_rain_agreement": ("Acord entre models", "🤝"),
    "ensemble_precip_spread": ("Acord entre models", "🤝"),
    "ensemble_temp_spread": ("Acord entre models", "🤝"),
    "ensemble_max_precip": ("Acord entre models", "🤝"),
    "ensemble_min_precip": ("Acord entre models", "🤝"),
    "ensemble_models_rain": ("Acord entre models", "🤝"),
    # Interaccions NWP-Ensemble
    "ensemble_surprise_rain": ("Acord entre models", "🤝"),
    "nwp_isolated_rain": ("Acord entre models", "🤝"),
    # Bias observació vs forecast
    "forecast_temp_bias": ("Correcció local", "🔧"),
    "forecast_humidity_bias": ("Correcció local", "🔧"),
    # Nivells de pressió (indicador)
    "has_pressure_levels": ("Dades disponibles", "📶"),
}
