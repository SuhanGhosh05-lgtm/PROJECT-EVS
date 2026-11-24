import os
import math
import time
import calendar
import datetime
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import joblib
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", os.getenv("WAQI_TOKEN", "")).strip().strip("'\"")
ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "").strip().strip("'\"")

app = FastAPI(
    title="EVS Climate AI - Live Data Engine",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. LOAD MACHINE-LEARNING MODELS SAFELY
# =========================================================

def load_model(filename):
    """Loads a pickled scikit-learn/joblib model from ./models if present."""
    path = BASE_DIR / "models" / filename
    if not path.exists():
        return None
    try:
        model = joblib.load(path)
        print(f"[MODEL] Loaded {filename}: {type(model).__name__}")
        return model
    except Exception as exc:
        print(f"[MODEL] Failed to load {filename}: {type(exc).__name__}: {exc}")
        return None


rainfall_ai = load_model("rainfall_ml_model.pkl")
climate_ai = load_model("climate_rf_model.pkl")
disaster_ai = load_model("disaster_rf_model.pkl")
wrf_ai = load_model("wrf_rf_model.pkl")


def _safe_float(val, default):
    """Converts a value to a finite float or returns the fallback default."""
    if val is None:
        return float(default)
    try:
        num = float(val)
        return num if math.isfinite(num) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(val, default):
    """Converts a value to an int via finite float or returns the fallback default."""
    if val is None:
        return int(default)
    try:
        num = float(val)
        return int(num) if math.isfinite(num) else int(default)
    except (TypeError, ValueError):
        return int(default)


def _run_safely(fn, fallback_fn, *args, **kwargs):
    """Executes a task safely and returns fallback_fn() on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[WORKER] {fn.__name__} failed ({type(exc).__name__}: {exc}), using fallback.")
        return fallback_fn()


def _is_snow_code(code):
    """Returns True for WMO snowfall and snow shower codes."""
    return _safe_int(code, 0) in [71, 73, 75, 77, 85, 86]


def _is_heavy_snow_code(code):
    """Returns True for heavy snowfall and heavy snow shower WMO codes."""
    return _safe_int(code, 0) in [75, 86]


def _is_snow_condition(code, temp, precip_now=0.0, rain_chance=0, hour_code=None):
    """Returns True when WMO code, hourly code, or freezing precipitation indicates snow."""
    code_val = _safe_int(code, 0)
    h_code_val = _safe_int(hour_code if hour_code is not None else code_val, code_val)
    temp_val = _safe_float(temp, 25.0)
    precip_val = _safe_float(precip_now, 0.0)
    rain_val = _safe_int(rain_chance, 0)
    return (
        _is_snow_code(code_val)
        or _is_snow_code(h_code_val)
        or (temp_val <= 1.0 and (precip_val > 0.05 or rain_val >= 35))
    )


def _is_thunder_condition(code, rain_chance=0, precip_now=0.0, temp=25.0, hour_code=None):
    """Returns True for thunderstorms or heavy rain/high precip (>=85%) when not snow."""
    code_val = _safe_int(code, 0)
    h_code_val = _safe_int(hour_code if hour_code is not None else code_val, code_val)
    chance_val = _safe_int(rain_chance, 0)
    precip_val = _safe_float(precip_now, 0.0)
    temp_val = _safe_float(temp, 25.0)

    if _is_snow_condition(code_val, temp_val, precip_val, chance_val, hour_code=h_code_val):
        return False

    return (
        code_val in [82, 95, 96, 99]
        or h_code_val in [82, 95, 96, 99]
        or precip_val >= 5.0
        or (chance_val >= 85 and code_val not in [0, 1])
    )


def _normalize_rain_chance(wmo_code, rain_chance, precip_now=0.0, hour_code=None):
    """Normalizes rain probability so clear/sunny dry skies report 0% on minor noise."""
    code_val = _safe_int(wmo_code, 0)
    h_code_val = _safe_int(hour_code if hour_code is not None else code_val, code_val)
    chance_val = max(0, min(100, _safe_int(rain_chance, 0)))
    precip_val = _safe_float(precip_now, 0.0)

    has_active_rain = (
        precip_val > 0.1
        or code_val in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]
        or h_code_val in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]
    )
    is_dry_sky = (
        not has_active_rain
        and not _is_snow_code(code_val)
        and not _is_snow_code(h_code_val)
        and code_val in [0, 1, 2, 3]
        and h_code_val in [0, 1, 2, 3]
        and precip_val <= 0.05
    )

    if is_dry_sky and chance_val <= 35:
        return 0
    if not has_active_rain and chance_val > 50:
        return round(chance_val * 0.45)
    return chance_val


def _format_rain_range(rain_chance):
    """Builds a display range string for a normalized rain probability percentage."""
    pct = max(0, min(100, _safe_int(rain_chance, 0)))
    if pct == 0:
        return 0, 0, "0%"
    low_pct = max(0, min(100, pct - 5))
    high_pct = max(low_pct, min(100, pct + 5))
    return low_pct, high_pct, f"{low_pct}-{high_pct}%"


def predict_rainfall_flag(temp, humidity, wind_speed, pressure):
    """Runs ML rainfall classifier if available; otherwise uses meteorological rules."""
    t_val = _safe_float(temp, 25.0)
    h_val = _safe_float(humidity, 60.0)
    w_val = _safe_float(wind_speed, 8.0)
    p_val = _safe_float(pressure, 1012.0)

    if rainfall_ai:
        try:
            features_df = pd.DataFrame(
                [[t_val, h_val, w_val, p_val]],
                columns=["tempC", "humidity", "windspeedKmph", "pressure"],
            )
            pred = int(rainfall_ai.predict(features_df)[0])
            verdict = "Rain Expected Today" if pred == 1 else "No Rain Expected Today"
            return pred, verdict, "ML Model (rainfall_ml_model.pkl)"
        except Exception as exc:
            print(f"[MODEL] Rainfall prediction error: {exc}")

    fallback_flag = 1 if (h_val >= 82.0 and p_val < 1009.0) else 0
    fallback_verdict = "Rain Expected Today" if fallback_flag == 1 else "No Rain Expected Today"
    return fallback_flag, fallback_verdict, "Meteorological Rule Fallback"


def build_rainfall_verdict(ai_pred, wmo_code, rain_chance, precip_now=0.0, temp=25.0, hour_code=None):
    """Aligns rainfall verdict and effective rain flag with live weather & Open-Meteo."""
    code_val = _safe_int(wmo_code, 0)
    h_code_val = _safe_int(hour_code if hour_code is not None else code_val, code_val)
    chance_val = _safe_int(rain_chance, 0)
    precip_val = _safe_float(precip_now, 0.0)
    temp_val = _safe_float(temp, 25.0)

    if _is_snow_condition(code_val, temp_val, precip_val, chance_val, hour_code=h_code_val):
        snow_word = (
            "Heavy Snowfall"
            if (_is_heavy_snow_code(code_val) or _is_heavy_snow_code(h_code_val) or precip_val >= 3.0)
            else "Snowfall"
        )
        return 0, f"{snow_word} Expected Today"

    if _is_thunder_condition(code_val, chance_val, precip_val, temp=temp_val, hour_code=h_code_val):
        return 1, "Thunderstorm & Heavy Rain Expected Today"

    if code_val in [0, 1] and precip_val <= 0.05 and chance_val < 25:
        return 0, "No Rain Expected Today (Sunny & Dry)"

    if precip_val > 0.1 or code_val in [63, 81] or chance_val >= 55:
        return 1, "Rain Expected Today"

    if ai_pred == 1 and chance_val >= 25:
        return 1, "Rain Expected Today"

    if chance_val >= 25:
        return 0, "Chance of Rain Today"

    return 0, "No Rain Expected Today"


# =========================================================
# 3. COMMON UTILITIES & GLOBAL/REGIONAL GEOCODING
# =========================================================

INDIAN_STATE_HUBS = {
    "andhra pradesh": "Amaravati",
    "arunachal pradesh": "Itanagar",
    "assam": "Guwahati",
    "bihar": "Patna",
    "chhattisgarh": "Raipur",
    "goa": "Panaji",
    "gujarat": "Ahmedabad",
    "haryana": "Gurugram",
    "himachal pradesh": "Shimla",
    "jharkhand": "Ranchi",
    "karnataka": "Bengaluru",
    "kerala": "Thiruvananthapuram",
    "madhya pradesh": "Bhopal",
    "maharashtra": "Mumbai",
    "manipur": "Imphal",
    "meghalaya": "Shillong",
    "mizoram": "Aizawl",
    "nagaland": "Kohima",
    "odisha": "Bhubaneswar",
    "punjab": "Chandigarh",
    "rajasthan": "Jaipur",
    "sikkim": "Gangtok",
    "tamil nadu": "Chennai",
    "telangana": "Hyderabad",
    "tripura": "Agartala",
    "uttar pradesh": "Lucknow",
    "uttarakhand": "Dehradun",
    "mussoorie": "Mussoorie",
    "nepal": "Kathmandu",
    "kathmandu": "Kathmandu",
    "west bengal": "Kolkata",
    "delhi": "New Delhi",
    "jammu and kashmir": "Srinagar",
    "ladakh": "Leh",
    "puducherry": "Puducherry",
    "chandigarh": "Chandigarh",
    "andaman and nicobar islands": "Port Blair",
    "dadra and nagar haveli and daman and diu": "Daman",
    "lakshadweep": "Kavaratti",
}

KNOWN_CITY_COORDS = {
    "salt lake city": (22.5867, 88.4172, "India", "Salt Lake City"),
    "salt lake": (22.5867, 88.4172, "India", "Salt Lake"),
    "bidhannagar": (22.5867, 88.4172, "India", "Salt Lake"),
    "howrah": (22.5958, 88.2636, "India", "Howrah"),
    "durgapur": (23.5204, 87.3119, "India", "Durgapur"),
    "asansol": (23.6739, 86.9524, "India", "Asansol"),
    "siliguri": (26.7271, 88.3953, "India", "Siliguri"),
    "darjeeling": (27.0360, 88.2627, "India", "Darjeeling"),
    "noida": (28.5355, 77.3910, "India", "Noida"),
    "ghaziabad": (28.6692, 77.4538, "India", "Ghaziabad"),
    "kanpur": (26.4499, 80.3319, "India", "Kanpur"),
    "varanasi": (25.3176, 82.9739, "India", "Varanasi"),
    "agra": (27.1767, 78.0081, "India", "Agra"),
    "prayagraj": (25.4358, 81.8463, "India", "Prayagraj"),
    "gorakhpur": (26.7606, 83.3732, "India", "Gorakhpur"),
    "meerut": (28.9845, 77.7064, "India", "Meerut"),
    "ayodhya": (26.7922, 82.1998, "India", "Ayodhya"),
    "pune": (18.5204, 73.8567, "India", "Pune"),
    "nagpur": (21.1458, 79.0882, "India", "Nagpur"),
    "nashik": (19.9975, 73.7898, "India", "Nashik"),
    "thane": (19.2183, 72.9781, "India", "Thane"),
    "aurangabad": (19.8762, 75.3433, "India", "Aurangabad"),
    "navi mumbai": (19.0330, 73.0297, "India", "Navi Mumbai"),
    "mysuru": (12.2958, 76.6394, "India", "Mysuru"),
    "mangaluru": (12.9141, 74.8560, "India", "Mangaluru"),
    "hubballi": (15.3647, 75.1240, "India", "Hubballi"),
    "coimbatore": (11.0168, 76.9558, "India", "Coimbatore"),
    "madurai": (9.9252, 78.1198, "India", "Madurai"),
    "tiruchirappalli": (10.7905, 78.7047, "India", "Tiruchirappalli"),
    "salem": (11.6643, 78.1460, "India", "Salem"),
    "kochi": (9.9312, 76.2673, "India", "Kochi"),
    "kozhikode": (11.2588, 75.7804, "India", "Kozhikode"),
    "thrissur": (10.5276, 76.2144, "India", "Thrissur"),
    "warangal": (17.9689, 79.5941, "India", "Warangal"),
    "visakhapatnam": (17.6868, 83.2185, "India", "Visakhapatnam"),
    "vijayawada": (16.5062, 80.6480, "India", "Vijayawada"),
    "guntur": (16.3067, 80.4365, "India", "Guntur"),
    "tirupati": (13.6288, 79.4192, "India", "Tirupati"),
    "surat": (21.1702, 72.8311, "India", "Surat"),
    "vadodara": (22.3072, 73.1812, "India", "Vadodara"),
    "rajkot": (22.3039, 70.8022, "India", "Rajkot"),
    "gandhinagar": (23.2156, 72.6369, "India", "Gandhinagar"),
    "jodhpur": (26.2389, 73.0243, "India", "Jodhpur"),
    "udaipur": (24.5854, 73.7125, "India", "Udaipur"),
    "kota": (25.2138, 75.8648, "India", "Kota"),
    "ajmer": (26.4499, 74.6399, "India", "Ajmer"),
    "indore": (22.7196, 75.8577, "India", "Indore"),
    "jabalpur": (23.1815, 79.9864, "India", "Jabalpur"),
    "gwalior": (26.2183, 78.1828, "India", "Gwalior"),
    "ujjain": (23.1765, 75.7885, "India", "Ujjain"),
    "gaya": (24.7914, 85.0002, "India", "Gaya"),
    "jamshedpur": (22.8046, 86.2029, "India", "Jamshedpur"),
    "dhanbad": (23.7957, 86.4304, "India", "Dhanbad"),
    "cuttack": (20.4625, 85.8830, "India", "Cuttack"),
    "rourkela": (22.2604, 84.8536, "India", "Rourkela"),
    "puri": (19.8135, 85.8312, "India", "Puri"),
    "bhilai": (21.2094, 81.4285, "India", "Bhilai"),
    "faridabad": (28.4089, 77.3178, "India", "Faridabad"),
    "ludhiana": (30.9010, 75.8573, "India", "Ludhiana"),
    "amritsar": (31.6340, 74.8723, "India", "Amritsar"),
    "jalandhar": (31.3260, 75.5762, "India", "Jalandhar"),
    "manali": (32.2396, 77.1887, "India", "Manali"),
    "dharamshala": (32.2190, 76.3234, "India", "Dharamshala"),
    "haridwar": (29.9457, 78.1642, "India", "Haridwar"),
    "rishikesh": (30.0869, 78.2676, "India", "Rishikesh"),
    "nainital": (29.3919, 79.4542, "India", "Nainital"),
    "jammu": (32.7266, 74.8570, "India", "Jammu"),
    "gulmarg": (34.0484, 74.3805, "India", "Gulmarg"),
    "dibrugarh": (27.4728, 94.9120, "India", "Dibrugarh"),
    "silchar": (24.8333, 92.7789, "India", "Silchar"),
    "dimapur": (25.9091, 93.7266, "India", "Dimapur"),
    "margao": (15.2832, 73.9862, "India", "Margao"),
    "port blair": (11.6234, 92.7265, "India", "Port Blair"),
    "daman": (20.3974, 72.8328, "India", "Daman"),
    "kavaratti": (10.5667, 72.6417, "India", "Kavaratti"),
}


def _build_retry_session():
    """Creates a pooled requests Session configured with bounded retries."""
    session = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=0.25,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_HTTP_SESSION = _build_retry_session()


def safe_get(url, params=None, timeout=8):
    """Performs a GET request via the shared session and parses JSON."""
    response = _HTTP_SESSION.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def degrees_to_compass(deg):
    """Converts wind direction degrees into an 8-point compass direction."""
    if deg is None:
        return "N"
    try:
        val = float(deg)
        if not math.isfinite(val):
            return "N"
    except (TypeError, ValueError):
        return "N"

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(val / 45) % 8]


def get_current_hour_slice(hourly: dict, current_time_str: str):
    """Finds the index in Open-Meteo hourly arrays matching the current local hour."""
    times = hourly.get("time") or []
    if not times:
        return 0
    target_prefix = (current_time_str or "")[:13]
    if target_prefix:
        for idx, t in enumerate(times):
            if str(t).startswith(target_prefix):
                return idx
    return 0


# =========================================================
# ACCUWEATHER ACCURATE CURRENT CONDITIONS & ALERTS
# =========================================================

_ACCU_LOC_CACHE: dict[tuple[float, float], tuple[float, str]] = {}
_ACCU_DATA_CACHE: dict[str, tuple[float, dict]] = {}
_ACCU_CACHE_TTL_SECONDS = 300


def _accu_cache_get(cache_store: dict, key):
    entry = cache_store.get(key)
    if entry and (time.monotonic() - entry[0]) < _ACCU_CACHE_TTL_SECONDS:
        return entry[1]
    return None


def _accu_cache_set(cache_store: dict, key, value):
    if len(cache_store) >= 256:
        oldest_key = next(iter(cache_store))
        cache_store.pop(oldest_key, None)
    cache_store[key] = (time.monotonic(), value)


def _accu_map_icon_to_wmo(weather_icon: int | None, is_precipitation: bool = False) -> int:
    """Maps AccuWeather icon numbers to WMO weather codes used across Klima."""
    if weather_icon is not None:
        icon = _safe_int(weather_icon, 1)
        if icon in (22, 23, 24, 25, 26, 29, 43, 44):
            return 75 if icon in (22, 24, 44) else 71
        if icon in (19, 20, 21):
            return 85
    if is_precipitation:
        return 61
    if weather_icon is None:
        return 1
    icon = _safe_int(weather_icon, 1)
    if icon in (1, 2, 33, 34):
        return 0
    if icon in (3, 4, 35, 36):
        return 2
    if icon in (5, 6, 7, 8, 37, 38):
        return 3
    if icon == 11:
        return 45
    if icon in (12, 13, 14, 18, 39, 40):
        return 80
    if icon in (15, 16, 17, 41, 42):
        return 95
    return 2


def get_accuweather_location_key(lat: float, lon: float) -> str | None:
    """Resolves latitude/longitude to an AccuWeather location key when configured."""
    if not ACCUWEATHER_API_KEY:
        return None
    coord_key = (round(lat, 2), round(lon, 2))
    cached_key = _accu_cache_get(_ACCU_LOC_CACHE, coord_key)
    if cached_key:
        return cached_key
    try:
        payload = safe_get(
            "https://dataservice.accuweather.com/locations/v1/cities/geoposition/search",
            params={"apikey": ACCUWEATHER_API_KEY, "q": f"{lat},{lon}"},
            timeout=8,
        )
        loc_key = payload.get("Key")
        if loc_key:
            _accu_cache_set(_ACCU_LOC_CACHE, coord_key, str(loc_key))
            return str(loc_key)
    except Exception as exc:
        print(f"[ACCUWEATHER] Geoposition lookup skipped: {exc}")
    return None


def fetch_accuweather_accuracy_and_alerts(lat: float, lon: float) -> dict:
    """Fetches AccuWeather current conditions, MinuteCast timing, and severe alerts."""
    cache_id = f"{round(lat, 2)}:{round(lon, 2)}"
    cached = _accu_cache_get(_ACCU_DATA_CACHE, cache_id)
    if cached:
        return cached

    if not ACCUWEATHER_API_KEY:
        result = {
            "enabled": False,
            "provider": "AccuWeather (API key not configured)",
            "current": None,
            "minutecast": {
                "available": False,
                "summary": "Add ACCUWEATHER_API_KEY in .env for MinuteCast precipitation timing.",
                "precipitation_window": "Standby",
            },
            "severe_alerts": [],
        }
        _accu_cache_set(_ACCU_DATA_CACHE, cache_id, result)
        return result

    loc_key = get_accuweather_location_key(lat, lon)
    if not loc_key:
        result = {
            "enabled": True,
            "provider": "AccuWeather (Location lookup fallback)",
            "current": None,
            "minutecast": {"available": False, "summary": "Unavailable", "precipitation_window": "Standby"},
            "severe_alerts": [],
        }
        _accu_cache_set(_ACCU_DATA_CACHE, cache_id, result)
        return result

    current_parsed = None
    try:
        cond_list = safe_get(
            f"https://dataservice.accuweather.com/currentconditions/v1/{loc_key}",
            params={"apikey": ACCUWEATHER_API_KEY, "details": "true"},
            timeout=8,
        )
        if isinstance(cond_list, list) and cond_list:
            item = cond_list[0]
            temp_c = _safe_float((item.get("Temperature") or {}).get("Metric", {}).get("Value"), None)
            feel_c = _safe_float((item.get("RealFeelTemperature") or {}).get("Metric", {}).get("Value"), temp_c)
            precip_mm = _safe_float((item.get("PrecipitationSummary") or {}).get("Precipitation", {}).get("Metric", {}).get("Value"), 0.0)
            wind_kmh = _safe_float(((item.get("Wind") or {}).get("Speed") or {}).get("Metric", {}).get("Value"), None)
            wind_deg = _safe_float(((item.get("Wind") or {}).get("Direction") or {}).get("Degrees"), 0.0)
            press_mb = _safe_float((item.get("Pressure") or {}).get("Metric", {}).get("Value"), None)
            vis_km = _safe_float((item.get("Visibility") or {}).get("Metric", {}).get("Value"), None)
            has_precip = bool(item.get("HasPrecipitation", precip_mm > 0.05))
            wmo_mapped = _accu_map_icon_to_wmo(item.get("WeatherIcon"), has_precip)
            current_parsed = {
                "weather_text": item.get("WeatherText"),
                "wmo_code": wmo_mapped,
                "temperature_c": temp_c,
                "real_feel_c": feel_c,
                "humidity": _safe_float(item.get("RelativeHumidity"), None),
                "precipitation_mm": precip_mm,
                "has_precipitation": has_precip,
                "wind_speed_kmh": wind_kmh,
                "wind_direction_compass": degrees_to_compass(wind_deg),
                "pressure_mb": press_mb,
                "visibility_km": vis_km,
                "cloud_cover_percent": _safe_float(item.get("CloudCover"), None),
                "uv_index": _safe_float(item.get("UVIndex"), None),
                "observation_time": item.get("LocalObservationDateTime"),
            }
    except Exception as exc:
        print(f"[ACCUWEATHER] Current conditions error: {exc}")

    minutecast_info = {
        "available": False,
        "summary": "No minute-by-minute precipitation alert.",
        "precipitation_window": "Next 120 min clear",
    }
    try:
        mc = safe_get(
            f"https://dataservice.accuweather.com/forecasts/v1/minute/{loc_key}",
            params={"apikey": ACCUWEATHER_API_KEY},
            timeout=8,
        )
        summary_obj = mc.get("Summary") or {}
        if summary_obj.get("Text"):
            minutecast_info = {
                "available": True,
                "summary": summary_obj.get("Text"),
                "precipitation_window": f"Next {summary_obj.get('Duration', 120)} min",
            }
    except Exception:
        pass

    alerts_list = []
    try:
        raw_alerts = safe_get(
            f"https://dataservice.accuweather.com/alerts/v1/{loc_key}",
            params={"apikey": ACCUWEATHER_API_KEY, "details": "true"},
            timeout=8,
        )
        if isinstance(raw_alerts, list):
            for al in raw_alerts[:5]:
                alerts_list.append({
                    "id": al.get("AlertID"),
                    "category": al.get("Category") or "Severe Weather",
                    "severity": al.get("Severity") or "Moderate",
                    "headline": al.get("Description", {}).get("Localized") or al.get("HeadlineText") or "Weather Alert",
                    "start": (al.get("Start") or {}).get("LocalTime"),
                    "end": (al.get("End") or {}).get("LocalTime"),
                })
    except Exception:
        pass

    result = {
        "enabled": True,
        "provider": "AccuWeather Live + MinuteCast + Alerts",
        "current": current_parsed,
        "minutecast": minutecast_info,
        "severe_alerts": alerts_list,
    }
    _accu_cache_set(_ACCU_DATA_CACHE, cache_id, result)
    return result


@lru_cache(maxsize=256)
def _cached_geocode(normalized_query: str):
    """Resolves a location name to (lat, lon, country, resolved_name) with caching."""
    q_lower = normalized_query.lower().strip()
    if q_lower in KNOWN_CITY_COORDS:
        return KNOWN_CITY_COORDS[q_lower]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    mapped_query = INDIAN_STATE_HUBS.get(q_lower, normalized_query)

    try:
        data = safe_get(
            url,
            params={
                "name": mapped_query,
                "count": 5,
                "format": "json",
            },
            timeout=6,
        )
    except requests.RequestException:
        if q_lower == "kolkata" or mapped_query.lower() == "kolkata":
            return (22.5726, 88.3639, "India", "Kolkata")
        raise

    results = data.get("results") or []
    if not results:
        return None

    if q_lower in INDIAN_STATE_HUBS and q_lower not in {"nepal", "kathmandu"}:
        india_match = next((item for item in results if item.get("country_code") == "IN"), None)
        place = india_match if india_match else max(results, key=lambda r: _safe_float(r.get("population"), 0.0))
    else:
        exact_matches = [r for r in results if str(r.get("name", "")).strip().lower() == q_lower]
        pool = exact_matches if exact_matches else results
        place = max(pool, key=lambda r: _safe_float(r.get("population"), 0.0))

    return (
        float(place["latitude"]),
        float(place["longitude"]),
        place.get("country", "Earth"),
        place.get("name", mapped_query),
    )


def get_live_coordinates(location: str):
    """Validates and resolves a location query or raises HTTPException."""
    raw_query = location.strip()
    try:
        resolved = _cached_geocode(raw_query)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding service error: {exc}")

    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location}' not found.",
        )

    return resolved


def _classify_sky_common(wmo_code, humidity, temp, rain_chance, precip_mm=0.0, hour_code=None, is_live=False):
    """Single source of truth for classifying snow, thunder, rain, cloud, dry, and sunny states."""
    code_val = _safe_int(wmo_code, 0)
    h_code_val = _safe_int(hour_code if hour_code is not None else code_val, code_val)
    hum_val = _safe_float(humidity, 50.0)
    temp_val = _safe_float(temp, 25.0)
    precip_val = _safe_float(precip_mm, 0.0)
    chance_val = _normalize_rain_chance(code_val, rain_chance, precip_now=precip_val, hour_code=h_code_val)

    effective_snow_code = (
        code_val if _is_snow_code(code_val)
        else (h_code_val if _is_snow_code(h_code_val) else None)
    )
    heavy_snow_mm = 3.0 if is_live else 4.0
    if _is_snow_condition(code_val, temp_val, precip_val, chance_val, hour_code=h_code_val):
        snow_label = (
            "Heavy Snowfall"
            if (_is_heavy_snow_code(effective_snow_code) or precip_val >= heavy_snow_mm)
            else "Snowfall"
        )
        return snow_label, "SNOW_EFFECT", chance_val

    if _is_thunder_condition(code_val, chance_val, precip_val, temp=temp_val, hour_code=h_code_val):
        return "Thunderstorm", "THUNDER_EFFECT", chance_val

    if code_val in [51, 53, 55, 56, 57]:
        return "Light Drizzle", "RAIN_EFFECT", chance_val

    heavy_precip_mm = 0.5 if is_live else 0.2
    rain_prob_min = 65 if is_live else 55
    if precip_val > heavy_precip_mm or code_val in [63, 81] or (chance_val >= rain_prob_min and code_val not in [0, 1]):
        label = "Rainy" if (precip_val > heavy_precip_mm or code_val in [63, 81] or chance_val >= 75) else "Passing Showers"
        return label, "RAIN_EFFECT", chance_val

    if code_val in [61, 66, 80]:
        if precip_val <= 0.1 and chance_val == 0:
            return "Mostly Sunny", "SUN_RAYS_EFFECT", chance_val
        return ("Passing Showers" if is_live else "Scattered Clouds"), ("RAIN_EFFECT" if is_live else "CLOUD_EFFECT"), chance_val

    if code_val in [45, 48]:
        return "Foggy / Hazy", "SMOG_FOG_EFFECT", chance_val

    if hum_val < 35 and temp_val > 28:
        return "Dry & Warm", "DRY_HEAT_EFFECT", chance_val

    if code_val in [2, 3]:
        if precip_val <= 0.1 and chance_val == 0 and hum_val < 85:
            return "Mostly Sunny", "SUN_RAYS_EFFECT", chance_val
        if code_val == 2:
            return "Scattered Clouds", "CLOUD_EFFECT", chance_val
        return ("Overcast & Dry" if chance_val == 0 else "Overcast"), "CLOUD_EFFECT", chance_val

    if code_val == 1:
        return "Mostly Sunny", "SUN_RAYS_EFFECT", chance_val

    return "Clear & Sunny", "SUN_RAYS_EFFECT", chance_val


def classify_current_weather(
    wmo_code,
    humidity,
    temp,
    precip_now=0.0,
    hourly_code=None,
    hourly_rain_prob=None,
):
    """Classifies current live weather from WMO code & active precipitation."""
    label, trigger, _ = _classify_sky_common(
        wmo_code,
        humidity,
        temp,
        hourly_rain_prob,
        precip_mm=precip_now,
        hour_code=hourly_code,
        is_live=True,
    )
    return label, trigger


def classify_weather_condition(wmo_code, humidity, temp, rain_chance, ai_rain_flag, precip_mm=0.0):
    """Classifies daily/forecast weather condition, visual trigger, and normalized rain chance."""
    return _classify_sky_common(
        wmo_code,
        humidity,
        temp,
        rain_chance,
        precip_mm=precip_mm,
        hour_code=wmo_code,
        is_live=False,
    )


# =========================================================
# 4. AQI CATEGORY, RANGE, PM CONVERSION & CPCB CALCULATIONS
# =========================================================

def _valid_aqi_number(aqi):
    """Returns a non-negative finite float AQI or None."""
    if aqi is None:
        return None
    try:
        val = float(aqi)
        return val if math.isfinite(val) and val >= 0 else None
    except (TypeError, ValueError):
        return None


def get_indian_aqi_category(aqi):
    """Maps an AQI number to Indian CPCB categories."""
    val = _valid_aqi_number(aqi)
    if val is None:
        return "Unavailable"
    if val <= 50:
        return "Good"
    if val <= 100:
        return "Satisfactory"
    if val <= 200:
        return "Moderate"
    if val <= 300:
        return "Poor"
    if val <= 400:
        return "Very Poor"
    return "Severe"


def get_waqi_category(aqi):
    """Maps an AQI number to standard WAQI/US-EPA categories."""
    val = _valid_aqi_number(aqi)
    if val is None:
        return "Unavailable"
    if val <= 50:
        return "Good"
    if val <= 100:
        return "Moderate"
    if val <= 150:
        return "Unhealthy for Sensitive Groups"
    if val <= 200:
        return "Unhealthy"
    if val <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_aqi_visual_trigger(aqi):
    """Returns a frontend visual effect key based on AQI severity."""
    val = _valid_aqi_number(aqi)
    if val is None:
        return "AQI_UNAVAILABLE"
    if val > 150:
        return "HEAVY_SMOG_EFFECT"
    if val > 50:
        return "LIGHT_HAZE_EFFECT"
    return "CLEAN_AIR_EFFECT"


def calculate_aqi_range(aqi, margin=15):
    """Computes a clamped [lower_limit, upper_limit] range dictionary for an AQI."""
    val = _valid_aqi_number(aqi)
    if val is None:
        return None

    lower_limit = max(0, round(val - margin))
    upper_limit = min(500, round(val + margin))

    return {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "display": f"{lower_limit}-{upper_limit}",
    }


def epa_aqi_to_ugm3(pollutant: str, aqi):
    """Converts a US-EPA sub-index value to approximate concentration in ug/m3."""
    val = _valid_aqi_number(aqi)
    if val is None:
        return None

    breakpoints = {
        "pm25": [
            (0, 50, 0.0, 12.0),
            (51, 100, 12.1, 35.4),
            (101, 150, 35.5, 55.4),
            (151, 200, 55.5, 150.4),
            (201, 300, 150.5, 250.4),
            (301, 500, 250.5, 500.4),
        ],
        "pm10": [
            (0, 50, 0, 54),
            (51, 100, 55, 154),
            (101, 150, 155, 254),
            (151, 200, 255, 354),
            (201, 300, 355, 424),
            (301, 500, 425, 604),
        ],
    }

    for ilo, ihi, clo, chi in breakpoints.get(pollutant, []):
        if ilo <= val <= ihi:
            conc = ((val - ilo) / (ihi - ilo)) * (chi - clo) + clo
            return round(conc, 1)

    return None


def calculate_pollutant_subindex(concentration, breakpoints):
    """Computes linear sub-index interpolation against a pollutant breakpoint table."""
    if concentration is None:
        return None
    try:
        conc = float(concentration)
        if not math.isfinite(conc) or conc < 0:
            return None
    except (TypeError, ValueError):
        return None

    for low_c, high_c, low_i, high_i in breakpoints:
        if low_c <= conc <= high_c:
            result = ((high_i - low_i) / (high_c - low_c)) * (conc - low_c) + low_i
            return round(result, 1)

    if conc > breakpoints[-1][1]:
        return 500.0

    return None


CPCB_PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 500, 400, 500),
]

CPCB_PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 250, 100, 200),
    (250, 350, 200, 300),
    (350, 430, 300, 400),
    (430, 600, 400, 500),
]


def calculate_indian_cpcb_aqi(pm25, pm10):
    """Calculates Indian CPCB AQI as the maximum of PM2.5 and PM10 sub-indices."""
    values = []

    pm25_index = calculate_pollutant_subindex(pm25, CPCB_PM25_BREAKPOINTS)
    pm10_index = calculate_pollutant_subindex(pm10, CPCB_PM10_BREAKPOINTS)

    if pm25_index is not None:
        values.append(pm25_index)
    if pm10_index is not None:
        values.append(pm10_index)

    return round(max(values), 1) if values else None


# =========================================================
# 5. AQI LIVE SOURCES (WAQI -> OPEN-METEO CAMS) & FORECAST
# =========================================================

def get_waqi_live_data(lat, lon):
    """Fetches live AQI directly from WAQI using geo coordinates as first priority."""
    if not WAQI_API_TOKEN:
        return {"available": False}

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"

    try:
        result = safe_get(url, params={"token": WAQI_API_TOKEN}, timeout=12)

        if result.get("status") != "ok":
            return {"available": False}

        data = result.get("data", {})
        aqi_value = _valid_aqi_number(data.get("aqi"))

        if aqi_value is None:
            return {"available": False}

        iaqi = data.get("iaqi", {})
        pm25_sub = iaqi.get("pm25", {}).get("v") if isinstance(iaqi.get("pm25"), dict) else None
        pm10_sub = iaqi.get("pm10", {}).get("v") if isinstance(iaqi.get("pm10"), dict) else None

        pm25_ugm3 = epa_aqi_to_ugm3("pm25", pm25_sub)
        pm10_ugm3 = epa_aqi_to_ugm3("pm10", pm10_sub)

        station = data.get("city", {})

        return {
            "available": True,
            "aqi": float(aqi_value),
            "scale": "WAQI",
            "pm25_ugm3": pm25_ugm3,
            "pm10_ugm3": pm10_ugm3,
            "station_name": station.get("name", "WAQI monitoring station"),
            "measurement_time": data.get("time", {}).get("s"),
            "dominant_pollutant": data.get("dominentpol"),
        }

    except Exception:
        return {"available": False}


def get_cams_live_aqi(lat, lon):
    """Fetches live CAMS PM2.5/PM10 from Open-Meteo and converts to Indian CPCB AQI."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    try:
        data = safe_get(
            url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "pm2_5,pm10",
                "timezone": "auto",
            },
            timeout=12,
        )

        current = data.get("current", {})
        pm25_raw = _valid_aqi_number(current.get("pm2_5"))
        pm10_raw = _valid_aqi_number(current.get("pm10"))

        if pm25_raw is None and pm10_raw is None:
            return {"available": False}

        pm25_val = round(pm25_raw, 1) if pm25_raw is not None else None
        pm10_val = round(pm10_raw, 1) if pm10_raw is not None else None

        aqi = calculate_indian_cpcb_aqi(pm25_val, pm10_val)
        if aqi is None:
            return {"available": False}

        return {
            "available": True,
            "aqi": aqi,
            "scale": "CPCB",
            "pm25_ugm3": pm25_val,
            "pm10_ugm3": pm10_val,
            "measurement_time": current.get("time"),
        }

    except Exception:
        return {"available": False}


def get_open_meteo_aqi_prediction(lat, lon):
    """Computes next-hour, today, and tomorrow AQI estimates from Open-Meteo hourly CAMS."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    try:
        data = safe_get(
            url,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm2_5,pm10",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=12,
        )

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        pm25_list = hourly.get("pm2_5", [])
        pm10_list = hourly.get("pm10", [])

        if not times or (not pm25_list and not pm10_list):
            return {"available": False}

        utc_offset = _safe_int(data.get("utc_offset_seconds"), 0)
        local_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=utc_offset)
        local_now = local_now.replace(tzinfo=None)

        today_str = local_now.strftime("%Y-%m-%d")
        tomorrow_str = (local_now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        current_hour_dt = local_now.replace(minute=0, second=0, microsecond=0)

        today_vals = []
        tomorrow_vals = []
        next_hour_val = None

        for idx, t in enumerate(times):
            pm25_v = _valid_aqi_number(pm25_list[idx]) if idx < len(pm25_list) else None
            pm10_v = _valid_aqi_number(pm10_list[idx]) if idx < len(pm10_list) else None
            cpcb_val = calculate_indian_cpcb_aqi(pm25_v, pm10_v)

            if cpcb_val is None:
                continue

            dt = datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = (dt.astimezone(datetime.timezone.utc) + datetime.timedelta(seconds=utc_offset)).replace(tzinfo=None)

            if dt < current_hour_dt:
                continue

            if next_hour_val is None and dt > current_hour_dt:
                next_hour_val = round(cpcb_val, 1)

            if t.startswith(today_str):
                today_vals.append(cpcb_val)
            elif t.startswith(tomorrow_str):
                tomorrow_vals.append(cpcb_val)

        if not today_vals and not tomorrow_vals:
            return {"available": False}

        pred_today = round(sum(today_vals) / len(today_vals), 1) if today_vals else None
        pred_tomorrow = round(sum(tomorrow_vals) / len(tomorrow_vals), 1) if tomorrow_vals else None

        return {
            "available": True,
            "source": "Open-Meteo Air Quality Forecast (CPCB Scale)",
            "predicted_aqi_next_hour": next_hour_val if next_hour_val is not None else pred_today,
            "predicted_aqi_today": pred_today,
            "predicted_aqi_tomorrow": pred_tomorrow,
        }

    except Exception:
        return {"available": False}


# =========================================================
# 6. WEATHER AND RAINFALL ENDPOINT (OPEN-METEO PRIORITY)
# =========================================================

_WEATHER_CACHE: dict[tuple[float, float, str], tuple[float, dict]] = {}
_WEATHER_CACHE_TTL_SECONDS = 300


def _get_live_weather_payload(lat: float, lon: float, hour_bucket: str) -> dict:
    """Fetches and caches Open-Meteo forecast data per location and UTC hour bucket."""
    key = (round(lat, 4), round(lon, 4), hour_bucket)
    now_ts = time.monotonic()
    cached = _WEATHER_CACHE.get(key)
    if cached and (now_ts - cached[0]) < _WEATHER_CACHE_TTL_SECONDS:
        return cached[1]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    payload = safe_get(
        weather_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,weather_code,"
                "cloud_cover,surface_pressure,wind_speed_10m,"
                "wind_direction_10m"
            ),
            "hourly": (
                "temperature_2m,weather_code,visibility,uv_index,"
                "precipitation_probability,wind_speed_10m,relative_humidity_2m"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "sunrise,sunset,uv_index_max,"
                "precipitation_probability_max,"
                "wind_speed_10m_max,wind_speed_10m_min"
            ),
            "timezone": "auto",
        },
    )

    for existing_key in list(_WEATHER_CACHE.keys()):
        if existing_key[2] != hour_bucket:
            _WEATHER_CACHE.pop(existing_key, None)

    active_keys = [k for k in _WEATHER_CACHE if k[2] == hour_bucket]
    if len(active_keys) >= 256:
        for stale_key in active_keys[: len(active_keys) - 255]:
            _WEATHER_CACHE.pop(stale_key, None)

    _WEATHER_CACHE[key] = (now_ts, payload)
    return payload


@app.get("/predict-rainfall")
def predict_live_rainfall(location: str = "Kolkata"):
    """Returns live weather telemetry (AccuWeather accuracy when available), normalized rain chance, verdict, and 7-day outlook."""
    lat, lon, country, resolved_location = get_live_coordinates(location)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    hour_bucket = utc_now.strftime("%Y-%m-%dT%H")

    try:
        data = _get_live_weather_payload(lat, lon, hour_bucket)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    accu_bundle = _run_safely(
        fetch_accuweather_accuracy_and_alerts,
        lambda: {
            "enabled": False,
            "provider": "Unavailable",
            "current": None,
            "minutecast": {"available": False, "summary": "Unavailable", "precipitation_window": "N/A"},
            "severe_alerts": [],
        },
        lat,
        lon,
    )
    accu_cur = accu_bundle.get("current") or {}

    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    utc_offset = _safe_int(data.get("utc_offset_seconds"), 0)
    local_now = utc_now + datetime.timedelta(seconds=utc_offset)
    current_time_str = str(
        accu_cur.get("observation_time") or current.get("time") or local_now.strftime("%Y-%m-%dT%H:%M")
    )

    cur_idx = get_current_hour_slice(hourly, current_time_str)
    hourly_temps = hourly.get("temperature_2m") or []
    hourly_hums = hourly.get("relative_humidity_2m") or []
    hourly_winds = hourly.get("wind_speed_10m") or []
    hourly_probs = hourly.get("precipitation_probability") or []
    hourly_codes = hourly.get("weather_code") or []
    hourly_vis = hourly.get("visibility") or []

    temp = _safe_float(
        accu_cur.get("temperature_c") if accu_cur.get("temperature_c") is not None else current.get("temperature_2m"),
        hourly_temps[cur_idx] if cur_idx < len(hourly_temps) else 25.0,
    )
    feels_like = _safe_float(
        accu_cur.get("real_feel_c") if accu_cur.get("real_feel_c") is not None else current.get("apparent_temperature"),
        temp,
    )
    humidity = _safe_float(
        accu_cur.get("humidity") if accu_cur.get("humidity") is not None else current.get("relative_humidity_2m"),
        hourly_hums[cur_idx] if cur_idx < len(hourly_hums) else 50.0,
    )
    precip_now = _safe_float(
        accu_cur.get("precipitation_mm") if accu_cur.get("precipitation_mm") is not None else current.get("precipitation"),
        0.0,
    )
    wind_speed = _safe_float(
        accu_cur.get("wind_speed_kmh") if accu_cur.get("wind_speed_kmh") is not None else current.get("wind_speed_10m"),
        hourly_winds[cur_idx] if cur_idx < len(hourly_winds) else 5.0,
    )
    wind_dir_deg = _safe_float(current.get("wind_direction_10m"), 0.0)
    pressure = _safe_float(
        accu_cur.get("pressure_mb") if accu_cur.get("pressure_mb") is not None else current.get("surface_pressure"),
        1013.0,
    )
    clouds = _safe_float(
        accu_cur.get("cloud_cover_percent") if accu_cur.get("cloud_cover_percent") is not None else current.get("cloud_cover"),
        0.0,
    )
    wmo_code = _safe_int(
        accu_cur.get("wmo_code") if accu_cur.get("wmo_code") is not None else current.get("weather_code"),
        0,
    )

    hour_code = _safe_int(hourly_codes[cur_idx] if cur_idx < len(hourly_codes) else None, wmo_code)
    live_rain_prob = _safe_int(
        hourly_probs[cur_idx] if cur_idx < len(hourly_probs) and hourly_probs[cur_idx] is not None else None,
        0,
    )
    corrected_rain = _normalize_rain_chance(
        wmo_code, live_rain_prob, precip_now=precip_now, hour_code=hour_code
    )

    max_list = daily.get("temperature_2m_max") or []
    min_list = daily.get("temperature_2m_min") or []
    temp_high = _safe_float(max_list[0] if max_list else None, temp)
    temp_low = _safe_float(min_list[0] if min_list else None, temp)

    sunrise_raw = (daily.get("sunrise") or ["06:00"])[0] or "06:00"
    sunset_raw = (daily.get("sunset") or ["18:00"])[0] or "18:00"
    sunrise = str(sunrise_raw).split("T")[-1]
    sunset = str(sunset_raw).split("T")[-1]

    uv_list = daily.get("uv_index_max") or []
    uv_index = _safe_float(
        accu_cur.get("uv_index") if accu_cur.get("uv_index") is not None else (uv_list[0] if uv_list else None),
        5.0,
    )

    vis_raw = _safe_float(
        accu_cur.get("visibility_km") * 1000.0
        if accu_cur.get("visibility_km") is not None
        else (hourly_vis[cur_idx] if cur_idx < len(hourly_vis) else None),
        10000.0,
    )
    visibility_km = round(vis_raw / 1000.0, 1)

    weather_label, visual_trigger = classify_current_weather(
        wmo_code,
        humidity,
        temp,
        precip_now,
        hourly_code=hour_code,
        hourly_rain_prob=corrected_rain,
    )
    if accu_cur.get("weather_text") and not accu_cur.get("has_precipitation"):
        accu_text = str(accu_cur["weather_text"]).strip()
        accu_lower = accu_text.lower()
        is_dry_zero_rain_cloud = (
            wmo_code in (0, 1, 2, 3)
            and precip_now <= 0.1
            and corrected_rain == 0
            and humidity < 85
            and "cloud" in accu_lower
        )
        if _is_snow_condition(wmo_code, temp, precip_now, corrected_rain, hour_code=hour_code) or "snow" in accu_lower:
            weather_label = (
                "Heavy Snowfall"
                if (_is_heavy_snow_code(wmo_code) or _is_heavy_snow_code(hour_code) or precip_now >= 3.0 or "heavy snow" in accu_lower)
                else "Snowfall"
            )
            visual_trigger = "SNOW_EFFECT"
        elif wmo_code in (51, 53, 55, 56, 57):
            weather_label = "Light Drizzle"
            visual_trigger = "RAIN_EFFECT"
        elif corrected_rain >= 55 and wmo_code not in (0, 1):
            weather_label = "Rainy" if corrected_rain >= 75 else "Passing Showers"
            visual_trigger = "RAIN_EFFECT"
        elif is_dry_zero_rain_cloud:
            weather_label = "Clear & Sunny" if wmo_code == 0 else "Mostly Sunny"
            visual_trigger = "SUN_RAYS_EFFECT"
        else:
            weather_label = accu_text

    raw_ai_pred, _, prediction_source = predict_rainfall_flag(temp, humidity, wind_speed, pressure)
    effective_rain_flag, ai_verdict = build_rainfall_verdict(
        raw_ai_pred, wmo_code, corrected_rain, precip_now, temp=temp, hour_code=hour_code
    )

    is_snow_active = _is_snow_condition(wmo_code, temp, precip_now, corrected_rain, hour_code=hour_code)
    if is_snow_active:
        weather_label = (
            "Heavy Snowfall"
            if (_is_heavy_snow_code(wmo_code) or _is_heavy_snow_code(hour_code) or precip_now >= 3.0)
            else "Snowfall"
        )
        visual_trigger = "SNOW_EFFECT"
    else:
        if effective_rain_flag == 1 and corrected_rain < 55 and wmo_code not in (0, 1):
            corrected_rain = max(corrected_rain, 60)

        is_sunny_dry_sky = (wmo_code in (0, 1) and precip_now <= 0.05 and corrected_rain == 0)
        is_wet_weather_code = (
            precip_now > 0.1
            or wmo_code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99)
            or hour_code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99)
        )
        if not is_sunny_dry_sky and (is_wet_weather_code or (corrected_rain >= 55 and wmo_code not in (0, 1))):
            if wmo_code in (51, 53, 55, 56, 57):
                weather_label = "Light Drizzle"
                visual_trigger = "RAIN_EFFECT"
            elif weather_label in (
                "Clear & Sunny",
                "Mostly Sunny",
                "Scattered Clouds",
                "Partly Cloudy",
                "Overcast & Dry",
                "Overcast",
            ):
                weather_label = "Rainy" if (precip_now > 0.5 or corrected_rain >= 75) else "Passing Showers"
                visual_trigger = "RAIN_EFFECT"

    wind_min = max(0.0, round(wind_speed - 2.0, 1))
    wind_max = round(wind_speed + 2.0, 1)
    hum_min = max(0, int(humidity - 5))
    hum_max = min(100, int(humidity + 5))
    rain_min, rain_max, rain_display = _format_rain_range(corrected_rain)

    seven_day_forecast = []
    days_time = daily.get("time") or []
    daily_codes = daily.get("weather_code") or []
    daily_rains = daily.get("precipitation_probability_max") or []
    daily_maxs = daily.get("temperature_2m_max") or []
    daily_mins = daily.get("temperature_2m_min") or []
    daily_winds = daily.get("wind_speed_10m_max") or []

    for i, date_string in enumerate(days_time[:7]):
        d_max = _safe_float(daily_maxs[i] if i < len(daily_maxs) else None, temp)
        d_min = _safe_float(daily_mins[i] if i < len(daily_mins) else None, temp)
        d_wind = _safe_float(daily_winds[i] if i < len(daily_winds) else None, wind_speed)
        d_code = _safe_int(daily_codes[i] if i < len(daily_codes) else None, 0)
        d_rain_raw = _safe_int(daily_rains[i] if i < len(daily_rains) else None, 0)

        d_condition, d_trigger, d_rain = classify_weather_condition(d_code, humidity, d_max, d_rain_raw, 0)
        if _is_snow_condition(d_code, d_max, d_rain_raw * 0.25, d_rain_raw):
            d_condition = "Heavy Snowfall" if (_is_heavy_snow_code(d_code) or d_rain_raw >= 75) else "Snowfall"
            d_trigger = "SNOW_EFFECT"
        d_rmin, d_rmax, d_rdisp = _format_rain_range(d_rain)
        parsed_date = datetime.date.fromisoformat(date_string)

        seven_day_forecast.append({
            "day_name": parsed_date.strftime("%a"),
            "date": date_string,
            "condition": d_condition,
            "visual_trigger": d_trigger,
            "temp_range_c": {
                "min": round(d_min, 1),
                "max": round(d_max, 1),
                "display": f"{round(d_min)}° - {round(d_max)}°C",
            },
            "rain_chance_range": {
                "min": d_rmin,
                "max": d_rmax,
                "display": d_rdisp,
            },
            "humidity_range": {
                "min": hum_min,
                "max": hum_max,
                "display": f"{hum_min}-{hum_max}%",
            },
            "wind_speed_range_kmh": {
                "min": max(0.0, round(d_wind - 2.0, 1)),
                "max": round(d_wind + 2.0, 1),
                "display": f"{max(0.0, round(d_wind - 2.0, 1))}-{round(d_wind + 2.0, 1)} km/h",
            },
        })

    hourly_codes_clean = [
        (2 if (_safe_int(v, 0) in [61, 66, 80] and precip_now <= 0.1) else _safe_int(v, 0))
        for v in hourly_codes[cur_idx : cur_idx + 6]
    ]
    hourly_rain_clean = [
        _normalize_rain_chance(
            hourly_codes_clean[i] if i < len(hourly_codes_clean) else wmo_code,
            v,
            precip_now=precip_now,
            hour_code=hourly_codes_clean[i] if i < len(hourly_codes_clean) else hour_code,
        )
        for i, v in enumerate(hourly_probs[cur_idx : cur_idx + 6])
    ]

    return {
        "date": str(local_now.date()),
        "observed_at": current_time_str,
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "weather_source": accu_bundle["provider"] if accu_cur else "Open-Meteo Live Weather API",
        "prediction_source": f"Open-Meteo / Forecast + {prediction_source}",
        "accuweather_alerts": accu_bundle,
        "weather_condition": weather_label,
        "frontend_visual_trigger": visual_trigger,
        "chance_of_rain_range": {
            "min": rain_min,
            "max": rain_max,
            "display": rain_display,
        },
        "ai_rainfall_verdict": ai_verdict,
        "taskbar_metrics": {
            "temperature_c": temp,
            "real_feel_c": feels_like,
            "precipitation_mm": precip_now,
            "temp_range": {
                "min": round(temp_low, 1),
                "max": round(temp_high, 1),
                "display": f"{round(temp_low)}° - {round(temp_high)}°C",
            },
            "humidity_range": {
                "min": hum_min,
                "max": hum_max,
                "display": f"{hum_min}-{hum_max}%",
            },
            "wind_speed_range": {
                "min": wind_min,
                "max": wind_max,
                "display": f"{wind_min}-{wind_max} km/h",
            },
            "wind_direction_compass": degrees_to_compass(wind_dir_deg),
            "pressure_mb": pressure,
            "uv_index": uv_index,
            "uv_risk": "High" if uv_index >= 6 else "Moderate",
            "visibility_km": visibility_km,
            "cloud_cover_percent": clouds,
            "sunrise_time": sunrise,
            "sunset_time": sunset,
        },
        "hourly_times_next_6h": (hourly.get("time") or [])[cur_idx : cur_idx + 6],
        "hourly_weather_codes_next_6h": hourly_codes_clean,
        "hourly_rain_chance_next_6h": hourly_rain_clean,
        "next_7_days_forecast": seven_day_forecast,
    }


# =========================================================
# 7. LIVE AQI & PREDICTION ENDPOINT (WAQI -> OPEN-METEO CAMS)
# =========================================================

@app.get("/predict-aqi")
def predict_live_aqi(location: str = "Kolkata"):
    """Returns live AQI and near-term forecast prioritizing WAQI live telemetry blended with Open-Meteo."""
    lat, lon, country, resolved_location = get_live_coordinates(location)

    live_aqi = None
    live_pm25 = None
    live_pm10 = None
    source_used = "Unavailable"
    scale_used = "Unavailable"
    station_name = None
    measurement_time = None
    dominant_pollutant = None
    category = "Unavailable"

    waqi = get_waqi_live_data(lat, lon)
    if waqi.get("available"):
        live_aqi = _valid_aqi_number(waqi.get("aqi"))
        live_pm25 = waqi.get("pm25_ugm3")
        live_pm10 = waqi.get("pm10_ugm3")
        source_used = "World Air Quality Index (WAQI)"
        scale_used = waqi.get("scale", "WAQI")
        station_name = waqi.get("station_name")
        measurement_time = waqi.get("measurement_time")
        dominant_pollutant = waqi.get("dominant_pollutant")
        category = get_waqi_category(live_aqi)
    else:
        cams = get_cams_live_aqi(lat, lon)
        if cams.get("available"):
            live_aqi = _valid_aqi_number(cams.get("aqi"))
            live_pm25 = cams.get("pm25_ugm3")
            live_pm10 = cams.get("pm10_ugm3")
            source_used = "Open-Meteo CAMS"
            scale_used = cams.get("scale", "CPCB")
            measurement_time = cams.get("measurement_time")
            category = get_indian_aqi_category(live_aqi)

    om_pred = get_open_meteo_aqi_prediction(lat, lon)
    if live_aqi is not None and source_used == "World Air Quality Index (WAQI)":
        base_live = round(live_aqi, 1)
        om_next = _valid_aqi_number(om_pred.get("predicted_aqi_next_hour")) if om_pred.get("available") else None
        om_today = _valid_aqi_number(om_pred.get("predicted_aqi_today")) if om_pred.get("available") else None
        om_tomorrow = _valid_aqi_number(om_pred.get("predicted_aqi_tomorrow")) if om_pred.get("available") else None
        if om_today is not None and om_today > 0:
            prediction_source = "WAQI Live + Open-Meteo CAMS Forecast Blend"
            if om_next is not None:
                next_ratio = max(0.75, min(1.35, om_next / om_today))
                predicted_next_hour = round(max(0.0, min(500.0, base_live * 0.7 + (base_live * next_ratio) * 0.3)), 1)
            else:
                predicted_next_hour = base_live
            predicted_today = round(max(0.0, min(500.0, base_live * 0.65 + om_today * 0.35)), 1)
            if om_tomorrow is not None:
                om_ratio = max(0.65, min(1.5, om_tomorrow / om_today))
                scaled_tmrw = base_live * om_ratio
                predicted_tomorrow = round(max(0.0, min(500.0, scaled_tmrw * 0.65 + om_tomorrow * 0.35)), 1)
            else:
                predicted_tomorrow = predicted_today
        else:
            prediction_source = "WAQI Live Anchor + Atmospheric Trend"
            predicted_next_hour = base_live
            predicted_today = base_live
            predicted_tomorrow = base_live
    elif om_pred.get("available"):
        prediction_source = om_pred["source"]
        predicted_next_hour = _valid_aqi_number(om_pred.get("predicted_aqi_next_hour"))
        predicted_today = _valid_aqi_number(om_pred.get("predicted_aqi_today"))
        predicted_tomorrow = _valid_aqi_number(om_pred.get("predicted_aqi_tomorrow"))
    elif live_aqi is not None:
        prediction_source = "Live Persistence Fallback"
        predicted_next_hour = round(live_aqi, 1)
        predicted_today = round(live_aqi, 1)
        predicted_tomorrow = round(live_aqi, 1)
    else:
        prediction_source = "Unavailable"
        predicted_next_hour = None
        predicted_today = None
        predicted_tomorrow = None

    aqi_range = calculate_aqi_range(live_aqi)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_date_str = str(utc_now.date())
    try:
        om_meta = safe_get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "pm2_5", "timezone": "auto"},
            timeout=6,
        )
        offset_s = _safe_int(om_meta.get("utc_offset_seconds"), 0)
        local_date_str = str((utc_now + datetime.timedelta(seconds=offset_s)).date())
    except Exception:
        pass

    return {
        "date": local_date_str,
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "data_source": source_used,
        "aqi_scale": scale_used,
        "station_name": station_name,
        "measurement_time": measurement_time,
        "aqi_available": live_aqi is not None,
        "live_current_aqi": round(live_aqi, 1) if live_aqi is not None else None,
        "aqi_range": aqi_range,
        "live_pm25_ugm3": live_pm25,
        "live_pm10_ugm3": live_pm10,
        "pollution_category": category,
        "dominant_pollutant": dominant_pollutant,
        "frontend_aqi_trigger": get_aqi_visual_trigger(live_aqi),
        "prediction_source": prediction_source,
        "predicted_aqi_next_hour": predicted_next_hour,
        "predicted_aqi_today": predicted_today,
        "predicted_aqi_tomorrow": predicted_tomorrow,
        "predicted_today_range": calculate_aqi_range(predicted_today),
        "predicted_tomorrow_range": calculate_aqi_range(predicted_tomorrow),
    }


# =========================================================
# 8. CLIMATE CHANGE PROJECTION
# =========================================================

@app.get("/forecast-climate")
def forecast_climate(location: str = "Kolkata", target_year: int = 2050):
    """Projects future temperature and adaptation risk for a target year."""
    lat, lon, country, resolved_location = get_live_coordinates(location)

    try:
        weather = safe_get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
        )
        current_temp = _safe_float(weather.get("current", {}).get("temperature_2m"), 28.0)
    except Exception:
        current_temp = 28.0

    years_ahead = _safe_int(target_year, 2050) - datetime.date.today().year
    co2_factor = 1.25
    projected_temp = None

    if climate_ai:
        try:
            pred_raw = float(climate_ai.predict([[current_temp, lat, years_ahead, co2_factor]])[0])
            if math.isfinite(pred_raw):
                projected_temp = round(pred_raw, 2)
        except Exception:
            projected_temp = None

    if projected_temp is None:
        projected_temp = round(current_temp + years_ahead * 0.04, 2)

    return {
        "region": resolved_location.upper(),
        "country": country,
        "baseline_year": datetime.date.today().year,
        "target_year": target_year,
        "current_temperature": current_temp,
        "projected_temperature": projected_temp,
        "temperature_range": {
            "min": round(projected_temp - 1.2, 1),
            "max": round(projected_temp + 1.5, 1),
            "display": f"{round(projected_temp - 1.2, 1)}° - {round(projected_temp + 1.5, 1)}°C",
        },
        "impact_risk": (
            "Severe Heat Stress & Drought Vulnerability"
            if projected_temp > 34
            else "Moderate Climate Adaptation Needed"
        ),
    }


# =========================================================
# 9. SATELLITE DISASTER RADAR (LIVE OPEN-METEO + USGS)
# =========================================================

@app.get("/get-satellite-map")
def get_satellite_map(location: str = "Kolkata"):
    """Computes flood, wildfire, and nearby seismic risk alongside satellite map links."""
    lat, lon, country, resolved_location = get_live_coordinates(location)

    try:
        weather_data = safe_get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "precipitation,surface_pressure,wind_speed_10m"
                ),
                "daily": "precipitation_probability_max,precipitation_sum",
                "timezone": "auto",
            },
            timeout=12,
        )
    except Exception:
        weather_data = {}

    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    temp = _safe_float(current.get("temperature_2m"), 28.0)
    humidity = _safe_float(current.get("relative_humidity_2m"), 60.0)
    precip_now = _safe_float(current.get("precipitation"), 0.0)
    wind_speed = _safe_float(current.get("wind_speed_10m"), 8.0)
    pressure = _safe_float(current.get("surface_pressure"), 1013.0)

    daily_rain_probs = daily.get("precipitation_probability_max") or []
    daily_rain_sums = daily.get("precipitation_sum") or []
    rain_prob = _safe_float(daily_rain_probs[0] if daily_rain_probs else None, 15.0)
    daily_rain_mm = _safe_float(daily_rain_sums[0] if daily_rain_sums else None, 0.0)

    pressure_drop = max(0.0, 1013.0 - pressure)
    flood_risk = min(
        98.0,
        round(
            (rain_prob * 0.35)
            + (daily_rain_mm * 1.5)
            + (precip_now * 5.0)
            + (pressure_drop * 2.5),
            1,
        ),
    )

    heat_score = max(0.0, (temp - 25.0) * 3.0)
    dryness_score = max(0.0, (60.0 - humidity) * 1.2)
    wind_spread = min(20.0, wind_speed * 0.6)
    rain_dampener = min(40.0, daily_rain_mm * 4.0)
    wildfire_risk = round(
        max(2.0, min(95.0, heat_score + dryness_score + wind_spread - rain_dampener)),
        1,
    )

    earthquake_risk = 2.0
    recent_earthquakes = []
    start_time = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    try:
        usgs_data = safe_get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start_time,
                "latitude": lat,
                "longitude": lon,
                "maxradiuskm": 500,
                "minmagnitude": 2.5,
                "eventtype": "earthquake",
                "orderby": "time",
                "limit": 5,
            },
            timeout=10,
        )
        features = usgs_data.get("features") or []
        max_mag = 0.0

        for item in features:
            props = item.get("properties") or {}
            mag_val = _valid_aqi_number(props.get("mag"))
            if mag_val is not None:
                max_mag = max(max_mag, mag_val)
                time_ms = _valid_aqi_number(props.get("time"))
                recent_earthquakes.append({
                    "magnitude": round(mag_val, 1),
                    "place": props.get("place", "Nearby region"),
                    "time_utc": datetime.datetime.fromtimestamp(
                        time_ms / 1000.0, tz=datetime.timezone.utc
                    ).isoformat() if time_ms is not None else None,
                })

        if max_mag >= 6.0:
            earthquake_risk = 85.0
        elif max_mag >= 5.0:
            earthquake_risk = 55.0
        elif max_mag >= 4.0:
            earthquake_risk = 30.0
        elif max_mag >= 2.5:
            earthquake_risk = 15.0
    except Exception:
        pass

    probabilities = {
        "Flood Risk": flood_risk,
        "Earthquake Tectonic Warning": earthquake_risk,
        "Wildfire Thermal Alert": wildfire_risk,
    }

    highest_threat, highest_val = max(probabilities.items(), key=lambda x: x[1])
    if highest_val >= 60.0:
        primary_threat = f"High Alert: {highest_threat}"
    elif highest_val >= 35.0:
        primary_threat = f"Moderate Watch: {highest_threat}"
    else:
        primary_threat = "Normal (No Disaster Imminent)"

    return {
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "satellite_map_link": f"https://maps.google.com/?q={lat},{lon}&t=k",
        "disaster_risk_assessment": {
            "primary_threat_status": primary_threat,
            "probability_percentages": probabilities,
            "thermal_anomaly_detected": wildfire_risk > 40.0,
            "recent_earthquakes_500km": recent_earthquakes,
        },
        "forest_fire_scan": {
            "status": "Active Meteorological Fire Index",
            "risk_level": "Elevated" if wildfire_risk >= 40.0 else "Normal",
        },
    }


# =========================================================
# 10. WRF ATMOSPHERIC MODEL SIMULATION (LIVE NWP DATA)
# =========================================================

@app.get("/wrf-model-data")
def get_wrf_model(location: str = "Kolkata"):
    """Returns boundary-layer wind shear, CAPE, pressure, stability, and real 12-hour NWP progression."""
    lat, lon, country, resolved_location = get_live_coordinates(location)

    try:
        weather = safe_get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,"
                    "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
                ),
                "hourly": (
                    "temperature_2m,dew_point_2m,surface_pressure,cape,boundary_layer_height,"
                    "wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_direction_10m,wind_gusts_10m"
                ),
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=12,
        )
        current = weather.get("current", {})
        hourly = weather.get("hourly", {})
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Atmospheric model error: {exc}")

    utc_offset = _safe_int(weather.get("utc_offset_seconds"), 0)
    local_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=utc_offset)
    cur_time_str = str(current.get("time") or local_now.strftime("%Y-%m-%dT%H:%M"))
    cur_idx = get_current_hour_slice(hourly, cur_time_str)

    def _h_val(key, idx, default_val):
        arr = hourly.get(key) or []
        if 0 <= idx < len(arr) and arr[idx] is not None:
            return _safe_float(arr[idx], default_val)
        return float(default_val)

    temp_c = round(_safe_float(current.get("temperature_2m"), 26.0), 1)
    dew_point_c = round(_safe_float(current.get("dew_point_2m"), _h_val("dew_point_2m", cur_idx, temp_c - 6.0)), 1)
    dew_spread_c = round(max(0.0, temp_c - dew_point_c), 1)

    wind_speed = round(_safe_float(current.get("wind_speed_10m"), 5.5), 1)
    wind_dir = round(_safe_float(current.get("wind_direction_10m"), 140.0), 1)
    wind_gusts = round(_safe_float(current.get("wind_gusts_10m"), wind_speed + 3.0), 1)
    pressure = round(_safe_float(current.get("surface_pressure"), 1013.0), 1)

    wind_80m = round(_h_val("wind_speed_80m", cur_idx, wind_speed * 1.28), 1)
    wind_120m = round(_h_val("wind_speed_120m", cur_idx, wind_speed * 1.42), 1)
    cape_jkg = round(_h_val("cape", cur_idx, 120.0), 0)
    pbl_height_m = round(_h_val("boundary_layer_height", cur_idx, 850.0), 0)
    shear_10_120 = round(max(0.0, wind_120m - wind_speed), 1)

    ml_model_used = False
    if wrf_ai:
        try:
            ml_pred = float(wrf_ai.predict([[temp_c, pressure, wind_speed, wind_dir]])[0])
            if math.isfinite(ml_pred) and ml_pred > 0:
                wind_gusts = round(max(wind_gusts, ml_pred), 1)
                ml_model_used = True
        except Exception:
            pass

    gust_spread = round(max(0.0, wind_gusts - wind_speed), 1)
    if wind_gusts >= 30.0 or gust_spread >= 15.0 or cape_jkg >= 1500:
        stability = "Severe Convective / Gust Warning"
    elif wind_speed >= 15.0 or gust_spread >= 8.0 or cape_jkg >= 600:
        stability = "Moderate Boundary-Layer Turbulence"
    else:
        stability = "Stable Laminar Flow"

    wind_min = round(max(0.0, wind_speed - 1.5), 1)
    wind_max = round(max(wind_speed + 1.5, wind_gusts), 1)

    h_times = hourly.get("time") or []
    progression_12h = []
    for h_off in [0, 3, 6, 9, 12]:
        s_idx = min(len(h_times) - 1, cur_idx + h_off) if h_times else -1
        slot_time = str(h_times[s_idx]) if s_idx >= 0 else ""
        slot_label = "Now" if h_off == 0 else (slot_time.split("T")[-1][:5] if "T" in slot_time else f"+{h_off}h")

        w10 = round(_h_val("wind_speed_10m", s_idx, wind_speed), 1) if s_idx >= 0 else wind_speed
        wg = round(max(w10 + 1.0, _h_val("wind_gusts_10m", s_idx, wind_gusts)), 1) if s_idx >= 0 else wind_gusts
        w80 = round(_h_val("wind_speed_80m", s_idx, w10 * 1.28), 1) if s_idx >= 0 else wind_80m
        w120 = round(_h_val("wind_speed_120m", s_idx, w10 * 1.42), 1) if s_idx >= 0 else wind_120m
        p_mb = round(_h_val("surface_pressure", s_idx, pressure), 1) if s_idx >= 0 else pressure
        c_val = round(_h_val("cape", s_idx, cape_jkg), 0) if s_idx >= 0 else cape_jkg
        d_deg = round(_h_val("wind_direction_10m", s_idx, wind_dir), 0) if s_idx >= 0 else wind_dir

        progression_12h.append({
            "horizon_label": "Now" if h_off == 0 else f"+{h_off}h",
            "local_clock": slot_label,
            "hours_ahead": h_off,
            "wind_speed_kmh": w10,
            "wind_gusts_kmh": wg,
            "wind_80m_kmh": w80,
            "wind_120m_kmh": w120,
            "surface_pressure_mb": p_mb,
            "cape_jkg": c_val,
            "wind_direction_degrees": d_deg,
            "wind_direction_compass": degrees_to_compass(d_deg),
        })

    return {
        "region": resolved_location.upper(),
        "country": country,
        "numerical_model": (
            "Open-Meteo ICON/GFS NWP + WRF-RF Hybrid"
            if ml_model_used
            else "Open-Meteo ICON / GFS Seamless NWP"
        ),
        "simulated_atmospheric_metrics": {
            "wind_speed_kmh": wind_speed,
            "wind_gusts_kmh": wind_gusts,
            "wind_80m_kmh": wind_80m,
            "wind_120m_kmh": wind_120m,
            "wind_shear_10_120_kmh": shear_10_120,
            "surface_pressure_mb": pressure,
            "dew_point_c": dew_point_c,
            "dew_point_spread_c": dew_spread_c,
            "cape_jkg": cape_jkg,
            "boundary_layer_height_m": pbl_height_m,
            "wind_speed_range": {
                "min": wind_min,
                "max": wind_max,
                "display": f"{wind_min}-{wind_max} km/h",
            },
            "wind_direction_degrees": wind_dir,
            "wind_direction_compass": degrees_to_compass(wind_dir),
            "atmospheric_stability_index": stability,
        },
        "progression_12h": progression_12h,
    }


# =========================================================
# 11. CALENDAR FORECAST ENGINE (YEAR + MONTH + DAY)
# =========================================================

def get_safe_archive_date(target_date: datetime.date, reference_year: int) -> datetime.date:
    """Clamps leap-day dates (Feb 29) to Feb 28 in non-leap reference years."""
    try:
        return target_date.replace(year=reference_year)
    except ValueError:
        return target_date.replace(year=reference_year, day=28)


def fetch_calendar_weather(lat: float, lon: float, target_date: datetime.date):
    """Fetches daily weather/climate projection for a specific calendar date."""
    today = datetime.date.today()
    days_diff = (target_date - today).days
    years_ahead = max(0, target_date.year - today.year)

    if 0 <= days_diff <= 15:
        url = "https://api.open-meteo.com/v1/forecast"
        query_date = target_date
        mode = "Live Forecast"
        source = "Open-Meteo Forecast API"
    elif -2 <= days_diff < 0:
        url = "https://api.open-meteo.com/v1/forecast"
        query_date = target_date
        mode = "Recent Observed"
        source = "Open-Meteo Recent Archive"
    elif days_diff < -2:
        url = "https://archive-api.open-meteo.com/v1/archive"
        query_date = target_date
        mode = "Historical Observed"
        source = "Open-Meteo Historical Archive"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        query_date = get_safe_archive_date(target_date, today.year - 1)
        mode = "Multi-Year Climate Prediction" if years_ahead > 0 else "Seasonal Prediction"
        source = "Open-Meteo Seasonal Climatology + AI"

    date_str = query_date.isoformat()

    try:
        data = safe_get(
            url,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,wind_speed_10m_max"
                ),
                "hourly": "relative_humidity_2m,surface_pressure",
                "timezone": "auto",
            },
            timeout=15,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar weather error: {exc}")

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    max_list = daily.get("temperature_2m_max") or []
    min_list = daily.get("temperature_2m_min") or []
    precip_list = daily.get("precipitation_sum") or []
    wind_list = daily.get("wind_speed_10m_max") or []
    code_list = daily.get("weather_code") or []

    t_max = _safe_float(max_list[0] if max_list else None, 28.0)
    t_min = _safe_float(min_list[0] if min_list else None, 22.0)
    precip_mm = _safe_float(precip_list[0] if precip_list else None, 0.0)
    wind_max = _safe_float(wind_list[0] if wind_list else None, 8.0)
    wmo_code = _safe_int(code_list[0] if code_list else None, 0)

    if years_ahead > 0:
        warming_offset = round(years_ahead * 0.04, 2)
        if climate_ai:
            try:
                base_mean = (t_max + t_min) / 2.0
                pred_raw = float(climate_ai.predict([[base_mean, lat, years_ahead, 1.25]])[0])
                if math.isfinite(pred_raw):
                    warming_offset = round(pred_raw - base_mean, 2)
            except Exception:
                pass
        t_max = round(t_max + warming_offset, 1)
        t_min = round(t_min + warming_offset, 1)

    t_mean = round((t_max + t_min) / 2.0, 1)

    hum_vals = [v for v in (_valid_aqi_number(x) for x in (hourly.get("relative_humidity_2m") or [])) if v is not None]
    pres_vals = [v for v in (_valid_aqi_number(x) for x in (hourly.get("surface_pressure") or [])) if v is not None]
    avg_humidity = round(sum(hum_vals) / len(hum_vals), 1) if hum_vals else 60.0
    avg_pressure = round(sum(pres_vals) / len(pres_vals), 1) if pres_vals else 1012.0

    rain_prob = min(95, max(10, int(precip_mm * 12))) if precip_mm > 0.2 else 0

    raw_ai_pred, _, _ = predict_rainfall_flag(t_mean, avg_humidity, wind_max, avg_pressure)

    condition, visual_trigger, corrected_rain = classify_weather_condition(
        wmo_code, avg_humidity, t_mean, rain_prob, 0, precip_mm=precip_mm
    )
    _, ai_verdict = build_rainfall_verdict(raw_ai_pred, wmo_code, corrected_rain, precip_now=precip_mm)

    return {
        "mode": mode,
        "data_source": source,
        "condition": condition,
        "visual_trigger": visual_trigger,
        "temp_c": {
            "min": round(t_min, 1),
            "max": round(t_max, 1),
            "mean": t_mean,
            "display": f"{round(t_min)}° - {round(t_max)}°C",
        },
        "humidity_percent": avg_humidity,
        "wind_speed_kmh": round(wind_max, 1),
        "precipitation_mm": round(precip_mm, 1),
        "rain_chance_percent": corrected_rain,
        "ai_rainfall_verdict": ai_verdict,
    }


def fetch_calendar_aqi(lat: float, lon: float, target_date: datetime.date):
    """Fetches or estimates AQI and particulate levels for a calendar date."""
    today = datetime.date.today()
    days_diff = (target_date - today).days

    if datetime.date(2022, 8, 1) <= target_date <= (today + datetime.timedelta(days=5)):
        query_date = target_date
        source = "Open-Meteo Air Quality Forecast" if days_diff >= 0 else "Open-Meteo Air Quality Archive"
    else:
        query_date = get_safe_archive_date(target_date, today.year - 1)
        source = "Open-Meteo Seasonal AQI Baseline"

    date_str = query_date.isoformat()

    try:
        data = safe_get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "pm2_5,pm10,us_aqi",
                "timezone": "auto",
            },
            timeout=12,
        )
        hourly = data.get("hourly", {})
        pm25_vals = [v for v in (_valid_aqi_number(x) for x in (hourly.get("pm2_5") or [])) if v is not None]
        pm10_vals = [v for v in (_valid_aqi_number(x) for x in (hourly.get("pm10") or [])) if v is not None]

        if pm25_vals or pm10_vals:
            avg_pm25 = round(sum(pm25_vals) / len(pm25_vals), 1) if pm25_vals else None
            avg_pm10 = round(sum(pm10_vals) / len(pm10_vals), 1) if pm10_vals else None
            avg_aqi = calculate_indian_cpcb_aqi(avg_pm25, avg_pm10)

            if avg_aqi is not None:
                return {
                    "available": True,
                    "data_source": source,
                    "predicted_aqi": avg_aqi,
                    "aqi_range": calculate_aqi_range(avg_aqi),
                    "pm25_ugm3": avg_pm25,
                    "pm10_ugm3": avg_pm10,
                    "category": get_indian_aqi_category(avg_aqi),
                    "visual_trigger": get_aqi_visual_trigger(avg_aqi),
                }
    except Exception:
        pass

    waqi = get_waqi_live_data(lat, lon)
    if waqi.get("available"):
        aqi_val = round(waqi["aqi"], 1)
        scale_used = waqi.get("scale", "CPCB")
        return {
            "available": True,
            "data_source": "WAQI Baseline Fallback",
            "predicted_aqi": aqi_val,
            "aqi_range": calculate_aqi_range(aqi_val),
            "pm25_ugm3": waqi.get("pm25_ugm3"),
            "pm10_ugm3": waqi.get("pm10_ugm3"),
            "category": get_waqi_category(aqi_val) if scale_used == "WAQI_RAW" else get_indian_aqi_category(aqi_val),
            "visual_trigger": get_aqi_visual_trigger(aqi_val),
        }

    return {
        "available": False,
        "data_source": "Unavailable",
        "predicted_aqi": None,
        "aqi_range": None,
        "pm25_ugm3": None,
        "pm10_ugm3": None,
        "category": "Unavailable",
        "visual_trigger": "AQI_UNAVAILABLE",
    }


@app.get("/calendar-forecast")
def get_calendar_forecast(
    month: int,
    day: int,
    year: int = datetime.date.today().year,
    location: str = "Kolkata",
):
    """Returns combined weather and AQI forecast for a validated year, month, and day."""
    if not (1950 <= year <= 2100):
        raise HTTPException(status_code=400, detail="Year must be between 1950 and 2100.")
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")

    days_in_month = calendar.monthrange(year, month)[1]
    if not (1 <= day <= days_in_month):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid day {day} for {calendar.month_name[month]} {year} (max {days_in_month}).",
        )

    target_date = datetime.date(year, month, day)
    lat, lon, country, resolved_location = get_live_coordinates(location)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_weather = pool.submit(fetch_calendar_weather, lat, lon, target_date)
        f_aqi = pool.submit(fetch_calendar_aqi, lat, lon, target_date)
        weather = f_weather.result()
        aqi = f_aqi.result()

    return {
        "meta": {
            "region": resolved_location.upper(),
            "country": country,
            "coordinates": {"latitude": lat, "longitude": lon},
            "selected_date": target_date.isoformat(),
            "year": year,
            "month": month,
            "month_name": calendar.month_name[month],
            "day": day,
            "day_name": target_date.strftime("%A"),
            "days_in_month": days_in_month,
            "forecast_mode": weather["mode"],
        },
        "weather": weather,
        "air_quality": aqi,
    }


# =========================================================
# 12. MASTER DASHBOARD SUMMARY (PARALLELIZED & RESILIENT)
# =========================================================

def _resolve_future(future, fallback_factory, timeout_s=8.0):
    """Waits for a worker future with timeout and falls back cleanly on error."""
    try:
        return future.result(timeout=timeout_s)
    except Exception as exc:
        print(f"[DASHBOARD] Future fallback triggered: {type(exc).__name__}: {exc}")
        return fallback_factory()


@app.get("/dashboard-summary")
def get_complete_dashboard(location: str = "Kolkata"):
    """Aggregates weather, AQI, 7-day forecast, climate 2050, satellite radar, and WRF."""
    lat, lon, country, resolved_location = get_live_coordinates(location)

    def _fallback_weather():
        return {
            "date": str(datetime.date.today()),
            "observed_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "region": resolved_location.upper(),
            "country": country,
            "coordinates": {"latitude": lat, "longitude": lon},
            "weather_condition": "Clear & Sunny",
            "frontend_visual_trigger": "SUN_RAYS_EFFECT",
            "chance_of_rain_range": {"min": 0, "max": 0, "display": "0%"},
            "ai_rainfall_verdict": "No Rain Expected Today (Sunny & Dry)",
            "taskbar_metrics": {
                "temperature_c": 25.0,
                "real_feel_c": 25.0,
                "precipitation_mm": 0.0,
                "temp_range": {"min": 22.0, "max": 29.0, "display": "22° - 29°C"},
                "humidity_range": {"min": 50, "max": 60, "display": "50-60%"},
                "wind_speed_range": {"min": 4.0, "max": 8.0, "display": "4.0-8.0 km/h"},
                "wind_direction_compass": "N",
                "pressure_mb": 1013.0,
                "uv_index": 5.0,
                "uv_risk": "Moderate",
                "visibility_km": 10.0,
                "cloud_cover_percent": 10.0,
                "sunrise_time": "06:00",
                "sunset_time": "18:00",
            },
            "hourly_times_next_6h": [],
            "hourly_weather_codes_next_6h": [],
            "hourly_rain_chance_next_6h": [],
            "next_7_days_forecast": [],
        }

    def _fallback_aqi():
        return {
            "live_current_aqi": None,
            "aqi_range": None,
            "live_pm25_ugm3": None,
            "live_pm10_ugm3": None,
            "aqi_available": False,
            "pollution_category": "Unavailable",
            "data_source": "Unavailable",
            "aqi_scale": "Unavailable",
            "station_name": None,
            "measurement_time": None,
            "dominant_pollutant": None,
            "prediction_source": "Unavailable",
            "predicted_aqi_next_hour": None,
            "predicted_aqi_today": None,
            "predicted_aqi_tomorrow": None,
            "predicted_today_range": None,
            "predicted_tomorrow_range": None,
            "frontend_aqi_trigger": "AQI_UNAVAILABLE",
        }

    def _fallback_climate():
        return {
            "region": resolved_location.upper(),
            "country": country,
            "baseline_year": datetime.date.today().year,
            "target_year": 2050,
            "current_temperature": 25.0,
            "projected_temperature": 26.1,
            "temperature_range": {"min": 24.9, "max": 27.6, "display": "24.9° - 27.6°C"},
            "impact_risk": "Moderate Climate Adaptation Needed",
        }

    def _fallback_sat():
        return {
            "region": resolved_location.upper(),
            "country": country,
            "coordinates": {"latitude": lat, "longitude": lon},
            "satellite_map_link": f"https://maps.google.com/?q={lat},{lon}&t=k",
            "disaster_risk_assessment": {
                "primary_threat_status": "Normal (No Disaster Imminent)",
                "probability_percentages": {
                    "Flood Risk": 5.0,
                    "Earthquake Tectonic Warning": 2.0,
                    "Wildfire Thermal Alert": 5.0,
                },
                "thermal_anomaly_detected": False,
                "recent_earthquakes_500km": [],
            },
            "forest_fire_scan": {
                "status": "Active Meteorological Fire Index",
                "risk_level": "Normal",
            },
        }

    def _fallback_wrf():
        return {
            "region": resolved_location.upper(),
            "country": country,
            "numerical_model": "Open-Meteo ICON / GFS Seamless NWP",
            "simulated_atmospheric_metrics": {
                "wind_speed_kmh": 5.5,
                "wind_gusts_kmh": 8.5,
                "surface_pressure_mb": 1013.0,
                "wind_speed_range": {"min": 4.0, "max": 10.0, "display": "4.0-10.0 km/h"},
                "wind_direction_degrees": 140.0,
                "wind_direction_compass": "SE",
                "atmospheric_stability_index": "Stable Laminar Flow",
            },
            "progression_12h": [],
        }

    with ThreadPoolExecutor(max_workers=5) as pool:
        f_weather = pool.submit(_run_safely, predict_live_rainfall, _fallback_weather, location)
        f_aqi = pool.submit(_run_safely, predict_live_aqi, _fallback_aqi, location)
        f_climate = pool.submit(_run_safely, forecast_climate, _fallback_climate, location, 2050)
        f_sat = pool.submit(_run_safely, get_satellite_map, _fallback_sat, location)
        f_wrf = pool.submit(_run_safely, get_wrf_model, _fallback_wrf, location)

        weather_info = _resolve_future(f_weather, _fallback_weather, timeout_s=8.0)
        aqi_info = _resolve_future(f_aqi, _fallback_aqi, timeout_s=8.0)
        climate_info = _resolve_future(f_climate, _fallback_climate, timeout_s=8.0)
        satellite_info = _resolve_future(f_sat, _fallback_sat, timeout_s=8.0)
        wrf_info = _resolve_future(f_wrf, _fallback_wrf, timeout_s=8.0)

    return {
        "location_header": f"{weather_info['region']}, {weather_info['country']}",
        "coordinates": weather_info["coordinates"],
        "observed_at": weather_info.get("observed_at"),
        "weather_condition": weather_info["weather_condition"],
        "visual_triggers": {
            "weather_effect": weather_info["frontend_visual_trigger"],
            "aqi_effect": aqi_info["frontend_aqi_trigger"],
        },
        "taskbar_cards": weather_info["taskbar_metrics"],
        "rain_analysis": {
            "ai_verdict": weather_info["ai_rainfall_verdict"],
            "chance_of_range": weather_info["chance_of_rain_range"],
            "humidity_range": weather_info["taskbar_metrics"]["humidity_range"],
            "wind_speed_range": weather_info["taskbar_metrics"]["wind_speed_range"],
            "hourly_times_next_6h": weather_info["hourly_times_next_6h"],
            "hourly_weather_codes_next_6h": weather_info["hourly_weather_codes_next_6h"],
            "hourly_rain_bars": weather_info["hourly_rain_chance_next_6h"],
            "minutecast": weather_info.get("accuweather_alerts", {}).get("minutecast"),
            "severe_alerts": weather_info.get("accuweather_alerts", {}).get("severe_alerts", []),
        },
        "air_quality": {
            "live_current_aqi": aqi_info["live_current_aqi"],
            "aqi_range": aqi_info["aqi_range"],
            "live_pm25_ugm3": aqi_info["live_pm25_ugm3"],
            "live_pm10_ugm3": aqi_info["live_pm10_ugm3"],
            "aqi_available": aqi_info["aqi_available"],
            "status": aqi_info["pollution_category"],
            "data_source": aqi_info["data_source"],
            "aqi_scale": aqi_info["aqi_scale"],
            "station_name": aqi_info["station_name"],
            "measurement_time": aqi_info["measurement_time"],
            "dominant_pollutant": aqi_info["dominant_pollutant"],
            "prediction_source": aqi_info["prediction_source"],
            "predicted_aqi_next_hour": aqi_info["predicted_aqi_next_hour"],
            "predicted_aqi_today": aqi_info["predicted_aqi_today"],
            "predicted_aqi_tomorrow": aqi_info["predicted_aqi_tomorrow"],
            "predicted_today_range": aqi_info["predicted_today_range"],
            "predicted_tomorrow_range": aqi_info["predicted_tomorrow_range"],
        },
        "weekly_strip": weather_info["next_7_days_forecast"],
        "climate_2050": climate_info,
        "satellite_radar": satellite_info,
        "wrf_simulation": wrf_info,
        "accuweather_intelligence": weather_info.get("accuweather_alerts"),
    }


@app.get("/accuweather-alerts")
def accuweather_alerts_endpoint(location: str = "Kolkata"):
    """Dedicated endpoint for AccuWeather MinuteCast timing and severe weather alerts."""
    lat, lon, country, resolved_location = get_live_coordinates(location)
    bundle = _run_safely(
        fetch_accuweather_accuracy_and_alerts,
        lambda: {
            "enabled": False,
            "provider": "Unavailable",
            "current": None,
            "minutecast": {"available": False, "summary": "Unavailable", "precipitation_window": "N/A"},
            "severe_alerts": [],
        },
        lat,
        lon,
    )
    return {
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "accuweather": bundle,
    }


# =========================================================
# 13. HEALTH CHECK & SINGLE-URL FRONTEND MOUNT
# =========================================================

@app.get("/health")
def health_check():
    """Health and capability summary endpoint."""
    return {
        "status": "EVS Climate AI backend is running",
        "rainfall_priority": "Open-Meteo Live Weather -> ML Model / Meteorological Fallback",
        "live_aqi_priority": ["WAQI (CPCB / Raw)", "Open-Meteo CAMS (CPCB)"],
        "aqi_prediction_priority": ["WAQI Live + Open-Meteo CAMS Forecast Blend", "Live Persistence Fallback"],
        "calendar_endpoint": "/calendar-forecast?location=Kolkata&year=2026&month=11&day=19",
        "waqi_token_configured": bool(WAQI_API_TOKEN),
        "accuweather_key_configured": bool(ACCUWEATHER_API_KEY),
    }


def _resolve_frontend_file() -> Path | None:
    """Finds index.html in standard project locations if present."""
    candidates = [
        BASE_DIR / "frontend" / "index.html",
        BASE_DIR / "index.html",
        BASE_DIR.parent / "frontend" / "index.html",
        BASE_DIR.parent / "index.html",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@app.get("/", response_class=HTMLResponse)
def serve_frontend_index():
    """Serves the frontend index.html or a clean backend status page."""
    index_file = _resolve_frontend_file()
    if index_file:
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse(
        content=(
            "<!DOCTYPE html><html><head><title>Klima Backend Running</title></head>"
            "<body style='font-family:sans-serif;background:#090d16;color:#f8fafc;padding:2rem;'>"
            "<h2>Klima FastAPI Backend is Running</h2>"
            "<p>Place <code>index.html</code> in <code>frontend/index.html</code> or next to <code>main.py</code>, "
            "or visit <a href='/docs' style='color:#38bdf8;'>/docs</a> to test API endpoints.</p>"
            "</body></html>"
        ),
        status_code=200,
    )


FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend-static")