import os
import math
import calendar
import datetime
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import joblib
import pandas as pd
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

WAQI_API_TOKEN = os.getenv("WAQI_API_TOKEN", os.getenv("WAQI_TOKEN", "")).strip().strip("'\"")

app = FastAPI(title="EVS Climate AI - Live Data Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. LOAD MACHINE-LEARNING MODELS SAFELY
# =========================================================

def load_model(filename):
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
    if val is None:
        return float(default)
    try:
        num = float(val)
        return num if math.isfinite(num) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(val, default):
    if val is None:
        return int(default)
    try:
        num = float(val)
        return int(num) if math.isfinite(num) else int(default)
    except (TypeError, ValueError):
        return int(default)


def predict_rainfall_flag(temp, humidity, wind_speed, pressure):
    if not rainfall_ai:
        return None, "Rainfall model unavailable"
    try:
        t_val = _safe_float(temp, 25.0)
        h_val = _safe_float(humidity, 60.0)
        w_val = _safe_float(wind_speed, 8.0)
        p_val = _safe_float(pressure, 1012.0)

        features_df = pd.DataFrame(
            [[t_val, h_val, w_val, p_val]],
            columns=["tempC", "humidity", "windspeedKmph", "pressure"],
        )
        pred = int(rainfall_ai.predict(features_df)[0])
        verdict = "Rain Expected Today" if pred == 1 else "No Rain Expected Today"
        return pred, verdict
    except Exception as exc:
        print(f"[MODEL] Rainfall prediction error: {exc}")
        return None, "Rainfall model prediction unavailable"


def build_rainfall_verdict(ai_pred, ai_verdict, wmo_code, rain_chance):
    code_val = _safe_int(wmo_code, 0)
    chance_val = _safe_int(rain_chance, 0)

    if ai_pred == 1 or code_val >= 51 or chance_val >= 55:
        return "Rain Expected Today"
    if chance_val >= 25:
        return "Chance of Rain Today"
    if ai_pred is None:
        return ai_verdict or "Forecast Unavailable"
    return "No Rain Expected Today"


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


def safe_get(url, params=None, timeout=12):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def _cached_geocode(normalized_query: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    mapped_query = INDIAN_STATE_HUBS.get(normalized_query.lower(), normalized_query)

    data = safe_get(
        url,
        params={
            "name": mapped_query,
            "count": 5,
            "format": "json",
        },
    )
    results = data.get("results") or []

    if not results:
        return None

    india_match = next((item for item in results if item.get("country_code") == "IN"), None)
    place = (
        india_match
        if (india_match and normalized_query.lower() in INDIAN_STATE_HUBS and normalized_query.lower() not in {"nepal", "kathmandu"})
        else results[0]
    )
    return (
        float(place["latitude"]),
        float(place["longitude"]),
        place.get("country", "Earth"),
        place.get("name", mapped_query),
    )


def get_live_coordinates(location: str):
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


def degrees_to_compass(deg):
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


def classify_current_weather(
    wmo_code,
    humidity,
    temp,
    precip_now=0.0,
    hourly_code=None,
    hourly_rain_prob=None,
):
    """Classifies what is happening RIGHT NOW using current + current-hour telemetry."""
    raw_code = hourly_code if hourly_code is not None else wmo_code
    code_val = _safe_int(raw_code, 0)
    hum_val = _safe_float(humidity, 50.0)
    temp_val = _safe_float(temp, 25.0)
    precip_val = _safe_float(precip_now, 0.0)
    prob_val = _safe_int(hourly_rain_prob, 0)

    if code_val >= 95:
        return "Thunderstorm", "THUNDER_EFFECT"
    elif 71 <= code_val <= 77:
        return "Snowfall", "SNOW_EFFECT"
    elif code_val >= 51 or precip_val > 0.1:
        return "Rainy", "RAIN_EFFECT"
    elif prob_val >= 75 and hum_val >= 80:
        return "Rainy Showers Likely", "RAIN_EFFECT"
    elif code_val in [45, 48]:
        return "Foggy / Hazy", "SMOG_FOG_EFFECT"
    elif hum_val < 35 and temp_val > 28:
        return "Dry", "DRY_HEAT_EFFECT"
    elif code_val in [2, 3]:
        return "Partly Cloudy", "CLOUD_EFFECT"
    elif code_val in [0, 1]:
        return "Clear", "SUN_RAYS_EFFECT"

    return "Clear", "SUN_RAYS_EFFECT"


def classify_weather_condition(wmo_code, humidity, temp, rain_chance, ai_rain_flag):
    code_val = _safe_int(wmo_code, 0)
    hum_val = _safe_float(humidity, 50.0)
    temp_val = _safe_float(temp, 25.0)
    chance_val = _safe_int(rain_chance, 0)
    ai_flag_val = _safe_int(ai_rain_flag, 0)

    if chance_val == 0:
        if code_val in [2, 3]:
            chance_val = 20
        elif hum_val > 75:
            chance_val = 20

    if ai_flag_val == 1 or code_val >= 51 or chance_val >= 55:
        return "Rainy", "RAIN_EFFECT", chance_val
    elif hum_val < 35 and temp_val > 28:
        return "Dry", "DRY_HEAT_EFFECT", chance_val
    elif code_val in [0, 1]:
        return "Sunny", "SUN_RAYS_EFFECT", chance_val
    elif code_val in [2, 3]:
        return "Partly Cloudy", "CLOUD_EFFECT", chance_val
    elif code_val in [45, 48]:
        return "Foggy / Hazy", "SMOG_FOG_EFFECT", chance_val

    return "Clear", "SUN_RAYS_EFFECT", chance_val


# =========================================================
# 4. AQI CATEGORY, RANGE, PM CONVERSION & CPCB CALCULATIONS
# =========================================================

def _valid_aqi_number(aqi):
    if aqi is None:
        return None
    try:
        val = float(aqi)
        return val if math.isfinite(val) and val >= 0 else None
    except (TypeError, ValueError):
        return None


def get_indian_aqi_category(aqi):
    val = _valid_aqi_number(aqi)
    if val is None:
        return "Unavailable"
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Satisfactory"
    elif val <= 200:
        return "Moderate"
    elif val <= 300:
        return "Poor"
    elif val <= 400:
        return "Very Poor"
    return "Severe"


def get_waqi_category(aqi):
    val = _valid_aqi_number(aqi)
    if val is None:
        return "Unavailable"
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Moderate"
    elif val <= 150:
        return "Unhealthy for Sensitive Groups"
    elif val <= 200:
        return "Unhealthy"
    elif val <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_aqi_visual_trigger(aqi):
    val = _valid_aqi_number(aqi)
    if val is None:
        return "AQI_UNAVAILABLE"
    elif val > 150:
        return "HEAVY_SMOG_EFFECT"
    elif val > 50:
        return "LIGHT_HAZE_EFFECT"
    return "CLEAN_AIR_EFFECT"


def calculate_aqi_range(aqi, margin=15):
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
        cpcb_aqi = calculate_indian_cpcb_aqi(pm25_ugm3, pm10_ugm3)

        station = data.get("city", {})

        return {
            "available": True,
            "aqi": cpcb_aqi if cpcb_aqi is not None else float(aqi_value),
            "pm25_ugm3": pm25_ugm3,
            "pm10_ugm3": pm10_ugm3,
            "station_name": station.get("name", "WAQI monitoring station"),
            "measurement_time": data.get("time", {}).get("s"),
            "dominant_pollutant": data.get("dominentpol"),
        }

    except Exception:
        return {"available": False}


def get_cams_live_aqi(lat, lon):
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
            "pm25_ugm3": pm25_val,
            "pm10_ugm3": pm10_val,
            "measurement_time": current.get("time"),
        }

    except Exception:
        return {"available": False}


def get_open_meteo_aqi_prediction(lat, lon):
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
            "source": "Open-Meteo Air Quality Forecast (WAQI Scale)",
            "predicted_aqi_next_hour": next_hour_val if next_hour_val is not None else pred_today,
            "predicted_aqi_today": pred_today,
            "predicted_aqi_tomorrow": pred_tomorrow,
        }

    except Exception:
        return {"available": False}


# =========================================================
# 6. WEATHER AND RAINFALL ENDPOINT (DYNAMIC PER HOUR)
# =========================================================

@lru_cache(maxsize=256)
def _cached_weather_payload(lat: float, lon: float, hour_bucket: str):
    weather_url = "https://api.open-meteo.com/v1/forecast"
    return safe_get(
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


@app.get("/predict-rainfall")
def predict_live_rainfall(location: str = "Kolkata"):
    lat, lon, country, resolved_location = get_live_coordinates(location)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    hour_bucket = utc_now.strftime("%Y-%m-%dT%H")

    try:
        data = _cached_weather_payload(round(lat, 4), round(lon, 4), hour_bucket)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    utc_offset = _safe_int(data.get("utc_offset_seconds"), 0)
    local_now = utc_now + datetime.timedelta(seconds=utc_offset)
    current_time_str = str(current.get("time") or local_now.strftime("%Y-%m-%dT%H:%M"))

    # Align hourly slice to the actual current local hour rather than midnight index 0
    cur_idx = get_current_hour_slice(hourly, current_time_str)
    hourly_temps = hourly.get("temperature_2m") or []
    hourly_hums = hourly.get("relative_humidity_2m") or []
    hourly_winds = hourly.get("wind_speed_10m") or []
    hourly_probs = hourly.get("precipitation_probability") or []
    hourly_codes = hourly.get("weather_code") or []
    hourly_vis = hourly.get("visibility") or []

    temp = _safe_float(
        current.get("temperature_2m"),
        hourly_temps[cur_idx] if cur_idx < len(hourly_temps) else 25.0,
    )
    feels_like = _safe_float(current.get("apparent_temperature"), temp)
    humidity = _safe_float(
        current.get("relative_humidity_2m"),
        hourly_hums[cur_idx] if cur_idx < len(hourly_hums) else 50.0,
    )
    precip_now = _safe_float(current.get("precipitation"), 0.0)
    wind_speed = _safe_float(
        current.get("wind_speed_10m"),
        hourly_winds[cur_idx] if cur_idx < len(hourly_winds) else 5.0,
    )
    wind_dir_deg = _safe_float(current.get("wind_direction_10m"), 0.0)
    pressure = _safe_float(current.get("surface_pressure"), 1013.0)
    clouds = _safe_float(current.get("cloud_cover"), 0.0)
    wmo_code = _safe_int(current.get("weather_code"), 0)

    hour_code = _safe_int(hourly_codes[cur_idx] if cur_idx < len(hourly_codes) else None, wmo_code)
    corrected_rain = _safe_int(
        hourly_probs[cur_idx] if cur_idx < len(hourly_probs) and hourly_probs[cur_idx] is not None else None,
        0,
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
    uv_index = _safe_float(uv_list[0] if uv_list else None, 5.0)

    vis_raw = _safe_float(
        hourly_vis[cur_idx] if cur_idx < len(hourly_vis) else None, 10000.0
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

    ai_pred, raw_ai_verdict = predict_rainfall_flag(temp, humidity, wind_speed, pressure)
    if corrected_rain < 20 and precip_now <= 0.1 and wmo_code < 51 and hour_code < 51:
        ai_verdict = "No Rain Currently Expected"
    else:
        ai_verdict = build_rainfall_verdict(ai_pred, raw_ai_verdict, max(wmo_code, hour_code), corrected_rain)

    wind_min = max(0.0, round(wind_speed - 2.0, 1))
    wind_max = round(wind_speed + 2.0, 1)
    hum_min = max(0, int(humidity - 5))
    hum_max = min(100, int(humidity + 5))
    rain_min = max(0, int(corrected_rain - 5))
    rain_max = min(100, int(corrected_rain + 5))

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
        d_rain = _safe_int(daily_rains[i] if i < len(daily_rains) else None, 10)

        d_condition, _, d_rain = classify_weather_condition(d_code, humidity, d_max, d_rain, 0)
        parsed_date = datetime.date.fromisoformat(date_string)

        seven_day_forecast.append({
            "day_name": parsed_date.strftime("%a"),
            "date": date_string,
            "condition": d_condition,
            "temp_range_c": {
                "min": round(d_min, 1),
                "max": round(d_max, 1),
                "display": f"{round(d_min)}° - {round(d_max)}°C",
            },
            "rain_chance_range": {
                "min": max(0, d_rain - 5),
                "max": min(100, d_rain + 5),
                "display": f"{max(0, d_rain - 5)}-{min(100, d_rain + 5)}%",
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

    hourly_rain_clean = [
        _safe_int(v, 0) for v in hourly_probs[cur_idx : cur_idx + 6]
    ]
    hourly_codes_clean = [
        _safe_int(v, 0) for v in hourly_codes[cur_idx : cur_idx + 6]
    ]

    return {
        "date": str(local_now.date()),
        "observed_at": current_time_str,
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "weather_condition": weather_label,
        "frontend_visual_trigger": visual_trigger,
        "chance_of_rain_range": {
            "min": rain_min,
            "max": rain_max,
            "display": f"{rain_min}-{rain_max}%",
        },
        "ai_rainfall_verdict": ai_verdict,
        "taskbar_metrics": {
            "temperature_c": temp,
            "real_feel_c": feels_like,
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
# 7. LIVE AQI & PREDICTION ENDPOINT
# =========================================================

@app.get("/predict-aqi")
def predict_live_aqi(location: str = "Kolkata"):
    lat, lon, country, resolved_location = get_live_coordinates(location)

    live_aqi = None
    live_pm25 = None
    live_pm10 = None
    source_used = "Unavailable"
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
            measurement_time = cams.get("measurement_time")
            category = get_waqi_category(live_aqi)

    om_pred = get_open_meteo_aqi_prediction(lat, lon)
    if om_pred.get("available"):
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

    return {
        "date": str(datetime.date.today()),
        "region": resolved_location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "data_source": source_used,
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
    lat, lon, country, resolved_location = get_live_coordinates(location)

    try:
        weather = safe_get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,surface_pressure,"
                    "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
                ),
                "timezone": "auto",
            },
            timeout=12,
        )
        current = weather.get("current", {})
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Atmospheric model error: {exc}")

    wind_speed = round(_safe_float(current.get("wind_speed_10m"), 5.5), 1)
    wind_dir = round(_safe_float(current.get("wind_direction_10m"), 140.0), 1)
    wind_gusts = round(_safe_float(current.get("wind_gusts_10m"), wind_speed + 3.0), 1)
    pressure = round(_safe_float(current.get("surface_pressure"), 1013.0), 1)

    gust_spread = round(max(0.0, wind_gusts - wind_speed), 1)
    if wind_gusts >= 30.0 or gust_spread >= 15.0:
        stability = "Severe Turbulent Gust Warning"
    elif wind_speed >= 15.0 or gust_spread >= 8.0:
        stability = "Moderate Atmospheric Turbulence"
    else:
        stability = "Stable Laminar Flow"

    wind_min = round(max(0.0, wind_speed - 1.5), 1)
    wind_max = round(max(wind_speed + 1.5, wind_gusts), 1)

    return {
        "region": resolved_location.upper(),
        "country": country,
        "numerical_model": "Open-Meteo ICON / GFS Seamless NWP",
        "simulated_atmospheric_metrics": {
            "wind_speed_kmh": wind_speed,
            "wind_gusts_kmh": wind_gusts,
            "surface_pressure_mb": pressure,
            "wind_speed_range": {
                "min": wind_min,
                "max": wind_max,
                "display": f"{wind_min}-{wind_max} km/h",
            },
            "wind_direction_degrees": wind_dir,
            "wind_direction_compass": degrees_to_compass(wind_dir),
            "atmospheric_stability_index": stability,
        },
        "progression_12h": [
            {
                "horizon_label": "Now" if h_off == 0 else f"+{h_off}h",
                "hours_ahead": h_off,
                "wind_speed_kmh": max(2.0, round(wind_speed + math.sin(idx * 1.15) * 2.2, 1)),
                "wind_gusts_kmh": max(3.5, round(wind_gusts + math.sin(idx * 1.15) * 3.1, 1)),
                "surface_pressure_mb": round(pressure - math.sin(idx * 1.15) * 0.8, 1),
            }
            for idx, h_off in enumerate([0, 3, 6, 12])
        ],
    }


# =========================================================
# 11. CALENDAR FORECAST ENGINE (YEAR + MONTH + DAY)
# =========================================================

def get_safe_archive_date(target_date: datetime.date, reference_year: int) -> datetime.date:
    try:
        return target_date.replace(year=reference_year)
    except ValueError:
        return target_date.replace(year=reference_year, day=28)


def fetch_calendar_weather(lat: float, lon: float, target_date: datetime.date):
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

    rain_prob = min(95, max(10, int(precip_mm * 12))) if precip_mm > 0.2 else 10

    ai_pred, raw_ai_verdict = predict_rainfall_flag(t_mean, avg_humidity, wind_max, avg_pressure)

    condition, visual_trigger, corrected_rain = classify_weather_condition(
        wmo_code, avg_humidity, t_mean, rain_prob, ai_pred if ai_pred is not None else 0
    )
    ai_verdict = build_rainfall_verdict(ai_pred, raw_ai_verdict, wmo_code, corrected_rain)

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
                    "category": get_waqi_category(avg_aqi),
                    "visual_trigger": get_aqi_visual_trigger(avg_aqi),
                }
    except Exception:
        pass

    waqi = get_waqi_live_data(lat, lon)
    if waqi.get("available"):
        aqi_val = round(waqi["aqi"], 1)
        return {
            "available": True,
            "data_source": "WAQI Baseline Fallback",
            "predicted_aqi": aqi_val,
            "aqi_range": calculate_aqi_range(aqi_val),
            "pm25_ugm3": waqi.get("pm25_ugm3"),
            "pm10_ugm3": waqi.get("pm10_ugm3"),
            "category": get_waqi_category(aqi_val),
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
# 12. MASTER DASHBOARD SUMMARY (PARALLELIZED)
# =========================================================

@app.get("/dashboard-summary")
def get_complete_dashboard(location: str = "Kolkata"):
    get_live_coordinates(location)

    with ThreadPoolExecutor(max_workers=5) as pool:
        f_weather = pool.submit(predict_live_rainfall, location)
        f_aqi = pool.submit(predict_live_aqi, location)
        f_climate = pool.submit(forecast_climate, location, 2050)
        f_sat = pool.submit(get_satellite_map, location)
        f_wrf = pool.submit(get_wrf_model, location)

        weather_info = f_weather.result()
        aqi_info = f_aqi.result()
        climate_info = f_climate.result()
        satellite_info = f_sat.result()
        wrf_info = f_wrf.result()

    return {
        "location_header": f"{weather_info['region']}, {weather_info['country']}",
        "coordinates": weather_info["coordinates"],
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
        },
        "air_quality": {
            "live_current_aqi": aqi_info["live_current_aqi"],
            "aqi_range": aqi_info["aqi_range"],
            "live_pm25_ugm3": aqi_info["live_pm25_ugm3"],
            "live_pm10_ugm3": aqi_info["live_pm10_ugm3"],
            "aqi_available": aqi_info["aqi_available"],
            "status": aqi_info["pollution_category"],
            "data_source": aqi_info["data_source"],
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
    }


# =========================================================
# 13. HEALTH CHECK & SINGLE-URL FRONTEND MOUNT
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "EVS Climate AI backend is running",
        "live_aqi_priority": ["WAQI (CPCB Breakpoints)", "Open-Meteo CAMS"],
        "aqi_prediction_priority": ["Open-Meteo Forecast (WAQI Scale)", "WAQI Fallback"],
        "calendar_endpoint": "/calendar-forecast?location=Kolkata&year=2026&month=11&day=19",
        "waqi_token_configured": bool(WAQI_API_TOKEN),
    }


FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")