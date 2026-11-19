import os
import math
import calendar
import datetime
from pathlib import Path

import joblib
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


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
    try:
        return joblib.load(path) if path.exists() else None
    except Exception:
        return None


rainfall_ai = load_model("rainfall_ml_model.pkl")
climate_ai = load_model("climate_rf_model.pkl")
disaster_ai = load_model("disaster_rf_model.pkl")
wrf_ai = load_model("wrf_rf_model.pkl")


# =========================================================
# 3. COMMON UTILITIES
# =========================================================

def safe_get(url, params=None, timeout=12):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_live_coordinates(location: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    try:
        data = safe_get(
            url,
            params={"name": location, "count": 1, "format": "json"}
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding service error: {exc}")

    results = data.get("results", [])

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location}' not found."
        )

    place = results[0]

    return (
        float(place["latitude"]),
        float(place["longitude"]),
        place.get("country", "India"),
        place.get("name", location)
    )


def degrees_to_compass(deg):
    if deg is None:
        return "N"

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(float(deg) / 45) % 8]


def classify_weather_condition(wmo_code, humidity, temp, rain_chance, ai_rain_flag):
    wmo_code = wmo_code or 0
    humidity = humidity or 50
    temp = temp or 25
    rain_chance = rain_chance or 0

    if rain_chance == 0:
        if wmo_code in [2, 3]:
            rain_chance = 25
        elif humidity > 70:
            rain_chance = 20

    if ai_rain_flag == 1 or wmo_code >= 51 or rain_chance >= 55:
        return "Rainy", "RAIN_EFFECT", rain_chance
    elif humidity < 35 and temp > 28:
        return "Dry", "DRY_HEAT_EFFECT", rain_chance
    elif wmo_code in [0, 1]:
        return "Sunny", "SUN_RAYS_EFFECT", rain_chance
    elif wmo_code in [2, 3]:
        return "Partly Cloudy", "CLOUD_EFFECT", rain_chance
    elif wmo_code in [45, 48]:
        return "Foggy / Hazy", "SMOG_FOG_EFFECT", rain_chance

    return "Clear", "SUN_RAYS_EFFECT", rain_chance


# =========================================================
# 4. AQI CATEGORY, RANGE, PM CONVERSION & CPCB CALCULATIONS
# =========================================================

def get_indian_aqi_category(aqi):
    if aqi is None:
        return "Unavailable"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


def get_waqi_category(aqi):
    if aqi is None:
        return "Unavailable"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_aqi_visual_trigger(aqi):
    if aqi is None:
        return "AQI_UNAVAILABLE"
    elif aqi > 200:
        return "HEAVY_SMOG_EFFECT"
    elif aqi > 50:
        return "LIGHT_HAZE_EFFECT"
    return "CLEAN_AIR_EFFECT"


def calculate_aqi_range(aqi, margin=15):
    """Returns AQI range with lower and upper limits."""
    if aqi is None:
        return None

    lower_limit = max(0, round(aqi - margin))
    upper_limit = min(500, round(aqi + margin))

    return {
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "display": f"{lower_limit}-{upper_limit}"
    }


def epa_aqi_to_ugm3(pollutant: str, aqi):
    """Reverse-converts WAQI US EPA AQI sub-index to approximate µg/m³."""
    if aqi is None:
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
        if ilo <= aqi <= ihi:
            conc = ((aqi - ilo) / (ihi - ilo)) * (chi - clo) + clo
            return round(conc, 1)

    return None


def calculate_pollutant_subindex(concentration, breakpoints):
    if concentration is None:
        return None

    for low_c, high_c, low_i, high_i in breakpoints:
        if low_c <= concentration <= high_c:
            result = ((high_i - low_i) / (high_c - low_c)) * (concentration - low_c) + low_i
            return round(result, 1)

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
        aqi_value = data.get("aqi")

        if not isinstance(aqi_value, (int, float)) or not math.isfinite(float(aqi_value)) or float(aqi_value) < 0:
            return {"available": False}

        iaqi = data.get("iaqi", {})
        pm25_sub = iaqi.get("pm25", {}).get("v") if isinstance(iaqi.get("pm25"), dict) else None
        pm10_sub = iaqi.get("pm10", {}).get("v") if isinstance(iaqi.get("pm10"), dict) else None

        station = data.get("city", {})

        return {
            "available": True,
            "aqi": float(aqi_value),
            "pm25_ugm3": epa_aqi_to_ugm3("pm25", pm25_sub),
            "pm10_ugm3": epa_aqi_to_ugm3("pm10", pm10_sub),
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
            timeout=12
        )

        current = data.get("current", {})
        pm25 = current.get("pm2_5")
        pm10 = current.get("pm10")

        if pm25 is None and pm10 is None:
            return {"available": False}

        pm25_val = round(float(pm25), 1) if pm25 is not None else None
        pm10_val = round(float(pm10), 1) if pm10 is not None else None

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
    """Priority 1 AQI Prediction: Open-Meteo 48-hour forecast."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    try:
        data = safe_get(
            url,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "us_aqi",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=12
        )

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        values = hourly.get("us_aqi", [])

        if not times or not values:
            return {"available": False}

        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        current_hour_dt = now.replace(minute=0, second=0, microsecond=0)

        today_vals = []
        tomorrow_vals = []
        next_hour_val = None

        for t, val in zip(times, values):
            if val is None:
                continue
            dt = datetime.datetime.fromisoformat(t)
            if dt < current_hour_dt:
                continue

            val_f = float(val)
            if next_hour_val is None and dt > current_hour_dt:
                next_hour_val = round(val_f, 1)

            if t.startswith(today_str):
                today_vals.append(val_f)
            elif t.startswith(tomorrow_str):
                tomorrow_vals.append(val_f)

        if not today_vals and not tomorrow_vals:
            return {"available": False}

        pred_today = round(sum(today_vals) / len(today_vals), 1) if today_vals else None
        pred_tomorrow = round(sum(tomorrow_vals) / len(tomorrow_vals), 1) if tomorrow_vals else None

        return {
            "available": True,
            "source": "Open-Meteo Air Quality Forecast",
            "predicted_aqi_next_hour": next_hour_val if next_hour_val is not None else pred_today,
            "predicted_aqi_today": pred_today,
            "predicted_aqi_tomorrow": pred_tomorrow,
        }

    except Exception:
        return {"available": False}


# =========================================================
# 6. WEATHER AND RAINFALL ENDPOINT
# =========================================================

@app.get("/predict-rainfall")
def predict_live_rainfall(location: str = "Kolkata"):
    lat, lon, country, resolved_location = get_live_coordinates(location)

    weather_url = "https://api.open-meteo.com/v1/forecast"

    try:
        data = safe_get(
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
                    "visibility,uv_index,precipitation_probability,"
                    "wind_speed_10m,relative_humidity_2m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "sunrise,sunset,uv_index_max,"
                    "precipitation_probability_max,"
                    "wind_speed_10m_max,wind_speed_10m_min"
                ),
                "timezone": "auto",
            }
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    temp = current.get("temperature_2m", 25.0)
    feels_like = current.get("apparent_temperature", temp)
    humidity = current.get("relative_humidity_2m", 50.0)
    wind_speed = current.get("wind_speed_10m", 5.0)
    wind_dir_deg = current.get("wind_direction_10m", 0)
    pressure = current.get("surface_pressure", 1013.0)
    clouds = current.get("cloud_cover", 0)
    wmo_code = current.get("weather_code", 0)

    temp_high = (daily.get("temperature_2m_max") or [temp])[0]
    temp_low = (daily.get("temperature_2m_min") or [temp])[0]

    sunrise_raw = (daily.get("sunrise") or ["06:00"])[0]
    sunset_raw = (daily.get("sunset") or ["18:00"])[0]
    sunrise = sunrise_raw.split("T")[-1]
    sunset = sunset_raw.split("T")[-1]

    uv_index = (daily.get("uv_index_max") or [5.0])[0]
    rain_chance = (daily.get("precipitation_probability_max") or [10])[0]

    visibility_list = hourly.get("visibility", [10000])
    visibility_km = round((visibility_list[0] or 10000) / 1000, 1)

    ai_pred = 0
    ai_verdict = "Rainfall model unavailable"

    if rainfall_ai:
        try:
            ai_pred = int(rainfall_ai.predict([[temp, humidity, wind_speed, pressure]])[0])
            ai_verdict = "Rain Expected Today" if ai_pred == 1 else "No Rain Expected Today"
        except Exception:
            ai_verdict = "Rainfall model prediction unavailable"

    weather_label, visual_trigger, corrected_rain = classify_weather_condition(
        wmo_code, humidity, temp, rain_chance, ai_pred
    )

    wind_min = max(0.0, round(wind_speed - 2, 1))
    wind_max = round(wind_speed + 2, 1)
    hum_min = max(0, int(humidity - 5))
    hum_max = min(100, int(humidity + 5))
    rain_min = max(0, int(corrected_rain - 5))
    rain_max = min(100, int(corrected_rain + 5))

    seven_day_forecast = []
    days_time = daily.get("time", [])
    daily_codes = daily.get("weather_code", [])
    daily_rains = daily.get("precipitation_probability_max", [])
    daily_maxs = daily.get("temperature_2m_max", [])
    daily_mins = daily.get("temperature_2m_min", [])
    daily_winds = daily.get("wind_speed_10m_max", [])

    for i, date_string in enumerate(days_time[:7]):
        d_max = daily_maxs[i] if i < len(daily_maxs) else temp
        d_min = daily_mins[i] if i < len(daily_mins) else temp
        d_wind = daily_winds[i] if i < len(daily_winds) else wind_speed
        d_code = daily_codes[i] if i < len(daily_codes) else 0
        d_rain = daily_rains[i] if i < len(daily_rains) else 10

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
                "min": max(0, round(d_wind - 2, 1)),
                "max": round(d_wind + 2, 1),
                "display": f"{max(0, round(d_wind - 2, 1))}-{round(d_wind + 2, 1)} km/h",
            },
        })

    return {
        "date": str(datetime.date.today()),
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
            "uv_risk": "High" if uv_index and uv_index >= 6 else "Moderate",
            "visibility_km": visibility_km,
            "cloud_cover_percent": clouds,
            "sunrise_time": sunrise,
            "sunset_time": sunset,
        },
        "hourly_rain_chance_next_6h": hourly.get("precipitation_probability", [])[:6],
        "next_7_days_forecast": seven_day_forecast,
    }


# =========================================================
# 7. LIVE AQI & PREDICTION ENDPOINT
#    Live Data: WAQI (1st) -> Open-Meteo CAMS (2nd)
#    Prediction: Open-Meteo (1st) -> WAQI (2nd)
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

    # --- LIVE DATA: Priority 1 = WAQI, Priority 2 = Open-Meteo CAMS ---
    waqi = get_waqi_live_data(lat, lon)
    if waqi.get("available"):
        live_aqi = waqi["aqi"]
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
            live_aqi = cams["aqi"]
            live_pm25 = cams.get("pm25_ugm3")
            live_pm10 = cams.get("pm10_ugm3")
            source_used = "Open-Meteo CAMS"
            measurement_time = cams.get("measurement_time")
            category = get_indian_aqi_category(live_aqi)

    # --- PREDICTION: Priority 1 = Open-Meteo, Priority 2 = WAQI ---
    om_pred = get_open_meteo_aqi_prediction(lat, lon)
    if om_pred.get("available"):
        prediction_source = om_pred["source"]
        predicted_next_hour = om_pred["predicted_aqi_next_hour"]
        predicted_today = om_pred["predicted_aqi_today"]
        predicted_tomorrow = om_pred["predicted_aqi_tomorrow"]
    elif waqi.get("available"):
        prediction_source = "WAQI Persistence Fallback"
        predicted_next_hour = round(waqi["aqi"], 1)
        predicted_today = round(waqi["aqi"], 1)
        predicted_tomorrow = round(waqi["aqi"], 1)
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
        current_temp = weather.get("current", {}).get("temperature_2m", 28.0)
    except Exception:
        current_temp = 28.0

    years_ahead = target_year - datetime.date.today().year
    co2_factor = 1.25
    projected_temp = None

    if climate_ai:
        try:
            projected_temp = round(
                float(climate_ai.predict([[current_temp, lat, years_ahead, co2_factor]])[0]),
                2,
            )
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

    temp = float(current.get("temperature_2m") or 28.0)
    humidity = float(current.get("relative_humidity_2m") or 60.0)
    precip_now = float(current.get("precipitation") or 0.0)
    wind_speed = float(current.get("wind_speed_10m") or 8.0)
    pressure = float(current.get("surface_pressure") or 1013.0)

    rain_prob = float((daily.get("precipitation_probability_max") or [15])[0] or 15)
    daily_rain_mm = float((daily.get("precipitation_sum") or [0.0])[0] or 0.0)

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
        features = usgs_data.get("features", [])
        max_mag = 0.0

        for item in features:
            props = item.get("properties", {})
            mag = props.get("mag")
            if isinstance(mag, (int, float)):
                max_mag = max(max_mag, float(mag))
                recent_earthquakes.append({
                    "magnitude": round(float(mag), 1),
                    "place": props.get("place", "Nearby region"),
                    "time_utc": datetime.datetime.fromtimestamp(
                        props["time"] / 1000, tz=datetime.timezone.utc
                    ).isoformat() if props.get("time") else None,
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

    wind_speed = round(float(current.get("wind_speed_10m") or 5.5), 1)
    wind_dir = round(float(current.get("wind_direction_10m") or 140.0), 1)
    wind_gusts = round(float(current.get("wind_gusts_10m") or (wind_speed + 3.0)), 1)
    pressure = round(float(current.get("surface_pressure") or 1013.0), 1)

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
    }


# =========================================================
# 11. CALENDAR FORECAST ENGINE (YEAR + MONTH + DAY)
#     Weather & AQI Prediction: Open-Meteo (1st) -> WAQI (2nd)
# =========================================================

def get_safe_archive_date(target_date: datetime.date, reference_year: int) -> datetime.date:
    """Safely maps a target month/day to a past archive year (handles Feb 29)."""
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

    t_max = float((daily.get("temperature_2m_max") or [28.0])[0] or 28.0)
    t_min = float((daily.get("temperature_2m_min") or [22.0])[0] or 22.0)
    precip_mm = float((daily.get("precipitation_sum") or [0.0])[0] or 0.0)
    wind_max = float((daily.get("wind_speed_10m_max") or [8.0])[0] or 8.0)
    wmo_code = int((daily.get("weather_code") or [0])[0] or 0)

    if years_ahead > 0:
        warming_offset = round(years_ahead * 0.04, 2)
        if climate_ai:
            try:
                base_mean = (t_max + t_min) / 2
                predicted_mean = float(climate_ai.predict([[base_mean, lat, years_ahead, 1.25]])[0])
                warming_offset = round(predicted_mean - base_mean, 2)
            except Exception:
                pass
        t_max = round(t_max + warming_offset, 1)
        t_min = round(t_min + warming_offset, 1)

    t_mean = round((t_max + t_min) / 2, 1)

    hum_vals = [v for v in hourly.get("relative_humidity_2m", []) if v is not None]
    pres_vals = [v for v in hourly.get("surface_pressure", []) if v is not None]
    avg_humidity = round(sum(hum_vals) / len(hum_vals), 1) if hum_vals else 60.0
    avg_pressure = round(sum(pres_vals) / len(pres_vals), 1) if pres_vals else 1012.0

    rain_prob = min(95, max(10, int(precip_mm * 12))) if precip_mm > 0.2 else 10

    ai_pred = 0
    ai_verdict = "No Rain Expected"
    if rainfall_ai:
        try:
            ai_pred = int(rainfall_ai.predict([[t_mean, avg_humidity, wind_max, avg_pressure]])[0])
            ai_verdict = "Rain Expected" if ai_pred == 1 else "No Rain Expected"
        except Exception:
            pass

    condition, visual_trigger, corrected_rain = classify_weather_condition(
        wmo_code, avg_humidity, t_mean, rain_prob, ai_pred
    )

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
        pm25_vals = [float(v) for v in hourly.get("pm2_5", []) if v is not None]
        pm10_vals = [float(v) for v in hourly.get("pm10", []) if v is not None]

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

    # Priority 2 Fallback: WAQI
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
    """
    Accepts any year (1950-2100), month (1-12), and day (1-31).
    Example: /calendar-forecast?location=Kolkata&year=2027&month=7&day=15
    """
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

    weather = fetch_calendar_weather(lat, lon, target_date)
    aqi = fetch_calendar_aqi(lat, lon, target_date)

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
# 12. MASTER DASHBOARD SUMMARY
# =========================================================

@app.get("/dashboard-summary")
def get_complete_dashboard(location: str = "Kolkata"):
    weather_info = predict_live_rainfall(location)
    aqi_info = predict_live_aqi(location)
    climate_info = forecast_climate(location, 2050)
    satellite_info = get_satellite_map(location)
    wrf_info = get_wrf_model(location)

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
# 13. HEALTH CHECK
# =========================================================

@app.get("/")
def health_check():
    return {
        "status": "EVS Climate AI backend is running",
        "live_aqi_priority": ["WAQI", "Open-Meteo CAMS"],
        "aqi_prediction_priority": ["Open-Meteo Forecast", "WAQI Fallback"],
        "calendar_endpoint": "/calendar-forecast?location=Kolkata&year=2026&month=11&day=19",
        "waqi_token_configured": bool(WAQI_API_TOKEN),
    }