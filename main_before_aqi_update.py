
import os
import json
import math
import datetime
from pathlib import Path

import joblib
import pandas as pd
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="EVS Climate AI - Production Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 2. MODEL LOADING
# =========================================================

def load_model(filename):
    path = BASE_DIR / "models" / filename

    try:
        model = joblib.load(path)
        print(f"Loaded model: {filename}")
        return model
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        return None


# Existing models are preserved.
rainfall_ai = load_model("rainfall_ml_model.pkl")
aqi_ai = load_model("aqi_timeseries_model.pkl")
climate_ai = load_model("climate_rf_model.pkl")
disaster_ai = load_model("disaster_rf_model.pkl")
wrf_ai = load_model("wrf_rf_model.pkl")


# Load the new six-hour rainfall model separately.
rainfall_v2_ai = None
rainfall_v2_features = []
rainfall_v2_threshold = 0.34

try:
    experimental_dir = BASE_DIR / "models" / "experimental"

    rainfall_v2_ai = joblib.load(
        experimental_dir / "rainfall_6h_hgb_v2.joblib"
    )

    with open(
        experimental_dir / "rainfall_6h_features_v2.json",
        "r",
        encoding="utf-8"
    ) as f:
        rainfall_v2_features = json.load(f)

    with open(
        experimental_dir / "rainfall_6h_metrics_v2.json",
        "r",
        encoding="utf-8"
    ) as f:
        rainfall_v2_metrics = json.load(f)

    rainfall_v2_threshold = float(
        rainfall_v2_metrics["threshold"]
    )

    print("V2 rainfall model loaded successfully.")
    print("V2 threshold:", rainfall_v2_threshold)
    print("V2 features:", rainfall_v2_features)

except Exception as e:
    print("V2 rainfall model could not be loaded:", e)


# =========================================================
# 3. CITY NORMALIZATION
# =========================================================

SUPPORTED_CITIES = {
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "jaipur": "Jaipur",
    "kanpur": "Kanpur",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "nagpur": "Nagpur",
    "pune": "Pune",
}


def normalize_city(location):
    cleaned = location.strip().lower()
    return SUPPORTED_CITIES.get(cleaned)


# =========================================================
# 4. LOCATION AND WEATHER HELPERS
# =========================================================

def get_live_coordinates(location: str):
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={requests.utils.quote(location)}"
        "&count=1&format=json"
    )

    try:
        response = requests.get(geo_url, timeout=15)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Geocoding service unavailable: {str(e)}"
        )

    if not result.get("results"):
        raise HTTPException(
            status_code=404,
            detail=f"Location '{location}' not found."
        )

    place = result["results"][0]

    return (
        place["latitude"],
        place["longitude"],
        place.get("country", "Unknown"),
        place.get("name", location)
    )


def get_weather_data(lat, lon):
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,"
        "apparent_temperature,dew_point_2m,precipitation,"
        "weather_code,cloud_cover,surface_pressure,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "surface_pressure,wind_speed_10m,wind_gusts_10m,"
        "cloud_cover,visibility,dew_point_2m,precipitation,"
        "precipitation_probability"
        "&daily=weather_code,temperature_2m_max,"
        "temperature_2m_min,sunrise,sunset,uv_index_max,"
        "precipitation_probability_max,wind_speed_10m_max,"
        "wind_speed_10m_min"
        "&timezone=auto"
        "&forecast_days=7"
    )

    try:
        response = requests.get(weather_url, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Weather service unavailable: {str(e)}"
        )


def get_hourly_current_index(hourly, current):
    times = hourly.get("time", [])
    current_time = current.get("time")

    if not times or not current_time:
        return 0

    # Open-Meteo current.time is generally aligned to an hourly entry.
    target = current_time[:13]

    for i, timestamp in enumerate(times):
        if timestamp[:13] == target:
            return i

    return 0


def get_hourly_value(hourly, key, index, default=None):
    values = hourly.get(key, [])

    if index < len(values):
        value = values[index]
        if value is not None:
            return value

    return default


def degrees_to_compass(deg):
    if deg is None:
        return "N"

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(deg / 45) % 8
    return directions[ix]


# =========================================================
# 5. V2 RAINFALL FEATURE BUILDER
# =========================================================

def build_rainfall_v2_input(location, current, hourly):
    city = normalize_city(location)

    if city is None:
        return None, None

    if rainfall_v2_ai is None:
        return None, "V2 model is not loaded."

    if not rainfall_v2_features:
        return None, "V2 feature list is empty."

    index = get_hourly_current_index(hourly, current)

    # Use live current values wherever available.
    # Use matching hourly values for variables not in current data.

    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    pressure = current.get("surface_pressure")
    wind_speed = current.get("wind_speed_10m")
    wind_gust = current.get("wind_gusts_10m")
    cloud_cover = current.get("cloud_cover")
    dew_point = current.get("dew_point_2m")
    precipitation = current.get("precipitation")

    if temp is None:
        temp = get_hourly_value(hourly, "temperature_2m", index)

    if humidity is None:
        humidity = get_hourly_value(
            hourly, "relative_humidity_2m", index
        )

    if pressure is None:
        pressure = get_hourly_value(
            hourly, "surface_pressure", index
        )

    if wind_speed is None:
        wind_speed = get_hourly_value(
            hourly, "wind_speed_10m", index
        )

    if wind_gust is None:
        wind_gust = get_hourly_value(
            hourly, "wind_gusts_10m", index
        )

    if cloud_cover is None:
        cloud_cover = get_hourly_value(
            hourly, "cloud_cover", index
        )

    if dew_point is None:
        dew_point = get_hourly_value(
            hourly, "dew_point_2m", index
        )

    if precipitation is None:
        precipitation = get_hourly_value(
            hourly, "precipitation", index
        )

    visibility_m = get_hourly_value(
        hourly, "visibility", index
    )

    # Open-Meteo visibility is in metres.
    # Historical dataset visibility is represented in kilometres.
    visibility_km = (
        float(visibility_m) / 1000.0
        if visibility_m is not None
        else None
    )

    raw_time = current.get("time")

    if raw_time:
        try:
            current_datetime = datetime.datetime.fromisoformat(
                raw_time
            )
        except ValueError:
            current_datetime = datetime.datetime.now()
    else:
        current_datetime = datetime.datetime.now()

    hour = current_datetime.hour
    month = current_datetime.month

    # Cyclical time features, matching training preparation.
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    month_sin = math.sin(2 * math.pi * (month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12)

    raw_values = {
        "tempC": temp,
        "humidity": humidity,
        "pressure": pressure,
        "windspeedKmph": wind_speed,
        "WindGustKmph": wind_gust,
        "cloudcover": cloud_cover,
        "visibility": visibility_km,
        "DewPointC": dew_point,
        "precipMM": precipitation,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    # Add the one-hot city fields.
    for feature in rainfall_v2_features:
        if feature.startswith("city_"):
            city_name = feature.replace("city_", "", 1)
            raw_values[feature] = int(city_name == city)

    # Do not silently invent missing weather observations.
    missing = [
        key for key in [
            "tempC",
            "humidity",
            "pressure",
            "windspeedKmph",
            "WindGustKmph",
            "cloudcover",
            "visibility",
            "DewPointC",
            "precipMM",
        ]
        if raw_values.get(key) is None
    ]

    if missing:
        return None, (
            "Missing live weather variables: " + ", ".join(missing)
        )

    # Build a DataFrame in exactly the feature order used in training.
    row = {}

    for feature in rainfall_v2_features:
        if feature not in raw_values:
            return None, f"Unknown model feature: {feature}"

        row[feature] = raw_values[feature]

    input_df = pd.DataFrame(
        [row],
        columns=rainfall_v2_features
    )

    return input_df, None


def predict_rainfall_v2(location, current, hourly):
    city = normalize_city(location)

    if city is None:
        return {
            "available": False,
            "message": (
                "The six-hour ML model currently supports "
                "Bengaluru, Delhi, Hyderabad, Jaipur, Kanpur, "
                "Mumbai, Nagpur and Pune."
            )
        }

    if rainfall_v2_ai is None:
        return {
            "available": False,
            "message": "V2 rainfall model is not loaded."
        }

    input_df, error = build_rainfall_v2_input(
        location, current, hourly
    )

    if error:
        return {
            "available": False,
            "message": error
        }

    try:
        probability = float(
            rainfall_v2_ai.predict_proba(input_df)[0][1]
        )

        prediction = int(
            probability >= rainfall_v2_threshold
        )

        return {
            "available": True,
            "model": "HistGradientBoosting V2",
            "prediction_horizon": "Next 6 hours",
            "city": city,
            "rain_expected": bool(prediction),
            "prediction_label": (
                "Rain expected in the next 6 hours"
                if prediction
                else "No significant rain expected in the next 6 hours"
            ),
            "model_score_percent": round(probability * 100, 1),
            "decision_threshold": rainfall_v2_threshold,
            "threshold_note": (
                "This is a model score, not a calibrated "
                "real-world probability."
            )
        }

    except Exception as e:
        return {
            "available": False,
            "message": f"V2 prediction failed: {str(e)}"
        }


# =========================================================
# 6. WEATHER CLASSIFICATION
# =========================================================

def classify_weather_condition(
    wmo_code, humidity, temp, rain_chance, ai_rain_flag
):
    if wmo_code is None:
        wmo_code = 0

    if rain_chance == 0:
        if wmo_code in [2, 3]:
            rain_chance = max(rain_chance, 25)
        elif humidity > 70:
            rain_chance = max(rain_chance, 20)

    if (
        ai_rain_flag == 1
        or wmo_code >= 51
        or rain_chance >= 55
    ):
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
# 7. CPCB AQI CALCULATOR
# =========================================================

def calculate_indian_cpcb_aqi(
    pm25: float, pm10: float = 0.0
) -> float:

    if pm25 <= 30:
        i_pm25 = (50 / 30) * pm25
    elif pm25 <= 60:
        i_pm25 = 50 + (50 / 30) * (pm25 - 30)
    elif pm25 <= 90:
        i_pm25 = 100 + (50 / 30) * (pm25 - 60)
    elif pm25 <= 120:
        i_pm25 = 200 + (100 / 30) * (pm25 - 90)
    elif pm25 <= 250:
        i_pm25 = 300 + (100 / 130) * (pm25 - 120)
    else:
        i_pm25 = 400 + (100 / 130) * (pm25 - 250)

    if pm10 <= 50:
        i_pm10 = pm10
    elif pm10 <= 100:
        i_pm10 = 50 + (pm10 - 50)
    elif pm10 <= 250:
        i_pm10 = 100 + (100 / 150) * (pm10 - 100)
    elif pm10 <= 350:
        i_pm10 = 200 + (pm10 - 250)
    elif pm10 <= 430:
        i_pm10 = 300 + (100 / 80) * (pm10 - 350)
    else:
        i_pm10 = 400 + (100 / 70) * (pm10 - 430)

    return round(max(i_pm25, i_pm10), 1)


# =========================================================
# 8. LIVE WEATHER AND RAINFALL ENDPOINT
# =========================================================

@app.get("/predict-rainfall")
def predict_live_rainfall(location: str = "Kolkata"):

    lat, lon, country, resolved_name = get_live_coordinates(
        location
    )

    data = get_weather_data(lat, lon)

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

    sunrise = (
        sunrise_raw.split("T")[-1]
        if "T" in sunrise_raw else sunrise_raw
    )

    sunset = (
        sunset_raw.split("T")[-1]
        if "T" in sunset_raw else sunset_raw
    )

    uv_index = (daily.get("uv_index_max") or [5.0])[0]

    raw_rain_chance = (
        daily.get("precipitation_probability_max") or [10]
    )[0]

    hourly_index = get_hourly_current_index(hourly, current)

    visibility_m = get_hourly_value(
        hourly, "visibility", hourly_index, 10000
    )

    visibility_km = round(visibility_m / 1000, 1)

    # New V2 prediction.
    v2_result = predict_rainfall_v2(
        location, current, hourly
    )

    ai_pred = (
        1 if v2_result.get("available")
        and v2_result.get("rain_expected")
        else 0
    )

    if v2_result.get("available"):
        ai_verdict = v2_result["prediction_label"]
    else:
        ai_verdict = "V2 prediction unavailable"

    # Weather classification continues to use live weather.
    weather_label, visual_trigger, corrected_rain_chance = (
        classify_weather_condition(
            wmo_code,
            humidity,
            temp,
            raw_rain_chance,
            ai_pred
        )
    )

    wind_min = max(0.0, round(wind_speed - 2.0, 1))
    wind_max = round(wind_speed + 2.0, 1)

    hum_min = max(0, int(humidity - 5))
    hum_max = min(100, int(humidity + 5))

    rain_min = max(0, int(corrected_rain_chance - 5))
    rain_max = min(100, int(corrected_rain_chance + 5))

    # Seven-day forecast.
    seven_day_forecast = []

    days_time = daily.get("time", [])
    daily_codes = daily.get("weather_code", [])
    daily_rain_probs = daily.get(
        "precipitation_probability_max", []
    )
    daily_maxs = daily.get("temperature_2m_max", [])
    daily_mins = daily.get("temperature_2m_min", [])
    daily_winds = daily.get("wind_speed_10m_max", [])

    for i in range(min(7, len(days_time))):

        day_date = datetime.datetime.strptime(
            days_time[i], "%Y-%m-%d"
        )

        d_code = (
            daily_codes[i] if i < len(daily_codes) else 0
        )

        d_raw_rain = (
            daily_rain_probs[i]
            if i < len(daily_rain_probs) else 10
        )

        d_max = (
            daily_maxs[i] if i < len(daily_maxs) else temp
        )

        d_min = (
            daily_mins[i] if i < len(daily_mins) else temp
        )

        d_wind = (
            daily_winds[i]
            if i < len(daily_winds) else wind_speed
        )

        d_cond, _, d_rain_chance = classify_weather_condition(
            d_code, humidity, d_max, d_raw_rain, 0
        )

        seven_day_forecast.append({
            "day_name": day_date.strftime("%a"),
            "date": days_time[i],
            "condition": d_cond,

            "temp_range_c": {
                "min": round(d_min, 1),
                "max": round(d_max, 1),
                "display": (
                    f"{round(d_min)}° - {round(d_max)}°C"
                )
            },

            "rain_chance_range": {
                "min": max(0, d_rain_chance - 5),
                "max": min(100, d_rain_chance + 5),
                "display": (
                    f"{max(0, d_rain_chance - 5)}-"
                    f"{min(100, d_rain_chance + 5)}%"
                )
            },

            "humidity_range": {
                "min": hum_min,
                "max": hum_max,
                "display": f"{hum_min}-{hum_max}%"
            },

            "wind_speed_range_kmh": {
                "min": max(0.0, round(d_wind - 2, 1)),
                "max": round(d_wind + 2, 1),
                "display": (
                    f"{max(0.0, round(d_wind - 2, 1))}-"
                    f"{round(d_wind + 2, 1)} km/h"
                )
            }
        })

    return {
        "date": str(datetime.date.today()),
        "region": location.upper(),
        "resolved_location": resolved_name,
        "country": country,

        "coordinates": {
            "latitude": lat,
            "longitude": lon
        },

        "weather_condition": weather_label,
        "frontend_visual_trigger": visual_trigger,

        "chance_of_rain_range": {
            "min": rain_min,
            "max": rain_max,
            "display": f"{rain_min}-{rain_max}%"
        },

        "ai_rainfall_verdict": ai_verdict,

        "rainfall_ml_prediction": v2_result,

        "taskbar_metrics": {
            "temperature_c": temp,
            "real_feel_c": feels_like,

            "temp_range": {
                "min": round(temp_low, 1),
                "max": round(temp_high, 1),
                "display": (
                    f"{round(temp_low)}° - {round(temp_high)}°C"
                )
            },

            "humidity_range": {
                "min": hum_min,
                "max": hum_max,
                "display": f"{hum_min}-{hum_max}%"
            },

            "wind_speed_range": {
                "min": wind_min,
                "max": wind_max,
                "display": f"{wind_min}-{wind_max} km/h"
            },

            "wind_direction_compass": degrees_to_compass(
                wind_dir_deg
            ),

            "pressure_mb": pressure,
            "uv_index": uv_index,

            "uv_risk": (
                "High" if uv_index and uv_index >= 6
                else "Moderate"
            ),

            "visibility_km": visibility_km,
            "cloud_cover_percent": clouds,
            "sunrise_time": sunrise,
            "sunset_time": sunset
        },

        "hourly_rain_chance_next_6h": (
            hourly.get("precipitation_probability", [0] * 6)[:6]
        ),

        "next_7_days_forecast": seven_day_forecast
    }


# =========================================================
# 9. AIR QUALITY ENDPOINT
# =========================================================

@app.get("/predict-aqi")
def predict_live_aqi(location: str = "Kolkata"):

    lat, lon, country, resolved_name = get_live_coordinates(
        location
    )

    WAQI_TOKEN = os.getenv("WAQI_TOKEN", "").strip(" '\"")
    OWM_API_KEY = os.getenv("OWM_API_KEY", "").strip(" '\"")

    live_pm25, live_pm10 = 0.0, 0.0
    source_used = ""

    # Tier 1: WAQI.
    if WAQI_TOKEN:
        waqi_url = (
            f"https://api.waqi.info/feed/geo:{lat};{lon}/"
            f"?token={WAQI_TOKEN}"
        )

        try:
            res = requests.get(waqi_url, timeout=5).json()

            if res.get("status") == "ok":
                waqi_data = res.get("data", {})

                pm25_val = float(
                    waqi_data.get("iaqi", {})
                    .get("pm25", {}).get("v", 0.0)
                )

                pm10_val = float(
                    waqi_data.get("iaqi", {})
                    .get("pm10", {}).get("v", 0.0)
                )

                if pm25_val > 0 or pm10_val > 0:
                    live_pm25 = pm25_val
                    live_pm10 = pm10_val
                    source_used = "WAQI"

        except Exception:
            pass

    # Tier 2: OpenWeatherMap.
    if (
        not source_used
        and OWM_API_KEY
        and OWM_API_KEY != "PASTE_YOUR_NEW_OWM_KEY_HERE"
    ):
        owm_url = (
            "https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
        )

        try:
            res = requests.get(owm_url, timeout=5).json()

            if res.get("list"):
                components = res["list"][0].get(
                    "components", {}
                )

                live_pm25 = float(
                    components.get("pm2_5", 0.0)
                )

                live_pm10 = float(
                    components.get("pm10", 0.0)
                )

                source_used = "OpenWeatherMap"

        except Exception:
            pass

    # Tier 3: Open-Meteo CAMS.
    if not source_used:
        om_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}"
            "&current=pm2_5,pm10"
        )

        try:
            res = requests.get(om_url, timeout=8).json()

            current_air = res.get("current", {})

            live_pm25 = float(
                current_air.get("pm2_5", 25.0)
            )

            live_pm10 = float(
                current_air.get("pm10", 35.0)
            )

            source_used = "Open-Meteo CAMS"

        except Exception:
            live_pm25, live_pm10 = 35.0, 45.0
            source_used = "Fallback estimate"

    live_indian_aqi = calculate_indian_cpcb_aqi(
        live_pm25, live_pm10
    )

    # Seven-day hourly history.
    trend_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=pm2_5,pm10&past_days=7"
    )

    daily_aqi_trend = []

    try:
        trend_response = requests.get(
            trend_url, timeout=10
        ).json()

        hourly_data = trend_response.get("hourly", {})

        hourly_pm25 = hourly_data.get("pm2_5", [])
        hourly_pm10 = hourly_data.get(
            "pm10", [0] * len(hourly_pm25)
        )

        trend_times = hourly_data.get("time", [])

        # Group by actual calendar date, not arbitrary 24-hour slices.
        daily_groups = {}

        for i, timestamp in enumerate(trend_times):
            date_key = timestamp[:10]

            pm25_value = (
                hourly_pm25[i]
                if i < len(hourly_pm25) else None
            )

            pm10_value = (
                hourly_pm10[i]
                if i < len(hourly_pm10) else None
            )

            if pm25_value is None or pm10_value is None:
                continue

            daily_groups.setdefault(
                date_key, {"pm25": [], "pm10": []}
            )

            daily_groups[date_key]["pm25"].append(
                float(pm25_value)
            )

            daily_groups[date_key]["pm10"].append(
                float(pm10_value)
            )

        for date_key in sorted(daily_groups)[-7:]:
            group = daily_groups[date_key]

            avg25 = (
                sum(group["pm25"]) / len(group["pm25"])
            )

            avg10 = (
                sum(group["pm10"]) / len(group["pm10"])
            )

            daily_aqi_trend.append(
                calculate_indian_cpcb_aqi(avg25, avg10)
            )

    except Exception:
        daily_aqi_trend = []

    if not daily_aqi_trend:
        daily_aqi_trend = [live_indian_aqi] * 7

    predicted_aqi = live_indian_aqi

    if aqi_ai:
        try:
            weather_url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=relative_humidity_2m,wind_speed_10m"
            )

            weather_res = requests.get(
                weather_url, timeout=8
            ).json()

            current_weather = weather_res.get(
                "current", {}
            )

            cur_humidity = current_weather.get(
                "relative_humidity_2m", 50.0
            )

            cur_wind = current_weather.get(
                "wind_speed_10m", 5.0
            )

            features = [[
                daily_aqi_trend[0],
                daily_aqi_trend[2],
                daily_aqi_trend[6],
                cur_humidity,
                cur_wind
            ]]

            raw_pred = float(
                aqi_ai.predict(features)[0]
            )

            predicted_aqi = round(
                (raw_pred * 0.2)
                + (live_indian_aqi * 0.8),
                1
            )

        except Exception:
            predicted_aqi = live_indian_aqi

    if live_indian_aqi > 400:
        category, visual_effect = (
            "Severe", "HEAVY_SMOG_EFFECT"
        )
    elif live_indian_aqi > 300:
        category, visual_effect = (
            "Very Poor", "HEAVY_SMOG_EFFECT"
        )
    elif live_indian_aqi > 200:
        category, visual_effect = (
            "Poor", "LIGHT_HAZE_EFFECT"
        )
    elif live_indian_aqi > 100:
        category, visual_effect = (
            "Moderate", "LIGHT_HAZE_EFFECT"
        )
    elif live_indian_aqi > 50:
        category, visual_effect = (
            "Satisfactory", "CLEAN_AIR_EFFECT"
        )
    else:
        category, visual_effect = (
            "Good", "CLEAN_AIR_EFFECT"
        )

    range_spread = round(live_indian_aqi * 0.10)

    return {
        "date": str(datetime.date.today()),
        "region": location.upper(),
        "country": country,
        "data_source": source_used,

        "live_current_aqi": live_indian_aqi,

        "aqi_range": {
            "min": max(
                0, round(live_indian_aqi - range_spread)
            ),
            "max": round(
                live_indian_aqi + range_spread
            ),
            "display": (
                f"{max(0, round(live_indian_aqi - range_spread))}-"
                f"{round(live_indian_aqi + range_spread)}"
            )
        },

        "live_pm25_ugm3": live_pm25,
        "live_pm10_ugm3": live_pm10,
        "live_past_7_days_trend": daily_aqi_trend,
        "ai_predicted_aqi_today": predicted_aqi,
        "pollution_category": category,
        "frontend_aqi_trigger": visual_effect
    }


# =========================================================
# 10. CLIMATE PROJECTION ENDPOINT
# =========================================================

@app.get("/forecast-climate")
def forecast_climate(
    location: str = "Kolkata",
    target_year: int = 2050
):

    lat, lon, country, resolved_name = get_live_coordinates(
        location
    )

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m"
    )

    try:
        weather_res = requests.get(
            weather_url, timeout=10
        ).json()

        current_temp = weather_res.get(
            "current", {}
        ).get("temperature_2m", 28.0)

    except Exception:
        current_temp = 28.0

    years_ahead = target_year - datetime.date.today().year
    co2_factor = 1.25

    if climate_ai:
        try:
            features = [[
                current_temp, lat, years_ahead, co2_factor
            ]]

            projected_temp = round(
                float(climate_ai.predict(features)[0]), 2
            )

        except Exception:
            projected_temp = round(
                current_temp + (years_ahead * 0.04), 2
            )

    else:
        projected_temp = round(
            current_temp + (years_ahead * 0.04), 2
        )

    return {
        "region": location.upper(),
        "country": country,
        "baseline_year": datetime.date.today().year,
        "target_year": target_year,
        "current_temperature": current_temp,
        "projected_temperature": projected_temp,

        "temperature_range": {
            "min": round(projected_temp - 1.2, 1),
            "max": round(projected_temp + 1.5, 1),
            "display": (
                f"{round(projected_temp - 1.2, 1)}° - "
                f"{round(projected_temp + 1.5, 1)}°C"
            )
        },

        "impact_risk": (
            "Severe Heat Stress & Drought Vulnerability"
            if projected_temp > 34
            else "Moderate Climate Adaptation Needed"
        )
    }


# =========================================================
# 11. DISASTER RADAR ENDPOINT
# =========================================================

@app.get("/get-satellite-map")
def get_satellite_map(location: str = "Kolkata"):

    lat, lon, country, resolved_name = get_live_coordinates(
        location
    )

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=surface_pressure,wind_speed_10m"
        "&daily=precipitation_probability_max"
    )

    try:
        weather_data = requests.get(
            weather_url, timeout=10
        ).json()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Weather data unavailable for disaster assessment."
        )

    rain_prob = (
        weather_data.get("daily", {})
        .get("precipitation_probability_max") or [15]
    )[0]

    current_weather = weather_data.get("current", {})

    wind_speed = current_weather.get(
        "wind_speed_10m", 8.0
    )

    pressure = current_weather.get(
        "surface_pressure", 1013.0
    )

    pressure_drop = max(0.0, 1013.0 - pressure)

    seismic_zone_flag = (
        1 if (
            (lat > 23.0 and lon > 88.0)
            or lat > 28.0
        ) else 0
    )

    disaster_probabilities = {
        "Flood Risk": 5.0,
        "Earthquake Tectonic Warning": 2.0,
        "Wildfire Thermal Alert": 3.0
    }

    primary_threat = "Normal (No Disaster Imminent)"

    if disaster_ai:
        try:
            features = [[
                rain_prob,
                wind_speed,
                pressure_drop,
                seismic_zone_flag
            ]]

            probabilities = disaster_ai.predict_proba(
                features
            )[0]

            classes = disaster_ai.classes_

            for idx, cls_code in enumerate(classes):
                prob_val = round(
                    float(probabilities[idx]) * 100, 1
                )

                if cls_code == 1:
                    disaster_probabilities["Flood Risk"] = prob_val
                elif cls_code == 2:
                    disaster_probabilities[
                        "Earthquake Tectonic Warning"
                    ] = prob_val
                elif cls_code == 3:
                    disaster_probabilities[
                        "Wildfire Thermal Alert"
                    ] = prob_val

            predicted_code = disaster_ai.predict(
                features
            )[0]

            if predicted_code == 1:
                primary_threat = (
                    "High Flood / Cyclone Storm Warning"
                )
            elif predicted_code == 2:
                primary_threat = (
                    "Seismic Activity / Minor Tectonic Alert"
                )
            elif predicted_code == 3:
                primary_threat = (
                    "Thermal Anomaly / Wildfire Risk Alert"
                )

        except Exception:
            pass

    return {
        "region": location.upper(),
        "country": country,

        "coordinates": {
            "latitude": lat,
            "longitude": lon
        },

        "satellite_map_link": (
            f"https://maps.google.com/?q={lat},{lon}&t=k"
        ),

        "disaster_risk_assessment": {
            "primary_threat_status": primary_threat,
            "probability_percentages": disaster_probabilities,
            "thermal_anomaly_detected": (
                disaster_probabilities[
                    "Wildfire Thermal Alert"
                ] > 40
            )
        },

        "forest_fire_scan": {
            "status": "Active Grid Radar",
            "risk_level": (
                "Normal"
                if disaster_probabilities[
                    "Wildfire Thermal Alert"
                ] < 40
                else "Elevated"
            )
        }
    }


# =========================================================
# 12. WRF ENDPOINT
# =========================================================

@app.get("/wrf-model-data")
def get_wrf_model(location: str = "Kolkata"):

    lat, lon, country, resolved_name = get_live_coordinates(
        location
    )

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,surface_pressure"
    )

    try:
        w_data = requests.get(
            weather_url, timeout=10
        ).json().get("current", {})
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Weather data unavailable for WRF endpoint."
        )

    base_temp = w_data.get("temperature_2m", 25.0)
    pressure = w_data.get("surface_pressure", 1013.0)

    sim_wind_speed = 5.5
    sim_wind_direction = 140.0

    if wrf_ai:
        try:
            features = [[
                lat, lon, base_temp, pressure
            ]]

            prediction = wrf_ai.predict(features)[0]

            sim_wind_speed = round(
                float(prediction[0]), 1
            )

            sim_wind_direction = round(
                float(prediction[1]), 1
            )

        except Exception:
            pass

    return {
        "region": location.upper(),
        "country": country,

        "numerical_model": (
            "WRF / ICON Seamless Model (Random Forest Regressor)"
        ),

        "simulated_atmospheric_metrics": {
            "wind_speed_kmh": sim_wind_speed,

            "wind_speed_range": {
                "min": max(0.0, sim_wind_speed - 1.5),
                "max": sim_wind_speed + 1.5,
                "display": (
                    f"{max(0.0, sim_wind_speed - 1.5)}-"
                    f"{sim_wind_speed + 1.5} km/h"
                )
            },

            "wind_direction_degrees": sim_wind_direction,

            "wind_direction_compass": degrees_to_compass(
                sim_wind_direction
            ),

            "atmospheric_stability_index": (
                "Stable Laminar Flow"
                if sim_wind_speed < 8
                else "Turbulent Gust Warning"
            )
        }
    }


# =========================================================
# 13. MASTER DASHBOARD SUMMARY
# =========================================================

@app.get("/dashboard-summary")
def get_complete_dashboard(location: str = "Kolkata"):

    weather_info = predict_live_rainfall(location)
    aqi_info = predict_live_aqi(location)
    climate_info = forecast_climate(location, 2050)
    sat_info = get_satellite_map(location)
    wrf_info = get_wrf_model(location)

    return {
        "location_header": (
            f"{weather_info['region']}, "
            f"{weather_info['country']}"
        ),

        "coordinates": weather_info["coordinates"],

        "weather_condition": weather_info["weather_condition"],

        "visual_triggers": {
            "weather_effect": weather_info[
                "frontend_visual_trigger"
            ],
            "aqi_effect": aqi_info[
                "frontend_aqi_trigger"
            ]
        },

        "taskbar_cards": weather_info["taskbar_metrics"],

        "rain_analysis": {
            "ai_verdict": weather_info[
                "ai_rainfall_verdict"
            ],

            "rainfall_ml_prediction": weather_info[
                "rainfall_ml_prediction"
            ],

            "chance_of_range": weather_info[
                "chance_of_rain_range"
            ],

            "humidity_range": weather_info[
                "taskbar_metrics"
            ]["humidity_range"],

            "wind_speed_range": weather_info[
                "taskbar_metrics"
            ]["wind_speed_range"],

            "hourly_rain_bars": weather_info[
                "hourly_rain_chance_next_6h"
            ]
        },

        "air_quality": {
            "live_current_aqi": aqi_info[
                "live_current_aqi"
            ],

            "aqi_range": aqi_info["aqi_range"],

            "ai_predicted_aqi": aqi_info[
                "ai_predicted_aqi_today"
            ],

            "status": aqi_info["pollution_category"],

            "past_7_days_trend": aqi_info.get(
                "live_past_7_days_trend", []
            )
        },

        "weekly_strip": weather_info[
            "next_7_days_forecast"
        ],

        "climate_2050": climate_info,
        "satellite_radar": sat_info,
        "wrf_simulation": wrf_info
    }


# =========================================================
# 14. HEALTH CHECK
# =========================================================

@app.get("/")
def home():
    return {
        "message": "EVS Climate AI API is running.",
        "models": {
            "old_rainfall_model_loaded": rainfall_ai is not None,
            "rainfall_v2_loaded": rainfall_v2_ai is not None,
            "aqi_model_loaded": aqi_ai is not None,
            "climate_model_loaded": climate_ai is not None,
            "disaster_model_loaded": disaster_ai is not None,
            "wrf_model_loaded": wrf_ai is not None
        }
    }
