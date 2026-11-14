from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests
import datetime

app = FastAPI(title="EVS Climate AI - Full Random Forest Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all trained Random Forest models
try:
    rainfall_ai = joblib.load("models/rainfall_ml_model.pkl")
except:
    rainfall_ai = None

try:
    aqi_ai = joblib.load("models/aqi_timeseries_model.pkl")
except:
    aqi_ai = None

try:
    climate_ai = joblib.load("models/climate_rf_model.pkl")
except:
    climate_ai = None

try:
    disaster_ai = joblib.load("models/disaster_rf_model.pkl")
except:
    disaster_ai = None


def get_live_coordinates(location: str):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&format=json"
    response = requests.get(geo_url).json()
    if "results" not in response:
        raise HTTPException(status_code=404, detail=f"Location '{location}' not found.")
    place = response["results"][0]
    return place["latitude"], place["longitude"], place.get("country", "")


def degrees_to_compass(deg):
    if deg is None:
        return "N"
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(deg / 45) % 8
    return directions[ix]


def classify_weather_condition(wmo_code, humidity, temp, rain_chance, ai_rain_flag):
    if wmo_code is None:
        wmo_code = 0
    if rain_chance == 0:
        if wmo_code in [2, 3]:
            rain_chance = max(rain_chance, 25)
        elif humidity > 70:
            rain_chance = max(rain_chance, 20)

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
    else:
        return "Clear", "SUN_RAYS_EFFECT", rain_chance


def calculate_indian_cpcb_aqi(pm25: float, pm10: float = 0.0) -> float:
    if pm25 <= 30:
        aqi_pm25 = pm25 * (50 / 30)
    elif pm25 <= 60:
        aqi_pm25 = 50 + (pm25 - 30) * (50 / 30)
    elif pm25 <= 90:
        aqi_pm25 = 100 + (pm25 - 60) * (100 / 30)
    elif pm25 <= 120:
        aqi_pm25 = 200 + (pm25 - 90) * (100 / 30)
    elif pm25 <= 250:
        aqi_pm25 = 300 + (pm25 - 120) * (100 / 130)
    else:
        aqi_pm25 = 400 + (pm25 - 250) * (100 / 130)

    if pm10 <= 50:
        aqi_pm10 = pm10
    elif pm10 <= 100:
        aqi_pm10 = pm10
    elif pm10 <= 250:
        aqi_pm10 = 100 + (pm10 - 100) * (100 / 150)
    elif pm10 <= 350:
        aqi_pm10 = 200 + (pm10 - 250)
    elif pm10 <= 430:
        aqi_pm10 = 300 + (pm10 - 350) * (100 / 80)
    else:
        aqi_pm10 = 400 + (pm10 - 430) * (100 / 80)

    base_aqi = max(aqi_pm25, aqi_pm10)
    if base_aqi > 100:
        base_aqi = base_aqi * 1.85
    return round(min(base_aqi, 495.0), 1)


# ---------------------------------------------------------
# 1. LIVE WEATHER, TASKBAR METRICS & RAINFALL AI ENDPOINT
# ---------------------------------------------------------
@app.get("/predict-rainfall")
def predict_live_rainfall(location: str = "Kolkata"):
    lat, lon, country = get_live_coordinates(location)
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,surface_pressure,wind_speed_10m,wind_direction_10m"
        f"&hourly=visibility,uv_index,precipitation_probability,wind_speed_10m,relative_humidity_2m"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset,uv_index_max,precipitation_probability_max,wind_speed_10m_max,wind_speed_10m_min"
        f"&timezone=auto"
    )
    data = requests.get(weather_url).json()
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
    sunrise = sunrise_raw.split("T")[-1] if "T" in sunrise_raw else sunrise_raw
    sunset = sunset_raw.split("T")[-1] if "T" in sunset_raw else sunset_raw
    uv_index = (daily.get("uv_index_max") or [5.0])[0]
    raw_rain_chance = (daily.get("precipitation_probability_max") or [10])[0]

    vis_list = hourly.get("visibility", [10000])
    visibility_km = round((vis_list[0] if vis_list and vis_list[0] else 10000) / 1000, 1)

    if rainfall_ai:
        ai_pred = int(rainfall_ai.predict([[temp, humidity, wind_speed, pressure]])[0])
        ai_verdict = "Rain Expected Today" if ai_pred == 1 else "No Rain Expected Today"
    else:
        ai_pred = 0
        ai_verdict = "Model missing"

    weather_label, visual_trigger, corrected_rain_chance = classify_weather_condition(
        wmo_code, humidity, temp, raw_rain_chance, ai_pred
    )

    wind_min = max(0.0, round(wind_speed - 2.0, 1))
    wind_max = round(wind_speed + 2.0, 1)
    hum_min = max(0, int(humidity - 5))
    hum_max = min(100, int(humidity + 5))
    rain_min = max(0, int(corrected_rain_chance - 5))
    rain_max = min(100, int(corrected_rain_chance + 5))

    seven_day_forecast = []
    days_time = daily.get("time", [])
    daily_codes = daily.get("weather_code", [0] * 7)
    daily_rain_probs = daily.get("precipitation_probability_max", [0] * 7)
    daily_maxs = daily.get("temperature_2m_max", [temp] * 7)
    daily_mins = daily.get("temperature_2m_min", [temp] * 7)
    daily_winds = daily.get("wind_speed_10m_max", [wind_speed] * 7)

    for i in range(min(7, len(days_time))):
        day_date = datetime.datetime.strptime(days_time[i], "%Y-%m-%d")
        d_code = daily_codes[i] if i < len(daily_codes) else 0
        d_raw_rain = daily_rain_probs[i] if i < len(daily_rain_probs) else 10
        d_max = daily_maxs[i] if i < len(daily_maxs) else temp
        d_min = daily_mins[i] if i < len(daily_mins) else temp
        d_wind = daily_winds[i] if i < len(daily_winds) else wind_speed

        d_cond, _, d_rain_chance = classify_weather_condition(d_code, humidity, d_max, d_raw_rain, 0)
        
        seven_day_forecast.append({
            "day_name": day_date.strftime("%a"),
            "date": days_time[i],
            "condition": d_cond,
            "temp_range_c": {"min": round(d_min, 1), "max": round(d_max, 1), "display": f"{round(d_min)}° - {round(d_max)}°C"},
            "rain_chance_range": {"min": max(0, d_rain_chance-5), "max": min(100, d_rain_chance+5), "display": f"{max(0, d_rain_chance-5)}-{min(100, d_rain_chance+5)}%"},
            "humidity_range": {"min": hum_min, "max": hum_max, "display": f"{hum_min}-{hum_max}%"},
            "wind_speed_range_kmh": {"min": max(0.0, round(d_wind-2, 1)), "max": round(d_wind+2, 1), "display": f"{max(0.0, round(d_wind-2, 1))}-{round(d_wind+2, 1)} km/h"}
        })

    return {
        "date": str(datetime.date.today()),
        "region": location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "weather_condition": weather_label,
        "frontend_visual_trigger": visual_trigger,
        "chance_of_rain_range": {"min": rain_min, "max": rain_max, "display": f"{rain_min}-{rain_max}%"},
        "ai_rainfall_verdict": ai_verdict,
        "taskbar_metrics": {
            "temperature_c": temp,
            "real_feel_c": feels_like,
            "temp_range": {"min": round(temp_low, 1), "max": round(temp_high, 1), "display": f"{round(temp_low)}° - {round(temp_high)}°C"},
            "humidity_range": {"min": hum_min, "max": hum_max, "display": f"{hum_min}-{hum_max}%"},
            "wind_speed_range": {"min": wind_min, "max": wind_max, "display": f"{wind_min}-{wind_max} km/h"},
            "wind_direction_compass": degrees_to_compass(wind_dir_deg),
            "pressure_mb": pressure,
            "uv_index": uv_index,
            "uv_risk": "High" if uv_index and uv_index >= 6 else "Moderate",
            "visibility_km": visibility_km,
            "cloud_cover_percent": clouds,
            "sunrise_time": sunrise,
            "sunset_time": sunset
        },
        "hourly_rain_chance_next_6h": hourly.get("precipitation_probability", [0] * 6)[:6],
        "next_7_days_forecast": seven_day_forecast
    }


# ---------------------------------------------------------
# 2. RANDOM FOREST AQI PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.get("/predict-aqi")
def predict_live_aqi(location: str = "Delhi"):
    lat, lon, _ = get_live_coordinates(location)
    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
        f"&current=pm2_5,pm10,us_aqi&past_days=7&hourly=pm2_5,pm10,relative_humidity_2m,wind_speed_10m"
    )
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m,wind_speed_10m"
    
    aqi_response = requests.get(aqi_url).json()
    weather_response = requests.get(weather_url).json()

    if "hourly" not in aqi_response or "pm2_5" not in aqi_response["hourly"]:
        raise HTTPException(status_code=500, detail="Could not fetch live regional air quality.")

    current_data = aqi_response.get("current", {})
    live_pm25 = current_data.get("pm2_5") or 60.0
    live_pm10 = current_data.get("pm10") or 120.0
    live_indian_aqi = calculate_indian_cpcb_aqi(live_pm25, live_pm10)

    hourly_pm25 = aqi_response["hourly"]["pm2_5"]
    hourly_pm10 = aqi_response["hourly"].get("pm10", [0] * len(hourly_pm25))

    daily_aqi_trend = []
    for i in range(0, 7):
        day_pm25 = [val for val in hourly_pm25[i * 24 : (i + 1) * 24] if val is not None]
        day_pm10 = [val for val in hourly_pm10[i * 24 : (i + 1) * 24] if val is not None]
        avg_25 = sum(day_pm25) / len(day_pm25) if day_pm25 else live_pm25
        avg_10 = sum(day_pm10) / len(day_pm10) if day_pm10 else live_pm10
        daily_aqi_trend.append(calculate_indian_cpcb_aqi(avg_25, avg_10))

    daily_aqi_trend[-1] = live_indian_aqi

    current_weather = weather_response.get("current", {})
    cur_humidity = current_weather.get("relative_humidity_2m", 50.0)
    cur_wind = current_weather.get("wind_speed_10m", 5.0)

    features = [[daily_aqi_trend[0], daily_aqi_trend[2], daily_aqi_trend[6], cur_humidity, cur_wind]]

    if aqi_ai:
        raw_pred = float(aqi_ai.predict(features)[0])
        predicted_aqi = round((raw_pred * 0.4) + (live_indian_aqi * 0.6), 1)
    else:
        predicted_aqi = live_indian_aqi

    if live_indian_aqi > 400:
        category, visual_effect = "Severe", "HEAVY_SMOG_EFFECT"
    elif live_indian_aqi > 300:
        category, visual_effect = "Very Poor", "HEAVY_SMOG_EFFECT"
    elif live_indian_aqi > 200:
        category, visual_effect = "Poor", "LIGHT_HAZE_EFFECT"
    elif live_indian_aqi > 100:
        category, visual_effect = "Moderate", "LIGHT_HAZE_EFFECT"
    else:
        category, visual_effect = "Satisfactory / Good", "CLEAN_AIR_EFFECT"

    return {
        "date": str(datetime.date.today()),
        "region": location.upper(),
        "live_current_aqi": live_indian_aqi,
        "ai_predicted_aqi_today": predicted_aqi,
        "pollution_category": category,
        "frontend_aqi_trigger": visual_effect
    }


# ---------------------------------------------------------
# 3. RANDOM FOREST CLIMATE CHANGE PROJECTION ENDPOINT
# ---------------------------------------------------------
@app.get("/forecast-climate")
def forecast_climate(location: str = "Kolkata", target_year: int = 2050):
    lat, lon, country = get_live_coordinates(location)
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
    current_temp = requests.get(weather_url).json().get("current", {}).get("temperature_2m", 28.0)

    years_ahead = target_year - datetime.date.today().year
    co2_factor = 1.25 # Escalation coefficient

    if climate_ai:
        features = [[current_temp, lat, years_ahead, co2_factor]]
        projected_temp = round(float(climate_ai.predict(features)[0]), 2)
    else:
        projected_temp = round(current_temp + (years_ahead * 0.04), 2)

    return {
        "region": location.upper(),
        "country": country,
        "baseline_year": datetime.date.today().year,
        "target_year": target_year,
        "current_temperature": current_temp,
        "projected_temperature": projected_temp,
        "temperature_range": {"min": round(projected_temp - 1.2, 1), "max": round(projected_temp + 1.5, 1), "display": f"{round(projected_temp - 1.2, 1)}° - {round(projected_temp + 1.5, 1)}°C"},
        "impact_risk": "Severe Heat Stress & Drought Vulnerability" if projected_temp > 34 else "Moderate Climate Adaptation Needed"
    }


# ---------------------------------------------------------
# 4. RANDOM FOREST SATELLITE DISASTER RISK RADAR
# ---------------------------------------------------------
@app.get("/get-satellite-map")
def get_satellite_map(location: str = "Kolkata"):
    lat, lon, country = get_live_coordinates(location)
    
    # Fetch live weather metrics to evaluate disaster probabilities
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=surface_pressure,wind_speed_10m&daily=precipitation_probability_max"
    weather_data = requests.get(weather_url).json()
    
    rain_prob = (weather_data.get("daily", {}).get("precipitation_probability_max") or [15])[0]
    wind_speed = weather_data.get("current", {}).get("wind_speed_10m", 8.0)
    pressure = weather_data.get("current", {}).get("surface_pressure", 1013.0)
    pressure_drop = max(0.0, 1013.0 - pressure)
    
    # Tectonic seismic zone heuristic flag for certain regions (e.g., North/Northeast India or Himalayan proximity)
    seismic_zone_flag = 1 if (lat > 23.0 and lon > 88.0) or (lat > 28.0) else 0

    disaster_probabilities = {"Flood Risk": 5.0, "Earthquake Tectonic Warning": 2.0, "Wildfire Thermal Alert": 3.0}
    primary_threat = "Normal (No Disaster Imminent)"

    if disaster_ai:
        features = [[rain_prob, wind_speed, pressure_drop, seismic_zone_flag]]
        probabilities = disaster_ai.predict_proba(features)[0]
        classes = disaster_ai.classes_
        
        # Map probabilities
        for idx, cls_code in enumerate(classes):
            prob_val = round(float(probabilities[idx]) * 100, 1)
            if cls_code == 1:
                disaster_probabilities["Flood Risk"] = prob_val
            elif cls_code == 2:
                disaster_probabilities["Earthquake Tectonic Warning"] = prob_val
            elif cls_code == 3:
                disaster_probabilities["Wildfire Thermal Alert"] = prob_val

        predicted_code = disaster_ai.predict(features)[0]
        if predicted_code == 1:
            primary_threat = "High Flood / Cyclone Storm Warning"
        elif predicted_code == 2:
            primary_threat = "Seismic Activity / Minor Tectonic Alert"
        elif predicted_code == 3:
            primary_threat = "Thermal Anomaly / Wildfire Risk Alert"

    return {
        "region": location.upper(),
        "country": country,
        "coordinates": {"latitude": lat, "longitude": lon},
        "satellite_map_link": f"https://maps.google.com/?q={lat},{lon}&t=k",
        "disaster_risk_assessment": {
            "primary_threat_status": primary_threat,
            "probability_percentages": disaster_probabilities,
            "thermal_anomaly_detected": True if disaster_probabilities["Wildfire Thermal Alert"] > 40 else False
        },
        "forest_fire_scan": {"status": "Active Grid Radar", "risk_level": "Normal" if disaster_probabilities["Wildfire Thermal Alert"] < 40 else "Elevated"}
    }


# ---------------------------------------------------------
# 5. WRF ATMOSPHERIC MODEL SIMULATION
# ---------------------------------------------------------
@app.get("/wrf-model-data")
def get_wrf_model(location: str = "Kolkata"):
    lat, lon, _ = get_live_coordinates(location)
    wrf_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_10m&models=icon_seamless"
    model_data = requests.get(wrf_url).json().get("current", {})

    return {
        "region": location.upper(),
        "numerical_model": "WRF / ICON Seamless Global Grid",
        "metrics": model_data
    }


# ---------------------------------------------------------
# 6. MASTER FRONTEND TASKBAR ENDPOINT (All-In-One)
# ---------------------------------------------------------
@app.get("/dashboard-summary")
def get_complete_dashboard(location: str = "Kolkata"):
    weather_info = predict_live_rainfall(location)
    aqi_info = predict_live_aqi(location)
    climate_info = forecast_climate(location, 2050)
    sat_info = get_satellite_map(location)

    return {
        "location_header": f"{weather_info['region']}, {weather_info['country']}",
        "coordinates": weather_info["coordinates"],
        "weather_condition": weather_info["weather_condition"],
        "visual_triggers": {
            "weather_effect": weather_info["frontend_visual_trigger"],
            "aqi_effect": aqi_info["frontend_aqi_trigger"]
        },
        "taskbar_cards": weather_info["taskbar_metrics"],
        "rain_analysis": {
            "ai_verdict": weather_info["ai_rainfall_verdict"],
            "chance_of_rain_range": weather_info["chance_of_rain_range"],
            "humidity_range": weather_info["taskbar_metrics"]["humidity_range"],
            "wind_speed_range": weather_info["taskbar_metrics"]["wind_speed_range"],
            "hourly_rain_bars": weather_info["hourly_rain_chance_next_6h"]
        },
        "air_quality": {
            "live_current_aqi": aqi_info["live_current_aqi"],
            "ai_predicted_aqi": aqi_info["ai_predicted_aqi_today"],
            "status": aqi_info["pollution_category"],
            "past_7_days_trend": aqi_info.get("live_past_7_days_trend", [])
        },
        "weekly_strip": weather_info["next_7_days_forecast"],
        "climate_2050": climate_info,
        "satellite_radar": sat_info
    }