import requests

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("=" * 60)
    print("TESTING EVS CLIMATE AI BACKEND (Universal & Accurate CPCB)")
    print("=" * 60)

    # 1. Test Master Dashboard Summary for Delhi (Severe AQI Check)
    print("\n[Test 1] Testing /dashboard-summary for DELHI...")
    try:
        res = requests.get(f"{BASE_URL}/dashboard-summary?location=Delhi")
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Success! Location: {data['location_header']}")
            print(f"   - Weather Condition: {data['weather_condition']} ({data['visual_triggers']['weather_effect']})")
            print(f"   - Temperature: {data['taskbar_cards']['temperature_c']}°C (Real Feel: {data['taskbar_cards']['real_feel_c']}°C)")
            print(f"   - Live AQI (Delhi): {data['air_quality']['live_current_aqi']} ({data['air_quality']['status']})")
            print(f"   - AI Predicted AQI: {data['air_quality']['ai_predicted_aqi']}")
            print(f"   - 2050 Projection: {data['climate_2050']['projected_temperature']}°C ({data['climate_2050']['impact_risk']})")
        else:
            print(f"❌ Failed with status code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")

    # 2. Test Rainfall & Taskbar Metrics for Kolkata
    print("\n[Test 2] Testing /predict-rainfall for KOLKATA...")
    try:
        res = requests.get(f"{BASE_URL}/predict-rainfall?location=Kolkata")
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Success! Region: {data['region']}, {data['country']}")
            print(f"   - AI Verdict: {data['ai_rainfall_verdict']}")
            print(f"   - Chance of Rain: {data['chance_of_rain_percent']}%")
            print(f"   - Humidity: {data['taskbar_cards']['humidity_percent']}% | Wind: {data['taskbar_cards']['wind_speed_kmh']} km/h ({data['taskbar_cards']['wind_direction_compass']})")
            print(f"   - Sunrise / Sunset: {data['taskbar_cards']['sunrise_time']} / {data['taskbar_cards']['sunset_time']}")
        else:
            print(f"❌ Failed: {res.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # 3. Test Satellite Map & WRF Data
    print("\n[Test 3] Testing /get-satellite-map & /wrf-model-data for MUMBAI...")
    try:
        sat_res = requests.get(f"{BASE_URL}/get-satellite-map?location=Mumbai")
        wrf_res = requests.get(f"{BASE_URL}/wrf-model-data?location=Mumbai")
        if sat_res.status_code == 200 and wrf_res.status_code == 200:
            print(f"✅ Success! Satellite Map Link: {sat_res.json()['satellite_map_link']}")
            print(f"   - WRF Model Grid: {wrf_res.json()['numerical_model']}")
        else:
            print("❌ Failed satellite or WRF test.")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED. Ready for frontend or presentation demo!")
    print("=" * 60)

if __name__ == "__main__":
    test_api()