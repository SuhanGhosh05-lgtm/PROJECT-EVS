
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_URL = "https://airquality.xkdr.org"

load_dotenv()

API_KEY = os.getenv("XKDR_AQI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key not found. Check your .env file "
        "for XKDR_AQI_API_KEY."
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

OUTPUT_DIR = Path("reports/aqi_source_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. API REQUEST FUNCTION
# ==========================================

def fetch_data(endpoint, params=None):

    url = f"{BASE_URL}{endpoint}"

    print(f"\nRequesting: {endpoint}")

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=60
    )

    print("HTTP Status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    filename = endpoint.strip("/").replace("/", "_")
    output_file = OUTPUT_DIR / f"{filename}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return data


# ==========================================
# 3. INSPECT API METADATA
# ==========================================

print("\n========== AQI DATABASE METADATA ==========")

metadata = fetch_data("/v1/meta")

print(json.dumps(metadata, indent=2)[:5000])


# ==========================================
# 4. INSPECT AVAILABLE POLLUTANTS
# ==========================================

print("\n========== AVAILABLE POLLUTANTS ==========")

parameters = fetch_data("/v1/parameters")

print(json.dumps(parameters, indent=2)[:5000])


# ==========================================
# 5. INSPECT KOLKATA STATIONS
# ==========================================

print("\n========== KOLKATA MONITORING STATIONS ==========")

kolkata_stations = fetch_data(
    "/v1/stations",
    params={"city": "Kolkata"}
)

print(json.dumps(kolkata_stations, indent=2)[:8000])


# ==========================================
# 6. INSPECT WEST BENGAL STATIONS
# ==========================================

print("\n========== WEST BENGAL MONITORING STATIONS ==========")

west_bengal_stations = fetch_data(
    "/v1/stations",
    params={"state": "West Bengal"}
)

print(json.dumps(west_bengal_stations, indent=2)[:8000])


print("\n==========================================")
print("AQI SOURCE INSPECTION COMPLETED")
print("Saved files in:", OUTPUT_DIR.resolve())
print("==========================================")
