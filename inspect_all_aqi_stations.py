
import os
import json
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("XKDR_AQI_API_KEY")

if not API_KEY:
    raise ValueError("XKDR_AQI_API_KEY not found in .env")

BASE_URL = "https://airquality.xkdr.org"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

OUTPUT_DIR = Path("data/aqi/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FETCH ALL STATIONS
# ==========================================

print("\nDownloading complete station inventory...")

response = requests.get(
    f"{BASE_URL}/v1/stations",
    headers=HEADERS,
    timeout=90
)

print("HTTP status:", response.status_code)

response.raise_for_status()

result = response.json()

metadata = result.get("meta", {})
stations = result.get("data", [])

print("Stations reported:", metadata.get("rows"))
print("Stations received:", len(stations))
print("Truncated:", metadata.get("truncated"))


# ==========================================
# 3. CHECK RESPONSE COMPLETENESS
# ==========================================

if not stations:
    raise RuntimeError("No station records returned.")

if metadata.get("truncated") is True:
    raise RuntimeError(
        "Station response is truncated. "
        "We must retrieve the remaining records "
        "before proceeding."
    )


# ==========================================
# 4. SAVE RAW JSON
# ==========================================

json_path = OUTPUT_DIR / "all_stations_raw.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

print("\nRaw JSON saved:", json_path)


# ==========================================
# 5. CONVERT TO DATAFRAME
# ==========================================

df = pd.DataFrame(stations)

if "parameters" not in df.columns:
    raise RuntimeError("Station parameter information missing.")

df["parameter_count"] = df["parameters"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

df["parameters"] = df["parameters"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else ""
)


# ==========================================
# 6. SAVE COMPLETE STATION CSV
# ==========================================

csv_path = OUTPUT_DIR / "all_india_stations.csv"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("Station CSV saved:", csv_path)


# ==========================================
# 7. STATE-WISE COVERAGE
# ==========================================

print("\n========== STATE-WISE STATION COVERAGE ==========")

state_summary = (
    df.groupby("state_name")
    .agg(
        stations=("station_id", "nunique"),
        cities=("city_name", "nunique")
    )
    .sort_values("stations", ascending=False)
)

print(state_summary.to_string())


# ==========================================
# 8. SOURCE-WISE COVERAGE
# ==========================================

print("\n========== SOURCE-WISE COVERAGE ==========")

print(
    df.groupby("source")["station_id"]
    .nunique()
    .sort_values(ascending=False)
)


# ==========================================
# 9. POLLUTANT COVERAGE
# ==========================================

print("\n========== POLLUTANT STATION COVERAGE ==========")

parameter_counts = {}

for _, station in df.iterrows():

    parameters = station["parameters"].split(", ")

    for parameter in parameters:

        if parameter:
            parameter_counts[parameter] = (
                parameter_counts.get(parameter, 0) + 1
            )

parameter_summary = (
    pd.Series(parameter_counts, name="station_count")
    .sort_values(ascending=False)
)

print(parameter_summary.to_string())


# ==========================================
# 10. CITY COVERAGE
# ==========================================

print("\n========== TOP 30 CITIES BY STATION COUNT ==========")

city_summary = (
    df.groupby(["state_name", "city_name"])
    ["station_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(30)
)

print(city_summary.to_string())


# ==========================================
# 11. WEST BENGAL COVERAGE
# ==========================================

print("\n========== WEST BENGAL STATIONS ==========")

west_bengal = df[
    df["state_name"].str.strip().str.lower()
    == "west bengal"
]

columns = [
    "station_id",
    "station_name",
    "city_name",
    "source",
    "first_seen",
    "last_seen",
    "parameter_count",
    "parameters"
]

print(
    west_bengal[columns].to_string(index=False)
)


# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n========== FINAL INVENTORY SUMMARY ==========")

print("Total stations:", df["station_id"].nunique())
print("Total states:", df["state_name"].nunique())
print("Total cities:", df["city_name"].nunique())

print("Truncated:", metadata.get("truncated"))

print("\nFiles saved in:", OUTPUT_DIR.resolve())

print("\nSTATION INVENTORY COMPLETED")
