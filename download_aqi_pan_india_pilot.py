
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

API_KEY = os.getenv("XKDR_AQI_API_KEY")

if not API_KEY:
    raise ValueError("XKDR_AQI_API_KEY missing from .env")

BASE_URL = "https://airquality.xkdr.org/v1/measurements"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

START_DATE = "2024-11-01"
END_DATE = "2024-11-07"

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone",
    "NH3"
]

INVENTORY_FILE = Path(
    "data/aqi/metadata/all_india_stations.csv"
)

OUTPUT_DIR = Path("data/aqi/raw/pan_india_pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD STATION INVENTORY
# =====================================================

print("\n========== LOADING STATION INVENTORY ==========")

stations = pd.read_csv(INVENTORY_FILE)

# Only genuine CPCB CAAQM monitoring stations.
stations = stations[
    stations["source"] == "cpcb_caaqm"
].copy()

stations["first_seen"] = pd.to_datetime(
    stations["first_seen"],
    errors="coerce"
)

stations["last_seen"] = pd.to_datetime(
    stations["last_seen"],
    errors="coerce"
)

# Only stations active during the selected period.
start = pd.Timestamp(START_DATE)
end = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)

stations = stations[
    (stations["first_seen"] < end) &
    (stations["last_seen"] >= start)
].copy()

# Parse available pollutants.
stations["parameters_list"] = stations["parameters"].apply(
    lambda value: [
        p.strip()
        for p in str(value).split(",")
        if p.strip()
    ]
)

# Require both particulate pollutants.
stations = stations[
    stations["parameters_list"].apply(
        lambda p: "PM2.5" in p and "PM10" in p
    )
].copy()

# Normalise known state-name variation for selection.
stations["state_normalized"] = (
    stations["state_name"]
    .astype(str)
    .str.strip()
    .str.lower()
)

stations["state_normalized"] = (
    stations["state_normalized"]
    .replace({
        "tamilnadu": "tamil nadu"
    })
)

print("Eligible CPCB stations:", len(stations))


# =====================================================
# 3. SELECT ONE STATION PER STATE
# =====================================================

selected = (
    stations
    .sort_values("n_rows", ascending=False)
    .drop_duplicates(
        subset=["state_normalized"],
        keep="first"
    )
    .copy()
)

print("Selected stations:", len(selected))
print("States represented:", selected["state_normalized"].nunique())

print("\n========== SELECTED STATIONS ==========")

print(
    selected[
        [
            "station_id",
            "station_name",
            "state_name",
            "city_name",
            "n_rows"
        ]
    ].to_string(index=False)
)


# =====================================================
# 4. DOWNLOAD STATION MEASUREMENTS
# =====================================================

all_data = []
download_log = []

for index, (_, station) in enumerate(
    selected.iterrows(),
    start=1
):

    station_id = station["station_id"]

    available = station["parameters_list"]

    requested = [
        p for p in POLLUTANTS
        if p in available
    ]

    print("\n--------------------------------------------")
    print(
        f"[{index}/{len(selected)}] "
        f"{station['station_name']}"
    )

    print("Station ID:", station_id)
    print("City:", station["city_name"])
    print("State:", station["state_name"])
    print("Pollutants:", requested)

    params = [
        ("station", station_id),
        *[("parameter", p) for p in requested],
        ("start", START_DATE),
        ("end", END_DATE),
        ("agg", "raw"),
        ("format", "csv")
    ]

    try:

        response = requests.get(
            BASE_URL,
            headers=HEADERS,
            params=params,
            timeout=90
        )

        print("HTTP status:", response.status_code)

        response.raise_for_status()

        row_count = response.headers.get("X-Row-Count")
        truncated = response.headers.get("X-Truncated")

        if str(truncated).lower() == "true":
            raise RuntimeError(
                "API reports truncated data. "
                "Do not use this response."
            )

        if not response.text.strip():
            raise RuntimeError("Empty API response.")

        data = pd.read_csv(
            StringIO(response.text)
        )

        print("Rows reported:", row_count)
        print("Rows received:", len(data))
        print("Truncated:", truncated)

        if len(data) > 0:

            data["state_name"] = station["state_name"]
            data["city_name"] = station["city_name"]
            data["station_name"] = station["station_name"]

            all_data.append(data)

        download_log.append({
            "station_id": station_id,
            "station_name": station["station_name"],
            "state": station["state_name"],
            "city": station["city_name"],
            "status": "success",
            "rows": len(data),
            "reported_rows": row_count,
            "truncated": truncated
        })

    except Exception as error:

        print("DOWNLOAD ERROR:", error)

        download_log.append({
            "station_id": station_id,
            "station_name": station["station_name"],
            "state": station["state_name"],
            "city": station["city_name"],
            "status": "failed",
            "rows": 0,
            "reported_rows": None,
            "truncated": None,
            "error": str(error)
        })

    time.sleep(0.2)


# =====================================================
# 5. COMBINE DATA
# =====================================================

print("\n========== COMBINING DATA ==========")

if not all_data:
    raise RuntimeError(
        "No measurements downloaded successfully."
    )

combined = pd.concat(
    all_data,
    ignore_index=True
)

combined["collected_at"] = pd.to_datetime(
    combined["collected_at"],
    errors="coerce"
)

output_file = OUTPUT_DIR / "pan_india_pilot.csv"

combined.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

log_file = OUTPUT_DIR / "download_log.csv"

pd.DataFrame(download_log).to_csv(
    log_file,
    index=False
)


# =====================================================
# 6. BASIC QUALITY SUMMARY
# =====================================================

print("\n========== PILOT DATA SUMMARY ==========")

print("Total records:", len(combined))

print(
    "Unique stations:",
    combined["station_id"].nunique()
)

print(
    "Unique cities:",
    combined["city_name"].nunique()
)

print(
    "Unique states:",
    combined["state_name"].nunique()
)

print("\nPollutant counts:")

print(
    combined["parameter_name"]
    .value_counts()
    .to_string()
)

print("\nMissing values:")

print(combined.isna().sum().to_string())

print("\nDownload status:")

print(
    pd.DataFrame(download_log)["status"]
    .value_counts()
)

print("\nSaved dataset:", output_file.resolve())
print("Saved log:", log_file.resolve())

print("\nPAN-INDIA PILOT DOWNLOAD COMPLETED")

