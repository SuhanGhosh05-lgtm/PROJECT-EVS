
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from dotenv import load_dotenv


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("XKDR_AQI_API_KEY")

API_URL = "https://airquality.xkdr.org/v1/measurements"

START_DATE = "2024-11-01"
END_DATE = "2024-11-30"

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone",
    "NH3",
]

PILOT_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "pan_india_pilot"
    / "pan_india_pilot.csv"
)

STATION_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "metadata"
    / "all_india_stations.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "monthly"
    / "2024-11"
)

STATION_DIR = OUTPUT_DIR / "stations"

STATION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. VALIDATE API KEY
# =========================================================

if not API_KEY:
    raise RuntimeError(
        "XKDR_AQI_API_KEY missing from .env file."
    )

print("XKDR API key detected.")


# =========================================================
# 3. LOAD PILOT STATIONS
# =========================================================

pilot = pd.read_csv(PILOT_FILE)

station_ids = sorted(
    pilot["station_id"].dropna().astype(str).unique()
)

stations = pd.read_csv(STATION_FILE)

stations["station_id"] = stations["station_id"].astype(str)

station_metadata = stations.drop_duplicates(
    subset=["station_id"]
).set_index("station_id")

print("\nStations to download:", len(station_ids))
print("Pollutants per station:", len(POLLUTANTS))
print("Date range:", START_DATE, "to", END_DATE)


# =========================================================
# 4. API REQUEST FUNCTION
# =========================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "text/csv",
})


def download_station(station_id):

    params = [
        ("station", station_id),
        ("start", START_DATE),
        ("end", END_DATE),
        ("agg", "raw"),
        ("format", "csv"),
    ]

    for pollutant in POLLUTANTS:
        params.append(("parameter", pollutant))

    for attempt in range(1, 4):

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=90,
            )

            response.raise_for_status()

            row_count = response.headers.get(
                "X-Row-Count"
            )

            truncated = response.headers.get(
                "X-Truncated",
                "unknown"
            )

            if truncated.lower() == "true":
                raise RuntimeError(
                    "API reports truncated response."
                )

            data = pd.read_csv(
                StringIO(response.text)
            )

            if data.empty:
                return data, row_count, truncated

            required_columns = {
                "station_id",
                "parameter_name",
                "unit",
                "collected_at",
                "value",
            }

            missing_columns = (
                required_columns - set(data.columns)
            )

            if missing_columns:
                raise RuntimeError(
                    f"Missing columns: {missing_columns}"
                )

            return data, row_count, truncated

        except Exception as error:

            print(
                f"Attempt {attempt}/3 failed "
                f"for {station_id}: {error}"
            )

            if attempt < 3:
                time.sleep(5 * attempt)

    return None, None, None


# =========================================================
# 5. DOWNLOAD ALL PILOT STATIONS
# =========================================================

all_data = []
download_log = []

for index, station_id in enumerate(station_ids, start=1):

    print(
        f"\n[{index}/{len(station_ids)}] "
        f"Downloading station {station_id}"
    )

    station_file = (
        STATION_DIR / f"{station_id}.csv"
    )

    # Reuse an already completed station download.
    if station_file.exists():

        print("Existing station file found. Reusing it.")

        data = pd.read_csv(station_file)

        all_data.append(data)

        download_log.append({
            "station_id": station_id,
            "status": "REUSED",
            "records": len(data),
            "api_row_count": None,
            "truncated": False,
        })

        continue

    data, row_count, truncated = download_station(
        station_id
    )

    if data is None:

        download_log.append({
            "station_id": station_id,
            "status": "FAILED",
            "records": 0,
            "api_row_count": row_count,
            "truncated": truncated,
        })

        continue

    # Preserve station metadata where available.
    metadata = station_metadata.loc[
        station_id
    ] if station_id in station_metadata.index else None

    if metadata is not None:

        for column in [
            "station_name",
            "state_name",
            "city_name",
            "source",
            "latitude",
            "longitude",
        ]:

            if column in metadata.index:
                data[column] = metadata[column]

    # Save the individual station file.
    data.to_csv(
        station_file,
        index=False
    )

    all_data.append(data)

    download_log.append({
        "station_id": station_id,
        "status": "SUCCESS",
        "records": len(data),
        "api_row_count": row_count,
        "truncated": truncated,
    })

    print("Records downloaded:", len(data))

    time.sleep(1)


# =========================================================
# 6. SAVE COMBINED MONTHLY DATASET
# =========================================================

if all_data:

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined["collected_at"] = pd.to_datetime(
        combined["collected_at"],
        errors="coerce"
    )

    combined["value"] = pd.to_numeric(
        combined["value"],
        errors="coerce"
    )

    combined = combined.sort_values(
        [
            "station_id",
            "parameter_name",
            "collected_at",
        ]
    )

    combined.to_csv(
        OUTPUT_DIR / "aqi_november_2024.csv",
        index=False
    )

    print("\nCombined records:", len(combined))

else:

    print("\nNo station data was downloaded.")


# =========================================================
# 7. SAVE DOWNLOAD LOG
# =========================================================

log = pd.DataFrame(download_log)

log.to_csv(
    OUTPUT_DIR / "download_log.csv",
    index=False
)


# =========================================================
# 8. FINAL REPORT
# =========================================================

print("\n========== DOWNLOAD SUMMARY ==========")

if not log.empty:

    print(
        log["status"].value_counts().to_string()
    )

    print("\nStation-level download log:")

    print(
        log.to_string(index=False)
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nDownload process completed.")
