
import os
import sys
import time
import calendar
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

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Ozone", "NH3"
]

PILOT_FILE = (
    BASE_DIR / "data" / "aqi" / "raw"
    / "pan_india_pilot" / "pan_india_pilot.csv"
)

STATION_FILE = (
    BASE_DIR / "data" / "aqi" / "metadata"
    / "all_india_stations.csv"
)

# Month is supplied as a command-line argument.
if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python download_aqi_monthly.py YYYY-MM\n"
        "Example: python download_aqi_monthly.py 2024-10"
    )

MONTH = sys.argv[1]

try:
    year, month = map(int, MONTH.split("-"))
    if len(MONTH) != 7 or not 1 <= month <= 12:
        raise ValueError
except ValueError:
    raise SystemExit("Invalid month. Use YYYY-MM, e.g. 2024-10")

START_DATE = f"{year:04d}-{month:02d}-01"
last_day = calendar.monthrange(year, month)[1]
END_DATE = f"{year:04d}-{month:02d}-{last_day:02d}"

OUTPUT_DIR = (
    BASE_DIR / "data" / "aqi" / "raw"
    / "monthly" / MONTH
)

STATION_DIR = OUTPUT_DIR / "stations"
STATION_DIR.mkdir(parents=True, exist_ok=True)

if not API_KEY:
    raise RuntimeError("XKDR_AQI_API_KEY missing from .env")

if not PILOT_FILE.exists():
    raise FileNotFoundError(PILOT_FILE)

if not STATION_FILE.exists():
    raise FileNotFoundError(STATION_FILE)

print("XKDR API key detected.")
print("Month:", MONTH)
print("Date range:", START_DATE, "to", END_DATE)


# =========================================================
# 2. LOAD THE SAME PILOT STATIONS
# =========================================================

pilot = pd.read_csv(PILOT_FILE)

station_ids = sorted(
    pilot["station_id"].dropna().astype(str).unique()
)

stations = pd.read_csv(STATION_FILE)
stations["station_id"] = stations["station_id"].astype(str)

station_metadata = (
    stations.drop_duplicates("station_id")
    .set_index("station_id")
)

print("Stations:", len(station_ids))
print("Pollutants:", len(POLLUTANTS))


# =========================================================
# 3. API SESSION
# =========================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "text/csv",
})

REQUIRED_COLUMNS = {
    "station_id",
    "parameter_name",
    "unit",
    "collected_at",
    "value",
}


# =========================================================
# 4. DOWNLOAD FUNCTION
# =========================================================

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

            row_count = response.headers.get("X-Row-Count")
            truncated = response.headers.get(
                "X-Truncated", "unknown"
            )

            if truncated.lower() == "true":
                raise RuntimeError("API response was truncated.")

            data = pd.read_csv(StringIO(response.text))

            missing = REQUIRED_COLUMNS - set(data.columns)

            if missing:
                raise RuntimeError(
                    f"Missing columns: {sorted(missing)}"
                )

            if not data.empty:
                data["station_id"] = data["station_id"].astype(str)

                if not data["station_id"].eq(station_id).all():
                    raise RuntimeError(
                        "Response contains unexpected station IDs."
                    )

                data["collected_at"] = pd.to_datetime(
                    data["collected_at"], errors="coerce"
                )

                data["value"] = pd.to_numeric(
                    data["value"], errors="coerce"
                )

                if data["collected_at"].isna().any():
                    raise RuntimeError(
                        "Invalid timestamps found in response."
                    )

                dates = data["collected_at"].dt.strftime("%Y-%m-%d")

                if not dates.between(START_DATE, END_DATE).all():
                    raise RuntimeError(
                        "Response contains records outside requested month."
                    )

            if row_count is not None:
                try:
                    if int(row_count) != len(data):
                        raise RuntimeError(
                            f"API row count {row_count} "
                            f"does not match downloaded rows {len(data)}."
                        )
                except ValueError:
                    raise RuntimeError(
                        f"Invalid X-Row-Count header: {row_count}"
                    )

            return data, row_count, truncated

        except Exception as error:
            print(
                f"Attempt {attempt}/3 failed for "
                f"{station_id}: {error}"
            )

            if attempt < 3:
                time.sleep(5 * attempt)

    return None, None, None


# =========================================================
# 5. LOAD PREVIOUS SUCCESSFUL DOWNLOAD LOG
# =========================================================

LOG_FILE = OUTPUT_DIR / "download_log.csv"

previous_log = {}

if LOG_FILE.exists():
    old_log = pd.read_csv(LOG_FILE, dtype={"station_id": str})

    for _, row in old_log.iterrows():
        previous_log[str(row["station_id"])] = row.to_dict()


# =========================================================
# 6. DOWNLOAD STATIONS
# =========================================================

all_data = []
download_log = []

for index, station_id in enumerate(station_ids, start=1):

    print(
        f"\n[{index}/{len(station_ids)}] "
        f"Station {station_id}"
    )

    station_file = STATION_DIR / f"{station_id}.csv"

    old_entry = previous_log.get(station_id, {})
    old_status = old_entry.get("status")

    # Reuse only a verified result from this month's own log.
    if (
        station_file.exists()
        and old_status in ["SUCCESS", "EMPTY", "REUSED"]
    ):
        print("Verified monthly download exists. Reusing.")

        data = pd.read_csv(
            station_file,
            dtype={"station_id": str}
        )

        all_data.append(data)

        download_log.append({
            "station_id": station_id,
            "status": "REUSED",
            "records": len(data),
            "api_row_count": old_entry.get("api_row_count"),
            "truncated": False,
        })

        continue

    data, row_count, truncated = download_station(station_id)

    if data is None:
        download_log.append({
            "station_id": station_id,
            "status": "FAILED",
            "records": 0,
            "api_row_count": row_count,
            "truncated": truncated,
        })
        continue

    # Attach station metadata.
    if station_id in station_metadata.index:
        metadata = station_metadata.loc[station_id]

        for column in [
            "station_name", "state_name", "city_name",
            "source", "latitude", "longitude"
        ]:
            if column in metadata.index:
                data[column] = metadata[column]

    data.to_csv(station_file, index=False)
    all_data.append(data)

    status = "SUCCESS" if len(data) else "EMPTY"

    download_log.append({
        "station_id": station_id,
        "status": status,
        "records": len(data),
        "api_row_count": row_count,
        "truncated": truncated,
    })

    print("Records:", len(data))
    time.sleep(1)


# =========================================================
# 7. SAVE MONTHLY DATASET
# =========================================================

if all_data:

    combined = pd.concat(all_data, ignore_index=True)

    if not combined.empty:
        combined["collected_at"] = pd.to_datetime(
            combined["collected_at"], errors="coerce"
        )

        combined["value"] = pd.to_numeric(
            combined["value"], errors="coerce"
        )

        combined = combined.sort_values([
            "station_id",
            "parameter_name",
            "collected_at",
        ])

    output_file = OUTPUT_DIR / f"aqi_{MONTH.replace('-', '_')}.csv"

    combined.to_csv(output_file, index=False)

    print("\nCombined records:", len(combined))
    print("Saved:", output_file)

else:
    print("\nNo station data downloaded.")


# =========================================================
# 8. SAVE DOWNLOAD LOG
# =========================================================

log = pd.DataFrame(download_log)
log.to_csv(LOG_FILE, index=False)

print("\n========== MONTHLY DOWNLOAD SUMMARY ==========")

if not log.empty:
    print(log["status"].value_counts().to_string())

    print("\nTotal records:", log["records"].sum())
    print("Successful stations:", log["status"].isin(
        ["SUCCESS", "REUSED", "EMPTY"]
    ).sum())
    print("Failed stations:", (log["status"] == "FAILED").sum())

print("\nOutput folder:", OUTPUT_DIR)
print("Download process completed.")
