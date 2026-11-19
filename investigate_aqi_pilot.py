
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data/aqi/raw/pan_india_pilot/pan_india_pilot.csv"
STATION_FILE = BASE_DIR / "data/aqi/metadata/all_india_stations.csv"

df = pd.read_csv(DATA_FILE)
stations = pd.read_csv(STATION_FILE)

# ---------------------------------------------------------
# 1. INVESTIGATE ALANDUR STATION
# ---------------------------------------------------------

print("\n========== ALANDUR STATION METADATA ==========")

print(
    stations[
        stations["station_id"].astype(str) == "site_293"
    ].to_string(index=False)
)

# ---------------------------------------------------------
# 2. INVESTIGATE EXTREME POLLUTANT VALUES
# ---------------------------------------------------------

print("\n========== EXTREME POLLUTANT READINGS ==========")

for pollutant in ["PM2.5", "PM10", "NH3", "NO2", "Ozone", "SO2", "CO"]:

    subset = df[df["parameter_name"] == pollutant]

    print(f"\n--- {pollutant} ---")

    print("Highest readings:")

    print(
        subset.nlargest(10, "value")[
            [
                "station_id",
                "station_name",
                "collected_at",
                "value",
                "unit"
            ]
        ].to_string(index=False)
    )

    print("\nRepeated extreme values:")

    print(
        subset["value"]
        .value_counts()
        .head(10)
        .to_string()
    )

# ---------------------------------------------------------
# 3. CHECK EXACT UPPER-END READINGS
# ---------------------------------------------------------

print("\n========== POSSIBLE UPPER-END FLAGS ==========")

for pollutant in ["PM2.5", "PM10"]:

    subset = df[df["parameter_name"] == pollutant]

    print(f"\n{pollutant} values >= 900:")

    print(
        subset[subset["value"] >= 900][
            [
                "station_id",
                "station_name",
                "collected_at",
                "value"
            ]
        ].to_string(index=False)
    )

# ---------------------------------------------------------
# 4. STATION-LEVEL PARTICULATE COVERAGE
# ---------------------------------------------------------

print("\n========== PARTICULATE COVERAGE ==========")

particulate = df[
    df["parameter_name"].isin(["PM2.5", "PM10"])
]

coverage = (
    particulate.groupby(["station_id", "station_name", "parameter_name"])
    .size()
    .unstack(fill_value=0)
)

print(coverage.to_string())

print("\nInvestigation completed.")
