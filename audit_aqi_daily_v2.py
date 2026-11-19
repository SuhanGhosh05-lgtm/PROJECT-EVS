
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "pan_india_pilot"
    / "aqi_november_2024_quality_checked.csv"
)

DAILY_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "pan_india_pilot"
    / "aqi_november_2024_daily_v2.csv"
)

raw = pd.read_csv(INPUT_FILE)
daily = pd.read_csv(DAILY_FILE)

pollutants = [
    "PM2.5", "PM10", "NO2", "SO2",
    "NH3", "Ozone", "CO"
]

print("\n========== AQI V2 VALIDATION AUDIT ==========")

# 1. Find the missing station
original_stations = set(raw["station_id"].dropna().unique())
daily_stations = set(daily["station_id"].dropna().unique())

missing = original_stations - daily_stations

print("\nSTATION COVERAGE")
print("Original stations:", len(original_stations))
print("Stations in daily dataset:", len(daily_stations))
print("Missing station IDs:", sorted(missing))

if missing:
    print("\nMissing station details:")
    print(
        raw[raw["station_id"].isin(missing)][
            ["station_id", "station_name", "city_name", "state_name"]
        ].drop_duplicates().to_string(index=False)
    )

# 2. Check concentration values without sub-indices
print("\nCONCENTRATION / SUB-INDEX CHECK")

for pollutant in pollutants:
    sub_col = f"{pollutant}_subindex"

    if pollutant not in daily.columns:
        print(f"{pollutant}: concentration column missing")
        continue

    if sub_col not in daily.columns:
        print(f"{pollutant}: sub-index column missing")
        continue

    has_value = daily[pollutant].notna()
    no_subindex = daily[sub_col].isna()

    affected = daily[has_value & no_subindex]

    print(f"\n{pollutant}")
    print("Concentration values available:", int(has_value.sum()))
    print("Missing sub-index despite concentration:", len(affected))

    if not affected.empty:
        print("Examples:")
        print(
            affected[
                ["station_id", "city_name", pollutant]
            ].head(8).to_string(index=False)
        )

# 3. Check valid AQI target distribution
print("\nAQI TARGET VALIDITY")

print(
    daily["aqi_target_valid"].value_counts(dropna=False)
)

print("\nValid targets by station:")

print(
    daily[daily["aqi_target_valid"]]
    .groupby(["station_id", "city_name"])
    .size()
    .sort_values()
    .to_string()
)

# 4. Check pollutant availability
print("\nPOLLUTANT AVAILABILITY")

for pollutant in pollutants:
    sub_col = f"{pollutant}_subindex"

    if sub_col in daily.columns:
        print(
            pollutant,
            "valid sub-indices:",
            int(daily[sub_col].notna().sum())
        )

# 5. Check AQI values outside the expected scale
valid_aqi = daily.loc[
    daily["aqi_target_valid"],
    "partial_cpcb_style_aqi"
].dropna()

print("\nAQI RANGE CHECK")

print("Minimum:", valid_aqi.min() if not valid_aqi.empty else None)
print("Maximum:", valid_aqi.max() if not valid_aqi.empty else None)
print("Values below zero:", int((valid_aqi < 0).sum()))
print("Values above 500:", int((valid_aqi > 500).sum()))

print("\nAudit completed.")
