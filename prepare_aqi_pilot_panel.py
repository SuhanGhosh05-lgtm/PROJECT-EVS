
from pathlib import Path

import pandas as pd


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
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
    / "processed"
    / "pan_india_pilot"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_HOURS = 7 * 24
MIN_COVERAGE = 0.80

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone",
    "NH3",
]


# =========================================================
# 2. LOAD DATA
# =========================================================

print("Loading AQI pilot data...")

df = pd.read_csv(INPUT_FILE)
stations = pd.read_csv(STATION_FILE)

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce"
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

df = df.dropna(
    subset=["station_id", "parameter_name", "collected_at"]
).copy()

print("Raw records:", len(df))
print("Stations:", df["station_id"].nunique())


# =========================================================
# 3. EXTREME VALUE FLAGS
# =========================================================

df["extreme_particulate_flag"] = (
    df["parameter_name"].isin(["PM2.5", "PM10"])
    & (df["value"] >= 900)
)

df["plateau_flag"] = False


# =========================================================
# 4. DETECT CONSECUTIVE IDENTICAL READINGS
# =========================================================

print("\nChecking consecutive repeated readings...")

df = df.sort_values(
    ["station_id", "parameter_name", "collected_at"]
).reset_index(drop=True)

group_columns = ["station_id", "parameter_name"]

previous_value = df.groupby(group_columns)["value"].shift(1)
previous_time = df.groupby(group_columns)["collected_at"].shift(1)

same_value = df["value"].eq(previous_value)

one_hour_gap = (
    df["collected_at"] - previous_time
) == pd.Timedelta(hours=1)

new_run = ~(same_value & one_hour_gap)

df["run_id"] = (
    new_run.groupby(
        [df["station_id"], df["parameter_name"]]
    ).cumsum()
)

run_sizes = df.groupby(
    ["station_id", "parameter_name", "run_id"]
)["value"].transform("size")

df["plateau_flag"] = run_sizes >= 6

print(
    "Readings in possible six-hour-or-longer plateaus:",
    int(df["plateau_flag"].sum())
)


# =========================================================
# 5. CREATE QUALITY FLAGS TABLE
# =========================================================

quality_flags = df[
    df["extreme_particulate_flag"] | df["plateau_flag"]
].copy()

quality_flags.to_csv(
    OUTPUT_DIR / "aqi_quality_flags.csv",
    index=False
)

print("Quality flags saved.")


# =========================================================
# 6. CREATE WIDE STATION-HOUR DATASET
# =========================================================

print("\nCreating wide hourly dataset...")

wide = df.pivot(
    index=["station_id", "collected_at"],
    columns="parameter_name",
    values="value"
).reset_index()

wide.columns.name = None

# Ensure all expected pollutant columns exist.
for pollutant in POLLUTANTS:
    if pollutant not in wide.columns:
        wide[pollutant] = pd.NA


# =========================================================
# 7. ADD STATION METADATA
# =========================================================

metadata_columns = [
    "station_id",
    "station_name",
    "state_name",
    "city_name",
    "source",
    "latitude",
    "longitude",
]

available_metadata = [
    column
    for column in metadata_columns
    if column in stations.columns
]

station_metadata = stations[available_metadata].drop_duplicates(
    subset=["station_id"]
)

wide = wide.merge(
    station_metadata,
    on="station_id",
    how="left"
)


# =========================================================
# 8. STATION COVERAGE REPORT
# =========================================================

print("\nCalculating station coverage...")

coverage = (
    df.groupby(["station_id", "parameter_name"])
    .size()
    .unstack(fill_value=0)
)

for pollutant in POLLUTANTS:
    if pollutant not in coverage.columns:
        coverage[pollutant] = 0

coverage = coverage[POLLUTANTS]

coverage["PM2.5_coverage"] = (
    coverage["PM2.5"] / EXPECTED_HOURS
)

coverage["PM10_coverage"] = (
    coverage["PM10"] / EXPECTED_HOURS
)

coverage["training_eligible"] = (
    (coverage["PM2.5_coverage"] >= MIN_COVERAGE)
    & (coverage["PM10_coverage"] >= MIN_COVERAGE)
)

coverage = coverage.reset_index()

coverage = coverage.merge(
    station_metadata,
    on="station_id",
    how="left"
)

coverage.to_csv(
    OUTPUT_DIR / "aqi_station_coverage.csv",
    index=False
)


# =========================================================
# 9. ADD TRAINING ELIGIBILITY TO WIDE DATA
# =========================================================

eligibility = coverage[
    ["station_id", "training_eligible"]
]

wide = wide.merge(
    eligibility,
    on="station_id",
    how="left"
)

wide["training_eligible"] = (
    wide["training_eligible"].fillna(False)
)


# =========================================================
# 10. SAVE OUTPUT
# =========================================================

wide = wide.sort_values(
    ["station_id", "collected_at"]
)

wide.to_csv(
    OUTPUT_DIR / "aqi_pilot_wide.csv",
    index=False
)


# =========================================================
# 11. FINAL SUMMARY
# =========================================================

print("\n========== AQI PILOT PANEL SUMMARY ==========")

print("Wide dataset rows:", len(wide))
print("Unique stations:", wide["station_id"].nunique())

print("\nPollutant missing-value counts:")

print(wide[POLLUTANTS].isna().sum())

print("\nStation training eligibility:")

print(
    coverage[
        [
            "station_id",
            "station_name",
            "PM2.5",
            "PM10",
            "PM2.5_coverage",
            "PM10_coverage",
            "training_eligible",
        ]
    ].to_string(index=False)
)

print("\nEligible stations:",
      int(coverage["training_eligible"].sum()))

print("Excluded stations:",
      int((~coverage["training_eligible"]).sum()))

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nAQI pilot panel preparation completed.")
