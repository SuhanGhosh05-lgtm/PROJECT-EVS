
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "monthly"
    / "2024-11"
    / "aqi_november_2024.csv"
)

OUTPUT_DIR = INPUT_FILE.parent

POLLUTANTS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Ozone",
    "NH3",
]

EXPECTED_HOURS = 30 * 24

print("Loading November AQI dataset...")

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce"
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

print("\n========== BASIC DATASET SUMMARY ==========")

print("Total records:", len(df))
print("Unique stations:", df["station_id"].nunique())
print("Unique pollutants:", df["parameter_name"].nunique())

print("Earliest timestamp:", df["collected_at"].min())
print("Latest timestamp:", df["collected_at"].max())

print("\nPollutants present:")
print(sorted(df["parameter_name"].dropna().unique()))


# =====================================================
# 1. DATE RANGE VALIDATION
# =====================================================

print("\n========== DATE VALIDATION ==========")

print("Records outside November:")

outside = df[
    (df["collected_at"] < "2024-11-01")
    | (df["collected_at"] >= "2024-12-01")
]

print(len(outside))

print("\nDaily record counts:")

daily = (
    df.assign(date=df["collected_at"].dt.date)
    .groupby("date")
    .size()
)

print(daily.to_string())


# =====================================================
# 2. DUPLICATE CHECK
# =====================================================

print("\n========== DUPLICATE CHECK ==========")

duplicates = df.duplicated(
    subset=[
        "station_id",
        "parameter_name",
        "collected_at",
    ],
    keep=False
)

print("Duplicate measurement rows:", int(duplicates.sum()))


# =====================================================
# 3. MISSING VALUES
# =====================================================

print("\n========== MISSING VALUES ==========")

print(df.isna().sum().to_string())


# =====================================================
# 4. POLLUTANT COVERAGE BY STATION
# =====================================================

print("\n========== STATION-POLLUTANT COVERAGE ==========")

coverage = (
    df.groupby(["station_id", "parameter_name"])
    .size()
    .unstack(fill_value=0)
)

for pollutant in POLLUTANTS:
    if pollutant not in coverage.columns:
        coverage[pollutant] = 0

coverage = coverage[POLLUTANTS]

print(coverage.to_string())

coverage.to_csv(
    OUTPUT_DIR / "november_station_pollutant_coverage.csv"
)


# =====================================================
# 5. STATION COVERAGE PERCENTAGE
# =====================================================

print("\n========== COVERAGE PERCENTAGE ==========")

coverage_percent = (
    coverage / EXPECTED_HOURS * 100
).round(2)

print(coverage_percent.to_string())

coverage_percent.to_csv(
    OUTPUT_DIR / "november_station_coverage_percent.csv"
)


# =====================================================
# 6. UNIT CONSISTENCY
# =====================================================

print("\n========== UNIT CONSISTENCY ==========")

print(
    df.groupby("parameter_name")["unit"]
    .value_counts(dropna=False)
    .to_string()
)


# =====================================================
# 7. NUMERIC VALUE QUALITY
# =====================================================

print("\n========== VALUE QUALITY ==========")

print(
    df.groupby("parameter_name")["value"]
    .agg(
        count="count",
        minimum="min",
        mean="mean",
        median="median",
        maximum="max",
        standard_deviation="std",
    )
    .round(3)
    .to_string()
)

print("\nNegative readings:")

print(
    df[df["value"] < 0]
    .groupby("parameter_name")
    .size()
    .to_string()
)


# =====================================================
# 8. STATION TIMESTAMP COVERAGE
# =====================================================

print("\n========== HOURLY TIMESTAMP COVERAGE ==========")

timestamp_counts = (
    df.groupby("station_id")["collected_at"]
    .nunique()
    .sort_values()
)

print(timestamp_counts.to_string())


# =====================================================
# 9. SAVE AUDIT SUMMARY
# =====================================================

summary = pd.DataFrame({
    "station_id": coverage.index,
    "total_records": [
        len(group)
        for _, group in df.groupby("station_id")
    ],
}).set_index("station_id")

summary["unique_timestamps"] = timestamp_counts

summary.to_csv(
    OUTPUT_DIR / "november_station_summary.csv"
)

print("\nAudit reports saved in:")
print(OUTPUT_DIR)

print("\nNovember AQI audit completed.")
