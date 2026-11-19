
import pandas as pd
from pathlib import Path

# =========================================================
# 1. FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "pan_india_pilot"
    / "pan_india_pilot.csv"
)

REPORT_DIR = BASE_DIR / "reports" / "aqi_source_audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading AQI pilot dataset...")

df = pd.read_csv(DATA_FILE)

# =========================================================
# 2. BASIC DATASET INFORMATION
# =========================================================

print("\n" + "=" * 65)
print("BASIC DATASET INFORMATION")
print("=" * 65)

print("Total records:", len(df))
print("Columns:", list(df.columns))
print("Unique stations:", df["station_id"].nunique())
print("Unique pollutants:", df["parameter_name"].nunique())

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce"
)

print("Earliest timestamp:", df["collected_at"].min())
print("Latest timestamp:", df["collected_at"].max())

# =========================================================
# 3. MISSING VALUES
# =========================================================

print("\n" + "=" * 65)
print("MISSING VALUES")
print("=" * 65)

print(df.isnull().sum())

# =========================================================
# 4. DUPLICATE MEASUREMENTS
# =========================================================

print("\n" + "=" * 65)
print("DUPLICATE CHECK")
print("=" * 65)

key_columns = [
    "station_id",
    "parameter_name",
    "collected_at"
]

duplicates = df.duplicated(
    subset=key_columns,
    keep=False
)

print("Duplicate measurement rows:", duplicates.sum())

if duplicates.sum() > 0:
    df[duplicates].sort_values(key_columns).to_csv(
        REPORT_DIR / "pilot_duplicates.csv",
        index=False
    )

# =========================================================
# 5. POLLUTANT COVERAGE BY STATION
# =========================================================

print("\n" + "=" * 65)
print("POLLUTANT COVERAGE BY STATION")
print("=" * 65)

coverage = pd.crosstab(
    index=[
        df["station_id"],
        df["station_name"],
        df["city_name"],
        df["state_name"]
    ],
    columns=df["parameter_name"]
)

print(coverage.to_string())

coverage.to_csv(
    REPORT_DIR / "pilot_station_pollutant_coverage.csv"
)

# =========================================================
# 6. RECORD COUNTS BY STATION
# =========================================================

print("\n" + "=" * 65)
print("RECORD COUNTS BY STATION")
print("=" * 65)

station_counts = (
    df.groupby(
        ["station_id", "station_name", "city_name", "state_name"],
        dropna=False
    )
    .size()
    .reset_index(name="record_count")
    .sort_values("record_count")
)

print(station_counts.to_string(index=False))

station_counts.to_csv(
    REPORT_DIR / "pilot_station_record_counts.csv",
    index=False
)

# =========================================================
# 7. TIME COVERAGE BY STATION AND POLLUTANT
# =========================================================

print("\n" + "=" * 65)
print("TIME COVERAGE")
print("=" * 65)

time_coverage = (
    df.groupby(
        ["station_id", "parameter_name"],
        dropna=False
    )
    .agg(
        records=("value", "count"),
        first_reading=("collected_at", "min"),
        last_reading=("collected_at", "max")
    )
    .reset_index()
)

print(time_coverage.to_string(index=False))

time_coverage.to_csv(
    REPORT_DIR / "pilot_time_coverage.csv",
    index=False
)

# =========================================================
# 8. VALUE QUALITY CHECK
# =========================================================

print("\n" + "=" * 65)
print("POLLUTANT VALUE QUALITY")
print("=" * 65)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

value_quality = (
    df.groupby("parameter_name")
    .agg(
        records=("value", "count"),
        missing_values=("value", lambda x: x.isna().sum()),
        negative_values=("value", lambda x: (x < 0).sum()),
        minimum=("value", "min"),
        mean=("value", "mean"),
        median=("value", "median"),
        maximum=("value", "max"),
        standard_deviation=("value", "std")
    )
    .reset_index()
)

print(value_quality.to_string(index=False))

value_quality.to_csv(
    REPORT_DIR / "pilot_value_quality.csv",
    index=False
)

# =========================================================
# 9. UNIT CONSISTENCY
# =========================================================

print("\n" + "=" * 65)
print("UNITS BY POLLUTANT")
print("=" * 65)

unit_check = (
    df.groupby("parameter_name")["unit"]
    .value_counts(dropna=False)
    .rename("count")
    .reset_index()
)

print(unit_check.to_string(index=False))

unit_check.to_csv(
    REPORT_DIR / "pilot_unit_check.csv",
    index=False
)

# =========================================================
# 10. STATIONS WITH MISSING LOCATION METADATA
# =========================================================

print("\n" + "=" * 65)
print("MISSING STATION METADATA")
print("=" * 65)

metadata_columns = [
    "station_id",
    "station_name",
    "city_name",
    "state_name"
]

metadata_check = (
    df[metadata_columns]
    .drop_duplicates()
)

missing_metadata = metadata_check[
    metadata_check[["city_name", "state_name"]].isna().any(axis=1)
]

print(missing_metadata.to_string(index=False))

missing_metadata.to_csv(
    REPORT_DIR / "pilot_missing_station_metadata.csv",
    index=False
)

# =========================================================
# 11. HOURLY INTERVAL CHECK
# =========================================================

print("\n" + "=" * 65)
print("TIME INTERVAL CHECK")
print("=" * 65)

interval_results = []

for (station, pollutant), group in df.groupby(
    ["station_id", "parameter_name"]
):
    timestamps = (
        group["collected_at"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    intervals = timestamps.diff().dropna()

    if len(intervals) == 0:
        continue

    interval_results.append({
        "station_id": station,
        "parameter_name": pollutant,
        "median_interval": str(intervals.median()),
        "hourly_intervals": int(
            (intervals == pd.Timedelta(hours=1)).sum()
        ),
        "total_intervals": len(intervals)
    })

interval_df = pd.DataFrame(interval_results)

print(interval_df.to_string(index=False))

interval_df.to_csv(
    REPORT_DIR / "pilot_interval_check.csv",
    index=False
)

# =========================================================
# 12. FINAL REPORT
# =========================================================

print("\n" + "=" * 65)
print("AUDIT COMPLETED")
print("=" * 65)

print("Reports saved to:", REPORT_DIR)
print("Original downloaded dataset was not modified.")
