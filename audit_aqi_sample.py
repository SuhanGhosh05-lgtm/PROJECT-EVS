
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================

FILE = Path(
    "data/aqi/raw/kolkata_ballygunge_sample.csv"
)

df = pd.read_csv(FILE)

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce"
)

print("\n========== BASIC DATASET INFORMATION ==========")

print("Total rows:", len(df))
print("Columns:", list(df.columns))
print("Stations:", df["station_id"].nunique())

print(
    "Start date:",
    df["collected_at"].min()
)

print(
    "End date:",
    df["collected_at"].max()
)

# ==========================================
# 2. POLLUTANT COVERAGE
# ==========================================

print("\n========== POLLUTANT COVERAGE ==========")

for pollutant, group in df.groupby("parameter_name"):

    print(
        pollutant,
        "| Rows:", len(group),
        "| Unit:", group["unit"].iloc[0],
        "| Start:", group["collected_at"].min(),
        "| End:", group["collected_at"].max()
    )

# ==========================================
# 3. DUPLICATE CHECK
# ==========================================

print("\n========== DUPLICATE CHECK ==========")

duplicate_count = df.duplicated(
    subset=[
        "station_id",
        "parameter_name",
        "collected_at"
    ]
).sum()

print("Duplicate records:", duplicate_count)

# ==========================================
# 4. MISSING TIMESTAMPS
# ==========================================

print("\n========== MISSING HOURLY RECORDS ==========")

for pollutant, group in df.groupby("parameter_name"):

    timestamps = (
        group["collected_at"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(timestamps) < 2:
        continue

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="1h"
    )

    missing = expected.difference(timestamps)

    print(
        pollutant,
        "| Expected:", len(expected),
        "| Actual:", len(timestamps),
        "| Missing:", len(missing)
    )

# ==========================================
# 5. INVALID VALUES
# ==========================================

print("\n========== INVALID VALUE CHECK ==========")

print(
    "Missing values:",
    df.isna().sum().to_dict()
)

print(
    "Negative readings:",
    (df["value"] < 0).sum()
)

print(
    "Non-numeric readings:",
    pd.to_numeric(
        df["value"],
        errors="coerce"
    ).isna().sum()
)

# ==========================================
# 6. STATISTICAL SUMMARY
# ==========================================

print("\n========== POLLUTANT STATISTICS ==========")

summary = df.groupby("parameter_name")["value"].agg(
    ["count", "mean", "std", "min", "median", "max"]
)

print(summary.round(3))

# ==========================================
# 7. TIME INTERVAL CHECK
# ==========================================

print("\n========== TIME INTERVAL CHECK ==========")

for pollutant, group in df.groupby("parameter_name"):

    timestamps = (
        group["collected_at"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    intervals = timestamps.diff().dropna()

    print(
        pollutant,
        "| Most common interval:",
        intervals.mode().iloc[0]
        if not intervals.empty else "Unavailable"
    )

# ==========================================
# 8. SAVE AUDIT REPORT
# ==========================================

report_dir = Path("reports/aqi_source_audit")
report_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    report_dir / "kolkata_sample_statistics.csv"
)

print("\n========== AUDIT COMPLETED ==========")
print("Original dataset was not modified.")
print("Statistics saved in:", report_dir.resolve())

