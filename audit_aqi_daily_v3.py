
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "pan_india_pilot"
)

HOURLY_FILE = DATA_DIR / "aqi_november_2024_hourly_v3.csv"
DAILY_FILE = DATA_DIR / "aqi_november_2024_daily_v3.csv"

POLLUTANTS = [
    "PM2.5", "PM10", "NO2", "SO2",
    "NH3", "Ozone", "CO",
]

print("\n========== AQI V3 VALIDATION AUDIT ==========")

hourly = pd.read_csv(HOURLY_FILE)
daily = pd.read_csv(DAILY_FILE)

hourly["window_end"] = pd.to_datetime(hourly["window_end"])
daily["daily_max_aqi_timestamp"] = pd.to_datetime(
    daily["daily_max_aqi_timestamp"]
)

daily["date"] = pd.to_datetime(daily["date"]).dt.date
hourly["date"] = hourly["window_end"].dt.date

# ---------------------------------------------------------
# 1. BASIC COUNTS
# ---------------------------------------------------------

print("\n========== DATASET COUNTS ==========")

print("Hourly rows:", len(hourly))
print("Daily rows:", len(daily))

print("Hourly stations:", hourly["station_id"].nunique())
print("Daily stations:", daily["station_id"].nunique())

print("Hourly date range:",
      hourly["date"].min(), "to", hourly["date"].max())

print("Daily date range:",
      daily["date"].min(), "to", daily["date"].max())

# ---------------------------------------------------------
# 2. DUPLICATE CHECK
# ---------------------------------------------------------

print("\n========== DUPLICATE CHECK ==========")

hourly_duplicates = hourly.duplicated(
    ["station_id", "window_end"]
).sum()

daily_duplicates = daily.duplicated(
    ["station_id", "date"]
).sum()

print("Duplicate station-hour rows:", hourly_duplicates)
print("Duplicate station-day rows:", daily_duplicates)

# ---------------------------------------------------------
# 3. HOURLY TARGET VALIDATION
# ---------------------------------------------------------

print("\n========== HOURLY TARGET VALIDATION ==========")

subindex_columns = [
    f"{p}_subindex" for p in POLLUTANTS
]

for column in subindex_columns:
    hourly[column] = pd.to_numeric(
        hourly[column], errors="coerce"
    )

hourly["partial_cpcb_style_aqi"] = pd.to_numeric(
    hourly["partial_cpcb_style_aqi"],
    errors="coerce",
)

hourly["recalculated_max"] = (
    hourly[subindex_columns].max(axis=1)
)

hourly["recalculated_count"] = (
    hourly[subindex_columns].notna().sum(axis=1)
)

hourly["recalculated_particulate"] = (
    hourly[
        ["PM2.5_subindex", "PM10_subindex"]
    ].notna().any(axis=1)
)

hourly["recalculated_valid"] = (
    (hourly["recalculated_count"] >= 3)
    & hourly["recalculated_particulate"]
)

hourly_mismatch = (
    hourly["aqi_target_valid"].astype(bool)
    != hourly["recalculated_valid"]
).sum()

target_mismatch = (
    hourly.loc[
        hourly["recalculated_valid"],
        "partial_cpcb_style_aqi"
    ]
    - hourly.loc[
        hourly["recalculated_valid"],
        "recalculated_max"
    ]
).abs().gt(0.01).sum()

print("Validity mismatches:", hourly_mismatch)
print("AQI calculation mismatches:", target_mismatch)

# ---------------------------------------------------------
# 4. DAILY TARGET VALIDATION
# ---------------------------------------------------------

print("\n========== DAILY TARGET VALIDATION ==========")

daily["daily_max_partial_aqi"] = pd.to_numeric(
    daily["daily_max_partial_aqi"],
    errors="coerce",
)

hourly_valid = hourly[
    hourly["recalculated_valid"]
].copy()

expected_daily = (
    hourly_valid.groupby(
        ["station_id", "date"]
    )["partial_cpcb_style_aqi"]
    .max()
    .reset_index(name="expected_daily_max")
)

comparison = daily.merge(
    expected_daily,
    on=["station_id", "date"],
    how="left",
)

comparison["daily_difference"] = (
    comparison["daily_max_partial_aqi"]
    - comparison["expected_daily_max"]
).abs()

daily_mismatches = (
    comparison["daily_difference"].fillna(999) > 0.01
).sum()

print("Daily target mismatches:", daily_mismatches)

# ---------------------------------------------------------
# 5. AQI RANGE AND MISSING VALUES
# ---------------------------------------------------------

print("\n========== RANGE CHECK ==========")

print(
    "Hourly AQI range:",
    hourly["partial_cpcb_style_aqi"].min(),
    "to",
    hourly["partial_cpcb_style_aqi"].max(),
)

print(
    "Daily AQI range:",
    daily["daily_max_partial_aqi"].min(),
    "to",
    daily["daily_max_partial_aqi"].max(),
)

print(
    "Daily missing targets:",
    daily["daily_max_partial_aqi"].isna().sum(),
)

print(
    "Daily targets equal to 500:",
    (daily["daily_max_partial_aqi"] == 500).sum(),
)

print(
    "Daily targets outside 0–500:",
    (
        (daily["daily_max_partial_aqi"] < 0)
        | (daily["daily_max_partial_aqi"] > 500)
    ).sum(),
)

# ---------------------------------------------------------
# 6. DAILY POLLUTANT COVERAGE
# ---------------------------------------------------------

print("\n========== POLLUTANT COVERAGE ==========")

print(
    daily["available_subindices"]
    .value_counts()
    .sort_index()
)

print("\nDominant pollutant distribution:")

print(daily["dominant_pollutant"].value_counts())

print("\nDaily records per station:")

print(
    daily.groupby("station_id").size()
    .sort_values()
)

# ---------------------------------------------------------
# 7. LOWEST COVERAGE DAILY TARGETS
# ---------------------------------------------------------

print("\n========== LOWEST COVERAGE DAILY TARGETS ==========")

print(
    daily[
        [
            "station_id",
            "city_name",
            "date",
            "daily_max_partial_aqi",
            "available_subindices",
            "dominant_pollutant",
        ]
    ]
    .sort_values("available_subindices")
    .head(15)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 8. FINAL RESULT
# ---------------------------------------------------------

print("\n========== FINAL AUDIT RESULT ==========")

checks = {
    "No duplicate hourly rows": hourly_duplicates == 0,
    "No duplicate daily rows": daily_duplicates == 0,
    "Hourly validity is consistent": hourly_mismatch == 0,
    "Hourly AQI calculations match": target_mismatch == 0,
    "Daily targets match hourly maxima": daily_mismatches == 0,
    "No missing daily targets": daily["daily_max_partial_aqi"].notna().all(),
    "All daily targets within 0–500": (
        daily["daily_max_partial_aqi"].between(0, 500).all()
    ),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

print(
    "\nOverall:",
    "ALL CHECKS PASSED" if all(checks.values())
    else "INVESTIGATION REQUIRED",
)

print("\nAQI V3 audit completed.")
