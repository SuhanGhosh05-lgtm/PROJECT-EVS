
from pathlib import Path

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "processed"
    / "pan_india_pilot"
    / "aqi_november_2024_quality_checked.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "aqi"
    / "processed"
    / "pan_india_pilot"
)

OUTPUT_FILE = OUTPUT_DIR / "aqi_november_2024_daily_v2.csv"

MIN_24H_READINGS = 16
MIN_8H_READINGS = 6

AVERAGING_WINDOWS = {
    "PM2.5": 24,
    "PM10": 24,
    "NO2": 24,
    "SO2": 24,
    "NH3": 24,
    "Ozone": 8,
    "CO": 8,
}

BREAKPOINTS = {
    
    "PM2.5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    
    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2400, 401, 500),
    ],
    "Ozone": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    "CO": [
        (0, 1, 0, 50),
        (1.1, 2, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
    
}
SEVERE_THRESHOLDS = {
    "PM2.5": 251,
    "PM10": 431,
    "NO2": 401,
    "SO2": 1601,
    "NH3": 1801,
    "Ozone": 749,
    "CO": 34.1,
}

# =========================================================
# 2. SUB-INDEX CALCULATION
# =========================================================

def calculate_subindex(pollutant, concentration):

    if pd.isna(concentration) or concentration < 0:
        return None

    if pollutant == "CO":
        concentration = float(
            Decimal(str(concentration)).quantize(
                Decimal("0.1"),
                rounding=ROUND_HALF_UP,
            )
        )
    else:
        concentration = int(
            Decimal(str(concentration)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )

    # Severe category: cap the AQI sub-index at 500.
    if concentration >= SEVERE_THRESHOLDS[pollutant]:
        return 500.0

    for bp_low, bp_high, i_low, i_high in BREAKPOINTS[pollutant]:

        if bp_low <= concentration <= bp_high:

            index = (
                (i_high - i_low)
                / (bp_high - bp_low)
                * (concentration - bp_low)
                + i_low
            )

            return round(index, 2)

    return None

# =========================================================
# 3. LOAD DATA
# =========================================================

print("\n========== AQI DAILY DATASET V2 ==========")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

print("Original records:", len(df))
print("Original stations:", df["station_id"].nunique())


# =========================================================
# 4. APPLY PRELIMINARY QUALITY FILTER
# =========================================================

df = df[
    df["preliminary_training_eligible"].astype(bool)
].copy()

df = df.dropna(subset=["value", "collected_at"])

# Preserve records even when station metadata is missing.
df["station_name"] = df["station_name"].fillna("Unknown station")
df["city_name"] = df["city_name"].fillna("Unknown city")
df["state_name"] = df["state_name"].fillna("Unknown state")

df = df.sort_values("collected_at")
print("Records after quality filtering:", len(df))


# =========================================================
# 5. CALCULATE DAILY 24-HOUR AVERAGES
# =========================================================

metadata_columns = [
    "station_id",
    "station_name",
    "state_name",
    "city_name",
]

daily_records = []

pollutants_24h = [
    pollutant
    for pollutant, hours in AVERAGING_WINDOWS.items()
    if hours == 24
]

for pollutant in pollutants_24h:

    subset = df[
        df["parameter_name"] == pollutant
    ].copy()

    subset["date"] = subset["collected_at"].dt.date

    grouped = (
        subset.groupby(metadata_columns + ["date"])
        .agg(
            concentration=("value", "mean"),
            valid_hourly_readings=("value", "count"),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["valid_hourly_readings"] >= MIN_24H_READINGS
    ].copy()

    grouped["parameter_name"] = pollutant

    daily_records.append(grouped)


# =========================================================
# 6. CALCULATE MAXIMUM ROLLING 8-HOUR AVERAGES
# =========================================================

pollutants_8h = [
    pollutant
    for pollutant, hours in AVERAGING_WINDOWS.items()
    if hours == 8
]

for pollutant in pollutants_8h:

    subset = df[
        df["parameter_name"] == pollutant
    ].copy()

    subset = subset.sort_values("collected_at")

    for station_id, station_data in subset.groupby("station_id"):

        station_data = station_data.sort_values("collected_at")

        station_data = station_data.set_index("collected_at")

        # Calculate the average over the preceding 8 hours.
        rolling_mean = station_data["value"].rolling(
            "8h",
            min_periods=MIN_8H_READINGS,
        ).mean()

        rolling_count = station_data["value"].rolling(
            "8h",
            min_periods=MIN_8H_READINGS,
        ).count()

        rolling_data = pd.DataFrame({
            "concentration": rolling_mean,
            "valid_hourly_readings": rolling_count,
        })

        rolling_data = rolling_data.dropna(
            subset=["concentration"]
        )

        if rolling_data.empty:
            continue

        rolling_data["date"] = rolling_data.index.date

        # Select the highest rolling 8-hour average per day.
        max_indices = rolling_data.groupby("date")[
            "concentration"
        ].idxmax()

        best_windows = rolling_data.loc[max_indices].copy()

        metadata = station_data.iloc[0]

        best_windows["station_id"] = station_id
        best_windows["station_name"] = metadata["station_name"]
        best_windows["state_name"] = metadata["state_name"]
        best_windows["city_name"] = metadata["city_name"]
        best_windows["parameter_name"] = pollutant

        best_windows = best_windows.reset_index(
            names="window_end"
        )

        daily_records.append(best_windows[
            metadata_columns
            + [
                "date",
                "parameter_name",
                "concentration",
                "valid_hourly_readings",
            ]
        ])


# =========================================================
# 7. COMBINE DAILY POLLUTANT DATA
# =========================================================

if not daily_records:
    raise ValueError("No valid daily pollutant averages found.")

daily = pd.concat(
    daily_records,
    ignore_index=True,
)

print("\nDaily pollutant records:", len(daily))


# =========================================================
# 8. PIVOT INTO STATION-DAY DATASET
# =========================================================

index_columns = metadata_columns + ["date"]

concentration_panel = daily.pivot_table(
    index=index_columns,
    columns="parameter_name",
    values="concentration",
    aggfunc="first",
).reset_index()

coverage_panel = daily.pivot_table(
    index=index_columns,
    columns="parameter_name",
    values="valid_hourly_readings",
    aggfunc="first",
).reset_index()

concentration_panel.columns.name = None
coverage_panel.columns.name = None

# Add every pollutant column, even if unavailable.
for pollutant in AVERAGING_WINDOWS:

    if pollutant not in concentration_panel.columns:
        concentration_panel[pollutant] = None

    if pollutant not in coverage_panel.columns:
        coverage_panel[pollutant] = None


# =========================================================
# 9. CALCULATE POLLUTANT SUB-INDICES
# =========================================================

for pollutant in AVERAGING_WINDOWS:

    concentration_panel[f"{pollutant}_subindex"] = (
        concentration_panel[pollutant].apply(
            lambda value: calculate_subindex(
                pollutant,
                value,
            )
        )
    )


# =========================================================
# 10. CALCULATE VALID AQI TARGETS
# =========================================================

subindex_columns = [
    f"{pollutant}_subindex"
    for pollutant in AVERAGING_WINDOWS
]

concentration_panel["available_subindices"] = (
    concentration_panel[subindex_columns]
    .notna()
    .sum(axis=1)
)

concentration_panel["particulate_available"] = (
    concentration_panel[
        ["PM2.5_subindex", "PM10_subindex"]
    ].notna().any(axis=1)
)

# Require at least three valid pollutant sub-indices,
# including at least one particulate pollutant.

concentration_panel["aqi_target_valid"] = (
    (concentration_panel["available_subindices"] >= 3)
    & concentration_panel["particulate_available"]
)

concentration_panel["partial_cpcb_style_aqi"] = (
    concentration_panel[subindex_columns].max(axis=1)
)

concentration_panel.loc[
    ~concentration_panel["aqi_target_valid"],
    "partial_cpcb_style_aqi",
] = None


# =========================================================
# 11. IDENTIFY DOMINANT POLLUTANT
# =========================================================

def get_dominant_pollutant(row):

    values = {
        pollutant: row[f"{pollutant}_subindex"]
        for pollutant in AVERAGING_WINDOWS
    }

    values = {
        pollutant: value
        for pollutant, value in values.items()
        if pd.notna(value)
    }

    if not values:
        return None

    return max(values, key=values.get)


concentration_panel["dominant_pollutant"] = (
    concentration_panel.apply(
        get_dominant_pollutant,
        axis=1,
    )
)


# =========================================================
# 12. ADD AUDIT INFORMATION
# =========================================================

concentration_panel["aqi_method"] = (
    "PARTIAL_CPCB_STYLE_INDICATIVE"
)

concentration_panel["official_cpcb_aqi"] = False

concentration_panel["missing_lead_pollutant"] = True

concentration_panel["minimum_pollutants_required"] = 3


# =========================================================
# 13. SAVE DATASET
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

concentration_panel.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# 14. FINAL SUMMARY
# =========================================================

valid = concentration_panel[
    concentration_panel["aqi_target_valid"]
]

print("\n========== V2 DATASET SUMMARY ==========")

print("Daily station records:", len(concentration_panel))
print("Stations:", concentration_panel["station_id"].nunique())

print("Valid AQI targets:", len(valid))

print(
    "Invalid AQI targets:",
    int((~concentration_panel["aqi_target_valid"]).sum())
)

print(
    "Unique cities:",
    concentration_panel["city_name"].nunique()
)

print("\nDominant pollutant distribution:")

print(
    valid["dominant_pollutant"].value_counts()
)

print("\nAQI target statistics:")

if not valid.empty:
    print(
        valid["partial_cpcb_style_aqi"].describe()
    )
else:
    print("No valid AQI targets found.")

print("\nAvailable pollutant sub-indices:")

print(
    valid["available_subindices"].value_counts().sort_index()
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nAQI daily dataset V2 completed.")
