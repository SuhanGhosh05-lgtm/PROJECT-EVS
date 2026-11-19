
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

OUTPUT_FILE = OUTPUT_DIR / "aqi_november_2024_daily.csv"

# Preliminary completeness requirements
MIN_24H_READINGS = 16
MIN_8H_READINGS = 6

# CPCB breakpoint table.
# Concentration units:
# PM2.5, PM10, NO2, SO2, NH3, O3 = µg/m³
# CO = mg/m³

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

AVERAGING_WINDOWS = {
    "PM2.5": 24,
    "PM10": 24,
    "NO2": 24,
    "SO2": 24,
    "NH3": 24,
    "Ozone": 8,
    "CO": 8,
}


# =========================================================
# 2. POLLUTANT SUB-INDEX CALCULATION
# =========================================================

def calculate_subindex(pollutant, concentration):

    if pd.isna(concentration) or concentration < 0:
        return None

    for bp_low, bp_high, i_low, i_high in BREAKPOINTS[pollutant]:

        if bp_low <= concentration <= bp_high:

            index = (
                (i_high - i_low)
                / (bp_high - bp_low)
                * (concentration - bp_low)
                + i_low
            )

            return round(index, 2)

    # Above the highest breakpoint: do not extrapolate.
    return None


# =========================================================
# 3. LOAD QUALITY-CONTROLLED MEASUREMENTS
# =========================================================

print("\n========== BUILDING DAILY AQI DATASET ==========")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

print("Input records:", len(df))
print("Stations:", df["station_id"].nunique())


# =========================================================
# 4. REMOVE PRELIMINARILY INELIGIBLE MEASUREMENTS
# =========================================================

df = df[
    df["preliminary_training_eligible"].astype(bool)
].copy()

# Zero values are retained at this stage.
# Agartala CO series, long plateaus and extreme particulate
# values are excluded from this preliminary training copy.

print("Records after preliminary quality filtering:", len(df))


# =========================================================
# 5. CREATE DAILY POLLUTANT AVERAGES
# =========================================================

df["date"] = df["collected_at"].dt.date

daily = (
    df.groupby(
        [
            "station_id",
            "station_name",
            "state_name",
            "city_name",
            "date",
            "parameter_name",
        ]
    )
    .agg(
        concentration=("value", "mean"),
        valid_hourly_readings=("value", "count"),
    )
    .reset_index()
)

daily["required_readings"] = daily["parameter_name"].map(
    lambda p: (
        MIN_24H_READINGS
        if AVERAGING_WINDOWS.get(p) == 24
        else MIN_8H_READINGS
    )
)

daily["average_valid"] = (
    daily["valid_hourly_readings"] >= daily["required_readings"]
)

daily.loc[~daily["average_valid"], "concentration"] = None

print("Daily station-pollutant records:", len(daily))


# =========================================================
# 6. PIVOT POLLUTANTS INTO COLUMNS
# =========================================================

concentration_panel = daily.pivot_table(
    index=[
        "station_id",
        "station_name",
        "state_name",
        "city_name",
        "date",
    ],
    columns="parameter_name",
    values="concentration",
    aggfunc="first",
).reset_index()

coverage_panel = daily.pivot_table(
    index=[
        "station_id",
        "date",
    ],
    columns="parameter_name",
    values="average_valid",
    aggfunc="first",
).reset_index()

concentration_panel.columns.name = None
coverage_panel.columns.name = None

# Ensure all seven pollutant columns exist.
for pollutant in AVERAGING_WINDOWS:
    if pollutant not in concentration_panel.columns:
        concentration_panel[pollutant] = None

    if pollutant not in coverage_panel.columns:
        coverage_panel[pollutant] = False


# =========================================================
# 7. CALCULATE POLLUTANT SUB-INDICES
# =========================================================

for pollutant in AVERAGING_WINDOWS:

    concentration_panel[f"{pollutant}_subindex"] = (
        concentration_panel[pollutant].apply(
            lambda value: calculate_subindex(pollutant, value)
        )
    )


# =========================================================
# 8. CALCULATE DAILY AQI TARGET
# =========================================================

subindex_columns = [
    f"{pollutant}_subindex"
    for pollutant in AVERAGING_WINDOWS
]

concentration_panel["available_subindices"] = (
    concentration_panel[subindex_columns].notna().sum(axis=1)
)

concentration_panel["partial_cpcb_style_aqi"] = (
    concentration_panel[subindex_columns].max(axis=1)
)

# At least one particulate sub-index must be available.
concentration_panel["particulate_available"] = (
    concentration_panel[
        ["PM2.5_subindex", "PM10_subindex"]
    ].notna().any(axis=1)
)

concentration_panel["aqi_target_valid"] = (
    concentration_panel["particulate_available"]
    & concentration_panel["partial_cpcb_style_aqi"].notna()
)

concentration_panel.loc[
    ~concentration_panel["aqi_target_valid"],
    "partial_cpcb_style_aqi",
] = None


# =========================================================
# 9. IDENTIFY DOMINANT POLLUTANT
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
    concentration_panel.apply(get_dominant_pollutant, axis=1)
)


# =========================================================
# 10. ADD DATASET LABELS
# =========================================================

concentration_panel["aqi_method"] = (
    "PARTIAL_CPCB_STYLE_INDICATIVE"
)

concentration_panel["official_cpcb_aqi"] = False

concentration_panel["missing_lead_pollutant"] = True


# =========================================================
# 11. SAVE DAILY DATASET
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

concentration_panel.to_csv(OUTPUT_FILE, index=False)


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

valid = concentration_panel[
    concentration_panel["aqi_target_valid"]
]

print("\n========== DAILY AQI DATASET SUMMARY ==========")

print("Daily station records:", len(concentration_panel))
print("Stations:", concentration_panel["station_id"].nunique())
print("Valid AQI targets:", len(valid))
print("Invalid AQI targets:", int((~concentration_panel["aqi_target_valid"]).sum()))

print(
    "Unique cities:",
    concentration_panel["city_name"].nunique()
)

print("\nDominant pollutant distribution:")
print(valid["dominant_pollutant"].value_counts())

print("\nAQI target statistics:")
if not valid.empty:
    print(valid["partial_cpcb_style_aqi"].describe())
else:
    print("No valid AQI targets were generated.")

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nDaily AQI dataset construction completed.")
