
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "multi_month"
    / "aqi_oct_nov_quality_checked.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "multi_month"
)

HOURLY_OUTPUT = OUTPUT_DIR / "aqi_oct_nov_hourly_v3.csv"
DAILY_OUTPUT = OUTPUT_DIR / "aqi_oct_nov_daily_v3.csv"

MIN_24H_READINGS = 16
MIN_8H_READINGS = 6

WINDOWS = {
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

METADATA_COLUMNS = [
    "station_id",
    "station_name",
    "state_name",
    "city_name",
]


# =========================================================
# 2. SUB-INDEX CALCULATION
# =========================================================



def calculate_subindex(pollutant, concentration):

    if pd.isna(concentration) or concentration < 0:
        return None

    if pollutant not in BREAKPOINTS:
        return None

    # Apply pollutant-specific rounding
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

    # Conservative handling of the Severe category:
    # do not invent an upper concentration breakpoint.
    if concentration >= SEVERE_THRESHOLDS[pollutant]:
        return 500.0

    # Interpolate only within the published lower categories
    for bp_low, bp_high, i_low, i_high in BREAKPOINTS[pollutant]:

        if concentration < SEVERE_THRESHOLDS[pollutant] and bp_low <= concentration <= bp_high:

            index = (
                (i_high - i_low)
                / (bp_high - bp_low)
                * (concentration - bp_low)
                + i_low
            )

            return round(index, 2)

    return None


# =========================================================
# 3. LOAD AND VALIDATE INPUT
# =========================================================

print("\n========== AQI DATASET V3 ==========")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "station_id",
    "collected_at",
    "parameter_name",
    "value",
    "preliminary_training_eligible",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce",
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce",
)

# Robustly interpret the eligibility field.
eligible = (
    df["preliminary_training_eligible"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

df = df[eligible].copy()

df = df.dropna(
    subset=[
        "station_id",
        "collected_at",
        "value",
    ]
)

df = df[df["parameter_name"].isin(WINDOWS)].copy()

df = df[df["value"] >= 0].copy()

for column in ["station_name", "state_name", "city_name"]:
    if column not in df.columns:
        df[column] = "Unknown"

    df[column] = df[column].fillna("Unknown")

# Measurements are hourly. Normalize timestamps to the hour.
df["collected_at"] = df["collected_at"].dt.floor("h")

print("Eligible input records:", len(df))
print("Stations:", df["station_id"].nunique())
print("Pollutants:", df["parameter_name"].nunique())


# =========================================================
# 4. STATION METADATA
# =========================================================

station_metadata = (
    df.sort_values("collected_at")
    .groupby("station_id")[METADATA_COLUMNS[1:]]
    .first()
)


# =========================================================
# 5. GENERATE ROLLING POLLUTANT WINDOWS
# =========================================================

rolling_records = []

for (station_id, pollutant), group in df.groupby(
    ["station_id", "parameter_name"]
):

    group = group.sort_values("collected_at")

    # Average duplicate station/pollutant/hour records.
    series = (
        group.groupby("collected_at")["value"]
        .mean()
    )

    # Construct a complete hourly timeline.
    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="h",
    )

    hourly = series.reindex(full_index)

    window_hours = WINDOWS[pollutant]

    if window_hours == 24:
        minimum = MIN_24H_READINGS
    else:
        minimum = MIN_8H_READINGS

    rolling = hourly.rolling(
        window=window_hours,
        min_periods=minimum,
    )

    rolling_mean = rolling.mean()
    rolling_count = rolling.count()

    valid = rolling_mean.notna()

    if not valid.any():
        continue

    result = pd.DataFrame({
        "station_id": station_id,
        "window_end": full_index[valid],
        "parameter_name": pollutant,
        "concentration": rolling_mean[valid].values,
        "valid_hourly_readings": rolling_count[valid].values,
    })

    result["subindex"] = result.apply(
        lambda row: calculate_subindex(
            row["parameter_name"],
            row["concentration"],
        ),
        axis=1,
    )

    rolling_records.append(result)

    print(
        f"{pollutant:7s} | {station_id:12s} | "
        f"{len(result)} valid windows"
    )

if not rolling_records:
    raise ValueError("No valid rolling pollutant windows generated.")

rolling_data = pd.concat(
    rolling_records,
    ignore_index=True,
)

print("\nTotal valid pollutant windows:", len(rolling_data))


# =========================================================
# 6. BUILD HOURLY STATION PANEL
# =========================================================

index_columns = ["station_id", "window_end"]

subindex_panel = rolling_data.pivot_table(
    index=index_columns,
    columns="parameter_name",
    values="subindex",
    aggfunc="first",
)

concentration_panel = rolling_data.pivot_table(
    index=index_columns,
    columns="parameter_name",
    values="concentration",
    aggfunc="first",
)

coverage_panel = rolling_data.pivot_table(
    index=index_columns,
    columns="parameter_name",
    values="valid_hourly_readings",
    aggfunc="first",
)

subindex_panel.columns = [
    f"{pollutant}_subindex"
    for pollutant in subindex_panel.columns
]

concentration_panel.columns = [
    f"{pollutant}_rolling_concentration"
    for pollutant in concentration_panel.columns
]

coverage_panel.columns = [
    f"{pollutant}_valid_readings"
    for pollutant in coverage_panel.columns
]

hourly = (
    subindex_panel
    .join(concentration_panel)
    .join(coverage_panel)
    .reset_index()
)

# Add missing pollutant columns consistently.
for pollutant in WINDOWS:

    for suffix in [
        "_subindex",
        "_rolling_concentration",
        "_valid_readings",
    ]:

        column = pollutant + suffix

        if column not in hourly.columns:
            hourly[column] = pd.NA

hourly = hourly.merge(
    station_metadata.reset_index(),
    on="station_id",
    how="left",
)

hourly["date"] = hourly["window_end"].dt.date


# =========================================================
# 7. CALCULATE HOURLY AQI-STYLE TARGET
# =========================================================

subindex_columns = [
    f"{pollutant}_subindex"
    for pollutant in WINDOWS
]

hourly["available_subindices"] = (
    hourly[subindex_columns].notna().sum(axis=1)
)

hourly["particulate_available"] = (
    hourly[
        ["PM2.5_subindex", "PM10_subindex"]
    ].notna().any(axis=1)
)

hourly["aqi_target_valid"] = (
    (hourly["available_subindices"] >= 3)
    & hourly["particulate_available"]
)

# Overall value = highest available pollutant sub-index.
hourly["partial_cpcb_style_aqi"] = (
    hourly[subindex_columns].max(axis=1)
)

hourly.loc[
    ~hourly["aqi_target_valid"],
    "partial_cpcb_style_aqi",
] = pd.NA


def get_dominant_pollutant(row):

    values = {
        pollutant: row[f"{pollutant}_subindex"]
        for pollutant in WINDOWS
    }

    values = {
        pollutant: value
        for pollutant, value in values.items()
        if pd.notna(value)
    }

    if not values:
        return None

    return max(values, key=values.get)


hourly["dominant_pollutant"] = hourly.apply(
    get_dominant_pollutant,
    axis=1,
)

hourly["aqi_method"] = "PARTIAL_CPCB_STYLE_INDICATIVE"
hourly["official_cpcb_aqi"] = False
hourly["missing_lead_pollutant"] = True

hourly = hourly.sort_values(
    ["station_id", "window_end"]
).reset_index(drop=True)


# =========================================================
# 8. CREATE DAILY MAXIMUM AQI-STYLE TARGET
# =========================================================

valid_hourly = hourly[
    hourly["aqi_target_valid"]
].copy()

if valid_hourly.empty:
    raise ValueError(
        "No valid hourly AQI-style targets were generated."
    )

valid_hourly["date"] = pd.to_datetime(
    valid_hourly["window_end"]
).dt.date

daily_group_columns = [
    "station_id",
    "date",
]

valid_hours_per_day = (
    valid_hourly.groupby(daily_group_columns)
    .size()
    .rename("valid_hourly_aqi_count")
)

# Select the hour with the highest valid AQI-style value.
maximum_indices = (
    valid_hourly.groupby(daily_group_columns)[
        "partial_cpcb_style_aqi"
    ].idxmax()
)

daily = valid_hourly.loc[maximum_indices].copy()

daily = daily.merge(
    valid_hours_per_day.reset_index(),
    on=daily_group_columns,
    how="left",
)

daily = daily.rename(
    columns={
        "window_end": "daily_max_aqi_timestamp",
        "partial_cpcb_style_aqi": "daily_max_partial_aqi",
    }
)

daily["aqi_target_valid"] = True
daily["aqi_method"] = "PARTIAL_CPCB_STYLE_INDICATIVE"
daily["official_cpcb_aqi"] = False
daily["missing_lead_pollutant"] = True

daily = daily.sort_values(
    ["station_id", "date"]
).reset_index(drop=True)


# =========================================================
# 9. SAVE BOTH DATASETS
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

hourly.to_csv(
    HOURLY_OUTPUT,
    index=False,
)

daily.to_csv(
    DAILY_OUTPUT,
    index=False,
)


# =========================================================
# 10. FINAL AUDIT
# =========================================================

print("\n========== V3 DATASET SUMMARY ==========")

print("Hourly station records:", len(hourly))
print("Hourly valid AQI-style targets:", len(valid_hourly))

print("Daily station records:", len(daily))
print("Stations:", daily["station_id"].nunique())
print("Unique cities:", daily["city_name"].nunique())

print("\nDaily target statistics:")

print(
    daily["daily_max_partial_aqi"].describe()
)

print("\nDominant pollutant distribution:")

print(
    daily["dominant_pollutant"].value_counts()
)

print("\nAvailable sub-indices at daily maximum:")

print(
    daily["available_subindices"].value_counts().sort_index()
)

print("\nValid hourly AQI targets by station:")

print(
    valid_hourly.groupby("station_id").size()
    .sort_values()
)

print("\nHourly output:")
print(HOURLY_OUTPUT)

print("\nDaily output:")
print(DAILY_OUTPUT)

print("\nAQI V3 generation completed.")
