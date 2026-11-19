
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

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(
    df["collected_at"],
    errors="coerce"
)

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

df = df.sort_values(
    ["station_id", "parameter_name", "collected_at"]
).reset_index(drop=True)


# =====================================================
# 1. DETECT CONSECUTIVE IDENTICAL READINGS
# =====================================================

previous_value = df.groupby(
    ["station_id", "parameter_name"]
)["value"].shift(1)

previous_time = df.groupby(
    ["station_id", "parameter_name"]
)["collected_at"].shift(1)

same_value = df["value"].eq(previous_value)

one_hour_gap = (
    df["collected_at"] - previous_time
) == pd.Timedelta(hours=1)

new_run = ~(same_value & one_hour_gap)

df["run_id"] = new_run.groupby(
    [df["station_id"], df["parameter_name"]]
).cumsum()

run_sizes = df.groupby(
    ["station_id", "parameter_name", "run_id"]
)["value"].transform("size")

df["plateau_flag"] = run_sizes >= 6

plateaus = df[df["plateau_flag"]].copy()


# =====================================================
# 2. SUMMARISE PLATEAU RUNS
# =====================================================

runs = (
    plateaus.groupby(
        [
            "station_id",
            "station_name",
            "parameter_name",
            "run_id",
        ],
        dropna=False
    )
    .agg(
        value=("value", "first"),
        readings=("value", "size"),
        start=("collected_at", "min"),
        end=("collected_at", "max"),
    )
    .reset_index()
)

runs["duration_hours"] = (
    (runs["end"] - runs["start"])
    .dt.total_seconds() / 3600 + 1
)

runs = runs.sort_values(
    "readings",
    ascending=False
)

runs.to_csv(
    OUTPUT_DIR / "november_plateau_runs.csv",
    index=False
)

print("\n========== PLATEAU SUMMARY ==========")

print("Flagged readings:", len(plateaus))
print("Distinct plateau runs:", len(runs))

print("\nBy pollutant:")
print(
    plateaus.groupby("parameter_name")
    .size()
    .sort_values(ascending=False)
)

print("\nBy station:")
print(
    plateaus.groupby(
        ["station_id", "station_name"]
    ).size().sort_values(ascending=False)
)

print("\nLongest 25 runs:")
print(runs.head(25).to_string(index=False))


# =====================================================
# 3. INVESTIGATE EXTREME PARTICULATE READINGS
# =====================================================

extremes = df[
    df["parameter_name"].isin(["PM2.5", "PM10"])
    & (df["value"] >= 900)
].copy()

extremes.to_csv(
    OUTPUT_DIR / "november_extreme_particulate.csv",
    index=False
)

print("\n========== EXTREME PARTICULATE READINGS ==========")

if extremes.empty:
    print("No readings >= 900 µg/m³")
else:
    print(
        extremes[
            [
                "station_id",
                "station_name",
                "parameter_name",
                "collected_at",
                "value",
            ]
        ].to_string(index=False)
    )

    print("\nExtreme readings by station:")
    print(
        extremes.groupby(
            ["station_id", "parameter_name"]
        ).size().to_string()
    )


# =====================================================
# 4. CHECK ZERO-VALUE POLLUTANT SERIES
# =====================================================


print("\n========== ZERO-VALUE INVESTIGATION ==========")

# Count zero readings for every station and pollutant
zero_summary = (
    df.assign(is_zero=df["value"] == 0)
    .groupby(["station_id", "station_name", "parameter_name"])
    .agg(
        total_readings=("value", "count"),
        zero_readings=("is_zero", "sum"),
    )
    .reset_index()
)

# Calculate percentage of zero readings
zero_summary["zero_percentage"] = (
    zero_summary["zero_readings"]
    / zero_summary["total_readings"] * 100
)

# Keep every combination with at least one zero
zero_flags = zero_summary[
    zero_summary["zero_readings"] > 0
].copy()

print("\nTotal station-pollutant combinations with zero readings:",
      len(zero_flags))

print("\nAll zero-reading combinations:")

if zero_flags.empty:
    print("No zero readings found.")
else:
    print(
        zero_flags.sort_values(
            ["zero_readings", "zero_percentage"],
            ascending=False
        ).to_string(index=False)
    )

# Save complete zero-reading investigation
zero_flags.to_csv(
    OUTPUT_DIR / "november_zero_value_flags.csv",
    index=False
)

# Highlight combinations where at least 50% are zero
high_zero_flags = zero_flags[
    zero_flags["zero_percentage"] >= 50
]

print("\nCombinations with at least 50% zero readings:")

if high_zero_flags.empty:
    print("None found.")
else:
    print(high_zero_flags.to_string(index=False))
