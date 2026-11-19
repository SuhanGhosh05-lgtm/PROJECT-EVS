
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "processed"
    / "pan_india_pilot"
    / "aqi_quality_flags.csv"
)

OUTPUT_DIR = INPUT_FILE.parent

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])

# Keep only repeated-value plateau flags.
plateaus = df[df["plateau_flag"] == True].copy()

print("\n========== PLATEAU SUMMARY ==========")
print("Flagged readings:", len(plateaus))

print("\nFlagged readings by pollutant:")
print(
    plateaus.groupby("parameter_name")
    .size()
    .sort_values(ascending=False)
)

print("\nFlagged readings by station:")
print(
    plateaus.groupby(["station_id", "station_name"])
    .size()
    .sort_values(ascending=False)
)

# Summarise each individual consecutive plateau.
runs = (
    plateaus.groupby(
        ["station_id", "station_name", "parameter_name", "run_id"],
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
    (runs["end"] - runs["start"]).dt.total_seconds() / 3600 + 1
)

runs = runs.sort_values(
    ["readings", "station_id"],
    ascending=[False, True]
)

print("\n========== LONGEST PLATEAU RUNS ==========")
print(runs.head(30).to_string(index=False))

runs.to_csv(
    OUTPUT_DIR / "aqi_plateau_runs.csv",
    index=False
)

# Inspect extreme particulate readings separately.
extremes = df[
    df["extreme_particulate_flag"] == True
].copy()

print("\n========== EXTREME PARTICULATE FLAGS ==========")
if extremes.empty:
    print("No extreme particulate readings.")
else:
    print(
        extremes[
            [
                "station_id",
                "station_name",
                "parameter_name",
                "collected_at",
                "value",
                "unit",
            ]
        ].to_string(index=False)
    )

print("\nPlateau run report saved:")
print(OUTPUT_DIR / "aqi_plateau_runs.csv")
