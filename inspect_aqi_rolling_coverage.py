

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "pan_india_pilot"
    / "aqi_november_2024_quality_checked.csv"
)

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df[
    df["preliminary_training_eligible"].astype(bool)
].dropna(subset=["collected_at", "value"])

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "NH3", "Ozone", "CO"]

print("\n========== ROLLING WINDOW COVERAGE AUDIT ==========")

for pollutant in pollutants:

    subset = df[df["parameter_name"] == pollutant].copy()

    print(f"\n========== {pollutant} ==========")

    station_results = []

    for station_id, group in subset.groupby("station_id"):

        group = group.sort_values("collected_at")

        # Count unique timestamps, not duplicate rows.
        timestamps = pd.DatetimeIndex(
            group["collected_at"].drop_duplicates()
        )

        if len(timestamps) == 0:
            continue

        expected_hours = int(
            (timestamps.max() - timestamps.min()).total_seconds()
            / 3600
        ) + 1

        observed_hours = len(timestamps)

        missing_hours = max(
            0,
            expected_hours - observed_hours
        )

        differences = timestamps.to_series().diff()

        gaps_over_one_hour = int(
            (differences > pd.Timedelta(hours=1)).sum()
        )

        station_results.append({
            "station_id": station_id,
            "observed_hours": observed_hours,
            "expected_hours": expected_hours,
            "missing_hours": missing_hours,
            "coverage_percent": round(
                observed_hours / expected_hours * 100, 2
            ) if expected_hours else 0,
            "gaps_over_1_hour": gaps_over_one_hour,
        })

    result = pd.DataFrame(station_results)

    if result.empty:
        print("No data available.")
        continue

    print(result.to_string(index=False))

    print("\nPollutant summary:")
    print(
        result[
            ["observed_hours", "expected_hours",
             "missing_hours", "coverage_percent"]
        ].describe().round(2)
    )

print("\nRolling coverage audit completed.")
