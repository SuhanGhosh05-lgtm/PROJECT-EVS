
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "pan_india_pilot"
    / "aqi_november_2024_quality_checked.csv"
)

POLLUTANT_WINDOWS = {
    "PM2.5": 24,
    "PM10": 24,
    "NO2": 24,
    "SO2": 24,
    "NH3": 24,
    "Ozone": 8,
    "CO": 8,
}

MINIMUM_READINGS = {
    24: 16,
    8: 6,
}

df = pd.read_csv(INPUT_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df[
    df["preliminary_training_eligible"].astype(bool)
].dropna(subset=["collected_at", "value"])

print("\n========== VALID ROLLING WINDOW AUDIT ==========")

all_results = []

for pollutant, window_hours in POLLUTANT_WINDOWS.items():

    minimum = MINIMUM_READINGS[window_hours]

    subset = df[
        df["parameter_name"] == pollutant
    ].copy()

    print(f"\n========== {pollutant} ({window_hours}H) ==========")

    for station_id, group in subset.groupby("station_id"):

        # Aggregate duplicate timestamps before reindexing.
        hourly = (
            group.groupby("collected_at")["value"]
            .mean()
            .sort_index()
        )

        if hourly.empty:
            continue

        full_index = pd.date_range(
            start=hourly.index.min().floor("h"),
            end=hourly.index.max().floor("h"),
            freq="h",
        )

        hourly.index = hourly.index.floor("h")

        hourly = hourly.groupby(level=0).mean()
        hourly = hourly.reindex(full_index)

        rolling = hourly.rolling(
            window=window_hours,
            min_periods=minimum,
        )

        rolling_mean = rolling.mean()
        rolling_count = rolling.count()

        valid_windows = (
            rolling_count >= minimum
        ) & rolling_mean.notna()

        valid_count = int(valid_windows.sum())

        valid_days = int(
            rolling_mean[valid_windows].index.normalize().nunique()
        )

        possible_windows = max(
    0,
    len(full_index) - minimum + 1
)

        result = {
            "pollutant": pollutant,
            "station_id": station_id,
            "valid_windows": valid_count,
            "possible_windows": possible_windows,
            "valid_window_percent": round(
                valid_count / possible_windows * 100, 2
            ) if possible_windows else 0,
            "days_with_valid_windows": valid_days,
        }

        all_results.append(result)

        print(
            f"{station_id}: "
            f"{valid_count}/{possible_windows} valid windows, "
            f"{valid_days} days"
        )

results = pd.DataFrame(all_results)

print("\n========== OVERALL SUMMARY ==========")

if not results.empty:

    print(
        results.groupby("pollutant")[
            [
                "valid_windows",
                "possible_windows",
                "days_with_valid_windows",
            ]
        ].sum().to_string()
    )

    print("\nLowest coverage station-pollutant combinations:")

    print(
        results.sort_values("valid_window_percent")
        .head(15)
        .to_string(index=False)
    )

    output_file = (
        BASE_DIR / "data" / "aqi" / "processed"
        / "pan_india_pilot"
        / "aqi_rolling_window_audit.csv"
    )

    results.to_csv(output_file, index=False)

    print("\nAudit report saved:")
    print(output_file)

print("\nRolling window audit completed.")
