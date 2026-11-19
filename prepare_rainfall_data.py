
import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(__file__).resolve().parent

city_files = {
    "Bengaluru": "bengaluru.csv",
    "Mumbai": "bombay.csv",
    "Delhi": "delhi.csv",
    "Hyderabad": "hyderabad.csv",
    "Jaipur": "jaipur.csv",
    "Kanpur": "kanpur.csv",
    "Nagpur": "nagpur.csv",
    "Pune": "pune.csv"
}

all_data = []

print("Preparing multi-city rainfall dataset...")

for city, filename in city_files.items():

    path = folder / filename

    print(f"\nProcessing {city}...")

    df = pd.read_csv(path)

    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    df = df.dropna(subset=["date_time"])
    df = df.sort_values("date_time")
    df = df.drop_duplicates(subset=["date_time"])

    df["city"] = city

    # Verify the next six hours exist continuously.
    valid_future = pd.Series(True, index=df.index)

    for hours in range(1, 7):
        future_time = df["date_time"].shift(-hours)

        valid_future &= (
            future_time == df["date_time"] +
            pd.Timedelta(hours=hours)
        )

    # Calculate rainfall over the next six hours.
    future_rain = pd.Series(0.0, index=df.index)

    for hours in range(1, 7):
        future_rain += df["precipMM"].shift(-hours)

    df["future_rain_6h_mm"] = future_rain

    # Binary prediction target.
    df["will_rain_next_6h"] = (
        df["future_rain_6h_mm"] > 0.1
    ).astype(int)

    # Calendar features.
    df["hour"] = df["date_time"].dt.hour
    df["month"] = df["date_time"].dt.month
    df["day_of_year"] = df["date_time"].dt.dayofyear

    # Cyclical time features.
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Only use information available at prediction time.
    feature_columns = [
        "tempC",
        "humidity",
        "pressure",
        "windspeedKmph",
        "WindGustKmph",
        "cloudcover",
        "visibility",
        "DewPointC",
        "precipMM",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
    ]

    df = df.loc[
        valid_future,
        feature_columns + [
            "date_time",
            "city",
            "future_rain_6h_mm",
            "will_rain_next_6h"
        ]
    ].copy()

    all_data.append(df)

    print("Usable records:", len(df))
    print(
        "Rain events:",
        int(df["will_rain_next_6h"].sum())
    )

# Combine all cities.
combined = pd.concat(all_data, ignore_index=True)

# Save prepared data without modifying original files.
output_path = folder / "rainfall_training_data.csv"

combined.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print("DATASET PREPARATION COMPLETE")

print("Total records:", len(combined))

print("\nRainfall target distribution:")
print(
    combined["will_rain_next_6h"].value_counts()
)

print("\nRainfall percentage:")
print(
    combined["will_rain_next_6h"].value_counts(
        normalize=True
    ) * 100
)

print("\nRecords per city:")
print(combined["city"].value_counts())

print("\nSaved to:", output_path)
