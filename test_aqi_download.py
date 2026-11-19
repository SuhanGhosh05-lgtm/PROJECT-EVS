
import os
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv
from io import StringIO


# Load API key
load_dotenv()

API_KEY = os.getenv("XKDR_AQI_API_KEY")

if not API_KEY:
    raise ValueError("XKDR API key missing from .env")


# API configuration
URL = "https://airquality.xkdr.org/v1/measurements"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

PARAMS = {
    "station": "site_5238",
    "parameter": [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "Ozone",
        "NH3"
    ],
    "start": "2024-11-01",
    "end": "2024-11-07",
    "agg": "raw",
    "format": "csv"
}


# Download sample
print("Downloading Kolkata pollutant data...")

response = requests.get(
    URL,
    headers=HEADERS,
    params=PARAMS,
    timeout=90
)

print("HTTP status:", response.status_code)
response.raise_for_status()

print("Rows reported:", response.headers.get("X-Row-Count"))
print("Truncated:", response.headers.get("X-Truncated"))


# Convert CSV response to DataFrame
df = pd.read_csv(StringIO(response.text))

print("\n========== DATA SAMPLE ==========")
print(df.head(10).to_string(index=False))

print("\n========== DATASET INFORMATION ==========")
print("Total rows:", len(df))
print("Columns:", list(df.columns))

if "parameter_name" in df.columns:
    print("\nPollutant counts:")
    print(df["parameter_name"].value_counts())

if "value" in df.columns:
    print("\nMissing values:", df["value"].isna().sum())

    print("\nPollutant statistics:")
    if "parameter_name" in df.columns:
        print(
            df.groupby("parameter_name")["value"]
            .agg(["count", "mean", "min", "max"])
            .round(3)
        )


# Save sample
output_dir = Path("data/aqi/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "kolkata_ballygunge_sample.csv"

df.to_csv(output_file, index=False)

print("\nSample saved to:", output_file.resolve())
print("\nDOWNLOAD TEST COMPLETED")
