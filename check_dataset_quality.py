
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

city_files = [
    "bengaluru.csv",
    "bombay.csv",
    "delhi.csv",
    "hyderabad.csv",
    "jaipur.csv",
    "kanpur.csv",
    "nagpur.csv",
    "pune.csv"
]

for filename in city_files:
    path = folder / filename

    print("\n" + "=" * 65)
    print("DATASET:", filename)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    df["date_time"] = pd.to_datetime(
        df["date_time"], errors="coerce"
    )

    print("Total records:", len(df))
    print("Start date:", df["date_time"].min())
    print("End date:", df["date_time"].max())

    print("Invalid timestamps:", df["date_time"].isna().sum())
    print("Duplicate timestamps:", df["date_time"].duplicated().sum())

    valid_dates = df["date_time"].dropna().sort_values()
    gaps = valid_dates.diff()

    print("Most common time interval:")
    print(gaps.value_counts().head(5))

    print("\nRainfall statistics (precipMM):")
    print(df["precipMM"].describe())

    print(
        "Percentage of zero-rainfall records:",
        round((df["precipMM"] == 0).mean() * 100, 2),
        "%"
    )

    print(
        "Records with rainfall above 0.1 mm:",
        (df["precipMM"] > 0.1).sum()
    )
