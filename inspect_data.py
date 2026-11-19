
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

files = list(folder.glob("*.csv"))

for file in files:
    print("\n" + "=" * 60)
    print("DATASET:", file.name)

    try:
        df = pd.read_csv(file, nrows=10000)

        print("Rows inspected:", len(df))
        print("Columns:", df.columns.tolist())

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        print("\nFirst 3 records:")
        print(df.head(3).to_string(index=False))

    except Exception as error:
        print("Could not inspect:", error)
