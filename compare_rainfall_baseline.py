


import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "experimental"

df = pd.read_csv(BASE / "rainfall_training_data.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

model = joblib.load(
    MODEL_DIR / "rainfall_6h_hgb_v2.joblib"
)

with open(MODEL_DIR / "rainfall_6h_features_v2.json") as f:
    features = json.load(f)

with open(MODEL_DIR / "rainfall_6h_metrics_v2.json") as f:
    metrics = json.load(f)

numeric = [
    "tempC", "humidity", "pressure",
    "windspeedKmph", "WindGustKmph",
    "cloudcover", "visibility", "DewPointC",
    "precipMM", "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]

for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=numeric + ["city", "date_time", "will_rain_next_6h"]
).copy()

df["will_rain_next_6h"] = df["will_rain_next_6h"].astype(int)

# Recreate the original V2 chronological split.
dates = np.sort(df["date_time"].unique())

cut1 = pd.Timestamp(dates[int(len(dates) * 0.70)])
cut2 = pd.Timestamp(dates[int(len(dates) * 0.85)])

test_mask = df["date_time"] >= cut2
test = df.loc[test_mask].copy()

# Recreate the exact feature schema.
X = pd.get_dummies(
    df[numeric + ["city"]],
    columns=["city"],
    dtype=np.int8
)

X = X.reindex(columns=features, fill_value=0)

X_test = X.loc[test_mask]

actual = test["will_rain_next_6h"]

# Model prediction.
probability = model.predict_proba(X_test)[:, 1]

ml_prediction = (
    probability >= metrics["threshold"]
).astype(int)

# Baseline 1: current-hour rainfall persistence.
persistence_prediction = (
    test["precipMM"] > 0.1
).astype(int)

# Baseline 2: always predict no rain.
majority_prediction = np.zeros(len(test), dtype=int)


def evaluate(name, prediction):

    return {
        "model": name,
        "accuracy": round(
            accuracy_score(actual, prediction), 4
        ),
        "precision": round(
            precision_score(actual, prediction, zero_division=0), 4
        ),
        "recall": round(
            recall_score(actual, prediction, zero_division=0), 4
        ),
        "f1": round(
            f1_score(actual, prediction, zero_division=0), 4
        )
    }


results = [
    evaluate("HistGradientBoosting V2", ml_prediction),
    evaluate("Current-hour persistence", persistence_prediction),
    evaluate("Always predict no rain", majority_prediction)
]

result_df = pd.DataFrame(results)

print("\nRAINFALL MODEL VS BASELINES")
print("=" * 65)
print(result_df.to_string(index=False))

output = BASE / "reports" / "rainfall_v2_validation"
output.mkdir(parents=True, exist_ok=True)

result_df.to_csv(
    output / "baseline_comparison.csv",
    index=False
)

print("\nSaved baseline_comparison.csv")
