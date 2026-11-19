
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "experimental"

df = pd.read_csv(BASE / "rainfall_training_data.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

model = joblib.load(MODEL_DIR / "rainfall_6h_hgb.joblib")

with open(MODEL_DIR / "rainfall_6h_features.json") as f:
    feature_columns = json.load(f)

numeric_features = [
    "tempC", "humidity", "pressure",
    "windspeedKmph", "WindGustKmph",
    "cloudcover", "visibility", "DewPointC",
    "precipMM", "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]

target = "will_rain_next_6h"

# Recreate the original chronological split.
unique_dates = np.sort(df["date_time"].unique())
cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])

train_mask = df["date_time"] < cutoff - pd.Timedelta(hours=6)
test_mask = df["date_time"] >= cutoff

test = df.loc[test_mask].copy()

# Recreate exactly the same model features.
X = pd.get_dummies(
    df[numeric_features + ["city"]],
    columns=["city"],
    dtype=np.int8
)

X = X.reindex(columns=feature_columns, fill_value=0)

X_test = X.loc[test_mask]

test["probability"] = model.predict_proba(X_test)[:, 1]

test["prediction"] = (
    test["probability"] >= 0.5
).astype(int)

print("\nCITY-WISE MODEL PERFORMANCE")
print("=" * 75)

results = []

for city, group in test.groupby("city"):

    y_true = group[target]
    y_pred = group["prediction"]
    y_prob = group["probability"]

    result = {
        "city": city,
        "records": len(group),
        "rain_prevalence": round(y_true.mean() * 100, 2),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4)
    }

    results.append(result)

print(pd.DataFrame(results).to_string(index=False))

# Compare against current-hour rainfall persistence.
baseline = (
    test["precipMM"] > 0.1
).astype(int)

print("\nCURRENT-HOUR RAINFALL BASELINE")
print("=" * 55)

print("Accuracy:", round(
    accuracy_score(test[target], baseline), 4
))

print("Precision:", round(
    precision_score(test[target], baseline, zero_division=0), 4
))

print("Recall:", round(
    recall_score(test[target], baseline, zero_division=0), 4
))

print("F1:", round(
    f1_score(test[target], baseline, zero_division=0), 4
))

# Explore thresholds without changing the saved model.
print("\nTHRESHOLD COMPARISON")
print("=" * 55)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    predictions = (
        test["probability"] >= threshold
    ).astype(int)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision_score(test[target], predictions, zero_division=0):.3f} | "
        f"Recall: {recall_score(test[target], predictions, zero_division=0):.3f} | "
        f"F1: {f1_score(test[target], predictions, zero_division=0):.3f}"
    )

print("\nEvaluation completed. Original model unchanged.")
