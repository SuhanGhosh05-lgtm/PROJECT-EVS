
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "models" / "experimental"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "rainfall_training_data.csv")
df["date_time"] = pd.to_datetime(df["date_time"])

features = [
    "tempC", "humidity", "pressure",
    "windspeedKmph", "WindGustKmph",
    "cloudcover", "visibility", "DewPointC",
    "precipMM", "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]

target = "will_rain_next_6h"

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(
    subset=features + ["city", "date_time", target]
).copy()

df[target] = df[target].astype(int)

X = pd.get_dummies(
    df[features + ["city"]],
    columns=["city"],
    dtype=np.int8
)

feature_columns = X.columns.tolist()
y = df[target]

# Chronological 70 / 15 / 15 split.
dates = np.sort(df["date_time"].unique())

cut1 = pd.Timestamp(dates[int(len(dates) * 0.70)])
cut2 = pd.Timestamp(dates[int(len(dates) * 0.85)])

train_mask = df["date_time"] < cut1 - pd.Timedelta(hours=6)

val_mask = (
    (df["date_time"] >= cut1) &
    (df["date_time"] < cut2 - pd.Timedelta(hours=6))
)

test_mask = df["date_time"] >= cut2

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Select threshold using validation data only.
val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.20, 0.81, 0.01)

threshold_results = []

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)

    score = f1_score(
        y_val, pred, zero_division=0
    )

    threshold_results.append((score, threshold))

best_f1, best_threshold = max(threshold_results)

print("\nSelected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_f1, 4))

# Evaluate only once on final test data.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

metrics = {
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_pred)),
    "precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "pr_auc": float(average_precision_score(y_test, test_prob)),
    "test_records": int(len(X_test)),
    "test_start": str(df.loc[test_mask, "date_time"].min()),
    "test_end": str(df.loc[test_mask, "date_time"].max())
}

print("\nFINAL TEST RESULTS")
print("=" * 45)

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))

# City-wise final test results.
print("\nCITY-WISE FINAL TEST RESULTS")

test_cities = df.loc[test_mask, "city"]

for city in sorted(test_cities.unique()):
    mask = test_cities == city

    actual = y_test.loc[mask]
    predicted = pd.Series(
        test_pred, index=y_test.index
    ).loc[mask]

    print(
        city,
        "| Precision:",
        round(precision_score(actual, predicted, zero_division=0), 3),
        "| Recall:",
        round(recall_score(actual, predicted, zero_division=0), 3),
        "| F1:",
        round(f1_score(actual, predicted, zero_division=0), 3)
    )

# Save separate versioned files.
joblib.dump(model, OUT / "rainfall_6h_hgb_v2.joblib")

with open(OUT / "rainfall_6h_features_v2.json", "w") as f:
    json.dump(feature_columns, f, indent=4)

with open(OUT / "rainfall_6h_metrics_v2.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nV2 model and evaluation files saved.")
