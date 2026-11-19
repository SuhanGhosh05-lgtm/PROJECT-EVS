
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "rainfall_training_data.csv"

MODEL_DIR = BASE_DIR / "models" / "experimental"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "rainfall_6h_hgb.joblib"
METRICS_PATH = MODEL_DIR / "rainfall_6h_metrics.json"
FEATURES_PATH = MODEL_DIR / "rainfall_6h_features.json"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading prepared rainfall dataset...")

df = pd.read_csv(DATA_PATH)

df["date_time"] = pd.to_datetime(df["date_time"])

print("Total records:", len(df))

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

numeric_features = [
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

target = "will_rain_next_6h"

required = numeric_features + [
    "city",
    "date_time",
    target
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert numeric columns and handle invalid values.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=numeric_features + ["city", "date_time", target]
).copy()

df[target] = df[target].astype(int)

# Encode city as one-hot features.
X = pd.get_dummies(
    df[numeric_features + ["city"]],
    columns=["city"],
    dtype=np.int8
)

y = df[target]

feature_columns = X.columns.tolist()

# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------

unique_dates = np.sort(df["date_time"].unique())

split_index = int(len(unique_dates) * 0.80)
cutoff = pd.Timestamp(unique_dates[split_index])

# Purge six hours before the test boundary.
train_mask = df["date_time"] < cutoff - pd.Timedelta(hours=6)
test_mask = df["date_time"] >= cutoff

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("\nTraining period:")
print(df.loc[train_mask, "date_time"].min())
print("to", df.loc[train_mask, "date_time"].max())

print("\nTesting period:")
print(df.loc[test_mask, "date_time"].min())
print("to", df.loc[test_mask, "date_time"].max())

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
    "pr_auc": float(
        average_precision_score(y_test, probabilities)
    ),
    "test_rainfall_prevalence": float(y_test.mean()),
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "test_start": str(df.loc[test_mask, "date_time"].min()),
    "test_end": str(df.loc[test_mask, "date_time"].max())
}

# Baseline: always predict no rain.
baseline_accuracy = float((y_test == 0).mean())

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nAlways-no-rain baseline accuracy:", baseline_accuracy)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Rain", "Rain"],
        zero_division=0
    )
)

# --------------------------------------------------
# 7. SAVE MODEL AND METADATA
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

with open(FEATURES_PATH, "w") as file:
    json.dump(feature_columns, file, indent=4)

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("\n" + "=" * 55)
print("MODEL SAVED SUCCESSFULLY")

print("Model:", MODEL_PATH)
print("Features:", FEATURES_PATH)
print("Metrics:", METRICS_PATH)
