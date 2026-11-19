
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =====================================================
# 1. PATHS AND FILES
# =====================================================

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models" / "experimental"

REPORT_DIR = BASE / "reports" / "rainfall_v2_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = BASE / "rainfall_training_data.csv"
MODEL_PATH = MODEL_DIR / "rainfall_6h_hgb_v2.joblib"
FEATURE_PATH = MODEL_DIR / "rainfall_6h_features_v2.json"
METRICS_PATH = MODEL_DIR / "rainfall_6h_metrics_v2.json"

print("=" * 65)
print("RAINFALL V2 MODEL VALIDATION")
print("=" * 65)

# =====================================================
# 2. LOAD DATA AND MODEL
# =====================================================

print("\nLoading dataset and model...")

df = pd.read_csv(DATA_PATH)
df["date_time"] = pd.to_datetime(df["date_time"])

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH) as f:
    feature_columns = json.load(f)

with open(METRICS_PATH) as f:
    saved_metrics = json.load(f)

numeric_features = [
    "tempC", "humidity", "pressure",
    "windspeedKmph", "WindGustKmph",
    "cloudcover", "visibility", "DewPointC",
    "precipMM", "hour_sin", "hour_cos",
    "month_sin", "month_cos"
]

target = "will_rain_next_6h"

# Reproduce the same cleaning performed during training.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=numeric_features + ["city", "date_time", target]
).copy()

df[target] = df[target].astype(int)

# =====================================================
# 3. RECREATE EXACT V2 FEATURES
# =====================================================

X = pd.get_dummies(
    df[numeric_features + ["city"]],
    columns=["city"],
    dtype=np.int8
)

X = X.reindex(
    columns=feature_columns,
    fill_value=0
)

y = df[target]

print("Dataset records:", len(df))
print("Model features:", len(feature_columns))
print("Model expected features:", model.n_features_in_)

if len(feature_columns) != model.n_features_in_:
    raise ValueError("Feature count does not match the trained model.")

# =====================================================
# 4. RECREATE ORIGINAL CHRONOLOGICAL SPLIT
# =====================================================

dates = np.sort(df["date_time"].unique())

cut1 = pd.Timestamp(dates[int(len(dates) * 0.70)])
cut2 = pd.Timestamp(dates[int(len(dates) * 0.85)])

train_mask = (
    df["date_time"] < cut1 - pd.Timedelta(hours=6)
)

val_mask = (
    (df["date_time"] >= cut1) &
    (df["date_time"] < cut2 - pd.Timedelta(hours=6))
)

test_mask = df["date_time"] >= cut2

X_train = X.loc[train_mask]
X_val = X.loc[val_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_val = y.loc[val_mask]
y_test = y.loc[test_mask]

test = df.loc[test_mask].copy()

print("\nCHRONOLOGICAL SPLIT")
print("-" * 45)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("Training end:", df.loc[train_mask, "date_time"].max())
print("Validation start:", df.loc[val_mask, "date_time"].min())
print("Validation end:", df.loc[val_mask, "date_time"].max())
print("Testing start:", df.loc[test_mask, "date_time"].min())

# =====================================================
# 5. CHECK SIX-HOUR BOUNDARY LEAKAGE
# =====================================================

print("\nTEMPORAL BOUNDARY CHECK")
print("-" * 45)

train_end = df.loc[train_mask, "date_time"].max()
val_start = df.loc[val_mask, "date_time"].min()

val_end = df.loc[val_mask, "date_time"].max()
test_start = df.loc[test_mask, "date_time"].min()

train_gap_ok = (
    train_end + pd.Timedelta(hours=6) < val_start
)

val_gap_ok = (
    val_end + pd.Timedelta(hours=6) < test_start
)

print("Training -> Validation gap:", train_gap_ok)
print("Validation -> Testing gap:", val_gap_ok)

if not train_gap_ok or not val_gap_ok:
    print("WARNING: Check temporal target boundaries.")

# =====================================================
# 6. VERIFY SAVED TEST METADATA
# =====================================================

print("\nSAVED METADATA COMPARISON")
print("-" * 45)

actual_test_start = str(test["date_time"].min())
actual_test_end = str(test["date_time"].max())

print("Saved test records:", saved_metrics["test_records"])
print("Recreated test records:", len(X_test))

print("Saved test start:", saved_metrics["test_start"])
print("Actual test start:", actual_test_start)

print("Saved test end:", saved_metrics["test_end"])
print("Actual test end:", actual_test_end)

# =====================================================
# 7. RUN V2 MODEL PREDICTIONS
# =====================================================

print("\nGenerating test predictions...")

threshold = float(saved_metrics["threshold"])

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

test["actual"] = y_test.values
test["probability"] = probabilities
test["prediction"] = predictions

# =====================================================
# 8. OVERALL PERFORMANCE
# =====================================================

def calculate_metrics(actual, predicted, probability):
    cm = confusion_matrix(actual, predicted, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    result = {
        "records": int(len(actual)),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(actual, probability)
        ) if len(np.unique(actual)) == 2 else None,
        "pr_auc": float(
            average_precision_score(actual, probability)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    return result


overall = calculate_metrics(
    y_test,
    predictions,
    probabilities
)

print("\nOVERALL V2 TEST PERFORMANCE")
print("=" * 55)

for name, value in overall.items():
    print(f"{name}: {value}")

with open(REPORT_DIR / "overall_metrics.json", "w") as f:
    json.dump(overall, f, indent=4)

# =====================================================
# 9. CITY-WISE PERFORMANCE
# =====================================================

print("\nCITY-WISE PERFORMANCE")
print("=" * 65)

city_results = []

for city, group in test.groupby("city"):

    result = calculate_metrics(
        group["actual"],
        group["prediction"],
        group["probability"]
    )

    result["city"] = city
    result["rain_prevalence_percent"] = round(
        group["actual"].mean() * 100, 2
    )

    city_results.append(result)

city_df = pd.DataFrame(city_results)

print(city_df.to_string(index=False))

city_df.to_csv(
    REPORT_DIR / "city_wise_metrics.csv",
    index=False
)

# =====================================================
# 10. SEASON-WISE PERFORMANCE
# =====================================================

print("\nSEASON-WISE PERFORMANCE")
print("=" * 65)

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Pre-Monsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


test["season"] = test["date_time"].dt.month.map(get_season)

season_results = []

for season, group in test.groupby("season"):

    result = calculate_metrics(
        group["actual"],
        group["prediction"],
        group["probability"]
    )

    result["season"] = season

    season_results.append(result)

season_df = pd.DataFrame(season_results)

print(season_df.to_string(index=False))

season_df.to_csv(
    REPORT_DIR / "season_wise_metrics.csv",
    index=False
)

# =====================================================
# 11. CITY + SEASON PERFORMANCE
# =====================================================

print("\nCITY AND SEASON PERFORMANCE")
print("=" * 65)

city_season_results = []

for (city, season), group in test.groupby(
    ["city", "season"]
):

    result = calculate_metrics(
        group["actual"],
        group["prediction"],
        group["probability"]
    )

    result["city"] = city
    result["season"] = season

    city_season_results.append(result)

city_season_df = pd.DataFrame(city_season_results)

print(city_season_df.to_string(index=False))

city_season_df.to_csv(
    REPORT_DIR / "city_season_metrics.csv",
    index=False
)

# =====================================================
# 12. THRESHOLD COMPARISON
# =====================================================

print("\nTHRESHOLD COMPARISON")
print("=" * 65)

threshold_results = []

thresholds = sorted(
    set(
        [0.20, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50, 0.60, 0.70]
        + [round(threshold, 4)]
    )
)

for value in thresholds:

    pred = (probabilities >= value).astype(int)

    result = calculate_metrics(
        y_test,
        pred,
        probabilities
    )

    result["threshold"] = value
    threshold_results.append(result)

threshold_df = pd.DataFrame(threshold_results)

print(
    threshold_df[
        ["threshold", "precision", "recall", "f1"]
    ].to_string(index=False)
)

threshold_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

# =====================================================
# 13. FALSE POSITIVE / FALSE NEGATIVE ANALYSIS
# =====================================================

print("\nPREDICTION ERROR ANALYSIS")
print("=" * 65)

errors = test[
    test["actual"] != test["prediction"]
].copy()

errors["error_type"] = np.where(
    errors["prediction"] == 1,
    "False Positive",
    "False Negative"
)

error_columns = [
    "date_time",
    "city",
    "season",
    "actual",
    "prediction",
    "probability",
    "future_rain_6h_mm",
    "error_type"
]

errors[error_columns].to_csv(
    REPORT_DIR / "prediction_errors.csv",
    index=False
)

print("Total errors:", len(errors))

print("\nError distribution:")
print(errors["error_type"].value_counts())

print("\nErrors by city:")
print(
    errors.groupby(["city", "error_type"])
    .size()
    .unstack(fill_value=0)
)

# =====================================================
# 14. SAVE FINAL VALIDATION SUMMARY
# =====================================================

summary = {
    "model": "HistGradientBoosting V2",
    "threshold": threshold,
    "dataset_records": len(df),
    "test_records": len(test),
    "test_start": actual_test_start,
    "test_end": actual_test_end,
    "temporal_train_validation_gap_ok": bool(train_gap_ok),
    "temporal_validation_test_gap_ok": bool(val_gap_ok),
    "saved_test_record_count_matches": (
        len(X_test) == saved_metrics["test_records"]
    ),
    "saved_test_dates_match": (
        actual_test_start == saved_metrics["test_start"]
        and actual_test_end == saved_metrics["test_end"]
    ),
    "overall_metrics": overall,
    "total_prediction_errors": len(errors)
}

with open(REPORT_DIR / "validation_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("RAINFALL V2 VALIDATION COMPLETED")
print("=" * 65)

print("\nReports saved inside:")
print(REPORT_DIR)

print("\nOriginal model and training files were not modified.")
