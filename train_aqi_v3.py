
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "data" / "aqi" / "processed"
    / "multi_month" / "aqi_oct_nov_daily_v3.csv"
)

MODEL_DIR = BASE_DIR / "models" / "aqi"
REPORT_DIR = BASE_DIR / "reports" / "aqi"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "daily_max_partial_aqi"
GROUP = "station_id"


def evaluate(name, actual, predicted):
    return {
        "model": name,
        "MAE": round(
            mean_absolute_error(actual, predicted), 3
        ),
        "RMSE": round(
            np.sqrt(mean_squared_error(actual, predicted)), 3
        ),
        "R2": round(
            r2_score(actual, predicted), 4
        ),
    }


print("\n========== AQI V3 MULTI-MODEL TRAINING ==========")

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

df = pd.read_csv(DATA_FILE)

required = [
    "date",
    TARGET,
    GROUP,
    "dominant_pollutant",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

df = df.dropna(
    subset=["date", TARGET, GROUP, "dominant_pollutant"]
).copy()

df = df.sort_values([GROUP, "date"]).reset_index(drop=True)

# -----------------------------------------------------
# One-day-ahead forecasting features
# -----------------------------------------------------

grouped = df.groupby(GROUP, sort=False)

df["lag_1"] = grouped[TARGET].shift(0)
df["lag_2"] = grouped[TARGET].shift(1)
df["lag_3"] = grouped[TARGET].shift(2)

df["next_target"] = grouped[TARGET].shift(-1)
df["next_date"] = grouped["date"].shift(-1)

df["date_gap"] = (
    df["next_date"] - df["date"]
).dt.days

# Do not train across missing calendar days.
df = df[df["date_gap"] == 1].copy()

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month

feature_columns = [
    "lag_1",
    "lag_2",
    "lag_3",
    "day_of_week",
    "day_of_month",
    "month",
]

for col in [
    "available_subindices",
    "valid_hourly_aqi_count",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )
        feature_columns.append(col)

if "particulate_available" in df.columns:
    df["particulate_available"] = (
        df["particulate_available"].astype(int)
    )
    feature_columns.append("particulate_available")

categorical_columns = [
    GROUP,
    "dominant_pollutant",
]

# Require all lag values and target to be present.
df = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "next_target",
        "next_date",
    ]
).copy()

X = pd.get_dummies(
    df[feature_columns + categorical_columns],
    columns=categorical_columns,
    dtype=int,
)

y = df["next_target"]

# -----------------------------------------------------
# Chronological train/test split
# -----------------------------------------------------

unique_dates = sorted(df["next_date"].unique())

split_index = int(len(unique_dates) * 0.70)

if split_index < 2 or split_index >= len(unique_dates):
    raise ValueError("Insufficient dates for train/test split.")

split_date = unique_dates[split_index]

train_mask = df["next_date"] < split_date
test_mask = df["next_date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Empty training or testing partition.")

print("\n========== DATASET ==========")
print("Input file:", DATA_FILE)
print("Daily records:", len(pd.read_csv(DATA_FILE)))
print("Prediction samples:", len(df))
print("Stations:", df[GROUP].nunique())
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Test starts:", split_date)

# -----------------------------------------------------
# Baseline
# -----------------------------------------------------

baseline_predictions = df.loc[test_mask, "lag_1"].to_numpy()

results = [
    evaluate(
        "Persistence baseline",
        y_test,
        baseline_predictions,
    )
]

# -----------------------------------------------------
# Train multiple models
# -----------------------------------------------------

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),
}

trained_models = {}
predictions_by_model = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.clip(predictions, 0, 500)

    trained_models[name] = model
    predictions_by_model[name] = predictions

    results.append(
        evaluate(name, y_test, predictions)
    )

# -----------------------------------------------------
# Compare models
# -----------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

baseline_mae = results[0]["MAE"]
baseline_rmse = results[0]["RMSE"]

model_results = results_df[
    results_df["model"] != "Persistence baseline"
].copy()

# Select the lowest-MAE model.
best_row = model_results.sort_values(
    ["MAE", "RMSE"]
).iloc[0]

best_name = best_row["model"]
best_model = trained_models[best_name]

best_predictions = predictions_by_model[best_name]

deployment_ready = (
    best_row["MAE"] < baseline_mae
    and best_row["RMSE"] < baseline_rmse
)

print("\n========== BEST MODEL ==========")
print("Selected:", best_name)
print("MAE:", best_row["MAE"])
print("RMSE:", best_row["RMSE"])
print("R2:", best_row["R2"])

print("\nBeats persistence on MAE:",
      best_row["MAE"] < baseline_mae)

print("Beats persistence on RMSE:",
      best_row["RMSE"] < baseline_rmse)

print("Deployment ready:", deployment_ready)

# -----------------------------------------------------
# Save experiment bundle
# -----------------------------------------------------

bundle = {
    "model": best_model,
    "model_name": best_name,
    "feature_columns": X.columns.tolist(),
    "target": TARGET,
    "group_column": GROUP,
    "prediction_horizon": "next_day",
    "deployment_ready": bool(deployment_ready),
    "experimental": not deployment_ready,
    "training_data": "October-November 2024",
    "baseline_mae": baseline_mae,
    "baseline_rmse": baseline_rmse,
    "model_mae": float(best_row["MAE"]),
    "model_rmse": float(best_row["RMSE"]),
}

model_path = MODEL_DIR / "aqi_best_v3.joblib"

joblib.dump(bundle, model_path)

# -----------------------------------------------------
# Save predictions and metrics
# -----------------------------------------------------

prediction_output = df.loc[
    test_mask,
    [GROUP, "date", "next_date"]
].copy()

prediction_output["actual_aqi"] = y_test.to_numpy()
prediction_output["persistence_prediction"] = (
    baseline_predictions
)

for name, predictions in predictions_by_model.items():
    safe_name = name.lower().replace(" ", "_")
    prediction_output[f"{safe_name}_prediction"] = predictions

prediction_output.to_csv(
    REPORT_DIR / "aqi_v3_test_predictions.csv",
    index=False,
)

with open(
    REPORT_DIR / "aqi_v3_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "results": results,
            "best_model": best_name,
            "deployment_ready": bool(deployment_ready),
            "test_start_date": str(split_date),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
        },
        f,
        indent=4,
    )

print("\n========== SAVED ==========")
print("Model:", model_path)
print("Predictions:", REPORT_DIR / "aqi_v3_test_predictions.csv")
print("Metrics:", REPORT_DIR / "aqi_v3_metrics.json")

if deployment_ready:
    print("\nAQI MODEL PASSED THE BASELINE COMPARISON.")
else:
    print(
        "\nAQI MODEL DID NOT BEAT BOTH BASELINE METRICS."
        "\nKeep it experimental; do not deploy it yet."
    )

print("\nAQI V3 TRAINING COMPLETED.")
