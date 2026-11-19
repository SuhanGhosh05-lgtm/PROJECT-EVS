

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "rainfall_training_data.csv"
MODEL_PATH = ROOT / "models" / "experimental" / "rainfall_6h_hgb_v2.joblib"
FEATURE_PATH = ROOT / "models" / "experimental" / "rainfall_6h_features_v2.json"
METRICS_PATH = ROOT / "models" / "experimental" / "rainfall_6h_metrics_v2.json"

OUTPUT_DIR = ROOT / "reports" / "rainfall_v2_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "tempC", "humidity", "pressure", "windspeedKmph",
    "WindGustKmph", "cloudcover", "visibility", "DewPointC",
    "precipMM", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

TARGET = "will_rain_next_6h"


# ============================================================
# 2. LOAD MODEL AND FEATURE CONFIGURATION
# ============================================================

print("\nLoading model and configuration...")

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as file:
    feature_data = json.load(file)

# Supports either a plain list or a dictionary containing features.
if isinstance(feature_data, list):
    FEATURES = feature_data
elif isinstance(feature_data, dict):
    FEATURES = feature_data.get("features")

    if FEATURES is None:
        raise ValueError("Could not find 'features' in feature JSON.")
else:
    raise ValueError("Unexpected feature JSON format.")

if len(FEATURES) != 21:
    raise ValueError(f"Expected 21 features, found {len(FEATURES)}.")

print("Expected model features:", len(FEATURES))


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

print("\nLoading rainfall dataset...")

df = pd.read_csv(DATA_PATH)

required = ["date_time", "city", TARGET] + BASE_FEATURES

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required dataset columns: {missing}")

df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

for column in BASE_FEATURES + [TARGET]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=required).copy()

df[TARGET] = df[TARGET].astype(int)

df = df[df[TARGET].isin([0, 1])].copy()

df = df.sort_values(["date_time", "city"]).reset_index(drop=True)

print("Clean records:", len(df))
print("Cities:", sorted(df["city"].unique()))


# ============================================================
# 4. RECREATE THE ORIGINAL CHRONOLOGICAL TEST SPLIT
# ============================================================

print("\nRecreating original 70/15/15 chronological split...")

unique_times = np.sort(df["date_time"].unique())

cut1 = unique_times[int(len(unique_times) * 0.70)]
cut2 = unique_times[int(len(unique_times) * 0.85)]

cut1 = pd.Timestamp(cut1)
cut2 = pd.Timestamp(cut2)

test_mask = df["date_time"] >= cut2

test_df = df.loc[test_mask].copy()

print("Test records:", len(test_df))
print("Test start:", test_df["date_time"].min())
print("Test end:", test_df["date_time"].max())


# ============================================================
# 5. REBUILD EXACT MODEL FEATURES
# ============================================================

X = test_df[BASE_FEATURES].copy()

city_dummies = pd.get_dummies(
    test_df["city"],
    prefix="city",
    dtype=int
)

X = pd.concat(
    [X.reset_index(drop=True), city_dummies.reset_index(drop=True)],
    axis=1
)

X = X.reindex(columns=FEATURES, fill_value=0)

y_true = test_df[TARGET].to_numpy()

if X.shape[1] != len(FEATURES):
    raise ValueError("Feature count mismatch.")

if hasattr(model, "n_features_in_"):
    if X.shape[1] != model.n_features_in_:
        raise ValueError("Model and dataset feature counts do not match.")

print("Feature matrix shape:", X.shape)


# ============================================================
# 6. GET MODEL SCORES
# ============================================================

print("\nGenerating rainfall scores...")

y_score = model.predict_proba(X)[:, 1]

y_score = np.clip(y_score, 0, 1)

print("Minimum score:", round(float(y_score.min()), 4))
print("Maximum score:", round(float(y_score.max()), 4))
print("Mean score:", round(float(y_score.mean()), 4))
print("Actual rain frequency:", round(float(y_true.mean()), 4))


# ============================================================
# 7. OVERALL CALIBRATION METRICS
# ============================================================

brier = brier_score_loss(y_true, y_score)

logloss = log_loss(y_true, y_score, labels=[0, 1])

overall = {
    "test_records": int(len(y_true)),
    "actual_rain_frequency": float(y_true.mean()),
    "mean_predicted_score": float(y_score.mean()),
    "brier_score": float(brier),
    "log_loss": float(logloss),
}

print("\n" + "=" * 65)
print("OVERALL CALIBRATION RESULTS")
print("=" * 65)

for key, value in overall.items():
    print(f"{key:30s}: {value:.6f}" if isinstance(value, float)
          else f"{key:30s}: {value}")

with open(OUTPUT_DIR / "overall_calibration.json", "w") as file:
    json.dump(overall, file, indent=4)


# ============================================================
# 8. CALIBRATION TABLE
# ============================================================

print("\nCreating reliability table...")

probability_bins = np.linspace(0, 1, 11)

bin_ids = pd.cut(
    y_score,
    bins=probability_bins,
    include_lowest=True,
    labels=False
)

calibration_df = pd.DataFrame({
    "predicted_score": y_score,
    "actual_rain": y_true,
    "bin": bin_ids
})

reliability = (
    calibration_df.groupby("bin", observed=False)
    .agg(
        records=("actual_rain", "size"),
        mean_predicted_score=("predicted_score", "mean"),
        actual_rain_frequency=("actual_rain", "mean")
    )
    .reset_index()
)

reliability.to_csv(
    OUTPUT_DIR / "reliability_table.csv",
    index=False
)

print(reliability.to_string(index=False))


# ============================================================
# 9. CITY-WISE CALIBRATION
# ============================================================

print("\nCalculating city-wise calibration...")

city_results = []

for city in sorted(test_df["city"].unique()):

    mask = test_df["city"].to_numpy() == city

    city_y = y_true[mask]
    city_scores = y_score[mask]

    result = {
        "city": city,
        "records": int(len(city_y)),
        "actual_rain_frequency": float(city_y.mean()),
        "mean_predicted_score": float(city_scores.mean()),
        "brier_score": float(
            brier_score_loss(city_y, city_scores)
        ),
        "log_loss": float(
            log_loss(city_y, city_scores, labels=[0, 1])
        )
    }

    city_results.append(result)

city_df = pd.DataFrame(city_results)

city_df.to_csv(
    OUTPUT_DIR / "city_calibration.csv",
    index=False
)

print(city_df.to_string(index=False))


# ============================================================
# 10. RELIABILITY DIAGRAM
# ============================================================

print("\nGenerating calibration plot...")

fraction_positive, mean_predicted = calibration_curve(
    y_true,
    y_score,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Rainfall model"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted score")
plt.ylabel("Actual rainfall frequency")
plt.title("Rainfall V2 - Reliability Diagram")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "reliability_diagram.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 65)
print("CALIBRATION VALIDATION COMPLETED")
print("=" * 65)

print("Output folder:", OUTPUT_DIR)
print("Files generated:")
print("- overall_calibration.json")
print("- reliability_table.csv")
print("- city_calibration.csv")
print("- reliability_diagram.png")

print("\nIMPORTANT:")
print("These scores are not automatically calibrated probabilities.")
print("No model training or modification was performed.")
