import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest Disaster & Risk Classification Model...")
os.makedirs("models", exist_ok=True)

# Features: [rain_prob, wind_speed, pressure_drop, seismic_zone]
# Classes: 0 = Normal, 1 = Flood Risk, 2 = Earthquake Tectonic Warning, 3 = Wildfire Thermal Alert
data = {
    "rain_prob": [10, 85, 95, 20, 15, 90, 5, 25, 88, 10, 30, 45, 70, 12, 18, 60],
    "wind_speed": [8, 45, 60, 12, 10, 55, 25, 8, 40, 14, 15, 22, 38, 28, 11, 18],
    "pressure_drop": [2, 25, 35, 1, 3, 30, 2, 1, 28, 4, 5, 8, 14, 3, 2, 6],
    "seismic_zone": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "disaster_code": [0, 1, 1, 2, 3, 1, 2, 2, 1, 0, 0, 0, 1, 3, 0, 0],
}

df = pd.DataFrame(data)
X = df[["rain_prob", "wind_speed", "pressure_drop", "seismic_zone"]].values
y = df["disaster_code"].values

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/disaster_rf_model.pkl")
print("Success! models/disaster_rf_model.pkl saved.")