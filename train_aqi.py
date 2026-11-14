import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Training Random Forest Regressor AQI Model...")

os.makedirs("models", exist_ok=True)

# Training dataset with multi-day AQI history + humidity + wind speed
data = {
    "day_1_aqi": [35, 60, 110, 180, 250, 310, 340, 420, 90, 210],
    "day_3_aqi": [38, 68, 115, 195, 265, 325, 346, 435, 82, 202],
    "day_7_aqi": [50, 90, 140, 240, 320, 348, 350, 480, 65, 175],
    "avg_humidity": [45, 55, 60, 75, 85, 90, 88, 40, 65, 80],
    "avg_wind_speed": [12, 10, 8, 5, 3, 2, 2, 15, 9, 4],
    "next_day_aqi": [52, 95, 145, 250, 330, 352, 351, 490, 62, 170]
}

df = pd.DataFrame(data)

# Features: Day 1, 3, 7 AQI trends + Humidity + Wind Speed
X = df[["day_1_aqi", "day_3_aqi", "day_7_aqi", "avg_humidity", "avg_wind_speed"]]
y = df["next_day_aqi"]

# Random Forest Regressor with 100 decision trees
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/aqi_timeseries_model.pkl")
print("Success! Random Forest AQI model successfully trained and saved to models/aqi_timeseries_model.pkl")