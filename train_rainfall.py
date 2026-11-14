import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Loading weather datasets from across India...")

# 1. List all the CSV files present in your project folder, including Kolkata
city_files = [
    "bengaluru.csv", "bombay.csv", "delhi.csv", 
    "hyderabad.csv", "jaipur.csv", "kanpur.csv", 
    "nagpur.csv", "pune.csv", "kolkata.csv"
]

all_data = []

# 2. Read each file and combine them
for file in city_files:
    if os.path.exists(file):
        print(f" -> Reading {file}...")
        temp_df = pd.read_csv(file)
        all_data.append(temp_df)
    else:
        print(f" -> Warning: {file} not found, skipping.")

if not all_data:
    print("Error: No CSV files were found. Please check your folder.")
    exit()

# Merge all the individual city spreadsheets into one giant spreadsheet
df = pd.concat(all_data, ignore_index=True)
print(f"Total combined weather records: {len(df)}")

# 3. Prepare the data for the AI
# Convert precipitation millimeters into a simple 1 (Rain) or 0 (No Rain)
if 'precipMM' in df.columns:
    df['will_rain'] = (df['precipMM'] > 0.1).astype(int)
else:
    print("Error: 'precipMM' column not found in the dataset.")
    exit()

# Select the exact columns the AI will use to learn
# Make sure these match the column names in your Kaggle CSVs exactly
try:
    X = df[["tempC", "humidity", "windspeedKmph", "pressure"]]
    y = df["will_rain"]
except KeyError as e:
    print(f"Error: Missing expected column in dataset: {e}")
    exit()

# Fill any missing/blank rows with average values so the training doesn't crash
X = X.fillna(X.mean())

# 4. Train the Machine Learning model
print("Training AI model on massive combined dataset. This might take a minute...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the trained model to your 'models' folder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rainfall_ml_model.pkl")

print("Success! Nationwide AI Model trained and saved to models/rainfall_ml_model.pkl")