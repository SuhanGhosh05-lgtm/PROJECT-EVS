
from pathlib import Path

import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "monthly"
    / "2024-11"
    / "aqi_november_2024.csv"
)

QUALITY_DIR = (
    BASE_DIR
    / "data"
    / "aqi"
    / "raw"
    / "monthly"
    / "2024-11"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "aqi"
    / "processed"
    / "pan_india_pilot"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "aqi_november_2024_quality_checked.csv"
)

PLATEAU_FILE = QUALITY_DIR / "november_plateau_runs.csv"


# =========================================================
# 2. LOAD ORIGINAL DATA
# =========================================================

print("\n========== AQI QUALITY CONTROL ==========")

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

if not PLATEAU_FILE.exists():
    raise FileNotFoundError(f"Plateau report not found: {PLATEAU_FILE}")

df = pd.read_csv(RAW_FILE)

df["collected_at"] = pd.to_datetime(df["collected_at"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")

print("Original records:", len(df))
print("Original stations:", df["station_id"].nunique())


# =========================================================
# 3. LOAD PLATEAU INVESTIGATION
# =========================================================

plateaus = pd.read_csv(PLATEAU_FILE)

plateaus["start"] = pd.to_datetime(plateaus["start"])
plateaus["end"] = pd.to_datetime(plateaus["end"])

df["plateau_flag"] = False

for _, run in plateaus.iterrows():

    mask = (
        (df["station_id"] == run["station_id"])
        & (df["parameter_name"] == run["parameter_name"])
        & (df["collected_at"] >= run["start"])
        & (df["collected_at"] <= run["end"])
        & (df["value"] == run["value"])
    )

    df.loc[mask, "plateau_flag"] = True

print("\nPlateau-flagged records:", int(df["plateau_flag"].sum()))


# =========================================================
# 4. FLAG EXTREME PARTICULATE READINGS
# =========================================================

df["extreme_particulate_flag"] = (
    df["parameter_name"].isin(["PM2.5", "PM10"])
    & (df["value"] >= 900)
)

print(
    "Extreme particulate records:",
    int(df["extreme_particulate_flag"].sum())
)


# =========================================================
# 5. FLAG ZERO READINGS
# =========================================================

df["zero_reading_flag"] = df["value"] == 0

print(
    "Zero-valued records:",
    int(df["zero_reading_flag"].sum())
)


# =========================================================
# 6. EXCLUDE AGARTALA CO SERIES FROM TRAINING
# =========================================================

# Agartala CO has a 299-hour zero plateau.
# Preserve its measurements but do not use this series
# for model training until independently verified.

df["series_exclusion_flag"] = (
    (df["station_id"] == "site_5587")
    & (df["parameter_name"] == "CO")
)

print(
    "Records in excluded Agartala CO series:",
    int(df["series_exclusion_flag"].sum())
)


# =========================================================
# 7. CREATE COMBINED QUALITY FLAGS
# =========================================================

flag_columns = [
    "plateau_flag",
    "extreme_particulate_flag",
    "zero_reading_flag",
    "series_exclusion_flag",
]

df["quality_flag_count"] = df[flag_columns].sum(axis=1)

def make_quality_notes(row):

    notes = []

    if row["plateau_flag"]:
        notes.append("LONG_IDENTICAL_VALUE_RUN")

    if row["extreme_particulate_flag"]:
        notes.append("EXTREME_PARTICULATE_VALUE")

    if row["zero_reading_flag"]:
        notes.append("ZERO_READING")

    if row["series_exclusion_flag"]:
        notes.append("AGARTALA_CO_SERIES_EXCLUDED")

    return ";".join(notes) if notes else "NO_FLAG"


df["quality_flags"] = df.apply(make_quality_notes, axis=1)


# =========================================================
# 8. PRELIMINARY TRAINING ELIGIBILITY
# =========================================================

# These are conservative preliminary rules.
# They are not the final AQI training eligibility rules.

df["preliminary_training_eligible"] = ~(
    df["plateau_flag"]
    | df["extreme_particulate_flag"]
    | df["series_exclusion_flag"]
)

# A zero reading alone is flagged but not automatically
# excluded, because zero can sometimes be a valid reading.


# =========================================================
# 9. SAVE QUALITY-CONTROLLED DATASET
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)


# =========================================================
# 10. FINAL REPORT
# =========================================================

print("\n========== QUALITY CONTROL SUMMARY ==========")

print("Original records:", len(df))
print("Quality-flagged records:", int((df["quality_flag_count"] > 0).sum()))
print(
    "Preliminary training-eligible records:",
    int(df["preliminary_training_eligible"].sum())
)
print(
    "Preliminary excluded records:",
    int((~df["preliminary_training_eligible"]).sum())
)

print("\nFlags by type:")

for column in flag_columns:
    print(f"{column}: {int(df[column].sum())}")

print("\nStations retained:", df["station_id"].nunique())

print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nAQI quality-controlled dataset created successfully.")
