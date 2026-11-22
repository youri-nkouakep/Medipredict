"""Data warehouse loading, encoding and the train / test split used everywhere."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data" / "data_warehouse.csv"
SEED = 42

NUMERIC = ["age", "bmi", "HbA1c_level", "glucose", "cholesterol", "sleep_hours", "triglycerides",
           "physical_activity", "family_history", "stress_level", "high_blood_pressure",
           "sugar_consumption", "crp_level"]
BINARY = ["gender_Male", "smoking_Former", "smoking_Never", "education_level_Secondary",
          "education_level_Tertiary", "employment_status_Retired", "employment_status_Unemployed"]
FEATURES = NUMERIC + BINARY

ORDINAL = {
    "physical_activity": {"Low": 0, "Moderate": 1, "High": 2},
    "stress_level": {"Low": 0, "Medium": 1, "High": 2},
    "sugar_consumption": {"Low": 0, "Medium": 1, "High": 2},
    "family_history": {"No": 0, "Yes": 1},
    "high_blood_pressure": {"No": 0, "Yes": 1},
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in ORDINAL.items():
        out[col] = out[col].map(mapping)
    out["gender_Male"] = (out["gender"] == "Male").astype(int)
    out["smoking_Former"] = (out["smoking"] == "Former").astype(int)
    out["smoking_Never"] = (out["smoking"] == "Never").astype(int)
    out["education_level_Secondary"] = (out["education_level"] == "Secondary").astype(int)
    out["education_level_Tertiary"] = (out["education_level"] == "Tertiary").astype(int)
    out["employment_status_Retired"] = (out["employment_status"] == "Retired").astype(int)
    out["employment_status_Unemployed"] = (out["employment_status"] == "Unemployed").astype(int)
    out["target"] = (out["label"] == "Abnormal").astype(int)
    return out


def load(path: Path = WAREHOUSE) -> pd.DataFrame:
    """Encoded warehouse, exact duplicate rows removed. Keeps source and gender for the analyses."""
    df = encode(pd.read_csv(path, encoding="utf-8-sig").drop_duplicates().reset_index(drop=True))
    return df[FEATURES + ["target", "source_dataset", "gender"]]


def split(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, stratify=df["target"], random_state=SEED)
