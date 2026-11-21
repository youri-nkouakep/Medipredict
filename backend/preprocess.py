import joblib
import numpy as np
import pandas as pd

from config import SCALER_PATH

SELECTED_FEATURES = [
    'age', 'bmi', 'HbA1c_level', 'glucose', 'cholesterol', 'sleep_hours',
    'triglycerides', 'physical_activity', 'family_history', 'stress_level',
    'high_blood_pressure', 'sugar_consumption', 'crp_level',
    'gender_Male', 'smoking_Former', 'smoking_Never',
    'education_level_Secondary', 'education_level_Tertiary',
    'employment_status_Retired', 'employment_status_Unemployed',
]


NUMERIC_FEATURES = [
    'age', 'bmi', 'HbA1c_level', 'glucose', 'cholesterol', 'sleep_hours',
    'triglycerides', 'physical_activity', 'family_history', 'stress_level',
    'high_blood_pressure', 'sugar_consumption', 'crp_level',
]
NUMERIC_IDX = [SELECTED_FEATURES.index(f) for f in NUMERIC_FEATURES]

# Scaler ajusté sur le jeu d'entraînement (notebook 01), appliqué aux 13 variables numériques
_scaler = joblib.load(SCALER_PATH)


def preprocess_input(data: dict) -> pd.DataFrame:
    """Ordonne les 20 variables comme à l'entraînement et standardise les variables numériques."""
    values = np.array(
        [data.get(col, 0) for col in SELECTED_FEATURES],
        dtype=float
    )
    values[NUMERIC_IDX] = _scaler.transform(
        values[NUMERIC_IDX].reshape(1, -1)
    )[0]

    # DataFrame avec les noms de colonnes de l'entraînement (attendus par XGBoost)
    return pd.DataFrame(values.reshape(1, -1), columns=SELECTED_FEATURES)
