"""Loads the exported pipeline and returns a probability, a decision and the factors behind it."""
import numpy as np
import pandas as pd
import shap

from config import MODEL_PATH
from model_io import load_bundle

LABELS = {
    "age": "Âge", "bmi": "IMC", "HbA1c_level": "HbA1c", "glucose": "Glycémie", "cholesterol": "Cholestérol",
    "sleep_hours": "Sommeil", "triglycerides": "Triglycérides", "physical_activity": "Activité physique",
    "family_history": "Antécédents familiaux", "stress_level": "Stress", "high_blood_pressure": "Hypertension connue",
    "sugar_consumption": "Consommation de sucre", "crp_level": "CRP", "gender_Male": "Sexe masculin",
    "smoking_Former": "Ancien fumeur", "smoking_Never": "Non-fumeur",
    "education_level_Secondary": "Études secondaires", "education_level_Tertiary": "Études supérieures",
    "employment_status_Retired": "Retraité", "employment_status_Unemployed": "Sans emploi",
}


class Predictor:
    def __init__(self, path=MODEL_PATH):
        bundle = load_bundle(path)
        self.pipeline, self.threshold, self.features = bundle["pipeline"], bundle["threshold"], bundle["features"]
        self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])

    def predict(self, data: dict, top: int = 3) -> dict:
        row = pd.DataFrame([[data[f] for f in self.features]], columns=self.features)
        p = float(self.pipeline.predict_proba(row)[0, 1])
        contrib = np.ravel(self.explainer.shap_values(self.pipeline.named_steps["pre"].transform(row)))
        order = np.argsort(-np.abs(contrib))[:top]
        factors = [{"feature": self.features[i], "label": LABELS[self.features[i]], "value": data[self.features[i]],
                    "impact": float(contrib[i]), "direction": "increases" if contrib[i] > 0 else "decreases"}
                   for i in order]
        return {"probability_abnormal": p, "prediction": int(p >= self.threshold),
                "threshold": self.threshold, "factors": factors}
