import numpy as np
import pandas as pd

from ml.data import ORDINAL, encode
from ml.drift import psi, report
from ml.metrics import metrics, youden_threshold


def test_encode():
    row = pd.DataFrame([{"physical_activity": "High", "stress_level": "Low", "sugar_consumption": "Medium",
                         "family_history": "Yes", "high_blood_pressure": "No", "gender": "Male",
                         "smoking": "Former", "education_level": "Primary", "employment_status": "Retired",
                         "label": "Abnormal"}])
    out = encode(row).iloc[0]
    assert out["physical_activity"] == ORDINAL["physical_activity"]["High"] == 2
    assert out["gender_Male"] == 1 and out["smoking_Former"] == 1 and out["smoking_Never"] == 0
    assert out["education_level_Secondary"] == 0 and out["employment_status_Retired"] == 1 and out["target"] == 1


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    assert 0.4 < youden_threshold(y, p) <= 0.6
    m = metrics(y, p, 0.5)
    assert m["auc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25


def test_drift_report_flags_age_shift():
    ref = pd.read_csv("data/reference_sample.csv")
    cur = ref.assign(age=ref["age"] + 15)
    table = report(ref, cur).set_index("feature")
    assert table.loc["age", "status"] == "major shift"
    assert table.loc["bmi", "status"] == "stable"
