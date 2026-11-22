"""Save and load the exported model.

The scikit-learn preprocessing and the metadata are stored with joblib. The XGBoost model is stored
separately in XGBoost's own JSON format, which loads identically on every OS and XGBoost build
(a pickled booster can fail on Windows with "XGBoostError: input stream corrupted").
"""
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def booster_path(path) -> Path:
    """medipredict.joblib -> medipredict.xgb.json"""
    return Path(path).with_suffix(".xgb.json")


def save_bundle(pipeline: Pipeline, threshold: float, features: list, path) -> None:
    model = pipeline.named_steps["model"]
    model.save_model(booster_path(path))
    joblib.dump({"pre": pipeline.named_steps["pre"], "model_params": model.get_params(),
                 "threshold": threshold, "features": features}, path, compress=3)


def load_bundle(path) -> dict:
    bundle = joblib.load(path)
    model = XGBClassifier(**bundle["model_params"])
    model.load_model(booster_path(path))
    pipeline = Pipeline([("pre", bundle["pre"]), ("model", model)])
    return {"pipeline": pipeline, "threshold": bundle["threshold"], "features": bundle["features"]}
