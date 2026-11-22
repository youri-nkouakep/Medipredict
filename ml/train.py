"""Train and compare the models on the same split, log every run to MLflow, export the final pipeline.

Usage: python -m ml.train [--only XGBoost]
"""
import argparse
import json
import time

import mlflow
import pandas as pd

from backend.model_io import booster_path, save_bundle
from ml.data import FEATURES, ROOT, load, split
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.models import bootstrap_auc, candidates, metrics, pipeline, youden_threshold

RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "backend" / "model" / "medipredict.joblib"
FINAL = "XGBoost"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("medipredict")

    df = load()
    train, test = split(df)
    rows = []
    for name, model in candidates().items():
        if args.only and name not in args.only:
            continue
        with mlflow.start_run(run_name=name):
            pipe = pipeline(model)
            t = time.perf_counter()
            pipe.fit(train[FEATURES], train["target"])
            fit_s = time.perf_counter() - t
            # Threshold chosen on out-of-fold predictions: the training predictions themselves are
            # over-confident for flexible models (a random forest scores ~1.0 on its own training set).
            oof = cross_val_predict(pipeline(model), train[FEATURES], train["target"], method="predict_proba",
                                    cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
            thr = youden_threshold(train["target"], oof)
            p = pipe.predict_proba(test[FEATURES])[:, 1]
            m = metrics(test["target"].to_numpy(), p, thr)
            m["auc_ci_low"], m["auc_ci_high"] = bootstrap_auc(test["target"], p)
            m["train_seconds"] = fit_s
            mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str))})
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
            rows.append({"model": name, **m})
            print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)
            if name == FINAL:
                save_bundle(pipe, thr, FEATURES, MODEL_PATH)
                pd.DataFrame({"y": test["target"].to_numpy(), "p": p, "source": test["source_dataset"].to_numpy(),
                              "gender": test["gender"].to_numpy(), "age": test["age"].to_numpy()}) \
                    .to_csv(RESULTS / "test_predictions.csv.gz", index=False)
                mlflow.log_artifact(str(MODEL_PATH))
                mlflow.log_artifact(str(booster_path(MODEL_PATH)))

    path = RESULTS / "model_comparison.csv"
    old = pd.read_csv(path) if path.exists() and args.only else pd.DataFrame()
    table = pd.concat([old[~old["model"].isin([r["model"] for r in rows])] if len(old) else old,
                       pd.DataFrame(rows)]).sort_values("auc", ascending=False)
    table.round(4).to_csv(path, index=False)
    (RESULTS / "final_model.json").write_text(json.dumps(
        {"model": FINAL, "features": FEATURES, "train_rows": len(train), "test_rows": len(test)}, indent=2))


if __name__ == "__main__":
    main()
