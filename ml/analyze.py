"""Analyses of the final model: generalization across data sources, calibration, SHAP, fairness.

Usage: python -m ml.analyze   (after python -m ml.train)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from backend.model_io import load_bundle  # noqa: E402
from ml.data import FEATURES, ROOT, load, split  # noqa: E402
from ml.metrics import metrics
from ml.models import XGB_PARAMS, pipeline  # noqa: E402

RESULTS, FIG = ROOT / "results", ROOT / "results" / "figures"
BLUE, NAVY, GREY = "#3b9fd8", "#0f3b5f", "#b8c7d3"
SOURCES = ["hypertension", "diabetes", "heart"]


def cross_source(df, train, test, final):
    """Leave-one-source-out: train on two sources, test on the third."""
    rows = []
    for held in SOURCES:
        tr = df[df["source_dataset"] != held]
        te = df[df["source_dataset"] == held]
        pipe = pipeline(XGBClassifier(**XGB_PARAMS)).fit(tr[FEATURES], tr["target"])
        in_dist = test[test["source_dataset"] == held]
        rows.append({
            "held_out_source": held, "rows": len(te), "abnormal_rate": float(te["target"].mean()),
            "auc_in_distribution": roc_auc_score(in_dist["target"], final.predict_proba(in_dist[FEATURES])[:, 1]),
            "auc_unseen_source": roc_auc_score(te["target"], pipe.predict_proba(te[FEATURES])[:, 1]),
        })
    src = LogisticRegression(max_iter=2000).fit(pd.get_dummies(train["source_dataset"]), train["target"])
    source_only = roc_auc_score(test["target"], src.predict_proba(pd.get_dummies(test["source_dataset"]))[:, 1])
    finder = pipeline(XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42))
    finder.fit(train[FEATURES], train["source_dataset"].map({s: i for i, s in enumerate(SOURCES)}))
    source_acc = float((finder.predict(test[FEATURES]) ==
                        test["source_dataset"].map({s: i for i, s in enumerate(SOURCES)})).mean())
    return pd.DataFrame(rows), {"auc_source_only": source_only, "source_identification_accuracy": source_acc}


def calibration(y, p):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    bins = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(10) if (idx == b).any())
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfect calibration")
    ax.plot(mean, frac, "o-", color=BLUE, label="XGBoost")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration (ECE {ece:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=130)
    plt.close(fig)
    return {"ece": float(ece)}


def shap_importance(final, test):
    sample = test[FEATURES].sample(3000, random_state=42)
    X = final.named_steps["pre"].transform(sample)
    values = shap.TreeExplainer(final.named_steps["model"]).shap_values(X)
    imp = pd.Series(np.abs(values).mean(0), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "shap_importance.png", dpi=130)
    plt.close(fig)
    return imp.sort_values(ascending=False).round(4).to_dict()


def fairness(pred, threshold):
    pred = pred.assign(age_group=pd.cut(pred["age"], [0, 40, 60, 200], labels=["< 40", "40-59", "60 +"], right=False))
    rows = []
    for col in ("gender", "age_group"):
        for g, part in pred.groupby(col, observed=True):
            m = metrics(part["y"].to_numpy(), part["p"].to_numpy(), threshold)
            rows.append({"attribute": col, "group": str(g), "rows": len(part), "abnormal_rate": part["y"].mean(),
                         **{k: m[k] for k in ("auc", "sensitivity", "specificity", "brier")}})
    return pd.DataFrame(rows)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(ROOT / "backend" / "model" / "medipredict.joblib")
    final, thr = bundle["pipeline"], bundle["threshold"]
    df = load()
    train, test = split(df)
    pred = pd.read_csv(RESULTS / "test_predictions.csv.gz")

    cs, extra = cross_source(df, train, test, final)
    cs.round(4).to_csv(RESULTS / "cross_source.csv", index=False)
    print(cs.round(3).to_string(), extra)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(cs))
    ax.bar(x - 0.18, cs["auc_in_distribution"], 0.36, color=NAVY, label="seen during training")
    ax.bar(x + 0.18, cs["auc_unseen_source"], 0.36, color=BLUE, label="source left out")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x, cs["held_out_source"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Does the model generalize to a new data source?")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "cross_source.png", dpi=130)
    plt.close(fig)

    cal = calibration(pred["y"].to_numpy(), pred["p"].to_numpy())
    imp = shap_importance(final, test)
    fair = fairness(pred, thr)
    fair.round(4).to_csv(RESULTS / "fairness.csv", index=False)
    print(fair.round(3).to_string())
    (RESULTS / "analysis.json").write_text(json.dumps({**extra, **cal, "shap_importance": imp}, indent=2))


if __name__ == "__main__":
    main()
