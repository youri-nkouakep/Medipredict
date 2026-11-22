"""CI quality gate: the exported model must keep its AUC on a fixed holdout sample.

Exits with an error if the AUC drops below the reference, so a worse model cannot be merged.
"""
import sys

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.data import ROOT

MIN_AUC = 0.93


def main():
    bundle = joblib.load(ROOT / "backend" / "model" / "medipredict.joblib")
    holdout = pd.read_csv(ROOT / "data" / "holdout_sample.csv")
    auc = roc_auc_score(holdout["target"], bundle["pipeline"].predict_proba(holdout[bundle["features"]])[:, 1])
    print(f"AUC on {len(holdout)} holdout patients: {auc:.4f} (minimum {MIN_AUC})")
    if auc < MIN_AUC:
        sys.exit("Quality gate failed: the model is worse than the reference")


if __name__ == "__main__":
    main()
