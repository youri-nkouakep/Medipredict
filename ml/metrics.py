"""Evaluation metrics shared by training, analysis and tests (no deep learning dependency)."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score, roc_curve

from ml.data import SEED


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def metrics(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return {"auc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p), "brier": brier_score_loss(y, p),
            "accuracy": float((pred == y).mean()), "sensitivity": tp / max(int((y == 1).sum()), 1),
            "specificity": tn / max(int((y == 0).sum()), 1), "mcc": matthews_corrcoef(y, pred),
            "threshold": threshold}


def bootstrap_auc(y, p, n: int = 1000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
