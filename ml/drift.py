"""Data drift monitoring: compares the patients seen in production with the training population.

Population Stability Index (PSI) per feature: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
Usage:
    python -m ml.drift --db sqlite:///backend/medipredict.db   # predictions stored by the API
    python -m ml.drift --simulate                              # demo: an older, less active population
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from ml.data import BINARY, FEATURES, NUMERIC, ROOT

REFERENCE = ROOT / "data" / "reference_sample.csv"
REPORT = ROOT / "results" / "drift_report.json"


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    c = np.histogram(np.clip(cur, edges[0], edges[-1]), edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        ref, cur = reference[f].to_numpy(float), current[f].to_numpy(float)
        value = psi(ref, cur) if f in NUMERIC else psi(ref, cur, bins=2)
        rows.append({"feature": f, "psi": value, "ks_pvalue": ks_2samp(ref, cur).pvalue if f not in BINARY else None,
                     "reference_mean": ref.mean(), "current_mean": cur.mean(),
                     "status": "major shift" if value > 0.25 else "moderate shift" if value > 0.1 else "stable"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def from_database(url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine
    rows = pd.read_sql("SELECT patient_data FROM predictions", create_engine(url))
    return pd.DataFrame([json.loads(r) for r in rows["patient_data"] if r])


def simulate(reference: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cur = reference.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    cur["age"] = np.clip(cur["age"] + rng.normal(12, 4, n), 18, 95)
    cur["physical_activity"] = np.clip(cur["physical_activity"] - rng.binomial(1, 0.4, n), 0, 2)
    cur["employment_status_Retired"] = np.maximum(cur["employment_status_Retired"], rng.binomial(1, 0.3, n))
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    reference = pd.read_csv(REFERENCE)
    current = simulate(reference) if args.simulate or not args.db else from_database(args.db)
    if len(current) < 50:
        raise SystemExit(f"Only {len(current)} predictions stored: not enough to measure drift")
    table = report(reference, current)
    REPORT.write_text(table.round(4).to_json(orient="records", indent=2))
    print(table.round(3).to_string(index=False))
    if (table["status"] == "major shift").any():
        print("\nALERT: major shift on", ", ".join(table.loc[table["status"] == "major shift", "feature"]))


if __name__ == "__main__":
    main()
