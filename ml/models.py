"""Model definitions and evaluation helpers shared by training and analysis."""
import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.data import BINARY, NUMERIC, SEED
from ml.metrics import bootstrap_auc, metrics, youden_threshold  # noqa: F401

XGB_PARAMS = dict(n_estimators=400, max_depth=10, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6,
                  gamma=0.5, min_child_weight=1, eval_metric="logloss", random_state=SEED, n_jobs=-1)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """Feed-forward network (PyTorch) with batch norm and dropout, early stopping on a validation split."""

    def __init__(self, hidden=(128, 64), dropout=0.2, lr=1e-3, epochs=30, batch_size=1024, patience=4,
                 seed=SEED):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.epochs, self.batch_size, self.patience, self.seed = epochs, batch_size, patience, seed

    def _net(self, n_in):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            d = h
        return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        idx = rng.permutation(len(X))
        n_val = len(X) // 10
        val, tr = idx[:n_val], idx[n_val:]
        Xv, yv = torch.from_numpy(X[val]), y[val]
        self.net_ = self._net(X.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best, best_state, wait = -1.0, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // self.batch_size)):
                opt.zero_grad()
                loss = loss_fn(self.net_(torch.from_numpy(X[b])).squeeze(1), torch.from_numpy(y[b]))
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                auc = roc_auc_score(yv, self.net_(Xv).squeeze(1).numpy())
            if auc > best:
                best, wait = auc, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(np.asarray(X, np.float32))).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def pipeline(model) -> Pipeline:
    pre = ColumnTransformer([("scale", StandardScaler(), NUMERIC), ("binary", "passthrough", BINARY)],
                            verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", model)])


def candidates() -> dict:
    return {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=127, subsample=0.9,
                                   subsample_freq=1, colsample_bytree=0.6, random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(**XGB_PARAMS),
        "MLP (PyTorch)": TorchMLP(),
    }
