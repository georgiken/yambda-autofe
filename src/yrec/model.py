"""GBDT wrapper: LightGBM if available, sklearn HistGradientBoosting fallback (for tests only)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, seed=0, num_threads=0)


class GBDT:
    def __init__(self, params: dict | None = None, num_boost_round: int = 3000, early_stopping: int = 100):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.nbr, self.es = num_boost_round, early_stopping
        self.model, self.best_iter, self.fit_time = None, None, None

    def fit(self, Xtr: pd.DataFrame, ytr, Xva: pd.DataFrame, yva, group_tr=None, group_va=None):
        t0 = time.time()
        self.features = list(Xtr.columns)
        if lgb is not None:
            p = dict(self.params)
            if p["objective"] == "lambdarank":
                dtr = lgb.Dataset(Xtr, ytr, group=_sizes(group_tr), free_raw_data=False)
                dva = lgb.Dataset(Xva, yva, group=_sizes(group_va), reference=dtr)
                p.setdefault("metric", "ndcg"); p.setdefault("eval_at", [10])
            else:
                dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
                dva = lgb.Dataset(Xva, yva, reference=dtr)
                p.setdefault("metric", "binary_logloss")
            self.model = lgb.train(p, dtr, self.nbr, valid_sets=[dva],
                                   callbacks=[lgb.early_stopping(self.es, verbose=False)])
            self.best_iter = self.model.best_iteration
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                        early_stopping=True, validation_fraction=0.15,
                                                        random_state=0)
            self.model.fit(Xtr, ytr)
            self.best_iter = self.model.n_iter_
        self.fit_time = time.time() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if lgb is not None:
            raw = self.model.predict(X, num_iteration=self.best_iter)
            if self.params["objective"] == "lambdarank":
                return 1 / (1 + np.exp(-raw))
            return raw
        return self.model.predict_proba(X)[:, 1]

    def importance(self) -> pd.Series:
        if lgb is not None:
            return pd.Series(self.model.feature_importance("gain"), index=self.features).sort_values(ascending=False)
        return pd.Series(np.ones(len(self.features)), index=self.features)


def _sizes(group) -> np.ndarray:
    g = np.asarray(group)
    change = np.r_[True, g[1:] != g[:-1]]
    idx = np.r_[np.nonzero(change)[0], len(g)]
    return np.diff(idx)
