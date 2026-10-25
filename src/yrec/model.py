"""GBDT wrapper with interchangeable backends.

    BACKEND = "lgb"  LightGBM (CPU)                      - default when installed
    BACKEND = "xgb"  XGBoost hist, DEVICE="cuda" for GPU - used for fast training on Colab/Kaggle GPU
    BACKEND = "sk"   sklearn HistGradientBoosting        - fallback for unit tests only
Set via `configure(backend, device)` (run_experiments.py --backend/--device).
All backends get equivalent hyper-parameters (lr 0.05, 63 leaves, row/col subsampling 0.8, L2 = 1,
early stopping on the validation set), so feature sets are compared under the same learner.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

BACKEND = "lgb" if lgb is not None else ("xgb" if xgb is not None else "sk")
DEVICE = "cpu"


def configure(backend: str = "auto", device: str = "cpu"):
    global BACKEND, DEVICE
    if backend == "auto":
        backend = "xgb" if (device == "cuda" and xgb is not None) else ("lgb" if lgb is not None else
                                                                       "xgb" if xgb is not None else "sk")
    if backend == "lgb" and lgb is None or backend == "xgb" and xgb is None:
        raise ImportError(f"backend {backend} is not installed")
    BACKEND, DEVICE = backend, device
    return BACKEND, DEVICE


DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      verbose=-1, seed=0, num_threads=0)


def _xgb_params(p: dict) -> dict:
    rank = p["objective"] == "lambdarank"
    return {"objective": "rank:ndcg" if rank else "binary:logistic",
            "eval_metric": "ndcg@10" if rank else "logloss",
            "tree_method": "hist", "device": DEVICE, "learning_rate": p["learning_rate"],
            "grow_policy": "lossguide", "max_depth": 0, "max_leaves": p["num_leaves"],
            "min_child_weight": 10.0, "subsample": p["bagging_fraction"], "colsample_bytree": p["feature_fraction"],
            "reg_lambda": p["lambda_l2"], "max_bin": 256, "seed": p["seed"], "verbosity": 0}


class GBDT:
    def __init__(self, params: dict | None = None, num_boost_round: int = 3000, early_stopping: int = 100):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.nbr, self.es = num_boost_round, early_stopping
        self.model, self.best_iter, self.fit_time = None, None, None
        self.backend = BACKEND

    def fit(self, Xtr: pd.DataFrame, ytr, Xva: pd.DataFrame, yva, group_tr=None, group_va=None):
        t0 = time.time()
        self.features = list(Xtr.columns)
        rank = self.params["objective"] == "lambdarank"
        if self.backend == "lgb":
            p = dict(self.params)
            if rank:
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
        elif self.backend == "xgb":
            kw_tr = {"qid": np.asarray(group_tr)} if rank else {}
            kw_va = {"qid": np.asarray(group_va)} if rank else {}
            dtr = xgb.QuantileDMatrix(Xtr, label=np.asarray(ytr), max_bin=256, **kw_tr)
            dva = xgb.QuantileDMatrix(Xva, label=np.asarray(yva), ref=dtr, **kw_va)
            self.model = xgb.train(_xgb_params(self.params), dtr, num_boost_round=self.nbr,
                                   evals=[(dva, "val")], early_stopping_rounds=self.es, verbose_eval=False)
            self.best_iter = int(self.model.best_iteration) + 1
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
        rank = self.params["objective"] == "lambdarank"
        if self.backend == "lgb":
            raw = self.model.predict(X, num_iteration=self.best_iter)
        elif self.backend == "xgb":
            raw = self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iter))
        else:
            return self.model.predict_proba(X)[:, 1]
        return 1 / (1 + np.exp(-raw)) if rank else raw

    def importance(self) -> pd.Series:
        if self.backend == "lgb":
            imp = pd.Series(self.model.feature_importance("gain"), index=self.features)
        elif self.backend == "xgb":
            imp = pd.Series(self.model.get_score(importance_type="total_gain")).reindex(self.features).fillna(0.0)
        else:
            imp = pd.Series(np.ones(len(self.features)), index=self.features)
        return imp.sort_values(ascending=False)


def _sizes(group) -> np.ndarray:
    g = np.asarray(group)
    change = np.r_[True, g[1:] != g[:-1]]
    idx = np.r_[np.nonzero(change)[0], len(g)]
    return np.diff(idx)
