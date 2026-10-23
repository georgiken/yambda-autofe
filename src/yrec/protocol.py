"""Task definition, temporal split and metrics.

Task: for every recommendation-driven listen (is_organic = 0) in a target day, predict whether the
user will listen to the track to >= 50% (Yambda's "listened" definition) — an impression -> engagement
setting. Ranking metrics are computed inside groups (uid, day) = the tracks served to the user that day.

Split (by day, last days of the log):
    ... history ... | train targets (n_train days) | val (n_val days) | test (n_test days)
Features for a target on day d only see events with day <= d-1 (+ strictly earlier events of the
same day for session features), so the same feature code is valid for every split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_days: tuple[int, int]  # inclusive
    val_days: tuple[int, int]
    test_days: tuple[int, int]

    def as_dict(self):
        return {"train": self.train_days, "val": self.val_days, "test": self.test_days}


def make_split(ev: pd.DataFrame, n_train: int = 21, n_val: int = 7, n_test: int = 7) -> Split:
    last = int(ev["day"].max())
    te = (last - n_test + 1, last)
    va = (te[0] - n_val, te[0] - 1)
    tr = (va[0] - n_train, va[0] - 1)
    assert tr[0] > 30, "need at least a month of history before the first train target"
    return Split(tr, va, te)


def make_targets(ev: pd.DataFrame, days: tuple[int, int], max_rows: int | None = None,
                 seed: int = 0) -> pd.DataFrame:
    m = (ev["etype"] == 0) & (ev["organic"] == 0) & ev["day"].between(*days)
    t = ev.loc[m, ["uid", "item_id", "artist_id", "album_id", "ts", "day", "tlen", "ratio"]].copy()
    t["label"] = (t["ratio"] >= 50).astype("int8")
    t = t.drop(columns="ratio")
    if max_rows is not None and len(t) > max_rows:
        # subsample whole (uid, day) groups so that ranking metrics stay meaningful
        g = t["uid"].astype("int64") * 1024 + t["day"].astype("int64")
        ug = np.unique(g)
        rng = np.random.default_rng(seed)
        keep = rng.choice(ug, size=int(len(ug) * max_rows / len(t)), replace=False)
        t = t[g.isin(keep)]
    return t.sort_values(["uid", "ts"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- metrics
def evaluate(y: np.ndarray, p: np.ndarray, group: np.ndarray, k: int = 10) -> dict:
    """LogLoss, ROC-AUC (global), GAUC (group-weighted AUC), NDCG@k and Precision@k inside groups.
    Groups with all-equal labels are skipped for GAUC/NDCG (no ranking information)."""
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    res = {"logloss": float(log_loss(y, p)), "auc": float(roc_auc_score(y, p)), "n": int(len(y)),
           "pos_rate": float(y.mean())}
    df = pd.DataFrame({"g": np.asarray(group), "y": y, "p": p})
    gstat = df.groupby("g")["y"].agg(["min", "max", "size", "sum"])
    informative = gstat.index[(gstat["min"] != gstat["max"]).to_numpy()]
    df = df[df["g"].isin(informative)].sort_values(["g", "p"], ascending=[True, False], kind="stable")
    pos = df.groupby("g").cumcount().to_numpy()                  # 0-based rank by prediction
    disc = np.where(pos < k, 1.0 / np.log2(pos + 2), 0.0)
    df["dcg"] = df["y"].to_numpy() * disc
    df["topk"] = (pos < k) * df["y"].to_numpy()
    df["inK"] = (pos < k).astype(float)
    # ideal DCG: n_pos ones at the top
    npos = df.groupby("g")["y"].sum()
    disc_k = np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))
    idcg = disc_k[np.minimum(npos.to_numpy(), k) - 1]
    agg = df.groupby("g").agg(dcg=("dcg", "sum"), topk=("topk", "sum"), inK=("inK", "sum"), size=("y", "size"))
    ndcg = agg["dcg"].to_numpy() / idcg
    prec = agg["topk"].to_numpy() / agg["inK"].to_numpy()
    # per-group AUC via average ranks (ties handled)
    df["r"] = df.groupby("g")["p"].rank(method="average")
    rpos = df[df["y"] == 1].groupby("g")["r"].sum().reindex(agg.index).to_numpy()
    nn = agg["size"].to_numpy() - npos.to_numpy()
    npv = npos.to_numpy()
    gauc = (rpos - npv * (npv + 1) / 2) / (npv * nn)
    res.update({f"ndcg@{k}": float(ndcg.mean()), f"prec@{k}": float(prec.mean()),
                "gauc": float(np.average(gauc, weights=agg["size"].to_numpy())), "n_groups": int(len(agg))})
    return res


def group_id(t: pd.DataFrame) -> np.ndarray:
    return (t["uid"].to_numpy().astype("int64") * 1024 + t["day"].to_numpy().astype("int64"))
