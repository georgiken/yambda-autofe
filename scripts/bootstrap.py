"""Paired bootstrap over USERS for test-metric differences vs a reference model.

    python scripts/bootstrap.py results/u5000 --ref manual+sess --n 1000

Each replicate resamples users with replacement (all their (user, day) groups and rows go together),
so the CI reflects user-level variance. Writes bootstrap.csv next to preds_test.npz.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def per_group(y, p, g, k=10):
    """Per-(user,day) NDCG@k and AUC for informative groups + per-row log-loss."""
    df = pd.DataFrame({"g": g, "y": y, "p": np.clip(p, 1e-7, 1 - 1e-7)})
    st = df.groupby("g")["y"].agg(["min", "max"])
    df = df[df["g"].isin(st.index[st["min"] != st["max"]])].sort_values(["g", "p"], ascending=[True, False])
    pos = df.groupby("g").cumcount().to_numpy()
    df["dcg"] = df["y"] * np.where(pos < k, 1 / np.log2(pos + 2), 0)
    df["r"] = df.groupby("g")["p"].rank(method="average")
    agg = df.groupby("g").agg(dcg=("dcg", "sum"), npos=("y", "sum"), size=("y", "size"))
    disc = np.cumsum(1 / np.log2(np.arange(2, k + 2)))
    agg["ndcg"] = agg["dcg"] / disc[np.minimum(agg["npos"], k) - 1]
    rpos = df[df["y"] == 1].groupby("g")["r"].sum().reindex(agg.index)
    agg["auc"] = (rpos - agg["npos"] * (agg["npos"] + 1) / 2) / (agg["npos"] * (agg["size"] - agg["npos"]))
    return agg[["ndcg", "auc", "size"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default="manual+sess")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    d = np.load(os.path.join(a.run_dir, "preds_test.npz"))
    y, uid, g = d["y"], d["uid"], d["group"]
    models = [m for m in d.files if m not in ("y", "uid", "group")]
    users = np.unique(uid)
    uidx = {u: i for i, u in enumerate(users)}
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))
                  for _ in range(a.n)])  # (n, users): multiplicity of each user in each replicate

    def user_level(m):
        p = d[m].astype("float64")
        ll = -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
        row_u = np.array([uidx[u] for u in uid]) if not hasattr(user_level, "ru") else user_level.ru
        user_level.ru = row_u
        ll_sum = np.bincount(row_u, weights=ll, minlength=len(users))
        n_rows = np.bincount(row_u, minlength=len(users))
        pg = per_group(y, p, g)
        gu = np.array([uidx[x] for x in (pg.index.to_numpy() // 1024)])
        nd_sum = np.bincount(gu, weights=pg["ndcg"].to_numpy(), minlength=len(users))
        n_g = np.bincount(gu, minlength=len(users))
        au_sum = np.bincount(gu, weights=(pg["auc"] * pg["size"]).to_numpy(), minlength=len(users))
        w_g = np.bincount(gu, weights=pg["size"].to_numpy(), minlength=len(users))
        return {"logloss": (ll_sum, n_rows), "ndcg@10": (nd_sum, n_g), "gauc": (au_sum, w_g)}

    def boot(stats):
        return {k: (W @ s) / (W @ n) for k, (s, n) in stats.items()}

    ref = user_level(a.ref)
    ref_b = boot(ref)
    ref_pt = {k: s.sum() / n.sum() for k, (s, n) in ref.items()}
    rows = []
    for m in models:
        st = user_level(m)
        b = boot(st)
        for k in ("logloss", "gauc", "ndcg@10"):
            pt = st[k][0].sum() / st[k][1].sum()
            diff = b[k] - ref_b[k]
            rows.append({"model": m, "metric": k, "value": pt, "diff_vs_ref": pt - ref_pt[k],
                         "ci_low": np.quantile(diff, .025), "ci_high": np.quantile(diff, .975),
                         "p_better": float(np.mean(diff < 0) if k == "logloss" else np.mean(diff > 0))})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.run_dir, "bootstrap.csv"), index=False)
    piv = out.pivot(index="model", columns="metric", values=["diff_vs_ref", "ci_low", "ci_high"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"reference: {a.ref}, users={len(users)}, replicates={a.n}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
