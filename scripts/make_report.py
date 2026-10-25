"""Collect results/*/results.json into markdown tables + figures (results/report/)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(ROOT, "report")
os.makedirs(OUT, exist_ok=True)

ORDER = ["pop_item_fullrate", "user_fullrate", "ctx_only", "manual", "autofe", "manual+sess",
         "random_sameN+sess", "manual_retuned_windows+sess", "autofe+sess", "autofe_sameN+sess",
         "manual∪autofe+sess", "manual+sess_lambdarank", "autofe+sess_lambdarank"]


def table(run_dir: str) -> pd.DataFrame:
    r = json.load(open(os.path.join(run_dir, "results.json")))
    rows = []
    for name in ORDER + [k for k in r["results"] if k not in ORDER]:
        if name not in r["results"]:
            continue
        x = r["results"][name]
        rows.append({"model": name, "n_feat": x.get("n_features"),
                     "val_logloss": x["val"]["logloss"], "val_ndcg@10": x["val"]["ndcg@10"],
                     "test_logloss": x["test"]["logloss"], "test_auc": x["test"]["auc"],
                     "test_gauc": x["test"]["gauc"], "test_ndcg@10": x["test"]["ndcg@10"],
                     "fit_s": x.get("fit_s")})
    return pd.DataFrame(rows), r


md = []
for run_dir in sorted(glob.glob(os.path.join(ROOT, "*/"))):
    if not os.path.exists(os.path.join(run_dir, "results.json")):
        continue
    name = os.path.basename(os.path.normpath(run_dir))
    df, r = table(run_dir)
    md.append(f"## {name}\n")
    md.append("data: " + json.dumps(r["data"], default=str) + "\n")
    md.append("timings (s): " + json.dumps({k: round(v, 1) for k, v in r["timings"].items()
                                           if isinstance(v, (int, float))}) + "\n")
    md.append(df.to_markdown(index=False, floatfmt=".4f") + "\n")
    bpath = os.path.join(run_dir, "bootstrap.csv")
    if os.path.exists(bpath):
        b = pd.read_csv(bpath)
        b["diff [95% CI]"] = b.apply(lambda r: f"{r.diff_vs_ref:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]", axis=1)
        md.append("Paired bootstrap по пользователям, разница с manual+sess (test):\n")
        md.append(b.pivot(index="model", columns="metric", values="diff [95% CI]").to_markdown() + "\n")
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cpath = os.path.join(run_dir, "autofe_curve.csv")
        if os.path.exists(cpath):
            c = pd.read_csv(cpath)
            plt.figure(figsize=(5, 3.2))
            plt.plot(c["n"], c["val_logloss"], marker="o")
            plt.xlabel("число отобранных признаков N"); plt.ylabel("val LogLoss"); plt.title("AutoFE: stage 2")
            plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{name}_autofe_curve.png"), dpi=130)
            plt.close()
        spath = os.path.join(run_dir, "autofe_stage1.csv")
        if os.path.exists(spath):
            s = pd.read_csv(spath)
            # time-scale profiles for a few families
            fams = ["cnt__user_artist__full", "rate__user__full_rec_per_listen_rec", "rate__item__full_rec_per_listen_rec",
                    "cnt__user_item__listen", "rate__user_artist__full_per_listen", "cnt__item__listen"]
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))
            for f in fams:
                d = s[(s.family == f) & s.kind.isin(["cnt", "rate"])].copy()
                if d.empty:
                    continue
                d["w"] = d["window"].fillna(365)
                d = d.sort_values("w")
                ax.plot(d["w"], d["gain"] * 1e3, marker=".", label=f)
            ax.set_xscale("log"); ax.set_xlabel("окно, дней (365 = вся история)")
            ax.set_ylabel("уменьшение val LogLoss ×1e3"); ax.legend(fontsize=6); ax.grid(alpha=.3)
            plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{name}_window_profiles.png"), dpi=130); plt.close()
            best = s.loc[s.groupby("family")["gain"].idxmax()]
            best["scale"] = np.where(best["half_life"].notna(), "h=" + best["half_life"].astype(str),
                                     np.where(best["window"].notna(), "w=" + best["window"].astype(str), "all/other"))
            md.append("Лучший масштаб времени по семействам (топ-25 по gain):\n")
            md.append(best.sort_values("gain", ascending=False).head(25)[["family", "feature", "gain"]]
                      .to_markdown(index=False, floatfmt=".5f") + "\n")
            md.append("Распределение лучших масштабов: " + json.dumps(best["scale"].value_counts().to_dict()) + "\n")
        for imp in ("manual+sess", "autofe+sess"):
            ip = os.path.join(run_dir, f"importance_{imp}.csv")
            if os.path.exists(ip):
                im = pd.read_csv(ip, index_col=0).iloc[:, 0]
                md.append(f"Топ-15 признаков по gain ({imp}):\n")
                md.append((im / im.sum()).head(15).rename("gain_share").to_frame().to_markdown(floatfmt=".3f") + "\n")
    except Exception as e:  # plotting is optional
        md.append(f"(figures skipped: {e})\n")

open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md))
print("\n".join(md))
