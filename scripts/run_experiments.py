"""Full experiment: manual baseline vs AutoFE (+ controls) on the same split / targets / model.

Example (Kaggle):
    python scripts/run_experiments.py --n-users 10000 --out results/u10000
    python scripts/run_experiments.py --synthetic --out results/synthetic   # smoke test, no data needed
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yrec.autofe import run_autofe  # noqa: E402
from yrec.data import load_yambda, make_synthetic  # noqa: E402
from yrec.feature_sets import manual_features, search_space  # noqa: E402
from yrec.features import FeatureEngine, FeatureSpec, context_features, session_features  # noqa: E402
from yrec.model import GBDT, configure  # noqa: E402
from yrec.protocol import evaluate, group_id, make_split, make_targets  # noqa: E402


def rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KB -> GB (peak)
    except Exception:
        return float("nan")


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[peak {rss_gb():.1f}GB]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="folder with Yambda parquet files (HF layout); default: download")
    ap.add_argument("--size", default="50m")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-train", type=int, default=21)
    ap.add_argument("--n-val", type=int, default=7)
    ap.add_argument("--n-test", type=int, default=7)
    ap.add_argument("--max-train-rows", type=int, default=1_500_000)
    ap.add_argument("--autofe-train-rows", type=int, default=400_000)
    ap.add_argument("--autofe-val-rows", type=int, default=300_000)
    ap.add_argument("--skip-autofe", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "lgb", "xgb", "sk"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cuda = train XGBoost on GPU")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny real-data run (200 users, small samples) to check that everything works end-to-end")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()
    if a.smoke:
        a.n_users = a.n_users or 200
        a.max_train_rows, a.autofe_train_rows, a.autofe_val_rows = 50_000, 20_000, 20_000
    backend, device = configure(a.backend, a.device)
    log(f"GBDT backend={backend} device={device}")
    os.makedirs(a.out, exist_ok=True)
    timings, results = {}, {}

    # ------------------------------------------------------------------ data
    t = time.time()
    if a.synthetic:
        ev = make_synthetic(n_users=a.n_users or 400, n_items=3000, n_artists=300, n_days=80, seed=a.seed)
        a.n_train, a.n_val, a.n_test = min(a.n_train, 14), min(a.n_val, 5), min(a.n_test, 5)
    else:
        ev = load_yambda(a.data_dir, a.size, a.n_users, a.seed,
                         cache=os.path.join(a.out, "..", f"events_{a.size}_{a.n_users or 'all'}_{a.seed}.parquet"))
    timings["load_s"] = time.time() - t
    timings["backend"] = f"{backend}/{device}"
    sp = make_split(ev, a.n_train, a.n_val, a.n_test)
    log(f"events={len(ev):,} users={ev.uid.nunique():,} items={ev.item_id.nunique():,} days={ev.day.min()}..{ev.day.max()} split={sp.as_dict()}")

    tr = make_targets(ev, sp.train_days, a.max_train_rows, seed=a.seed)
    va = make_targets(ev, sp.val_days)
    te = make_targets(ev, sp.test_days)
    for n, df in (("train", tr), ("val", va), ("test", te)):
        log(f"{n}: rows={len(df):,} pos_rate={df.label.mean():.3f} users={df.uid.nunique():,}")
    data_stats = {"events": len(ev), "users": int(ev.uid.nunique()), "items": int(ev.item_id.nunique()),
                  "split": sp.as_dict(), "rows": {"train": len(tr), "val": len(va), "test": len(te)},
                  "pos_rate": {"train": float(tr.label.mean()), "val": float(va.label.mean()), "test": float(te.label.mean())},
                  "share_rec_listens": float(((ev.etype == 0) & (ev.organic == 0)).sum() / (ev.etype == 0).sum())}

    # ------------------------------------------------------------------ engine + fixed blocks
    t = time.time()
    eng = FeatureEngine(ev)
    timings["engine_build_s"] = time.time() - t
    log(f"engine built in {timings['engine_build_s']:.1f}s")

    t = time.time()
    ctx = {k: context_features(df) for k, df in (("tr", tr), ("va", va), ("te", te))}
    sess = {k: session_features(ev, df) for k, df in (("tr", tr), ("va", va), ("te", te))}
    timings["session_s"] = time.time() - t
    # AutoFE samples need the raw log for session features -> build them now, then free the raw log
    if not a.skip_autofe:
        tr_s = make_targets(ev, sp.train_days, a.autofe_train_rows, seed=a.seed + 1)
        va_s = make_targets(ev, sp.val_days, a.autofe_val_rows, seed=a.seed + 1)
        extra_tr = pd.concat([context_features(tr_s), session_features(ev, tr_s)], axis=1)
        extra_va = pd.concat([context_features(va_s), session_features(ev, va_s)], axis=1)
    del ev
    gc.collect()
    log("raw event log released")

    def block(specs: list[FeatureSpec]):
        t0 = time.time()
        out = {k: eng.frame(specs, df) for k, df in (("tr", tr), ("va", va), ("te", te))}
        eng.clear_cache()
        return out, time.time() - t0

    def run(name, blocks, params=None, save_importance=False):
        Xs = {k: pd.concat([b[k] for b in blocks], axis=1) for k in ("tr", "va", "te")}
        m = GBDT(params).fit(Xs["tr"], tr.label.values, Xs["va"], va.label.values, group_id(tr), group_id(va))
        r = {"n_features": Xs["tr"].shape[1], "fit_s": m.fit_time, "best_iter": m.best_iter,
             "val": evaluate(va.label.values, m.predict(Xs["va"]), group_id(va)),
             "test": evaluate(te.label.values, m.predict(Xs["te"]), group_id(te))}
        results[name] = r
        log(f"[{name}] nf={r['n_features']} val ll={r['val']['logloss']:.4f} auc={r['val']['auc']:.4f} "
            f"ndcg={r['val']['ndcg@10']:.4f} | test ll={r['test']['logloss']:.4f} auc={r['test']['auc']:.4f} "
            f"gauc={r['test']['gauc']:.4f} ndcg={r['test']['ndcg@10']:.4f}")
        if save_importance:
            m.importance().to_csv(os.path.join(a.out, f"importance_{name}.csv"))
        json.dump({"results": results, "timings": timings, "data": data_stats},
                  open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
        return m

    # ------------------------------------------------------------------ trivial baselines (no learning)
    for nm, spec in (("pop_item_fullrate", FeatureSpec("rate", "item", "full_rec", "listen_rec")),
                     ("user_fullrate", FeatureSpec("rate", "user", "full_rec", "listen_rec"))):
        p_va, p_te = eng.compute(spec, va), eng.compute(spec, te)
        fill = tr.label.mean()
        results[nm] = {"n_features": 1,
                       "val": evaluate(va.label.values, np.nan_to_num(p_va, nan=fill), group_id(va)),
                       "test": evaluate(te.label.values, np.nan_to_num(p_te, nan=fill), group_id(te))}
        log(f"[{nm}] test auc={results[nm]['test']['auc']:.4f} ndcg={results[nm]['test']['ndcg@10']:.4f}")

    # ------------------------------------------------------------------ manual baseline
    man_specs = manual_features()
    man, timings["manual_features_s"] = block(man_specs)
    log(f"manual: {len(man_specs)} features in {timings['manual_features_s']:.1f}s")
    run("ctx_only", [ctx])
    run("manual", [man, ctx])
    run("manual+sess", [man, ctx, sess], save_importance=True)
    run("manual+sess_lambdarank", [man, ctx, sess], params={"objective": "lambdarank"})
    del man
    gc.collect()

    if a.skip_autofe:
        return

    # ------------------------------------------------------------------ AutoFE (search on train/val samples only)
    eng.clear_cache()
    t = time.time()
    res = run_autofe(eng, tr_s, va_s, extra_tr, extra_va,
                     n_grid=(10, 30) if a.smoke else (10, 20, 30, 45, 60, 80, 100))
    timings["autofe_search_s"] = time.time() - t
    timings.update({f"autofe_{k}": v for k, v in res.timings.items()})
    res.stage1.to_csv(os.path.join(a.out, "autofe_stage1.csv"), index=False)
    res.curve.to_csv(os.path.join(a.out, "autofe_curve.csv"), index=False)
    pd.Series(res.ranking).to_csv(os.path.join(a.out, "autofe_ranking.csv"), index=False)
    pd.Series([s.name for s in res.selected]).to_csv(os.path.join(a.out, "autofe_selected.csv"), index=False)
    log(f"AutoFE: {res.timings}")

    auto, timings["auto_features_s"] = block(res.selected)
    run("autofe", [auto, ctx])
    run("autofe+sess", [auto, ctx, sess], save_importance=True)
    del auto, extra_tr, extra_va, tr_s, va_s
    gc.collect()

    # equal feature budget: top-|manual| AutoFE features vs |manual| random grammar features
    by_name = {s.name: s for s in res.pool}
    n_man = len(man_specs)
    autoN, _ = block([by_name[f] for f in res.ranking[:n_man]])
    run("autofe_sameN+sess", [autoN, ctx, sess])
    del autoN
    allspecs = [s for v in search_space().values() for s in v]
    rng = np.random.default_rng(a.seed)
    rnd, _ = block([allspecs[i] for i in rng.choice(len(allspecs), n_man, replace=False)])
    run("random_sameN+sess", [rnd, ctx, sess])
    del rnd

    # union
    union_specs = list({s.name: s for s in man_specs + res.selected}.values())
    uni, _ = block(union_specs)
    run("manual∪autofe+sess", [uni, ctx, sess], save_importance=True)
    del uni
    gc.collect()

    # manual set with windows re-tuned by AutoFE stage 1 (same families, best time scale)
    s1 = res.stage1
    best_scale = s1.loc[s1.groupby("family")["gain"].idxmax()].set_index("family")["feature"].to_dict()
    fam_specs = {s.name: s for v in search_space().values() for s in v}
    retuned = {}
    for s in man_specs:
        b = fam_specs.get(best_scale.get(s.family, ""), s)
        retuned[b.name] = b
    retuned = list(retuned.values())
    pd.Series([s.name for s in retuned]).to_csv(os.path.join(a.out, "manual_retuned.csv"), index=False)
    ret, _ = block(retuned)
    run("manual_retuned_windows+sess", [ret, ctx, sess])
    timings["peak_rss_gb"] = rss_gb()
    json.dump({"results": results, "timings": timings, "data": data_stats},
              open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
