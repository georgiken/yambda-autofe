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
from yrec.features import (FeatureEngine, FeatureSpec, context_features, session_family,  # noqa: E402
                           session_features, session_grammar)
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
    ap.add_argument("--autofe-from", default=None,
                    help="reuse AutoFE selection from another run dir (e.g. found on 1000 users) instead of searching")
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
    timings, results, preds = {}, {}, {}

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
    ctx = {k: context_features(df) for k, df in (("tr", tr), ("va", va), ("te", te))}
    sess = {k: session_features(ev, df) for k, df in (("tr", tr), ("va", va), ("te", te))}
    timings["session_s"] = time.time() - t
    # AutoFE samples need the raw log for session features -> build them now, then free the raw log
    if not a.skip_autofe:
        t = time.time()
        sg = {k: session_grammar(ev, df) for k, df in (("tr", tr), ("va", va), ("te", te))}
        fsel = os.path.join(a.autofe_from or "", "autofe_full_selected.csv")
        if a.autofe_from and os.path.exists(fsel):  # keep only the transferred columns (memory)
            keep = [c for c in pd.read_csv(fsel).iloc[:, 0] if c in sg["tr"].columns]
            sg = {k: v[keep] for k, v in sg.items()}
        timings["session_grammar_s"] = time.time() - t
    if not a.skip_autofe and not a.autofe_from:
        tr_s = make_targets(ev, sp.train_days, a.autofe_train_rows, seed=a.seed + 1)
        va_s = make_targets(ev, sp.val_days, a.autofe_val_rows, seed=a.seed + 1)
        ctx_s = (context_features(tr_s), context_features(va_s))
        extra_tr = pd.concat([ctx_s[0], session_features(ev, tr_s)], axis=1)
        extra_va = pd.concat([ctx_s[1], session_features(ev, va_s)], axis=1)
        pre_s = (session_grammar(ev, tr_s), session_grammar(ev, va_s))
    gc.collect()
    # build the aggregate engine last: session features above need only the raw log, so peak memory is
    # max(raw log + session grammar, raw log + engine) instead of their sum
    t = time.time()
    eng = FeatureEngine(ev)
    timings["engine_build_s"] = time.time() - t
    log(f"engine built in {timings['engine_build_s']:.1f}s")
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
        p_te = m.predict(Xs["te"])
        r = {"n_features": Xs["tr"].shape[1], "fit_s": m.fit_time, "best_iter": m.best_iter,
             "val": evaluate(va.label.values, m.predict(Xs["va"]), group_id(va)),
             "test": evaluate(te.label.values, p_te, group_id(te))}
        results[name] = r
        preds[name] = p_te.astype("float32")
        np.savez_compressed(os.path.join(a.out, "preds_test.npz"), y=te.label.values, uid=te.uid.values,
                            group=group_id(te), **preds)
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
    registry = {x.name: x for v in search_space().values() for x in v}
    registry.update({x.name: x for x in man_specs})
    if a.autofe_from:
        src = a.autofe_from
        rd = lambda f: list(pd.read_csv(os.path.join(src, f)).iloc[:, 0])  # noqa: E731
        selected = [registry[n] for n in rd("autofe_selected.csv")]
        ranking = rd("autofe_ranking.csv")
        retuned = [registry[n] for n in rd("manual_retuned.csv")]
        full_names = rd("autofe_full_selected.csv") if os.path.exists(os.path.join(src, "autofe_full_selected.csv")) else None
        timings["autofe_from"] = src
        log(f"AutoFE selection reused from {src}: {len(selected)} features (no search)")
    else:
        eng.clear_cache()
        t = time.time()
        res = run_autofe(eng, tr_s, va_s, extra_tr, extra_va,
                         n_grid=(10, 30) if a.smoke else (10, 20, 30, 45, 60, 80, 100))
        timings["autofe_search_s"] = time.time() - t
        timings.update({f"autofe_{k}": v for k, v in res.timings.items()})
        res.stage1.to_csv(os.path.join(a.out, "autofe_stage1.csv"), index=False)
        res.curve.to_csv(os.path.join(a.out, "autofe_curve.csv"), index=False)
        pd.Series(res.ranking).to_csv(os.path.join(a.out, "autofe_ranking.csv"), index=False)
        pd.Series([x.name for x in res.selected]).to_csv(os.path.join(a.out, "autofe_selected.csv"), index=False)
        log(f"AutoFE: {res.timings}")
        selected, ranking = res.selected, res.ranking
        # manual set with windows re-tuned by AutoFE stage 1 (same families, best time scale)
        s1 = res.stage1
        best_scale = s1.loc[s1.groupby("family")["gain"].idxmax()].set_index("family")["feature"].to_dict()
        rt = {}
        for x in man_specs:
            b = registry.get(best_scale.get(x.family, ""), x)
            rt[b.name] = b
        retuned = list(rt.values())
        pd.Series([x.name for x in retuned]).to_csv(os.path.join(a.out, "manual_retuned.csv"), index=False)
        del res
        # ---- AutoFE-full: aggregate grammar + SESSION grammar searched jointly, only context is fixed
        eng.clear_cache()
        t = time.time()
        res2 = run_autofe(eng, tr_s, va_s, ctx_s[0], ctx_s[1], pre_tr=pre_s[0], pre_va=pre_s[1],
                          pre_family=session_family,
                          n_grid=(10, 30) if a.smoke else (10, 20, 30, 45, 60, 80, 100))
        timings["autofe_full_search_s"] = time.time() - t
        timings.update({f"autofe_full_{k}": v for k, v in res2.timings.items()})
        res2.stage1.to_csv(os.path.join(a.out, "autofe_full_stage1.csv"), index=False)
        res2.curve.to_csv(os.path.join(a.out, "autofe_full_curve.csv"), index=False)
        pd.Series(res2.ranking).to_csv(os.path.join(a.out, "autofe_full_ranking.csv"), index=False)
        full_names = [x.name for x in res2.selected]
        pd.Series(full_names).to_csv(os.path.join(a.out, "autofe_full_selected.csv"), index=False)
        log(f"AutoFE-full: {res2.timings}")
        del res2, extra_tr, extra_va, tr_s, va_s, pre_s, ctx_s
        gc.collect()
        eng.clear_cache()

    auto, timings["auto_features_s"] = block(selected)
    run("autofe", [auto, ctx])
    run("autofe+sess", [auto, ctx, sess], save_importance=True)
    del auto
    gc.collect()

    # equal feature budget: top-|manual| AutoFE features vs |manual| random grammar features
    n_man = len(man_specs)
    autoN, _ = block([registry[f] for f in ranking[:n_man]])
    run("autofe_sameN+sess", [autoN, ctx, sess])
    del autoN
    allspecs = [x for v in search_space().values() for x in v]
    rng = np.random.default_rng(a.seed)
    rnd, _ = block([allspecs[i] for i in rng.choice(len(allspecs), n_man, replace=False)])
    run("random_sameN+sess", [rnd, ctx, sess])
    del rnd

    union_specs = list({x.name: x for x in man_specs + selected}.values())
    uni, _ = block(union_specs)
    run("manual∪autofe+sess", [uni, ctx, sess], save_importance=True)
    del uni
    gc.collect()

    ret, _ = block(retuned)
    run("manual_retuned_windows+sess", [ret, ctx, sess])
    del ret
    run("autofe+sess_lambdarank", [block(selected)[0], ctx, sess], params={"objective": "lambdarank"})
    # ---- AutoFE-full (aggregates + session searched; no hand-written session features)
    if full_names:
        t = time.time()
        agg_specs = [registry[n] for n in full_names if n in registry]
        sg_cols = [n for n in full_names if n not in registry]
        full_blk = block(agg_specs)[0] if agg_specs else {k: pd.DataFrame(index=df.index) for k, df in
                                                           (("tr", tr), ("va", va), ("te", te))}
        full_blk = {k: pd.concat([full_blk[k], sg[k][sg_cols]], axis=1) for k in full_blk}
        timings["autofe_full_features_s"] = time.time() - t
        log(f"AutoFE-full set: {len(agg_specs)} aggregate + {len(sg_cols)} session features")
        run("autofe_full", [full_blk, ctx], save_importance=True)
        run("autofe_full+sess", [full_blk, ctx, sess])
        run("autofe_full_lambdarank", [full_blk, ctx], params={"objective": "lambdarank"})
        del full_blk
    timings["peak_rss_gb"] = rss_gb()
    json.dump({"results": results, "timings": timings, "data": data_stats},
              open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    main()
