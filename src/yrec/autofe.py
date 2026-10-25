"""AutoFE: automatic search over the feature grammar.

Search space: ~150 families (entity x statistic / rate) x time scales (14 windows + 6 exponential
half-lives) ~ 2400 candidate features (see feature_sets.search_space).

Stage 1 - time-scale search inside each family (cheap, univariate, honest):
    every variant is target-encoded on the TRAIN sample (quantile bins, smoothed mean label)
    and scored by the LogLoss reduction it gives on the VAL sample. The best `k` variants of each
    family (with |corr| < max_corr between them) survive. This answers "why 7 days and not 10?"
    from data instead of by convention.
Stage 2 - model-based selection:
    GBDT on the whole stage-1 pool; features ranked by gain; the prefix size N is picked from a grid
    by VAL LogLoss (re-training each time). Optionally restrict to N = size of manual set.
Everything is measured on train/val only; test is never touched here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FeatureEngine, FeatureSpec
from .feature_sets import search_space
from .model import GBDT


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def univariate_gain(x_tr, y_tr, x_va, y_va, n_bins: int = 32, m: float = 50.0) -> float:
    """LogLoss reduction on val of a binned target encoding fitted on train."""
    prior = y_tr.mean()
    base = _logloss(y_va, np.full(len(y_va), prior))
    fin = np.isfinite(x_tr)
    if fin.sum() < 10:
        edges = np.array([])
    else:
        edges = np.unique(np.quantile(x_tr[fin], np.linspace(0, 1, n_bins + 1)[1:-1]))

    def bins(x):
        b = np.searchsorted(edges, x, side="right")
        return np.where(np.isfinite(x), b, len(edges) + 1)

    btr, bva = bins(x_tr), bins(x_va)
    nb = len(edges) + 2
    s = np.bincount(btr, weights=y_tr, minlength=nb)
    c = np.bincount(btr, minlength=nb)
    enc = (s + m * prior) / (c + m)
    return base - _logloss(y_va, enc[bva])


@dataclass
class AutoFEResult:
    stage1: pd.DataFrame                         # per-variant scores
    pool: list[FeatureSpec]                      # stage-1 survivors
    ranking: list[str]                           # stage-2 order (by gain)
    curve: pd.DataFrame                          # N -> val metrics
    selected: list[FeatureSpec]
    timings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Precomputed:
    """A candidate whose values are given as a precomputed column (e.g. session grammar)."""
    name: str
    family: str
    kind: str = "session"
    entity: str = "session"
    window: int | None = None
    half_life: float | None = None


def run_autofe(engine: FeatureEngine, tr: pd.DataFrame, va: pd.DataFrame, extra_tr: pd.DataFrame | None = None,
               extra_va: pd.DataFrame | None = None, k_per_family: int = 2, max_corr: float = 0.97,
               n_grid=(10, 20, 30, 45, 60, 80, 100), fixed_n: int | None = None, max_n: int = 100,
               gbdt_params: dict | None = None, verbose: bool = True,
               pre_tr: pd.DataFrame | None = None, pre_va: pd.DataFrame | None = None,
               pre_family=None) -> AutoFEResult:
    """tr/va: target frames (with 'label'); extra_*: fixed feature blocks always included in stage 2
    (e.g. context / session), so AutoFE selects features that add value on top of them.
    pre_tr/pre_va: extra candidate columns (e.g. session grammar) searched together with the aggregate
    grammar; pre_family(name) -> (family, k) groups them into families."""
    t0 = time.time()
    fam = {f: [(s, None) for s in v] for f, v in search_space().items()}
    if pre_tr is not None:
        for c in pre_tr.columns:
            f, k = pre_family(c)
            fam.setdefault(f, []).append((Precomputed(c, f, window=k), c))
    ytr, yva = tr["label"].to_numpy().astype(float), va["label"].to_numpy().astype(float)
    rows, pool, cache_tr, cache_va = [], [], {}, {}
    for i, (fname, specs) in enumerate(fam.items()):
        scored = []
        for s, col in specs:
            if col is None:
                xtr, xva = engine.compute(s, tr), engine.compute(s, va)
            else:
                xtr, xva = pre_tr[col].to_numpy(), pre_va[col].to_numpy()
            g = univariate_gain(xtr.astype("float64"), ytr, xva.astype("float64"), yva)
            rows.append({"family": fname, "feature": s.name, "kind": s.kind, "entity": s.entity,
                         "window": s.window, "half_life": s.half_life, "gain": g})
            scored.append((g, s, xtr, xva))
        scored.sort(key=lambda z: -z[0])
        kept = []
        for g, s, xtr, xva in scored:
            if g <= 0 or len(kept) >= k_per_family:
                continue
            ok = True
            for _, _, ktr, _ in kept:
                a, b = np.nan_to_num(xtr), np.nan_to_num(ktr)
                if a.std() > 0 and b.std() > 0 and abs(np.corrcoef(a, b)[0, 1]) > max_corr:
                    ok = False
                    break
            if ok:
                kept.append((g, s, xtr, xva))
        for g, s, xtr, xva in kept:
            pool.append(s)
            cache_tr[s.name], cache_va[s.name] = xtr, xva
        if verbose and i % 25 == 0:
            print(f"  stage1 {i}/{len(fam)} families, pool={len(pool)}, {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    stage1 = pd.DataFrame(rows)

    Xtr = pd.DataFrame(cache_tr, index=tr.index)
    Xva = pd.DataFrame(cache_va, index=va.index)
    if extra_tr is not None:
        Xtr, Xva = pd.concat([Xtr, extra_tr], axis=1), pd.concat([Xva, extra_va], axis=1)
    fixed_cols = [] if extra_tr is None else list(extra_tr.columns)
    m = GBDT(gbdt_params).fit(Xtr, ytr, Xva, yva)
    imp = m.importance()
    ranking = [f for f in imp.index if f not in fixed_cols]
    grid = sorted({n for n in n_grid if n < len(ranking)} | {min(len(ranking), max_n)})
    if fixed_n is not None:
        grid = [min(fixed_n, len(ranking))]
    curve = []
    for n in grid:
        cols = ranking[:n] + fixed_cols
        mm = GBDT(gbdt_params).fit(Xtr[cols], ytr, Xva[cols], yva)
        p = mm.predict(Xva[cols])
        curve.append({"n": n, "val_logloss": _logloss(yva, p), "iters": mm.best_iter})
        if verbose:
            print(f"  stage2 N={n}: val logloss {curve[-1]['val_logloss']:.5f}", flush=True)
    curve = pd.DataFrame(curve)
    best_n = int(curve.loc[curve["val_logloss"].idxmin(), "n"])
    by_name = {s.name: s for s in pool}
    selected = [by_name[f] for f in ranking[:best_n]]
    return AutoFEResult(stage1, pool, ranking, curve, selected,
                        {"stage1_s": t1 - t0, "stage2_s": time.time() - t1, "n_candidates": len(stage1),
                         "n_pool": len(pool), "best_n": best_n})
