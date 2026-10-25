"""Leakage & correctness tests (run: `pytest -q` or `python tests/test_leakage.py`).

1. Oracle test: engine values == naive pandas computation over events with day < d.
2. Future-perturbation test: arbitrarily changing events on/after the target day does not change
   any aggregate feature of that day.
3. Session test: removing every event at/after the target timestamp (except the target itself)
   does not change session features.
4. Split test: train < val < test in time; targets of a split never leak into earlier splits.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yrec.data import make_synthetic  # noqa: E402
from yrec.feature_sets import manual_features, search_space  # noqa: E402
from yrec.features import (FeatureEngine, FeatureSpec, entity_key, event_stats, session_features,  # noqa: E402
                           session_grammar)
from yrec.protocol import make_split, make_targets  # noqa: E402

EV = make_synthetic(n_users=60, n_items=400, n_artists=40, n_days=70, events_per_user_day=8, seed=1)


def _sample_specs(n=120, seed=0):
    fam = search_space()
    allspecs = [s for v in fam.values() for s in v] + manual_features()
    rng = np.random.default_rng(seed)
    return [allspecs[i] for i in rng.choice(len(allspecs), n, replace=False)] + manual_features()


def _naive(spec: FeatureSpec, ev: pd.DataFrame, row) -> float:
    st = event_stats(ev)
    d = int(row["day"])
    key = entity_key(pd.DataFrame([row]), spec.entity)[0]
    if key < 0:
        return np.nan
    ek = entity_key(ev, spec.entity)
    past = (ev["day"].to_numpy() < d) & (ek == key)

    def wsum(stat, w, k=key, ekk=ek):
        m = (ev["day"].to_numpy() < d) & (ekk == k)
        if w is not None:
            m &= ev["day"].to_numpy() >= d - w
        return st[stat].to_numpy()[m].sum()

    def dsum(stat, h):
        return (st[stat].to_numpy()[past] * np.exp2(-(d - ev["day"].to_numpy()[past]) / h)).sum()

    if spec.kind == "cnt":
        return wsum(spec.stat, spec.window)
    if spec.kind == "rate":
        den = wsum(spec.den, spec.window)
        return wsum(spec.stat, spec.window) / den if den > 0 else np.nan
    if spec.kind == "dcnt":
        return dsum(spec.stat, spec.half_life)
    if spec.kind == "drate":
        den = dsum(spec.den, spec.half_life)
        return dsum(spec.stat, spec.half_life) / den if den > 1e-3 else np.nan
    if spec.kind == "share":
        par = {"user_item": "user", "user_artist": "user", "user_album": "user"}[spec.entity]
        pk = entity_key(pd.DataFrame([row]), par)[0]
        den = wsum(spec.stat, spec.window, pk, entity_key(ev, par))
        return wsum(spec.stat, spec.window) / den if den > 0 else np.nan
    if spec.kind == "rec":
        m = past if spec.stat == "any" else past & (st[spec.stat].to_numpy() > 0)
        return d - ev["day"].to_numpy()[m].max() if m.any() else np.nan
    if spec.kind == "ten":
        return d - ev["day"].to_numpy()[past].min() if past.any() else np.nan
    raise ValueError


def test_oracle():
    sp = make_split(EV, n_train=10, n_val=5, n_test=5)
    tgt = make_targets(EV, sp.test_days).sample(25, random_state=0).reset_index(drop=True)
    eng = FeatureEngine(EV)
    specs = _sample_specs(80)
    bad = []
    for s in specs:
        v = eng.compute(s, tgt)
        for i in range(len(tgt)):
            ref = _naive(s, EV, tgt.iloc[i])
            if not (np.isnan(ref) and np.isnan(v[i])) and not np.isclose(v[i], ref, rtol=1e-4, atol=1e-4):
                bad.append((s.name, i, v[i], ref))
    assert not bad, bad[:10]


def test_future_perturbation():
    sp = make_split(EV, n_train=10, n_val=5, n_test=5)
    d0 = sp.val_days[0]
    tgt = make_targets(EV, (d0, d0))
    specs = _sample_specs(150, seed=1)
    f1 = FeatureEngine(EV).frame(specs, tgt)
    rng = np.random.default_rng(0)
    ev2 = EV.copy()
    fut = ev2["day"] >= d0
    # shuffle items, flip organic flags, rescale ratios, drop 30% of future events
    ev2.loc[fut, "item_id"] = rng.permutation(ev2.loc[fut, "item_id"].to_numpy())
    ev2.loc[fut, "organic"] = 1 - ev2.loc[fut, "organic"]
    ev2.loc[fut, "ratio"] = rng.uniform(0, 100, fut.sum()).astype("float32")
    ev2 = ev2[~(fut & (rng.random(len(ev2)) < 0.3))]
    f2 = FeatureEngine(ev2).frame(specs, tgt)
    pd.testing.assert_frame_equal(f1, f2)


def test_session_no_future():
    sp = make_split(EV, n_train=10, n_val=5, n_test=5)
    tgt = make_targets(EV, sp.test_days)
    full = session_features(EV, tgt)
    full_g = session_grammar(EV, tgt)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(tgt), 40, replace=False):
        r = tgt.iloc[i]
        # keep only events strictly before r.ts, plus the target event itself
        ev_cut = EV[(EV["uid"] != r["uid"]) | (EV["ts"] < r["ts"]) |
                    ((EV["ts"] == r["ts"]) & (EV["item_id"] == r["item_id"]) & (EV["etype"] == 0))]
        one = session_features(ev_cut, tgt.iloc[[i]])
        a, b = full.iloc[i].to_numpy(), one.iloc[0].to_numpy()
        assert np.allclose(a, b, equal_nan=True), (i, a, b)
        g1, g2 = full_g.iloc[i].to_numpy(), session_grammar(ev_cut, tgt.iloc[[i]]).iloc[0].to_numpy()
        assert np.allclose(g1, g2, equal_nan=True), (i, full_g.columns[~np.isclose(g1, g2, equal_nan=True)])


def test_split_order():
    sp = make_split(EV, n_train=10, n_val=5, n_test=5)
    assert sp.train_days[1] < sp.val_days[0] <= sp.val_days[1] < sp.test_days[0]
    tr = make_targets(EV, sp.train_days)
    te = make_targets(EV, sp.test_days)
    assert tr["day"].max() < te["day"].min()
    assert set(tr["label"].unique()) <= {0, 1}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
