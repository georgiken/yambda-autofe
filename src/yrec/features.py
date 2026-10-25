"""Leak-free aggregate feature engine.

Every aggregate feature for a target row (keys, d) uses ONLY events with day <= d-1
(snapshot at the start of the target day). This is guaranteed structurally: the engine
stores per-entity cumulative daily sums and looks them up with `searchsorted` at
composite position key*1024 + (d-1).

Feature grammar (FeatureSpec):
    cnt    entity, stat, window w (days, None = all history)      -> sum of stat over [d-w, d-1]
    rate   entity, stat / den, window                             -> cnt(stat)/cnt(den)   (NaN if den==0)
    dcnt   entity, stat, half-life h                              -> sum stat * 2^{-(d-day)/h}
    drate  entity, stat / den, half-life h                        -> dcnt(stat)/dcnt(den)
    share  entity (pair), stat, window; den on parent entity      -> cnt_pair/cnt_parent  (e.g. user-artist / user)
    rec    entity, stat                                           -> days since last day with stat>0
    ten    entity                                                 -> days since first event of entity
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import TS_PER_DAY

DAY_BITS = 1024  # composite key = key * 1024 + day  (day must be < 1024)
SHIFT = np.int64(1 << 24)

# --- entities: name -> columns forming the key
ENTITIES = {
    "user": ("uid",),
    "item": ("item_id",),
    "artist": ("artist_id",),
    "album": ("album_id",),
    "user_item": ("uid", "item_id"),
    "user_artist": ("uid", "artist_id"),
    "user_album": ("uid", "album_id"),
}
PARENT = {"user_item": "user", "user_artist": "user", "user_album": "user"}

STATS = ["listen", "listen_org", "listen_rec", "full", "full_rec", "skip", "early_skip",
         "ratio_sum", "like", "dislike", "unlike"]


def event_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-event indicator/weight columns used by all aggregates."""
    lis = ev["etype"].to_numpy() == 0
    org = ev["organic"].to_numpy() == 1
    r = np.nan_to_num(ev["ratio"].to_numpy(dtype="float64"), nan=0.0)
    full = lis & (r >= 50)
    out = {
        "listen": lis, "listen_org": lis & org, "listen_rec": lis & ~org,
        "full": full, "full_rec": full & ~org, "skip": lis & (r < 50), "early_skip": lis & (r < 10),
        "ratio_sum": np.where(lis, np.clip(r, 0, 100) / 100.0, 0.0),
        "like": ev["etype"].to_numpy() == 1, "dislike": ev["etype"].to_numpy() == 2,
        "unlike": ev["etype"].to_numpy() == 3,
    }
    return pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in out.items()})


def entity_key(df: pd.DataFrame, entity: str) -> np.ndarray:
    """int64 key for an entity; -1 where any component is unknown (<0)."""
    cols = ENTITIES[entity]
    if len(cols) == 1:
        k = df[cols[0]].to_numpy().astype("int64")
        return np.where(k < 0, -1, k)
    u = df[cols[0]].to_numpy().astype("int64")
    o = df[cols[1]].to_numpy().astype("int64")
    assert (o < SHIFT).all() and (u < (1 << 28)).all(), "ids too large for composite key"
    return np.where(o < 0, -1, u * SHIFT + o)


@dataclass(frozen=True)
class FeatureSpec:
    kind: str                 # cnt | rate | dcnt | drate | share | rec | ten
    entity: str
    stat: str = "listen"
    den: str | None = None
    window: int | None = None  # days; None = all history
    half_life: float | None = None

    @property
    def name(self) -> str:
        w = "all" if self.window is None else f"w{self.window}"
        if self.kind == "cnt":
            return f"cnt__{self.entity}__{self.stat}__{w}"
        if self.kind == "rate":
            return f"rate__{self.entity}__{self.stat}_per_{self.den}__{w}"
        if self.kind == "dcnt":
            return f"dcnt__{self.entity}__{self.stat}__h{self.half_life:g}"
        if self.kind == "drate":
            return f"drate__{self.entity}__{self.stat}_per_{self.den}__h{self.half_life:g}"
        if self.kind == "share":
            return f"share__{self.entity}__{self.stat}__{w}"
        if self.kind == "rec":
            return f"rec__{self.entity}__{self.stat}"
        if self.kind == "ten":
            return f"ten__{self.entity}"
        raise ValueError(self.kind)

    @property
    def family(self) -> str:
        """Family = everything except the time scale (window / half-life)."""
        if self.kind in ("cnt", "dcnt"):
            return f"cnt__{self.entity}__{self.stat}"
        if self.kind in ("rate", "drate"):
            return f"rate__{self.entity}__{self.stat}_per_{self.den}"
        return self.name if self.kind != "share" else f"share__{self.entity}__{self.stat}"


class _EntityStore:
    """Cumulative daily sums for one entity."""

    def __init__(self, keys: np.ndarray, day: np.ndarray, stats: pd.DataFrame):
        m = keys >= 0
        comp = keys[m] * DAY_BITS + day[m].astype("int64")
        uniq, inv = np.unique(comp, return_inverse=True)
        self.comp = uniq
        key = uniq // DAY_BITS
        self.day = (uniq % DAY_BITS).astype(np.int16)
        n = len(uniq)
        self.daily = {s: np.bincount(inv, weights=stats[s].to_numpy()[m], minlength=n).astype("float32")
                      for s in stats.columns}
        # key-start index for each row -> to make cumsums restart per key
        newkey = np.ones(n, dtype=bool)
        newkey[1:] = key[1:] != key[:-1]
        del key, inv, comp
        self.start = np.maximum.accumulate(np.where(newkey, np.arange(n), 0)).astype(np.int32)
        self._cum: OrderedDict = OrderedDict()
        self.max_decay_cache = 4
        self._last: dict = {}
        self.d0 = DAY_BITS / 2  # fixed reference -> decay sums depend only on the past (and stay in float64 range)

    def _cumsum(self, x: np.ndarray) -> np.ndarray:
        # per-key cumulative sum (no global cumsum -> no cancellation error for tiny decayed weights)
        return pd.Series(x).groupby(self.start).cumsum().to_numpy()

    def cum(self, stat: str, h: float | None = None) -> np.ndarray:
        """Window cumsums are cached permanently; decayed cumsums in a small LRU (memory bound:
        pair entities have tens of millions of daily rows)."""
        k = (stat, h)
        if k in self._cum:
            if h is not None:
                self._cum.move_to_end(k)
            return self._cum[k]
        x = self.daily[stat].astype("float64")
        if h is not None:
            x = x * np.exp2((self.day - self.d0) / h)
        c = self._cumsum(x)
        if h is None:
            c = c.astype(np.float32)  # window sums: per-key counts < 2^24 are exact in float32; halves memory
        self._cum[k] = c
        dk = [kk for kk in self._cum if kk[1] is not None]
        if len(dk) > self.max_decay_cache:
            self._cum.pop(dk[0])
        return c

    def clear(self):
        self._cum.clear()
        self._last.clear()

    def locate(self, q_key: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index of the last row with same key and day <= t; valid mask."""
        pos = np.searchsorted(self.comp, q_key * DAY_BITS + t, side="right") - 1
        posc = np.clip(pos, 0, max(len(self.comp) - 1, 0))
        valid = (pos >= 0) & (q_key >= 0) & (len(self.comp) > 0)
        if len(self.comp):
            valid &= (self.comp[posc] // DAY_BITS) == q_key
        return posc, valid

    def loc(self, q_key, d, off, memo=None):
        """locate(q_key, d - off) with optional memoisation (memo: dict owned by the caller)."""
        if memo is None:
            return self.locate(q_key, d - off)
        k = (id(self), off)
        if k not in memo:
            p, v = self.locate(q_key, d - off)
            memo[k] = (p.astype(np.int32 if len(self.comp) < 2**31 else np.int64), v)
        return memo[k]

    def window_sum(self, stat, q_key, d, w, memo=None):
        p1, v1 = self.loc(q_key, d, 1, memo)
        c = self.cum(stat)
        s = np.where(v1, c[p1], 0.0)
        if w is not None:
            p0, v0 = self.loc(q_key, d, 1 + w, memo)
            s = s - np.where(v0, c[p0], 0.0)
        return s

    def decay_sum(self, stat, q_key, d, h, memo=None):
        p1, v1 = self.loc(q_key, d, 1, memo)
        c = self.cum(stat, h)
        return np.where(v1, c[p1] * np.exp2(-(d - self.d0) / h), 0.0)

    def last_day(self, stat: str | None, q_key, d, memo=None):
        """Last day <= d-1 with stat>0 (stat=None: any event)."""
        if stat is None:
            p, v = self.loc(q_key, d, 1, memo)
            return np.where(v, self.day[p], np.nan)
        if stat not in self._last:
            m = self.daily[stat] > 0
            self._last[stat] = (self.comp[m], self.comp[m] // DAY_BITS, self.day[m])
        comp, key, day = self._last[stat]
        pos = np.searchsorted(comp, q_key * DAY_BITS + (d - 1), side="right") - 1
        posc = np.clip(pos, 0, max(len(comp) - 1, 0))
        valid = (pos >= 0) & (q_key >= 0) & (len(comp) > 0)
        if len(comp):
            valid &= key[posc] == q_key
            return np.where(valid, day[posc], np.nan)
        return np.full(len(q_key), np.nan)

    def first_day(self, q_key, d, memo=None):
        p, v = self.loc(q_key, d, 1, memo)
        return np.where(v, self.day[self.start[p]], np.nan)


class FeatureEngine:
    """Build once on the full event log; compute any FeatureSpec for target rows.
    Leakage-safety: lookups only ever read cumulative state at day <= d-1."""

    def __init__(self, ev: pd.DataFrame, entities: Iterable[str] = tuple(ENTITIES)):
        assert ev["day"].max() < DAY_BITS, "time span must be < 1024 days"
        self._memo = {}
        st = event_stats(ev)
        day = ev["day"].to_numpy()
        self.stores = {}
        for e in entities:
            self.stores[e] = _EntityStore(entity_key(ev, e), day, st)
        del st

    def clear_cache(self):
        """Drop per-frame memo and cumulative caches (they are rebuilt lazily on demand)."""
        self._memo.clear()
        for st in self.stores.values():
            st.clear()

    def _ctx(self, tgt: pd.DataFrame, entity: str):
        """Per-target-frame memo of keys and searchsorted positions (huge speed-up when many
        features are computed for the same frame, e.g. in AutoFE). Keeps at most 2 frames."""
        tid = id(tgt)
        if tid not in self._memo or self._memo[tid][0] is not tgt:
            if len(self._memo) >= 2:
                self._memo.pop(next(iter(self._memo)))
            self._memo[tid] = (tgt, {})
        m = self._memo[tid][1]
        if ("key", entity) not in m:
            m[("key", entity)] = entity_key(tgt, entity)
        if "d" not in m:
            m["d"] = tgt["day"].to_numpy().astype("int64")
        return m[("key", entity)], m["d"], m

    def compute(self, spec: FeatureSpec, tgt: pd.DataFrame, memo: bool = True) -> np.ndarray:
        st = self.stores[spec.entity]
        q, d, mm = self._ctx(tgt, spec.entity)
        M = mm if memo else None
        unknown = q < 0
        if spec.kind == "cnt":
            v = st.window_sum(spec.stat, q, d, spec.window, M)
        elif spec.kind == "rate":
            num = st.window_sum(spec.stat, q, d, spec.window, M)
            den = st.window_sum(spec.den, q, d, spec.window, M)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        elif spec.kind == "dcnt":
            v = st.decay_sum(spec.stat, q, d, spec.half_life, M)
        elif spec.kind == "drate":
            num = st.decay_sum(spec.stat, q, d, spec.half_life, M)
            den = st.decay_sum(spec.den, q, d, spec.half_life, M)
            v = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        elif spec.kind == "share":
            par = self.stores[PARENT[spec.entity]]
            num = st.window_sum(spec.stat, q, d, spec.window, M)
            pq, _, _ = self._ctx(tgt, PARENT[spec.entity])
            den = par.window_sum(spec.stat, pq, d, spec.window, M)
            v = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        elif spec.kind == "rec":
            v = d - st.last_day(None if spec.stat == "any" else spec.stat, q, d, M)
        elif spec.kind == "ten":
            v = d - st.first_day(q, d, M)
        else:
            raise ValueError(spec.kind)
        v = v.astype("float32")
        if unknown.any():
            v[unknown] = np.nan
        return v

    def frame(self, specs: Iterable[FeatureSpec], tgt: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({s.name: self.compute(s, tgt) for s in specs}, index=tgt.index)


# --------------------------------------------------------------------------- context & session
def context_features(tgt: pd.DataFrame) -> pd.DataFrame:
    """Known-at-request-time context: hour of day (UTC proxy), weekday proxy, track length."""
    tod = (tgt["ts"].to_numpy() % TS_PER_DAY) / 3600.0
    return pd.DataFrame({"ctx_hour": tod.astype("float32"),
                         "ctx_dow": (tgt["day"].to_numpy() % 7).astype("float32"),
                         "item_tlen": tgt["tlen"].to_numpy().astype("float32")}, index=tgt.index)


def session_features(ev: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """Intra-day features from the user's LISTEN events with timestamp STRICTLY earlier than the target
    (same 5-second bin excluded). Likes/dislikes are deliberately not used intra-day: a like on the current
    track may be logged before its listen event, which would leak the label. Time gap to the previous event is
    also not used (if ts marks the END of a listen, gap ~ current listen duration -> leak)."""
    days = np.unique(tgt["day"].to_numpy())
    lis = ev[(ev["etype"] == 0) & ev["day"].isin(days)][["uid", "ts", "day", "item_id", "artist_id", "organic", "ratio"]]
    r = np.nan_to_num(lis["ratio"].to_numpy(dtype="float64"))
    g = pd.DataFrame({
        "uid": lis["uid"].to_numpy(), "day": lis["day"].to_numpy(), "ts": lis["ts"].to_numpy(),
        "n": 1.0, "full": (r >= 50).astype(float), "rec": (lis["organic"].to_numpy() == 0).astype(float),
        "full_rec": ((r >= 50) & (lis["organic"].to_numpy() == 0)).astype(float),
        "esk": (r < 10).astype(float), "rsum": np.clip(r, 0, 100) / 100,
    })
    # aggregate per (uid, day, ts) bin, then exclusive cumulative sums within (uid, day)
    b = g.groupby(["uid", "day", "ts"], sort=True).sum().reset_index()
    cols = ["n", "full", "rec", "full_rec", "esk", "rsum"]
    cs = b.groupby(["uid", "day"])[cols].cumsum() - b[cols]
    b[[f"c_{c}" for c in cols]] = cs.to_numpy()
    # previous-bin values: last 1 / last 3 played ratios via rolling over bins (bin-mean ratio)
    b["bin_r"] = b["rsum"] / b["n"]
    grp = b.groupby(["uid", "day"])["bin_r"]
    b["prev1_r"] = grp.shift(1)
    k = b.groupby(["uid", "day"]).cumcount().to_numpy()
    csr = b.groupby(["uid", "day"])["bin_r"].cumsum()
    c1 = csr.groupby([b["uid"], b["day"]]).shift(1).fillna(0).to_numpy()
    c4 = csr.groupby([b["uid"], b["day"]]).shift(4).fillna(0).to_numpy()
    b["prev3_r"] = np.where(k > 0, (c1 - c4) / np.minimum(k, 3).clip(min=1), np.nan)
    # last listened artist before the target bin
    la = lis[["uid", "day", "ts", "artist_id"]].drop_duplicates(["uid", "day", "ts"], keep="last")
    b = b.merge(la, on=["uid", "day", "ts"], how="left")
    b["prev_artist"] = b.groupby(["uid", "day"])["artist_id"].shift(1)
    t = tgt[["uid", "day", "ts", "artist_id"]].merge(
        b[["uid", "day", "ts"] + [f"c_{c}" for c in cols] + ["prev1_r", "prev3_r", "prev_artist"]],
        on=["uid", "day", "ts"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "sess_n_prev": t["c_n"].to_numpy(),
            "sess_full_rate": np.where(t["c_n"] > 0, t["c_full"] / t["c_n"], np.nan),
            "sess_rec_full_rate": np.where(t["c_rec"] > 0, t["c_full_rec"] / t["c_rec"], np.nan),
            "sess_early_skip_rate": np.where(t["c_n"] > 0, t["c_esk"] / t["c_n"], np.nan),
            "sess_mean_ratio": np.where(t["c_n"] > 0, t["c_rsum"] / t["c_n"], np.nan),
            "sess_prev1_ratio": t["prev1_r"].to_numpy(),
            "sess_prev3_ratio": t["prev3_r"].to_numpy(),
            "sess_same_artist_prev": np.where(t["prev_artist"].isna(), np.nan,
                                              (t["prev_artist"] == t["artist_id"]).astype(float)),
        }, index=tgt.index).astype("float32")
    return out
