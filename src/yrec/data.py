"""Loading Yambda (flat/50m multi_event + artist/album mappings), user sampling, preprocessing.

Unified event table (one row per event) after `prepare_events`:
    uid      int64
    item_id  int64
    ts       int64   # Yambda timestamp, 5-second bins
    day      int16   # ts // 17280 (relative day index)
    organic  int8    # 1 = organic, 0 = recommendation-driven
    etype    int8    # 0 listen, 1 like, 2 dislike, 3 unlike, 4 undislike
    ratio    float32 # played_ratio_pct (listens only, NaN otherwise)
    tlen     float32 # track_length_seconds (listens only)
    artist_id int64  # -1 if unknown
    album_id  int64  # -1 if unknown
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

TS_PER_DAY = 17280  # 86400 s / 5 s bins
EVENT_CODES = {"listen": 0, "like": 1, "dislike": 2, "unlike": 3, "undislike": 4}
HF_REPO = "yandex/yambda"


# --------------------------------------------------------------------------- download
def download_yambda(size: str = "50m", cache_dir: str | None = None) -> dict[str, str]:
    """Download needed parquet files from HuggingFace. Returns local paths."""
    from huggingface_hub import hf_hub_download

    files = {
        "events": f"flat/{size}/multi_event.parquet",
        "artist": "artist_item_mapping.parquet",
        "album": "album_item_mapping.parquet",
    }
    out = {}
    for k, f in files.items():
        out[k] = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f, cache_dir=cache_dir)
    return out


def _read_mapping(path: str | None, name: str) -> pd.DataFrame | None:
    """Mapping item_id -> <name>_id. Robust to list-valued columns and multiple rows per item
    (keeps the smallest id per item = deterministic 'primary' artist/album)."""
    if path is None or not os.path.exists(path):
        return None
    m = pd.read_parquet(path)
    col = f"{name}_id" if f"{name}_id" in m.columns else [c for c in m.columns if c != "item_id"][0]
    m = m[["item_id", col]].rename(columns={col: f"{name}_id"})
    if m[f"{name}_id"].dtype == object:  # list-valued
        m = m.explode(f"{name}_id").dropna()
    m = m.astype({"item_id": "int64", f"{name}_id": "int64"})
    return m.groupby("item_id", as_index=False)[f"{name}_id"].min()


def load_raw_events(path: str, n_users: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Read multi_event parquet. If n_users is set, sample users uniformly at random and keep
    their FULL history (as required by the task)."""
    import pyarrow.parquet as pq

    cols = ["uid", "item_id", "timestamp", "is_organic", "event_type", "played_ratio_pct", "track_length_seconds"]
    if n_users is not None:
        uids = pq.read_table(path, columns=["uid"]).column("uid")
        uniq = np.unique(uids.to_numpy())
        rng = np.random.default_rng(seed)
        keep = rng.choice(uniq, size=min(n_users, len(uniq)), replace=False)
        tbl = pq.read_table(path, columns=cols, filters=[("uid", "in", keep.tolist())],
                            read_dictionary=["event_type"])
    else:
        tbl = pq.read_table(path, columns=cols, read_dictionary=["event_type"])
    return tbl.to_pandas()


def prepare_events(raw: pd.DataFrame, artist_map: pd.DataFrame | None = None,
                   album_map: pd.DataFrame | None = None) -> pd.DataFrame:
    ev = pd.DataFrame({
        "uid": raw["uid"].astype("int64").to_numpy(),
        "item_id": raw["item_id"].astype("int64").to_numpy(),
        "ts": raw["timestamp"].astype("int64").to_numpy(),
    })
    ev["day"] = (ev["ts"] // TS_PER_DAY).astype("int16")
    ev["organic"] = raw["is_organic"].astype("int8").to_numpy()
    et = raw["event_type"]
    if isinstance(et.dtype, pd.CategoricalDtype):
        lut = np.array([EVENT_CODES[str(c)] for c in et.cat.categories], dtype="int8")
        ev["etype"] = lut[et.cat.codes.to_numpy()]
    elif np.issubdtype(np.asarray(et).dtype, np.integer):
        ev["etype"] = np.asarray(et).astype("int8")
    else:
        ev["etype"] = pd.Series(np.asarray(et).astype(str)).map(EVENT_CODES).to_numpy().astype("int8")
    ev["ratio"] = pd.to_numeric(raw["played_ratio_pct"], errors="coerce").astype("float32").to_numpy()
    ev["tlen"] = pd.to_numeric(raw["track_length_seconds"], errors="coerce").astype("float32").to_numpy()
    for name, m in (("artist", artist_map), ("album", album_map)):
        if m is not None:
            ev = ev.merge(m, on="item_id", how="left")
            ids = ev[f"{name}_id"]
            # dense re-indexing keeps composite keys (uid << 24 | id) small; -1 = unknown
            codes, _ = pd.factorize(ids, use_na_sentinel=True)
            ev[f"{name}_id"] = codes.astype("int64")
        else:
            ev[f"{name}_id"] = np.int64(-1)
    ev = ev.sort_values(["uid", "ts"], kind="stable").reset_index(drop=True)
    return ev


def load_yambda(data_dir: str | None = None, size: str = "50m", n_users: int | None = None,
                seed: int = 0, cache: str | None = None) -> pd.DataFrame:
    """End-to-end loader. data_dir: folder with already downloaded files (Kaggle dataset)
    using HF layout; if None, downloads from HuggingFace."""
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    if data_dir is not None:
        d = Path(data_dir)
        cand = list(d.rglob(f"flat/{size}/multi_event.parquet")) or list(d.rglob("multi_event.parquet"))
        paths = {"events": str(cand[0]),
                 "artist": next(map(str, d.rglob("artist_item_mapping.parquet")), None),
                 "album": next(map(str, d.rglob("album_item_mapping.parquet")), None)}
    else:
        paths = download_yambda(size)
    raw = load_raw_events(paths["events"], n_users=n_users, seed=seed)
    ev = prepare_events(raw, _read_mapping(paths["artist"], "artist"), _read_mapping(paths["album"], "album"))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(cache, index=False)
    return ev


# --------------------------------------------------------------------------- synthetic
def make_synthetic(n_users: int = 300, n_items: int = 3000, n_artists: int = 300, n_days: int = 80,
                   events_per_user_day: float = 12.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic events with the same schema and planted signal (user-artist affinity, item quality,
    user skip propensity, repeat consumption, drifting tastes). Used for unit tests & smoke runs."""
    rng = np.random.default_rng(seed)
    item_artist = rng.integers(0, n_artists, n_items)
    item_album = item_artist * 10 + rng.integers(0, 3, n_items)
    item_q = rng.normal(0, 1, n_items)
    item_pop = rng.zipf(1.6, n_items).astype(float)
    item_pop /= item_pop.sum()
    tlen = rng.integers(120, 360, n_items).astype("float32")
    user_bias = rng.normal(0, 1, n_users)
    rows = []
    for u in range(n_users):
        fav = rng.choice(n_artists, 8, replace=False)
        aff = np.zeros(n_artists)
        aff[fav] = rng.normal(1.5, 0.5, 8)
        active = rng.random(n_days) < rng.uniform(0.3, 0.9)
        for d in np.nonzero(active)[0]:
            n = rng.poisson(events_per_user_day)
            if n == 0:
                continue
            if d > n_days // 2 and rng.random() < 0.05:  # taste drift
                aff[rng.integers(0, n_artists)] += 1.0
            org = (rng.random(n) < 0.35).astype(np.int8)
            items = rng.choice(n_items, n, p=item_pop)
            # organic listens biased to favourite artists
            fav_items = np.nonzero(np.isin(item_artist, fav))[0]
            if len(fav_items):
                sel = org == 1
                items[sel] = rng.choice(fav_items, sel.sum())
            ts = d * TS_PER_DAY + np.sort(rng.integers(0, TS_PER_DAY, n))
            logit = 0.3 + user_bias[u] + item_q[items] + aff[item_artist[items]] + 0.8 * org
            p = 1 / (1 + np.exp(-logit))
            full = rng.random(n) < p
            ratio = np.where(full, rng.integers(50, 101, n), rng.integers(1, 50, n)).astype("float32")
            for k in range(n):
                rows.append((u, items[k], ts[k], org[k], 0, ratio[k], tlen[items[k]]))
                if full[k] and rng.random() < 0.08 * (1 + aff[item_artist[items[k]]]):
                    rows.append((u, items[k], ts[k] + 1, org[k], 1, np.nan, np.nan))
                if (not full[k]) and rng.random() < 0.03:
                    rows.append((u, items[k], ts[k] + 1, org[k], 2, np.nan, np.nan))
    raw = pd.DataFrame(rows, columns=["uid", "item_id", "timestamp", "is_organic", "event_type",
                                      "played_ratio_pct", "track_length_seconds"])
    am = pd.DataFrame({"item_id": np.arange(n_items), "artist_id": item_artist})
    alm = pd.DataFrame({"item_id": np.arange(n_items), "album_id": item_album})
    return prepare_events(raw, am, alm)
