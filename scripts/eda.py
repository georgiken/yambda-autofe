"""Short EDA of Yambda-50M that motivates the task setup and the feature design.

    python scripts/eda.py --n-users 5000 --out results/eda
    python scripts/eda.py --synthetic --out /tmp/eda      # smoke test

Writes eda.md (tables + conclusions-ready numbers), summary.json and figures *.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from yrec.data import TS_PER_DAY, load_yambda, make_synthetic  # noqa: E402
from yrec.protocol import make_split  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ETYPES = {0: "listen", 1: "like", 2: "dislike", 3: "unlike", 4: "undislike"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=5000)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = make_synthetic(300, 3000, 300, 80) if a.synthetic else load_yambda(a.data_dir, "50m", a.n_users, 0)
    S, md = {}, ["# EDA: Yambda-50M" + (" (synthetic)" if a.synthetic else f", {a.n_users} случайных пользователей") + "\n"]

    def fig(name):
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, name), dpi=130)
        plt.close()
        md.append(f"![{name}]({name})\n")

    # ------------------------------------------------------------------ 1. overview
    lis = ev[ev.etype == 0]
    S["events"], S["users"], S["items"] = len(ev), int(ev.uid.nunique()), int(ev.item_id.nunique())
    S["artists"] = int(ev.loc[ev.artist_id >= 0, "artist_id"].nunique())
    S["days"] = int(ev.day.max() - ev.day.min() + 1)
    S["event_types"] = {ETYPES[k]: int(v) for k, v in ev.etype.value_counts().sort_index().items()}
    S["share_listens_recommended"] = float((lis.organic == 0).mean())
    S["artist_known_share"] = float((ev.artist_id >= 0).mean())
    md.append("## 1. Общая картина\n")
    md.append(pd.Series({"событий": S["events"], "пользователей": S["users"], "треков": S["items"],
                         "артистов": S["artists"], "дней": S["days"],
                         "доля рекомендательных прослушиваний": round(S["share_listens_recommended"], 3)})
              .to_frame("значение").to_markdown() + "\n")
    md.append(pd.Series(S["event_types"]).to_frame("число событий").to_markdown() + "\n")

    per_user = ev[ev.etype == 0].groupby("uid").size()
    S["listens_per_user_quantiles"] = per_user.quantile([.1, .25, .5, .75, .9, .99]).round(0).to_dict()
    daily = lis.groupby(["day", "organic"]).size().unstack(fill_value=0)
    fig_, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(np.log10(per_user), bins=50, color="#4C72B0")
    ax[0].set_xlabel("log10(прослушиваний на пользователя)"); ax[0].set_ylabel("пользователей")
    ax[0].set_title("Активность пользователей")
    for c, lab in ((0, "рекомендации"), (1, "органика")):
        if c in daily:
            ax[1].plot(daily.index, daily[c], label=lab)
    sp = make_split(ev, 21, 7, 7)
    for (lo, hi), col in ((sp.train_days, "#55A868"), (sp.val_days, "#C44E52"), (sp.test_days, "#8172B2")):
        ax[1].axvspan(lo, hi + 1, color=col, alpha=.15)
    ax[1].set_xlabel("день"); ax[1].set_ylabel("прослушиваний"); ax[1].legend(fontsize=8)
    ax[1].set_title("Прослушивания по дням; фон: train / val / test")
    fig("eda_activity.png")

    # ------------------------------------------------------------------ 2. target
    r = lis.ratio.clip(0, 150)
    S["ratio_quantiles"] = r.quantile([.1, .25, .5, .75, .9]).to_dict()
    S["full_rate_rec"] = float((lis.loc[lis.organic == 0, "ratio"] >= 50).mean())
    S["full_rate_org"] = float((lis.loc[lis.organic == 1, "ratio"] >= 50).mean())
    S["share_ratio_le_10"] = float((r <= 10).mean())
    S["share_ratio_ge_95"] = float((r >= 95).mean())
    fig_, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for c, lab in ((0, "рекомендации"), (1, "органика")):
        ax[0].hist(lis.loc[lis.organic == c, "ratio"].clip(0, 120), bins=60, alpha=.6, density=True, label=lab)
    ax[0].axvline(50, color="k", ls="--", lw=1); ax[0].set_xlabel("played_ratio_pct"); ax[0].legend(fontsize=8)
    ax[0].set_title("Доля прослушанного трека (порог 50%)")
    rec = lis[lis.organic == 0]
    dr = rec.assign(y=rec.ratio >= 50).groupby("day")["y"].mean()
    ax[1].plot(dr.index, dr.values)
    ax[1].set_xlabel("день"); ax[1].set_ylabel("доля дослушиваний"); ax[1].set_title("Таргет во времени (рекомендации)")
    fig("eda_target.png")
    md.append("## 2. Таргет\n")
    md.append(f"Распределение доли прослушанного бимодально: {S['share_ratio_le_10']:.1%} прослушиваний ≤10% "
              f"(мгновенный скип), {S['share_ratio_ge_95']:.1%} ≥95% (дослушан). Порог 50% разделяет моды. "
              f"Доля дослушиваний: рекомендации {S['full_rate_rec']:.3f}, органика {S['full_rate_org']:.3f}.\n")

    # ------------------------------------------------------------------ 3. repeat consumption & cold start
    # for each recommended listen: was this item / artist / album listened by the user on an EARLIER day?
    first_ui = lis.groupby(["uid", "item_id"])["day"].min().rename("first_ui")
    first_ua = lis[lis.artist_id >= 0].groupby(["uid", "artist_id"])["day"].min().rename("first_ua")
    first_i = lis.groupby("item_id")["day"].min().rename("first_i")
    t = rec[["uid", "item_id", "artist_id", "day", "ratio"]].copy()
    t = t.join(first_ui, on=["uid", "item_id"]).join(first_ua, on=["uid", "artist_id"]).join(first_i, on="item_id")
    t["seen_item"] = t.first_ui < t.day
    t["seen_artist"] = t.first_ua < t.day
    t["seen_item_global"] = t.first_i < t.day
    t["y"] = t.ratio >= 50
    tt = t[t.day >= sp.test_days[0]]
    S["rec_repeat_item_share"] = float(t.seen_item.mean())
    S["rec_known_artist_share"] = float(t.seen_artist.mean())
    S["test_cold_item_share"] = float(1 - tt.seen_item_global.mean())
    grp = {
        "трек уже слушал": t[t.seen_item],
        "новый трек, знакомый артист": t[~t.seen_item & t.seen_artist],
        "новый трек, новый артист": t[~t.seen_item & ~t.seen_artist],
    }
    tab = pd.DataFrame({k: {"доля строк": len(v) / len(t), "доля дослушиваний": v.y.mean()} for k, v in grp.items()}).T
    S["repeat_table"] = tab.round(3).to_dict()
    md.append("## 3. Повторное потребление и косвенные связи (рекомендательные прослушивания)\n")
    md.append(tab.to_markdown(floatfmt=".3f") + "\n")
    md.append(f"В test {S['test_cold_item_share']:.1%} строк — треки, которых не было в истории этой выборки "
              f"(холодные для item-признаков).\n")

    # ------------------------------------------------------------------ 4. popularity tail
    cnt = lis.groupby("item_id").size().sort_values(ascending=False).to_numpy()
    cum = np.cumsum(cnt) / cnt.sum()
    S["top1pct_items_listen_share"] = float(cum[max(int(len(cnt) * .01) - 1, 0)])
    S["items_with_1_listen_share"] = float((cnt == 1).mean())
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(np.arange(1, len(cum) + 1) / len(cum), cum)
    plt.xscale("log"); plt.xlabel("доля треков (по убыванию популярности)"); plt.ylabel("доля прослушиваний")
    plt.title("Длинный хвост"); plt.grid(alpha=.3)
    fig("eda_long_tail.png")
    md.append("## 4. Популярность\n")
    md.append(f"Топ-1% треков дают {S['top1pct_items_listen_share']:.1%} прослушиваний; "
              f"{S['items_with_1_listen_share']:.1%} треков прослушаны ровно один раз.\n")

    # ------------------------------------------------------------------ 5. session dynamics
    s = lis[["uid", "day", "ts", "ratio", "organic"]].copy()
    s["full"] = (s.ratio >= 50).astype(float)
    s["prev_full"] = s.groupby(["uid", "day"])["full"].shift(1)
    s["pos"] = s.groupby(["uid", "day"]).cumcount()
    sr = s[s.organic == 0]
    S["p_full_given_prev_full"] = float(sr.loc[sr.prev_full == 1, "full"].mean())
    S["p_full_given_prev_skip"] = float(sr.loc[sr.prev_full == 0, "full"].mean())
    S["p_full_first_of_day"] = float(sr.loc[sr.prev_full.isna(), "full"].mean())
    hour = ((sr.ts % TS_PER_DAY) // 3600).astype(int)
    by_hour = sr.groupby(hour)["full"].mean()
    by_pos = sr[sr.pos < 50].groupby("pos")["full"].mean()
    fig_, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(by_pos.index, by_pos.values); ax[0].set_xlabel("номер трека за день")
    ax[0].set_ylabel("доля дослушиваний"); ax[0].set_title("Внутридневная динамика")
    ax[1].bar(by_hour.index, by_hour.values); ax[1].set_ylim(by_hour.min() * .95, by_hour.max() * 1.02)
    ax[1].set_xlabel("час (UTC-сдвиг неизвестен)"); ax[1].set_title("Дослушивания по часу")
    fig("eda_session.png")
    md.append("## 5. Сессия\n")
    md.append(f"P(дослушает | предыдущий дослушан) = {S['p_full_given_prev_full']:.3f}, "
              f"P(дослушает | предыдущий скипнут) = {S['p_full_given_prev_skip']:.3f}, "
              f"первый трек дня: {S['p_full_first_of_day']:.3f}. Состояние пользователя внутри дня — сильный сигнал.\n")

    # ------------------------------------------------------------------ 6. user heterogeneity & likes
    ur = sr.groupby("uid")["full"].agg(["mean", "size"])
    ur = ur[ur["size"] >= 20]
    S["user_full_rate_quantiles"] = ur["mean"].quantile([.1, .25, .5, .75, .9]).to_dict()
    lk = ev[ev.etype == 1][["uid", "item_id"]].drop_duplicates()
    listened = lis[["uid", "item_id"]].drop_duplicates()
    S["likes_without_listen_share"] = float(1 - lk.merge(listened, on=["uid", "item_id"]).shape[0] / max(len(lk), 1))
    plt.figure(figsize=(4.5, 3.2))
    plt.hist(ur["mean"], bins=40); plt.xlabel("доля дослушиваний пользователя"); plt.ylabel("пользователей")
    plt.title("Разброс между пользователями")
    fig("eda_users.png")
    md.append("## 6. Различия между пользователями и лайки\n")
    q = S["user_full_rate_quantiles"]
    md.append(f"Доля дослушиваний у пользователей (≥20 рек. прослушиваний): 10% = {q[.1]:.2f}, медиана = {q[.5]:.2f}, "
              f"90% = {q[.9]:.2f} — большая часть глобального AUC объясняется «кто слушает», а не «что». "
              f"{S['likes_without_listen_share']:.1%} лайкнутых (user, track) не имеют прослушивания в логе — "
              f"поэтому like/listen не является долей и требует аккуратной нормировки.\n")

    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float, ensure_ascii=False)
    open(os.path.join(a.out, "eda.md"), "w").write("\n".join(md))
    print("\n".join(md))
    print(json.dumps(S, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
