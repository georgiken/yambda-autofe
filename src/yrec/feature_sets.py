"""Feature sets: (1) hand-crafted baseline as an engineer would write it; (2) AutoFE search space."""
from __future__ import annotations

from .features import FeatureSpec as F

W_ALL = None


def manual_features() -> list[F]:
    """Typical hand-made set: fixed 'round' windows (1/7/30 days, all history), the obvious
    stats and conversion rates for user, item, artist/album and user-x-{item, artist, album}."""
    S = []
    # --- user
    S += [F("cnt", "user", "listen", window=w) for w in (1, 7, 30, W_ALL)]
    S += [F("rate", "user", "full_rec", "listen_rec", window=w) for w in (7, 30, W_ALL)]
    S += [F("rate", "user", "full", "listen", window=30),
          F("rate", "user", "like", "listen", window=30),
          F("rate", "user", "dislike", "listen", window=W_ALL),
          F("rate", "user", "early_skip", "listen", window=7),
          F("rate", "user", "listen_org", "listen", window=30),
          F("rec", "user", "any"), F("ten", "user")]
    # --- item
    S += [F("cnt", "item", "listen", window=w) for w in (1, 7, 30, W_ALL)]
    S += [F("rate", "item", "full_rec", "listen_rec", window=w) for w in (7, 30, W_ALL)]
    S += [F("rate", "item", "full", "listen", window=W_ALL),
          F("rate", "item", "like", "listen", window=30), F("rate", "item", "like", "listen", window=W_ALL),
          F("rate", "item", "dislike", "listen", window=W_ALL),
          F("rate", "item", "listen_org", "listen", window=30),
          F("ten", "item"), F("rec", "item", "any")]
    # --- artist / album
    S += [F("cnt", "artist", "listen", window=w) for w in (7, 30)]
    S += [F("rate", "artist", "full", "listen", window=w) for w in (30, W_ALL)]
    S += [F("rate", "artist", "like", "listen", window=W_ALL)]
    S += [F("cnt", "album", "listen", window=30), F("rate", "album", "full", "listen", window=30)]
    # --- user x item (direct)
    S += [F("cnt", "user_item", "listen", window=w) for w in (30, W_ALL)]
    S += [F("cnt", "user_item", s, window=W_ALL) for s in ("full", "like", "dislike")]
    S += [F("rate", "user_item", "full", "listen", window=W_ALL), F("rec", "user_item", "any")]
    # --- user x artist / album (indirect)
    S += [F("cnt", "user_artist", "listen", window=w) for w in (7, 30, W_ALL)]
    S += [F("cnt", "user_artist", "full", window=w) for w in (30, W_ALL)]
    S += [F("cnt", "user_artist", s, window=W_ALL) for s in ("like", "dislike")]
    S += [F("rate", "user_artist", "full", "listen", window=w) for w in (30, W_ALL)]
    S += [F("share", "user_artist", "listen", window=30), F("rec", "user_artist", "any")]
    S += [F("cnt", "user_album", "listen", window=W_ALL), F("rate", "user_album", "full", "listen", window=W_ALL)]
    return S


# --------------------------------------------------------------------------- AutoFE search space
WINDOWS = (1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, None)
HALF_LIVES = (1, 3, 7, 14, 30, 90)
CNT_STATS = ("listen", "listen_org", "listen_rec", "full", "full_rec", "early_skip", "ratio_sum", "like", "dislike")
RATES = (("full", "listen"), ("full_rec", "listen_rec"), ("early_skip", "listen"), ("like", "listen"),
         ("dislike", "listen"), ("ratio_sum", "listen"), ("listen_org", "listen"))
REC_STATS = ("any", "full", "like", "dislike")
ENTS = ("user", "item", "artist", "album", "user_item", "user_artist", "user_album")
PAIRS = ("user_item", "user_artist", "user_album")


def search_space() -> dict[str, list[F]]:
    """Family -> list of variants (time scales). Families are the units of AutoFE search."""
    fam: dict[str, list[F]] = {}

    def add(spec: F):
        fam.setdefault(spec.family, []).append(spec)

    for e in ENTS:
        for s in CNT_STATS:
            for w in WINDOWS:
                add(F("cnt", e, s, window=w))
            for h in HALF_LIVES:
                add(F("dcnt", e, s, half_life=h))
        for a, b in RATES:
            for w in WINDOWS:
                add(F("rate", e, a, b, window=w))
            for h in HALF_LIVES:
                add(F("drate", e, a, b, half_life=h))
        for s in REC_STATS:
            add(F("rec", e, s))
        add(F("ten", e))
    for e in PAIRS:
        for s in ("listen", "full", "like"):
            for w in WINDOWS:
                add(F("share", e, s, window=w))
    return fam
