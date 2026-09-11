"""
ranker_features.py — THE canonical ranker feature definition.

Training (backend/scripts/) imports this module. Serving (this package) imports
this module. There is exactly one definition of the feature vector, in one file,
so training/serving skew cannot arise from two copies drifting apart.

The module exposes two builders that must agree exactly:

  build_row(...)     one candidate, used on the serving path
  build_matrix(...)  a whole candidate frame, vectorised, used in training

`tests/test_feature_parity.py` asserts they produce identical vectors on sampled
rows. If someone edits one and not the other, that test fails.

Historical note: the previous serving code built a 16-dim vector out of
different quantities than the 18-dim vector the training script built, then
called `predict_proba` on a LightGBM Booster (which has no such method) inside a
bare `except Exception: pass`. The ranker therefore never ran in production, and
`als_score`, `ranker_score` and `score` were all the same number.
"""
from __future__ import annotations
import math
from typing import Any, Mapping, Optional

FEATURE_COLS = [
    "als_score",
    "als_rank_norm",
    "item_pop_log",
    "item_pos_rate",
    "item_avg_rating",
    "item_recency_days",
    "item_cnt_30d_log",
    "item_year",
    "user_cnt_log",
    "user_avg_rating",
    "user_pos_rate",
    "user_tenure_days",
    "genre_match",
    "user_genre_affinity",
    "genre_share",
]
N_FEATURES = len(FEATURE_COLS)

# Defaults for an item or user the model never saw. Chosen to be neutral (a
# cold item looks unpopular rather than popular), never to flatter the model.
ITEM_DEFAULTS = {"item_cnt_total": 0.0, "item_pos_rate": 0.0, "item_avg_rating": 0.0,
                 "item_recency_days": 0.0, "item_cnt_30d": 0.0, "year": 0.0}
USER_DEFAULTS = {"user_cnt_total": 0.0, "user_avg_rating": 0.0, "user_pos_rate": 0.0,
                 "user_tenure_days": 0.0}


def build_row(als_score: float, als_rank_norm: float,
              item_f: Optional[Mapping[str, Any]],
              user_f: Optional[Mapping[str, Any]],
              genre_affinity: float, genre_share: float,
              genre_match: int) -> list[float]:
    """One candidate -> one feature vector, in FEATURE_COLS order."""
    i = dict(ITEM_DEFAULTS); i.update(item_f or {})
    u = dict(USER_DEFAULTS); u.update(user_f or {})
    return [
        float(als_score),
        float(als_rank_norm),
        math.log1p(float(i["item_cnt_total"] or 0.0)),
        float(i["item_pos_rate"] or 0.0),
        float(i["item_avg_rating"] or 0.0),
        float(i["item_recency_days"] or 0.0),
        math.log1p(float(i["item_cnt_30d"] or 0.0)),
        float(i["year"] or 0.0),
        math.log1p(float(u["user_cnt_total"] or 0.0)),
        float(u["user_avg_rating"] or 0.0),
        float(u["user_pos_rate"] or 0.0),
        float(u["user_tenure_days"] or 0.0),
        float(genre_match),
        float(genre_affinity),
        float(genre_share),
    ]


def build_matrix(cands, item_feat, user_feat, item_meta,
                 user_genre_stats, user_top_genres):
    """
    Vectorised equivalent of build_row over a candidate DataFrame.

    cands            user_id item_id als_score rank
    item_feat        DataFrame indexed by item_id
    user_feat        DataFrame indexed by user_id
    item_meta        DataFrame indexed by item_id (primary_genre, year)
    user_genre_stats {(user_id, genre): (mean_rating, share)}
    user_top_genres  {user_id: set of top-3 genres}
    """
    import numpy as np

    df = cands.copy()
    df = df.join(item_feat, on="item_id").join(user_feat, on="user_id")
    df = df.join(item_meta[["primary_genre", "year"]], on="item_id")

    span = max(int(df["rank"].max()), 1)
    df["als_rank_norm"]    = df["rank"] / span
    df["item_pop_log"]     = np.log1p(df["item_cnt_total"].fillna(0.0))
    df["item_cnt_30d_log"] = np.log1p(df["item_cnt_30d"].fillna(0.0))
    df["item_year"]        = df["year"].fillna(0.0)
    df["user_cnt_log"]     = np.log1p(df["user_cnt_total"].fillna(0.0))
    for c in ("item_pos_rate", "item_avg_rating", "item_recency_days",
              "user_avg_rating", "user_pos_rate", "user_tenure_days"):
        df[c] = df[c].fillna(0.0)

    keys = list(zip(df["user_id"].to_numpy(),
                    df["primary_genre"].fillna("?").to_numpy()))
    df["user_genre_affinity"] = np.fromiter(
        (user_genre_stats.get(k, (0.0, 0.0))[0] for k in keys),
        dtype=np.float64, count=len(keys))
    df["genre_share"] = np.fromiter(
        (user_genre_stats.get(k, (0.0, 0.0))[1] for k in keys),
        dtype=np.float64, count=len(keys))
    df["genre_match"] = np.fromiter(
        (float(g in user_top_genres.get(u, ())) for u, g in keys),
        dtype=np.float64, count=len(keys))

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"features missing after join: {missing}")
    return df


def rank_norm(position: int, n_candidates: int) -> float:
    """Serving-side equivalent of the training rank normalisation."""
    return float(position) / max(int(n_candidates) - 1, 1)
