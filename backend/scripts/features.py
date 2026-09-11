"""
features.py — training-side entry point to the canonical feature definition.

The definition itself lives in src/recsys/serving/ranker_features.py, which the
serving process also imports. This file only adds the training-only helper that
derives per-user genre statistics from the train split.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.serving.ranker_features import (        # noqa: F401
    FEATURE_COLS, N_FEATURES, build_row, build_matrix, rank_norm,
)

# Backwards-compatible alias used by train_ranker.py / evaluate.py.
build_feature_frame = build_matrix


def build_user_genre_stats(train_df, item_meta):
    """
    Per-user genre affinity and share, computed from the TRAIN split only.

    Returns
      stats {(user_id, genre): (mean_rating, share_of_history)}
      top   {user_id: set of the user's three most-watched genres}
    """
    import pandas as pd
    t = train_df.join(item_meta[["primary_genre"]], on="item_id")
    g = t.groupby(["user_id", "primary_genre"])
    stats = pd.DataFrame({"mean_rating": g["rating"].mean(), "cnt": g["rating"].size()})
    stats["share"] = stats["cnt"] / stats.groupby(level=0)["cnt"].transform("sum")
    out = {(u, gen): (float(r.mean_rating), float(r.share))
           for (u, gen), r in stats.iterrows()}
    top = (stats.reset_index()
                .sort_values(["user_id", "cnt"], ascending=[True, False])
                .groupby("user_id")["primary_genre"]
                .apply(lambda s: set(s.head(3))).to_dict())
    return out, top
