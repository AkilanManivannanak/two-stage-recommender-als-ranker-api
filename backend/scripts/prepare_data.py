"""
prepare_data.py — Stage 0 of the CineWave pipeline.

Reads raw MovieLens-1M (ratings.dat / movies.dat / users.dat) and produces the
parquet splits and feature tables every downstream stage consumes.

Design decisions, stated explicitly because the evaluation depends on them:

  Split        Per-user CHRONOLOGICAL 80/10/10. Each user's interactions are
               sorted by timestamp; the earliest 80% are train, the next 10%
               val, the final 10% test. This mirrors the production setting
               (predict the future from the past) and avoids the optimistic
               bias of a random split, where a model can see a user's later
               behaviour while scoring their earlier behaviour.

  Relevance    An item is relevant for ranking metrics iff rating >= 4.0.
               ML-1M is explicit-feedback; this is the standard binarisation
               (Rendle et al., He et al.). Lower-rated interactions are kept
               in the interaction history (they are still signal about what a
               user chose to watch) but do not count as a hit.

  Leakage      All feature tables are computed from the TRAIN split only.
               Item popularity, user activity counts and recency are exactly
               the quantities a serving system would have at prediction time.

Outputs (backend/data/processed/):
  train.parquet val.parquet test.parquet   user_id item_id rating timestamp label
  items.parquet   item_id title genres primary_genre year
  users.parquet   user_id gender age occupation zip
  features/user_features.parquet  activity + recency, train-only
  features/item_features.parquet  popularity + age, train-only
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
RAW       = Path(os.environ.get("RAW_DIR",  ROOT / "data" / "raw"))
OUT       = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
FEAT      = OUT / "features"
POS_THRESHOLD = float(os.environ.get("POS_THRESHOLD", "4.0"))
SEED      = int(os.environ.get("SEED", "42"))

OUT.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"[prepare] raw={RAW}  out={OUT}  positive_threshold={POS_THRESHOLD}")

# ── Load ──────────────────────────────────────────────────────────────────────
ratings = pd.read_csv(RAW / "ratings.dat", sep="::", engine="python",
                      names=["user_id", "item_id", "rating", "timestamp"],
                      encoding="latin-1")
movies  = pd.read_csv(RAW / "movies.dat", sep="::", engine="python",
                      names=["item_id", "title", "genres"], encoding="latin-1")
users   = pd.read_csv(RAW / "users.dat", sep="::", engine="python",
                      names=["user_id", "gender", "age", "occupation", "zip"],
                      encoding="latin-1")

print(f"  loaded {len(ratings):,} ratings · {ratings.user_id.nunique():,} users "
      f"· {ratings.item_id.nunique():,} rated items · {len(movies):,} catalog items")

# ── Items ─────────────────────────────────────────────────────────────────────
movies["primary_genre"] = movies["genres"].str.split("|").str[0]
movies["year"] = (movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
                  .astype("float").fillna(0).astype(int))
movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

# ── Chronological per-user split ──────────────────────────────────────────────
ratings = ratings.sort_values(["user_id", "timestamp"], kind="mergesort")
rank_in_user = ratings.groupby("user_id").cumcount()
n_per_user   = ratings.groupby("user_id")["item_id"].transform("size")
frac         = rank_in_user / n_per_user

split = np.where(frac < 0.80, "train", np.where(frac < 0.90, "val", "test"))
ratings["split"] = split
ratings["label"] = (ratings["rating"] >= POS_THRESHOLD).astype(int)

train = ratings[ratings.split == "train"].drop(columns="split").reset_index(drop=True)
val   = ratings[ratings.split == "val"].drop(columns="split").reset_index(drop=True)
test  = ratings[ratings.split == "test"].drop(columns="split").reset_index(drop=True)

for name, df in [("train", train), ("val", val), ("test", test)]:
    pos = int(df.label.sum())
    print(f"  {name:<5} {len(df):>8,} rows · {df.user_id.nunique():>5,} users "
          f"· {pos:>7,} positives ({pos/max(len(df),1):.1%})")

# Guard: a chronological split must not leak. Every train timestamp for a user
# must precede that user's first val timestamp.
_tr_max = train.groupby("user_id")["timestamp"].max()
_va_min = val.groupby("user_id")["timestamp"].min()
_shared = _tr_max.index.intersection(_va_min.index)
_bad = int((_tr_max.loc[_shared] > _va_min.loc[_shared]).sum())
assert _bad == 0, f"temporal leakage: {_bad} users have train after val"
print(f"  leakage check passed on {len(_shared):,} users")

# ── Feature tables — TRAIN ONLY ───────────────────────────────────────────────
now_ts = int(train["timestamp"].max())
DAY = 86400.0

g = train.groupby("user_id")
user_features = pd.DataFrame({
    "user_cnt_total":    g["item_id"].size(),
    "user_cnt_pos":      g["label"].sum(),
    "user_avg_rating":   g["rating"].mean(),
    "user_first_ts":     g["timestamp"].min(),
    "user_last_ts":      g["timestamp"].max(),
}).reset_index()
user_features["user_tenure_days"]  = (user_features.user_last_ts - user_features.user_first_ts) / DAY
user_features["user_recency_days"] = (now_ts - user_features.user_last_ts) / DAY
user_features["user_pos_rate"]     = user_features.user_cnt_pos / user_features.user_cnt_total

_w7  = train[train.timestamp >= now_ts - 7 * DAY].groupby("user_id").size()
_w30 = train[train.timestamp >= now_ts - 30 * DAY].groupby("user_id").size()
user_features["user_cnt_7d"]  = user_features.user_id.map(_w7).fillna(0).astype(int)
user_features["user_cnt_30d"] = user_features.user_id.map(_w30).fillna(0).astype(int)

gi = train.groupby("item_id")
item_features = pd.DataFrame({
    "item_cnt_total":  gi["user_id"].size(),
    "item_cnt_pos":    gi["label"].sum(),
    "item_avg_rating": gi["rating"].mean(),
    "item_first_ts":   gi["timestamp"].min(),
    "item_last_ts":    gi["timestamp"].max(),
}).reset_index()
item_features["item_age_days"]     = (now_ts - item_features.item_first_ts) / DAY
item_features["item_recency_days"] = (now_ts - item_features.item_last_ts) / DAY
item_features["item_pos_rate"]     = item_features.item_cnt_pos / item_features.item_cnt_total
_i7  = train[train.timestamp >= now_ts - 7 * DAY].groupby("item_id").size()
_i30 = train[train.timestamp >= now_ts - 30 * DAY].groupby("item_id").size()
item_features["item_cnt_7d"]  = item_features.item_id.map(_i7).fillna(0).astype(int)
item_features["item_cnt_30d"] = item_features.item_id.map(_i30).fillna(0).astype(int)

# Exposure propensity for IPS-corrected evaluation: P(item shown) under the
# logging policy, approximated by its empirical train share. Floored so a rare
# item cannot produce an unbounded 1/p weight.
share = item_features.item_cnt_total / item_features.item_cnt_total.sum()
item_features["propensity"] = np.maximum(share, 1e-5)

# ── Write ─────────────────────────────────────────────────────────────────────
train.to_parquet(OUT / "train.parquet", index=False)
val.to_parquet(OUT / "val.parquet", index=False)
test.to_parquet(OUT / "test.parquet", index=False)
movies.to_parquet(OUT / "items.parquet", index=False)
users.to_parquet(OUT / "users.parquet", index=False)
user_features.to_parquet(FEAT / "user_features.parquet", index=False)
item_features.to_parquet(FEAT / "item_features.parquet", index=False)

manifest = {
    "dataset":           "MovieLens-1M",
    "n_ratings":         int(len(ratings)),
    "n_users":           int(ratings.user_id.nunique()),
    "n_items_rated":     int(ratings.item_id.nunique()),
    "n_items_catalog":   int(len(movies)),
    "positive_threshold": POS_THRESHOLD,
    "split":             "per-user chronological 80/10/10",
    "n_train":           int(len(train)),
    "n_val":             int(len(val)),
    "n_test":            int(len(test)),
    "train_positives":   int(train.label.sum()),
    "val_positives":     int(val.label.sum()),
    "test_positives":    int(test.label.sum()),
    "reference_ts":      now_ts,
    "prepared_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"[prepare] wrote {OUT}  in {time.time()-t0:.1f}s")
print(json.dumps(manifest, indent=2))
