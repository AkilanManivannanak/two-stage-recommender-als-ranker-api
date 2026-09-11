"""
build_rl_dataset.py — build offline RL training data from real interactions.

The /rl/train/offline endpoint used to synthesise its own training set:

    sessions = [{"user_id": rng.randint(1, 1000),
                 "slates": [{"items": rng.sample(item_ids, 10),
                             "order": list(range(10)),
                             "reward": rng.uniform(0.0, 3.0)}]}
                for _ in range(body.n_sessions)]

The reward was drawn independently of the slate, so no ordering of those items
was better than any other and the policy gradient had nothing to learn. Every
user was also given identical activity ({n_ratings: 50, avg_rating: 3.5}), which
pinned two of the eight state features to a constant.

This builds the dataset from data instead:

  slate       the ALS candidate list actually generated for that user
  reward      derived from the user's OWN held-out rating of the item
                  rating >= 4  -> 1.0   (played and liked)
                  rating == 3  -> 0.3   (watched, indifferent)
                  rating <= 2  -> 0.0   (skip)
              items with no rating in the window are not scored and are
              excluded from the reward rather than assigned a default
  order       the logging policy's order (ALS rank), which is what the
              off-policy correction needs to know
  propensity  the item's empirical exposure share, from the train split
  activity    that user's real interaction count and mean rating

Writes artifacts/bundle/rl_sessions.jsonl
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
DATA   = Path(os.environ.get("DATA_DIR",      ROOT / "data" / "processed"))
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
SLATE  = int(os.environ.get("RL_SLATE_SIZE", "10"))
MAX_U  = int(os.environ.get("RL_MAX_USERS", "3000"))
SEED   = int(os.environ.get("SEED", "42"))
BUNDLE.mkdir(parents=True, exist_ok=True)

t0 = time.time()
val       = pd.read_parquet(DATA / "val.parquet")
items     = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
item_feat = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")
user_feat = pd.read_parquet(DATA / "features" / "user_features.parquet").set_index("user_id")
cands     = pd.read_parquet(DATA / "candidates_val.parquet")

def reward_from_rating(r: float) -> float:
    if r >= 4.0: return 1.0
    if r >= 3.0: return 0.3
    return 0.0

rated = {(int(u), int(i)): float(r) for u, i, r in
         zip(val.user_id, val.item_id, val.rating)}
propensity = item_feat["propensity"].to_dict()
genre_of   = items["primary_genre"].to_dict()
year_of    = items["year"].to_dict()
pop_of     = item_feat["item_cnt_total"].to_dict()
avg_of     = item_feat["item_avg_rating"].to_dict()

rng = np.random.default_rng(SEED)
users = np.sort(val.user_id.unique())
if len(users) > MAX_U:
    users = rng.choice(users, size=MAX_U, replace=False)
users = np.sort(users)

top = (cands[cands["rank"] < SLATE]
       .sort_values(["user_id", "rank"], kind="mergesort")
       .groupby("user_id"))

out_path = BUNDLE / "rl_sessions.jsonl"
n_sessions = n_scored = 0
reward_hist: list[float] = []

with open(out_path, "w") as fh:
    for uid in users:
        uid = int(uid)
        if uid not in top.groups: continue
        grp = top.get_group(uid)
        slate_items, rewards = [], []
        for pos, (iid, als) in enumerate(zip(grp.item_id, grp.als_score)):
            iid = int(iid)
            key = (uid, iid)
            if key not in rated:
                continue            # unobserved: excluded, not defaulted
            rew = reward_from_rating(rated[key])
            slate_items.append({
                "item_id":       iid,
                "position":      pos,
                "als_score":     float(als),
                "primary_genre": genre_of.get(iid, "?"),
                "year":          int(year_of.get(iid, 0) or 0),
                "popularity":    float(pop_of.get(iid, 0.0)),
                "avg_rating":    float(avg_of.get(iid, 0.0)),
                "propensity":    float(propensity.get(iid, 1e-4)),
            })
            rewards.append(rew)
        if not slate_items:
            continue

        uf = user_feat.loc[uid] if uid in user_feat.index else None
        session = {
            "user_id": uid,
            "activity": {
                "n_ratings":  int(uf["user_cnt_total"]) if uf is not None else 0,
                "avg_rating": float(uf["user_avg_rating"]) if uf is not None else 0.0,
                "pos_rate":   float(uf["user_pos_rate"]) if uf is not None else 0.0,
            },
            "slates": [{
                "items":  slate_items,
                "order":  list(range(len(slate_items))),
                "rewards": rewards,
                "reward": float(np.mean(rewards)),
            }],
        }
        fh.write(json.dumps(session) + "\n")
        n_sessions += 1
        n_scored += len(slate_items)
        reward_hist.extend(rewards)

stats = {
    "source":          "MovieLens-1M validation split, ALS candidate slates",
    "n_sessions":      n_sessions,
    "n_scored_items":  n_scored,
    "slate_size":      SLATE,
    "mean_reward":     round(float(np.mean(reward_hist)), 4) if reward_hist else 0.0,
    "reward_std":      round(float(np.std(reward_hist)), 4) if reward_hist else 0.0,
    "pct_positive":    round(float(np.mean([r == 1.0 for r in reward_hist])), 4) if reward_hist else 0.0,
    "note": ("Rewards come from each user's own held-out rating. Items the user "
             "never rated are excluded rather than given a default reward."),
    "built_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(BUNDLE / "rl_dataset_stats.json").write_text(json.dumps(stats, indent=2))
print(f"[rl] wrote {out_path}")
print(json.dumps(stats, indent=2))
