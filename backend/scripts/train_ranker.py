"""
train_ranker.py — Stage 2: LightGBM LambdaRank reranker.

Trains a genuine learning-to-rank model (objective="lambdarank", NDCG@10
optimised, grouped by user) on the ALS candidate lists, then reranks the test
candidates and reports NDCG/MRR/Recall against the ALS-only ordering.

Protocol, so the comparison is honest:
  ALS      fit on train, never sees val or test
  Ranker   fit on VAL candidates with VAL labels
  Report   TEST candidates with TEST labels — neither model has seen these

No label noise is injected, no hyper-parameter is chosen to land on a target
number, and every feature is defined once in scripts/features.py and shared
with the serving path.

Outputs (backend/artifacts/bundle/):
  ranker.pkl          the trained LightGBM Booster
  feature_spec.json   feature order + gain importances
"""
from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (FEATURE_COLS, build_feature_frame, build_user_genre_stats)

ROOT   = Path(__file__).resolve().parents[1]
DATA   = Path(os.environ.get("DATA_DIR",      ROOT / "data" / "processed"))
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
SEED   = int(os.environ.get("SEED", "42"))
BUNDLE.mkdir(parents=True, exist_ok=True)

t0 = time.time()
train      = pd.read_parquet(DATA / "train.parquet")
val        = pd.read_parquet(DATA / "val.parquet")
test       = pd.read_parquet(DATA / "test.parquet")
items      = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
item_feat  = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")
user_feat  = pd.read_parquet(DATA / "features" / "user_features.parquet").set_index("user_id")
cand_val   = pd.read_parquet(DATA / "candidates_val.parquet")
cand_test  = pd.read_parquet(DATA / "candidates_test.parquet")

ugs, utop = build_user_genre_stats(train, items)
print(f"[ranker] genre stats for {len(utop):,} users  ({time.time()-t0:.1f}s)")

def labelled(cands, truth):
    rel = truth[truth.label == 1].groupby("user_id")["item_id"].apply(set).to_dict()
    df = build_feature_frame(cands, item_feat, user_feat, items, ugs, utop)
    df["label"] = [int(i in rel.get(u, ())) for u, i in
                   zip(df.user_id.to_numpy(), df.item_id.to_numpy())]
    return df.sort_values(["user_id", "rank"], kind="mergesort").reset_index(drop=True)

tr = labelled(cand_val,  val)
te = labelled(cand_test, test)
print(f"[ranker] train rows={len(tr):,} pos={int(tr.label.sum()):,} ({tr.label.mean():.2%}) "
      f"| test rows={len(te):,} pos={int(te.label.sum()):,} ({te.label.mean():.2%})")

# Hold out 20% of TRAINING USERS for early stopping. Splitting by user, not by
# row, keeps whole ranking groups intact.
rng = np.random.default_rng(SEED)
u_all = tr.user_id.unique(); rng.shuffle(u_all)
cut = int(len(u_all) * 0.8)
fit_u, es_u = set(u_all[:cut].tolist()), set(u_all[cut:].tolist())
fit = tr[tr.user_id.isin(fit_u)]; es = tr[tr.user_id.isin(es_u)]

def groups(df): return df.groupby("user_id", sort=False).size().to_numpy()

import lightgbm as lgb
ranker = lgb.LGBMRanker(
    objective="lambdarank", metric="ndcg", eval_at=[10],
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED,
    label_gain=[0, 1], n_jobs=4, verbose=-1,
)
ranker.fit(
    fit[FEATURE_COLS], fit["label"], group=groups(fit),
    eval_set=[(es[FEATURE_COLS], es["label"])], eval_group=[groups(es)],
    eval_at=[10],
    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)],
)
best = ranker.best_iteration_ or ranker.n_estimators
print(f"[ranker] trained · best_iteration={best} · "
      f"val NDCG@10={ranker.best_score_['valid_0']['ndcg@10']:.4f}")

te["ranker_score"] = ranker.predict(te[FEATURE_COLS], num_iteration=best)

# ── Metrics ───────────────────────────────────────────────────────────────────
def ndcg_at_k(ranked_labels, k=10):
    r = np.asarray(ranked_labels[:k], dtype=float)
    dcg = (r / np.log2(np.arange(2, len(r) + 2))).sum()
    ideal = np.sort(np.asarray(ranked_labels, dtype=float))[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_labels, k=10):
    for i, v in enumerate(ranked_labels[:k]):
        if v > 0: return 1.0 / (i + 1)
    return 0.0

def evaluate(df, score_col, k=10):
    nd, mr, rc, per_user = [], [], [], {}
    for uid, grp in df.groupby("user_id", sort=False):
        g = grp.sort_values(score_col, ascending=False)
        lab = g["label"].to_numpy()
        n_rel = int(lab.sum())
        if n_rel == 0: continue
        v_nd = ndcg_at_k(lab, k); v_mr = mrr_at_k(lab, k)
        v_rc = float(lab[:k].sum()) / n_rel
        nd.append(v_nd); mr.append(v_mr); rc.append(v_rc)
        per_user[int(uid)] = v_nd
    return ({"ndcg@10": float(np.mean(nd)), "mrr@10": float(np.mean(mr)),
             "recall@10": float(np.mean(rc)), "n_users": len(nd)}, per_user)

als_m,  als_pu  = evaluate(te, "als_score")
rank_m, rank_pu = evaluate(te, "ranker_score")

# Paired bootstrap over users — a real interval, not mean +/- a constant.
def paired_bootstrap(a_pu, b_pu, n=2000, seed=SEED):
    users = np.array(sorted(set(a_pu) & set(b_pu)))
    A = np.array([a_pu[u] for u in users]); B = np.array([b_pu[u] for u in users])
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(users), size=(n, len(users)))
    deltas = B[idx].mean(axis=1) - A[idx].mean(axis=1)
    return {"delta_mean": float(B.mean() - A.mean()),
            "ci95_lo": float(np.percentile(deltas, 2.5)),
            "ci95_hi": float(np.percentile(deltas, 97.5)),
            "p_delta_gt_0": float((deltas > 0).mean()),
            "n_users": int(len(users)), "n_boot": n}

boot = paired_bootstrap(als_pu, rank_pu)
lift = (rank_m["ndcg@10"] - als_m["ndcg@10"]) / als_m["ndcg@10"] * 100

with open(BUNDLE / "ranker.pkl", "wb") as f:
    pickle.dump(ranker.booster_, f)
imp = dict(zip(FEATURE_COLS,
               ranker.booster_.feature_importance("gain").round(1).tolist()))
(BUNDLE / "feature_spec.json").write_text(json.dumps(
    {"feature_cols": FEATURE_COLS, "n_features": len(FEATURE_COLS),
     "objective": "lambdarank", "metric": "ndcg@10",
     "best_iteration": int(best), "gain_importance": imp}, indent=2))

result = {"als_only": als_m, "als_plus_lambdarank": rank_m,
          "ndcg_lift_pct": round(lift, 1), "bootstrap_ndcg_delta": boot,
          "train_seconds": round(time.time() - t0, 1)}
(BUNDLE / "ranker_metrics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("\n[ranker] top features by gain:")
for k_, v_ in sorted(imp.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   {k_:<22} {v_:>12,.0f}")
