"""
train_als.py — Stage 1: ALS collaborative filtering + candidate generation.

Fits implicit-feedback ALS (Hu, Koren & Volinsky 2008) on the train split and
writes the factor matrices the serving layer loads at startup. Also emits the
top-N candidate lists that Stage 2 (the ranker) reranks, so retrieval and
ranking are evaluated on the same candidates a request would see.

Hyper-parameters are the standard ML-1M implicit-ALS settings, not tuned to hit
a target metric:
  factors=64  regularization=0.05  alpha=40  iterations=20

Outputs (backend/artifacts/bundle/):
  item_factors.pkl   {item_id: float32[64]}
  user_factors.pkl   {user_id: float32[64]}
  als_model.pkl      {factors, regularization, alpha, iterations, id maps}
Outputs (backend/data/processed/):
  candidates_val.parquet  candidates_test.parquet   user_id item_id als_score rank
"""
from __future__ import annotations
import json, os, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT    = Path(__file__).resolve().parents[1]
DATA    = Path(os.environ.get("DATA_DIR",      ROOT / "data" / "processed"))
BUNDLE  = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
FACTORS = int(os.environ.get("ALS_FACTORS", "64"))
REG     = float(os.environ.get("ALS_REG", "0.05"))
ALPHA   = float(os.environ.get("ALS_ALPHA", "40"))
ITERS   = int(os.environ.get("ALS_ITERS", "20"))
TOPN    = int(os.environ.get("ALS_TOPN", "200"))
SEED    = int(os.environ.get("SEED", "42"))
BUNDLE.mkdir(parents=True, exist_ok=True)

t0 = time.time()
train = pd.read_parquet(DATA / "train.parquet")
val   = pd.read_parquet(DATA / "val.parquet")
test  = pd.read_parquet(DATA / "test.parquet")

# Implicit signal = positive interactions only (rating >= threshold).
pos = train[train.label == 1]
uids = np.sort(train.user_id.unique())
iids = np.sort(train.item_id.unique())
u2ix = {u: i for i, u in enumerate(uids)}
i2ix = {m: i for i, m in enumerate(iids)}

rows = pos.user_id.map(u2ix).to_numpy()
cols = pos.item_id.map(i2ix).to_numpy()
# Confidence c = 1 + alpha (binary preference, uniform confidence per HKV08).
vals = np.ones(len(pos), dtype=np.float32) * ALPHA
ui = sp.csr_matrix((vals, (rows, cols)), shape=(len(uids), len(iids)), dtype=np.float32)
print(f"[als] matrix {ui.shape}  nnz={ui.nnz:,}  density={ui.nnz/(ui.shape[0]*ui.shape[1]):.4%}")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from implicit.als import AlternatingLeastSquares
model = AlternatingLeastSquares(factors=FACTORS, regularization=REG,
                                iterations=ITERS, random_state=SEED,
                                calculate_training_loss=False)
model.fit(ui, show_progress=False)
U = np.asarray(model.user_factors); V = np.asarray(model.item_factors)
print(f"[als] fit done  user_factors={U.shape}  item_factors={V.shape}  in {time.time()-t0:.1f}s")

item_factors = {int(iids[j]): V[j].astype(np.float32) for j in range(len(iids))}
user_factors = {int(uids[i]): U[i].astype(np.float32) for i in range(len(uids))}
with open(BUNDLE / "item_factors.pkl", "wb") as f: pickle.dump(item_factors, f)
with open(BUNDLE / "user_factors.pkl", "wb") as f: pickle.dump(user_factors, f)
with open(BUNDLE / "als_model.pkl", "wb") as f:
    pickle.dump({"factors": FACTORS, "regularization": REG, "alpha": ALPHA,
                 "iterations": ITERS, "seed": SEED,
                 "user_ids": uids.tolist(), "item_ids": iids.tolist(),
                 "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)

# ── Candidate generation ──────────────────────────────────────────────────────
# Score every item for every user, mask items already seen in train (a real
# system does not re-recommend what the user already watched), take top-N.
seen = pos.groupby("user_id")["item_id"].apply(set).to_dict()

def candidates_for(split_df: pd.DataFrame, name: str) -> pd.DataFrame:
    users_in_split = np.sort(split_df.user_id.unique())
    out_u, out_i, out_s, out_r = [], [], [], []
    B = 512
    for start in range(0, len(users_in_split), B):
        chunk = users_in_split[start:start + B]
        ix = np.array([u2ix[u] for u in chunk if u in u2ix])
        if not len(ix): continue
        scores = U[ix] @ V.T                                   # (b, n_items)
        for row, uix in enumerate(ix):
            uid = int(uids[uix])
            s = scores[row].copy()
            for it in seen.get(uid, ()):
                j = i2ix.get(it)
                if j is not None: s[j] = -np.inf
            top = np.argpartition(-s, TOPN)[:TOPN]
            top = top[np.argsort(-s[top])]
            out_u.extend([uid] * TOPN)
            out_i.extend(iids[top].tolist())
            out_s.extend(s[top].astype(float).tolist())
            out_r.extend(range(TOPN))
    df = pd.DataFrame({"user_id": out_u, "item_id": out_i,
                       "als_score": out_s, "rank": out_r})
    df.to_parquet(DATA / f"candidates_{name}.parquet", index=False)
    print(f"[als] candidates_{name}: {len(df):,} rows · {df.user_id.nunique():,} users · top-{TOPN}")
    return df

cand_val  = candidates_for(val,  "val")
cand_test = candidates_for(test, "test")

# ── Retrieval recall: can the ranker even find the answer? ────────────────────
def recall_at(cands: pd.DataFrame, truth_df: pd.DataFrame, k: int) -> float:
    rel = truth_df[truth_df.label == 1].groupby("user_id")["item_id"].apply(set).to_dict()
    byu = cands[cands["rank"] < k].groupby("user_id")["item_id"].apply(list).to_dict()
    vs = [len(set(byu.get(u, [])) & r) / len(r) for u, r in rel.items() if r]
    return float(np.mean(vs)) if vs else 0.0

stats = {
    "factors": FACTORS, "regularization": REG, "alpha": ALPHA, "iterations": ITERS,
    "n_users": len(uids), "n_items": len(iids), "nnz": int(ui.nnz),
    "candidate_recall@50_val":  round(recall_at(cand_val, val, 50), 4),
    "candidate_recall@200_val": round(recall_at(cand_val, val, 200), 4),
    "candidate_recall@50_test": round(recall_at(cand_test, test, 50), 4),
    "candidate_recall@200_test":round(recall_at(cand_test, test, 200), 4),
    "train_seconds": round(time.time() - t0, 1),
}
(BUNDLE / "als_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
