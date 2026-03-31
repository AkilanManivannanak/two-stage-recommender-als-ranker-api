"""
Script: train_als_and_candidates.py
====================================
Stage 1 of the two-stage pipeline.

Trains an ALS model and generates candidate item sets per user.
Also builds:
  - Co-occurrence neighbor table (hybrid retrieval)
  - User feature table
  - Item feature table
  - Popularity scores

Writes to artifacts/als_{env}/ and data/processed/candidates_{split}.parquet.

Netflix Standard: ALS generates a diverse candidate pool (N=800).
Co-occurrence plugs the long-tail gap.
"""
import json, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR      = Path(os.environ.get("DATA_DIR",      "data/processed"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
ENV           = os.environ.get("ENV", "dev")
N_FACTORS     = int(os.environ.get("N_FACTORS",  "64"))
N_ITER        = int(os.environ.get("N_ITER",     "15"))
REG           = float(os.environ.get("REG",      "0.01"))
TOPN_CANDS    = int(os.environ.get("TOPN_CANDS", "800"))
SEED          = int(os.environ.get("SEED",       "42"))

als_dir = ARTIFACTS_DIR / f"als_{ENV}"
als_dir.mkdir(parents=True, exist_ok=True)
feat_dir = ARTIFACTS_DIR / f"features_{ENV}"
feat_dir.mkdir(parents=True, exist_ok=True)

print(f"[als] Training ALS  factors={N_FACTORS}  iter={N_ITER}  reg={REG}")
t0 = time.time()

train = pd.read_parquet(DATA_DIR/"train.parquet")
items = pd.read_parquet(DATA_DIR/"items.parquet")

# ── ALS ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
users = train["user_id"].unique()
mids  = train["item_id"].unique()
user_idx = {int(u):i for i,u in enumerate(users)}
item_idx = {int(m):i for i,m in enumerate(mids)}
rev_item = {v:k for k,v in item_idx.items()}
nu, ni = len(users), len(mids)
gmean = float(train["rating"].mean())
ub = np.zeros(nu, np.float32); ib = np.zeros(ni, np.float32)
UF = rng.normal(0,.1,(nu,N_FACTORS)).astype(np.float32)
IF = rng.normal(0,.1,(ni,N_FACTORS)).astype(np.float32)
u_ints: dict = defaultdict(list); i_ints: dict = defaultdict(list)
for row in train.itertuples(index=False):
    u=user_idx[int(row.user_id)]; m=item_idx[int(row.item_id)]; r=float(row.rating)
    u_ints[u].append((m,r)); i_ints[m].append((u,r))
reg_eye = REG*np.eye(N_FACTORS,dtype=np.float32)
for it in range(N_ITER):
    for u,pairs in u_ints.items():
        mids_=np.array([p[0] for p in pairs]); rv=np.array([p[1] for p in pairs],np.float32)
        res=rv-gmean-ib[mids_]; ub[u]=float(np.mean(res))/(1+REG); res-=ub[u]
        I=IF[mids_]; UF[u]=np.linalg.solve(I.T@I+reg_eye, I.T@res)
    for m,pairs in i_ints.items():
        uids_=np.array([p[0] for p in pairs]); rv=np.array([p[1] for p in pairs],np.float32)
        res=rv-gmean-ub[uids_]; ib[m]=float(np.mean(res))/(1+REG); res-=ib[m]
        U=UF[uids_]; IF[m]=np.linalg.solve(U.T@U+reg_eye, U.T@res)
    if (it+1)%5==0: print(f"  ALS iter {it+1}/{N_ITER}")

# Save ALS artifacts
np.save(als_dir/"user_factors.npy", UF)
np.save(als_dir/"item_factors.npy", IF)
mappings = {
    "user_id_to_index": {str(k):v for k,v in user_idx.items()},
    "item_id_to_index": {str(k):v for k,v in item_idx.items()},
    "index_to_user_id": [int(users[i]) for i in range(nu)],
    "index_to_item_id": [rev_item[i] for i in range(ni)],
}
with open(als_dir/"mappings.json","w") as f: json.dump(mappings, f)
print(f"  ALS saved → {als_dir}  ({time.time()-t0:.1f}s)")

# ── Popularity ────────────────────────────────────────────────────────
pop = Counter(train["item_id"].values)
pop_items = [int(x[0]) for x in pop.most_common(500)]
pop_scores = [float(pop[x]) for x in pop_items]
with open(feat_dir/"popularity.json","w") as f:
    json.dump({"items": pop_items, "scores": pop_scores}, f)

# ── User features ─────────────────────────────────────────────────────
ts_now = int(time.time())
uf = train.groupby("user_id").agg(
    user_cnt_total=("rating","count"),
    user_avg_rating=("rating","mean"),
).reset_index()
uf["user_tenure_days"]  = rng.integers(1,1000,len(uf)).astype(float)
uf["user_recency_days"] = rng.integers(0,30,len(uf)).astype(float)
uf["user_cnt_7d"]  = (uf["user_cnt_total"] * 0.05).astype(float)
uf["user_cnt_30d"] = (uf["user_cnt_total"] * 0.2).astype(float)
uf.to_parquet(feat_dir/"user_features.parquet", index=False)

# ── Item features ─────────────────────────────────────────────────────
it_feat = train.groupby("item_id").agg(item_cnt_total=("rating","count")).reset_index()
it_feat = it_feat.merge(items[["item_id","year"]], on="item_id", how="left")
it_feat["item_cnt_7d"]       = (it_feat["item_cnt_total"]*0.05).astype(float)
it_feat["item_cnt_30d"]      = (it_feat["item_cnt_total"]*0.2).astype(float)
it_feat["item_age_days"]     = ((2025 - it_feat["year"].fillna(2010))*365).astype(float)
it_feat["item_recency_days"] = rng.integers(0,60,len(it_feat)).astype(float)
it_feat.to_parquet(feat_dir/"item_features.parquet", index=False)

# ── Item metadata ─────────────────────────────────────────────────────
meta = items[["item_id","title","genres"]].copy()
meta.to_parquet(feat_dir/"item_metadata.parquet", index=False)

# ── User recent items ──────────────────────────────────────────────────
recent = (train.sort_values("timestamp").groupby("user_id")["item_id"]
          .apply(lambda x: list(x[-100:])).reset_index())
recent.columns = ["user_id","item_id"]
recent.to_parquet(feat_dir/"user_recent_items.parquet", index=False)

# ── Co-occurrence neighbors ────────────────────────────────────────────
pair_cnt: Counter = Counter()
for uid, grp in train.groupby("user_id"):
    tail = list(grp.sort_values("timestamp").tail(20)["item_id"].values)
    for i in range(len(tail)):
        for j in range(i+1,len(tail)):
            a,b=min(int(tail[i]),int(tail[j])),max(int(tail[i]),int(tail[j]))
            pair_cnt[(a,b)]+=1
co_nb: dict = defaultdict(list)
for (a,b),c in pair_cnt.most_common(200000):
    co_nb[a].append((b,float(c))); co_nb[b].append((a,float(c)))
nb_rows=[]
for it,nbs in co_nb.items():
    nbs_sorted=sorted(nbs,key=lambda x:-x[1])[:50]
    nb_rows.append({"item_id":it,"neighbor_ids":[x[0] for x in nbs_sorted],"neighbor_scores":[x[1] for x in nbs_sorted]})
pd.DataFrame(nb_rows).to_parquet(feat_dir/"cooccurrence_neighbors.parquet",index=False)

# ── Generate candidates ────────────────────────────────────────────────
print(f"  Generating candidates for val split ...")
val   = pd.read_parquet(DATA_DIR/"val.parquet")
recent_set = train.groupby("user_id")["item_id"].apply(set).to_dict()
cand_rows=[]
for uid in val["user_id"].unique():
    uid=int(uid)
    if uid not in user_idx: continue
    u=user_idx[uid]; uvec=UF[u]
    scores=IF@uvec+ib+ub[u]+gmean
    topk_idx=np.argpartition(-scores,min(TOPN_CANDS,ni-1))[:TOPN_CANDS]
    seen=recent_set.get(uid,set())
    for idx in topk_idx:
        iid=rev_item[int(idx)]
        if iid in seen: continue
        cand_rows.append({"user_id":uid,"item_id":iid,"als_score":float(scores[idx]),"split":"val"})
pd.DataFrame(cand_rows).to_parquet(DATA_DIR/"candidates_val.parquet",index=False)
print(f"  {len(cand_rows):,} val candidates generated")

print(f"[als] All done in {time.time()-t0:.1f}s")
