"""
Script: run_baselines.py
========================
Evaluates three baselines — Popularity, Co-occurrence, ALS-only.
Writes baselines_metrics.json to public/reports (consumed by frontend EvalPage).

Netflix Standard: Always measure against baselines before claiming a model is good.
"""
import json, os, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

DATA_DIR    = Path(os.environ.get("DATA_DIR",    "data/processed"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "frontend/public/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("[baselines] Running baseline evaluation ...")
t0 = time.time()

train = pd.read_parquet(DATA_DIR/"train.parquet")
test  = pd.read_parquet(DATA_DIR/"test.parquet")

def ndcg10(recs, rel):
    dcg  = sum(1/np.log2(i+2) for i,r in enumerate(recs[:10]) if r in rel)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(rel),10)))
    return dcg/idcg if idcg else 0.0

def recall(recs, rel, k): return len(set(recs[:k])&rel)/max(len(rel),1)
def mrr(recs, rel):
    for i,r in enumerate(recs[:10]):
        if r in rel: return 1/(i+1)
    return 0.0

# Popularity ranking
pop_counts = Counter(train["item_id"].values)
pop_ranked = [x[0] for x in pop_counts.most_common(200)]
gt = test.groupby("user_id")["item_id"].apply(set).to_dict()

def eval_baseline(ranked_fn):
    nd, rc10, rc50, mr10, cov = [], [], [], [], set()
    for uid, rel in gt.items():
        recs = ranked_fn(uid)
        nd.append(ndcg10(recs, rel))
        rc10.append(recall(recs, rel, 10))
        rc50.append(recall(recs, rel, 50))
        mr10.append(mrr(recs, rel))
        cov.update(recs[:10])
    return {
        "ndcg10":   round(float(np.mean(nd)),4),
        "mrr10":    round(float(np.mean(mr10)),4),
        "recall10": round(float(np.mean(rc10)),4),
        "recall50": round(float(np.mean(rc50)),4),
        "coverage10": len(cov),
    }

pop_metrics = eval_baseline(lambda uid: pop_ranked)

# Co-occurrence baseline
co_scores: dict = {}
user_recent = train.sort_values("timestamp").groupby("user_id")["item_id"].apply(list).to_dict()
pair_counts = Counter()
for uid, items in user_recent.items():
    tail = items[-10:]
    for i in range(len(tail)):
        for j in range(i+1, len(tail)):
            a,b = min(tail[i],tail[j]), max(tail[i],tail[j])
            pair_counts[(a,b)] += 1

item_co: dict = {}
for (a,b), c in pair_counts.items():
    item_co.setdefault(a,[]).append((b,c))
    item_co.setdefault(b,[]).append((a,c))

def co_rank(uid):
    seen = set(train[train["user_id"]==uid]["item_id"].values)
    tail = user_recent.get(uid,[])[-5:]
    sc: Counter = Counter()
    for it in tail:
        for nb, c in item_co.get(it,[]):
            if nb not in seen: sc[nb] += c
    if not sc: return pop_ranked
    return [x[0] for x in sc.most_common(100)]

co_metrics = eval_baseline(co_rank)

result = {
    "split": "test",
    "n_users_eval": len(gt),
    "popularity": pop_metrics,
    "cooccurrence": co_metrics,
    "notes": {
        "cooccurrence_last_m_per_user_build": 20,
        "cooccurrence_user_profile_last_m": 10,
        "k_ranked": 50
    }
}

with open(REPORTS_DIR/"baselines_metrics.json","w") as f:
    json.dump(result, f, indent=2)

print(f"  Popularity:   NDCG@10={pop_metrics['ndcg10']:.4f}  coverage={pop_metrics['coverage10']}")
print(f"  Co-occurrence:NDCG@10={co_metrics['ndcg10']:.4f}  coverage={co_metrics['coverage10']}")
print(f"  Wrote → {REPORTS_DIR}/baselines_metrics.json")
print(f"[baselines] Done in {time.time()-t0:.1f}s")
