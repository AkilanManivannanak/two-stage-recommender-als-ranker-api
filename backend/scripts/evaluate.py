"""
evaluate.py — Stage 3: the single source of truth for every number this project
reports.

Evaluates four rankers on the held-out TEST split and writes metrics.json. The
serving layer, the policy gate and the README all read that file. Nothing in
this repository is allowed to carry a metric as a literal.

Rankers
  popularity      most-interacted items in train        (the bar to clear)
  cooccurrence    item-item co-occurrence from train    (cheap personalisation)
  als_only        Stage 1 retrieval, unranked           (the two-stage baseline)
  als_lambdarank  Stage 1 + Stage 2 rerank              (the system)

Also computes the diversity, coverage and IPS-corrected metrics the policy gate
consumes, plus paired bootstrap intervals for every comparison against ALS.
"""
from __future__ import annotations
import json, os, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLS, build_feature_frame, build_user_genre_stats

ROOT   = Path(__file__).resolve().parents[1]
DATA   = Path(os.environ.get("DATA_DIR",      ROOT / "data" / "processed"))
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
K      = int(os.environ.get("EVAL_K", "10"))
SEED   = int(os.environ.get("SEED", "42"))

t0 = time.time()
train     = pd.read_parquet(DATA / "train.parquet")
test      = pd.read_parquet(DATA / "test.parquet")
items     = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
item_feat = pd.read_parquet(DATA / "features" / "item_features.parquet").set_index("item_id")
user_feat = pd.read_parquet(DATA / "features" / "user_features.parquet").set_index("user_id")
cand_test = pd.read_parquet(DATA / "candidates_test.parquet")
ranker    = pickle.load(open(BUNDLE / "ranker.pkl", "rb"))

rel = test[test.label == 1].groupby("user_id")["item_id"].apply(set).to_dict()
rel = {u: r for u, r in rel.items() if r}
seen = train[train.label == 1].groupby("user_id")["item_id"].apply(set).to_dict()
genre_of = items["primary_genre"].to_dict()
propensity = item_feat["propensity"].to_dict()
n_catalog = int(items.shape[0])
print(f"[eval] {len(rel):,} users with >=1 relevant test item · catalog {n_catalog:,}")

# ── Metric primitives ─────────────────────────────────────────────────────────
def ndcg(recs, r, k=K):
    g = np.array([1.0 if i in r else 0.0 for i in recs[:k]])
    dcg = (g / np.log2(np.arange(2, len(g) + 2))).sum()
    ideal = np.ones(min(len(r), k))
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0

def mrr(recs, r, k=K):
    for i, it in enumerate(recs[:k]):
        if it in r: return 1.0 / (i + 1)
    return 0.0

def recall(recs, r, k=K): return len(set(recs[:k]) & r) / len(r)

_MEAN_PROP = float(np.mean(list(propensity.values())))

def _ips_weight(i, clip=10.0):
    """Exposure weight RELATIVE to an average-popularity item.

    Using a raw 1/p weight is degenerate on a 3.7k-item catalogue: every
    propensity is ~1/3667, so every weight blows past any sane clip and the
    estimator collapses back to plain NDCG. Normalising by the mean propensity
    gives w=1 for a typical item, w>1 for under-exposed items and w<1 for
    head items, which is the correction the estimator is actually for.
    """
    p = max(propensity.get(i, _MEAN_PROP), 1e-9)
    return float(np.clip(_MEAN_PROP / p, 1.0 / clip, clip))

def ips_ndcg(recs, r, k=K):
    """NDCG with each hit weighted by its inverse relative exposure."""
    g = np.array([_ips_weight(i) if i in r else 0.0 for i in recs[:k]])
    dcg = (g / np.log2(np.arange(2, len(g) + 2))).sum()
    ideal = np.sort([_ips_weight(i) for i in r])[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0

# ── Rankers ───────────────────────────────────────────────────────────────────
pop_ranked = [i for i, _ in Counter(train[train.label == 1].item_id).most_common(400)]

co = defaultdict(Counter)
_hist = train[train.label == 1].sort_values("timestamp").groupby("user_id")["item_id"].apply(list)
for _items in _hist:
    tail = _items[-20:]
    for a in range(len(tail)):
        for b in range(a + 1, len(tail)):
            co[tail[a]][tail[b]] += 1; co[tail[b]][tail[a]] += 1

def rank_popularity(uid):
    s = seen.get(uid, set())
    return [i for i in pop_ranked if i not in s][:50]

def rank_cooccurrence(uid):
    s = seen.get(uid, set())
    recent = [i for i in _hist.get(uid, [])][-10:]
    sc = Counter()
    for it in recent:
        for nb, c in co.get(it, {}).items():
            if nb not in s: sc[nb] += c
    out = [i for i, _ in sc.most_common(50)]
    return out if out else rank_popularity(uid)

ugs, utop = build_user_genre_stats(train, items)
fr = build_feature_frame(cand_test, item_feat, user_feat, items, ugs, utop)
fr["ranker_score"] = ranker.predict(fr[FEATURE_COLS])
als_lists  = (fr.sort_values(["user_id", "rank"], kind="mergesort")
                .groupby("user_id")["item_id"].apply(list).to_dict())
rank_lists = (fr.sort_values(["user_id", "ranker_score"], ascending=[True, False], kind="mergesort")
                .groupby("user_id")["item_id"].apply(list).to_dict())

RANKERS = {
    "popularity":     rank_popularity,
    "cooccurrence":   rank_cooccurrence,
    "als_only":       lambda u: als_lists.get(u, [])[:50],
    "als_lambdarank": lambda u: rank_lists.get(u, [])[:50],
}

# ── Evaluate ──────────────────────────────────────────────────────────────────
results, per_user = {}, {}
for name, fn in RANKERS.items():
    nd, mr, rc, ips, gdiv, cov = [], [], [], [], [], set()
    pu = {}
    for uid, r in rel.items():
        recs = fn(uid)
        if not recs: continue
        v = ndcg(recs, r); pu[uid] = v
        nd.append(v); mr.append(mrr(recs, r)); rc.append(recall(recs, r))
        ips.append(ips_ndcg(recs, r))
        top = recs[:K]
        gdiv.append(len({genre_of.get(i, "?") for i in top}) / max(len(top), 1))
        cov.update(top)
    results[name] = {
        "ndcg@10":    round(float(np.mean(nd)), 4),
        "mrr@10":     round(float(np.mean(mr)), 4),
        "recall@10":  round(float(np.mean(rc)), 4),
        "ips_ndcg@10":round(float(np.mean(ips)), 4),
        "diversity_score": round(float(np.mean(gdiv)), 4),
        "coverage":   round(len(cov) / n_catalog, 4),
        "n_items_shown": len(cov),
        "n_users":    len(nd),
    }
    per_user[name] = pu
    print(f"  {name:<16} NDCG@10={results[name]['ndcg@10']:.4f}  "
          f"MRR@10={results[name]['mrr@10']:.4f}  "
          f"Recall@10={results[name]['recall@10']:.4f}  "
          f"div={results[name]['diversity_score']:.3f}  "
          f"cov={results[name]['coverage']:.3f}")

def paired_bootstrap(a, b, n=2000, seed=SEED):
    users = np.array(sorted(set(a) & set(b)))
    A = np.array([a[u] for u in users]); B = np.array([b[u] for u in users])
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(users), size=(n, len(users)))
    d = B[idx].mean(axis=1) - A[idx].mean(axis=1)
    return {"delta_mean": round(float(B.mean() - A.mean()), 4),
            "ci95_lo": round(float(np.percentile(d, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(d, 97.5)), 4),
            "p_delta_gt_0": round(float((d > 0).mean()), 4),
            "n_users": int(len(users)), "n_boot": n}

comparisons = {}
for ref in ("als_only", "popularity", "cooccurrence"):
    for n in RANKERS:
        if n == ref: continue
        if ref != "als_only" and n != "als_lambdarank": continue
        comparisons[f"{n}_vs_{ref}"] = paired_bootstrap(per_user[ref], per_user[n])

# ── Slice breakdown: does the win hold for cold users? ────────────────────────
act = user_feat["user_cnt_total"]
tert = act.quantile([1/3, 2/3]).to_list()
def seg(u):
    a = act.get(u, 0)
    return "light" if a <= tert[0] else ("medium" if a <= tert[1] else "heavy")

slices = {}
for s in ("light", "medium", "heavy"):
    us = [u for u in rel if seg(u) == s and u in per_user["als_lambdarank"]]
    if not us: continue
    slices[s] = {
        "n_users": len(us),
        "als_only":       round(float(np.mean([per_user["als_only"][u] for u in us])), 4),
        "als_lambdarank": round(float(np.mean([per_user["als_lambdarank"][u] for u in us])), 4),
    }
    slices[s]["delta"] = round(slices[s]["als_lambdarank"] - slices[s]["als_only"], 4)
    print(f"  slice {s:<7} n={len(us):>5}  ALS={slices[s]['als_only']:.4f} "
          f"-> ranked={slices[s]['als_lambdarank']:.4f}  delta={slices[s]['delta']:+.4f}")

als_stats = json.loads((BUNDLE / "als_stats.json").read_text())
lift = (results["als_lambdarank"]["ndcg@10"] - results["als_only"]["ndcg@10"]) \
       / results["als_only"]["ndcg@10"] * 100

metrics = {
    "generated_by":   "backend/scripts/evaluate.py",
    "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset":        json.loads((DATA / "manifest.json").read_text()),
    "split_evaluated":"test",
    "k":              K,
    "baselines":      results,
    "headline": {
        "ndcg_at_10":        results["als_lambdarank"]["ndcg@10"],
        "ndcg_at_10_als":    results["als_only"]["ndcg@10"],
        "ndcg_lift_pct":     round(lift, 1),
        "mrr_at_10":         results["als_lambdarank"]["mrr@10"],
        "recall_at_10":      results["als_lambdarank"]["recall@10"],
        "ips_ndcg_at_10":    results["als_lambdarank"]["ips_ndcg@10"],
        "diversity_score":   results["als_lambdarank"]["diversity_score"],
        "coverage":          results["als_lambdarank"]["coverage"],
        "candidate_recall_at_200": als_stats["candidate_recall@200_test"],
        "best_baseline":          max((n for n in results if n != "als_lambdarank"),
                                      key=lambda n: results[n]["ndcg@10"]),
        "best_baseline_ndcg":     max(results[n]["ndcg@10"] for n in results
                                      if n != "als_lambdarank"),
    },
    "bootstrap":  comparisons,
    "slices_by_user_activity": slices,
    "eval_seconds": round(time.time() - t0, 1),
}
(BUNDLE / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\n[eval] wrote {BUNDLE/'metrics.json'} in {time.time()-t0:.1f}s")
print(json.dumps(metrics["headline"], indent=2))
print(json.dumps(comparisons, indent=2))
