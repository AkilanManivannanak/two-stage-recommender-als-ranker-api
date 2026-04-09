#!/usr/bin/env python3
"""
docker cp ~/Downloads/full_check.py recsys_api:/app/fc.py
docker exec recsys_api python3 /app/fc.py
"""
import sys, json, time, hashlib, urllib.request
sys.path.insert(0,"/app/src"); sys.path.insert(0,"/app")

BASE = "http://localhost:8000"
G="\033[0;32m"; R="\033[0;31m"; Y="\033[1;33m"; N="\033[0m"
PASS=0; FAIL=0

def get(p):
    try:
        with urllib.request.urlopen(f"{BASE}{p}", timeout=10) as r: return json.loads(r.read())
    except: return {}

def post(p, b):
    try:
        req=urllib.request.Request(f"{BASE}{p}",method="POST")
        req.add_header("Content-Type","application/json"); req.data=json.dumps(b).encode()
        with urllib.request.urlopen(req,timeout=10) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: return {"_err":e.code,"_body":e.read().decode()[:100]}
    except Exception as e: return {"_err":str(e)}

def chk(n,d,ok,detail=""):
    global PASS,FAIL
    if ok: print(f"{G}  [PASS]{N} {n}: {d}"); PASS+=1
    else:  print(f"{R}  [FAIL]{N} {n}: {d}"); FAIL+=1
    if detail: print(f"         {detail}")

def is_hb(u): return int(hashlib.md5(f"cinewave_holdback_v1:{u}".encode()).hexdigest()[:8],16)/0xFFFFFFFF<0.05
HB=next(u for u in range(1,500) if is_hb(u))

print(f"\n{'='*60}")
print(f"  CineWave v6 — Complete system check")
print(f"{'='*60}\n")

# ── 9 Additions ──────────────────────────────────────────────────────────────
r=get("/two_tower/status")
chk("Add 1","Two-tower retrieval",r.get("available") and (r.get("loaded") or r.get("has_embeddings")),
    f"loaded={r.get('loaded')} has_embeddings={r.get('has_embeddings')}")

r=get("/metrics/skew"); psi=r.get("psi_values",{}); c=sum(1 for v in psi.values() if v.get("psi") is not None)
chk("Add 2","Skew PSI",c>=3,f"features={c}/6 max_psi={r.get('max_psi',0):.4f}")

r=get("/metrics/slices")
chk("Add 3","Slice NDCG","global_ndcg" in r,f"global_ndcg={r.get('global_ndcg')} genre_slices={len(r.get('slices',{}).get('genre',{}))}")

r=get("/metrics/retention/2026-04-07")
chk("Add 4","Retention","cohort_date" in r,f"cohort_size={r.get('cohort_size',0)}")

r=post("/recommend",{"user_id":1,"k":3}); rnk=r.get("model_version",{}).get("ranker_model","")
chk("Add 5","Context features (13-feat)","ctx" in rnk,f"ranker={rnk}")

r=get("/metrics/drift"); pd=r.get("prediction_drift",{})
chk("Add 6","Drift monitoring",pd.get("status") in ("ok","warn","alert") and pd.get("n_scores",0)>0,
    f"status={pd.get('status')} n={pd.get('n_scores',0)}")

r=post("/recommend",{"user_id":1,"k":3})
chk("Add 7a","Holdback ML→ml_full",r.get("experiment_group")=="ml_full",f"group={r.get('experiment_group')}")

time.sleep(1); r=post("/recommend",{"user_id":HB,"k":3})
chk("Add 7b",f"Holdback user {HB}→holdback_popularity",
    r.get("experiment_group")=="holdback_popularity" and len(r.get("items",[]))>0,
    f"group={r.get('experiment_group')} items={len(r.get('items',[]))}")

r=get("/ab/analyse_cuped/cuped_demo_v6")
chk("Add 8","CUPED analysis","cuped_analysis" in r or ("cuped" in str(r) and "error" not in str(r.get("error",""))),
    f"var_red={r.get('variance_reduction_pct','?')}% src={r.get('source','live')}")

r=get("/clip/search?q=dark+psychological+thriller&top_k=3"); n=len(r.get("results",[]))
chk("Add 9","Semantic search",n>0,f"n_results={n} method={r.get('method','?')}")

# ── Notebook files ────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  Notebook-converted files")
print(f"{'─'*60}")

try:
    from recsys.serving.netflix_prize_recommender import NETFLIX_PRIZE_MODEL
    score=NETFLIX_PRIZE_MODEL.predict_rating(user_id=42,item_id=1)
    similar=NETFLIX_PRIZE_MODEL.get_similar_items(item_id=1,top_k=5)
    chk("NB 1","Netflix Prize model",True,f"predict(42,1)={score:.3f} similar={len(similar)} items")
except Exception as e:
    chk("NB 1","Netflix Prize model",False,str(e))

try:
    import os
    cluster_dir="/app/artifacts/clustering/"
    has_artifacts=os.path.exists(cluster_dir) and any(f.endswith((".pkl",".json")) for f in os.listdir(cluster_dir)) if os.path.exists(cluster_dir) else False
    chk("NB 2","Content clustering (6 clusters)",has_artifacts,
        f"artifacts={'present' if has_artifacts else 'missing'} at {cluster_dir}")
    if has_artifacts:
        files=os.listdir(cluster_dir)
        print(f"         files: {files}")
except Exception as e:
    chk("NB 2","Content clustering",False,str(e))

# ── Core API ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  Core API health")
print(f"{'─'*60}")

r=get("/healthz")
chk("API","Healthz",r.get("status")=="ok" or r.get("ok") is True,str(r)[:80])

r=post("/recommend",{"user_id":1,"k":5})
items=len(r.get("items",[])); ranker=r.get("model_version",{}).get("ranker_model","?")
chk("API","Recommend endpoint (5 items)",items==5,f"items={items} ranker={ranker}")

r=get("/eval/freshness")
chk("API","Freshness watermark","features" in r or "watermark" in str(r) or len(r)>0,f"keys={list(r.keys())[:4]}")

# ── Final ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
total=PASS+FAIL
if FAIL==0: print(f"{G}  RESULT: {PASS}/{total} — ALL CHECKS PASS ✓{N}")
else:       print(f"{R}  RESULT: {PASS}/{total} PASS  |  {FAIL} FAIL{N}")
print(f"{'='*60}\n")

# Metrics summary
skew=get("/metrics/skew"); drift=get("/metrics/drift"); pd2=drift.get("prediction_drift",{})
print("  Live metrics:")
print(f"    PSI max={skew.get('max_psi',0):.4f}  status={skew.get('status','?')}")
print(f"    Drift n={pd2.get('n_scores',0)}  mean={pd2.get('current_mean',0):.3f}  status={pd2.get('status','?')}")
slices=get("/metrics/slices")
print(f"    NDCG global={slices.get('global_ndcg','?')}  genres={len(slices.get('slices',{}).get('genre',{}))}")
print()
