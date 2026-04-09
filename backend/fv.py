#!/usr/bin/env python3
import sys, json, time, hashlib
sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app")
import urllib.request
BASE = "http://localhost:8000"
G="\033[0;32m"; R="\033[0;31m"; Y="\033[1;33m"; N="\033[0m"
PASS=0; FAIL=0
def get(p):
    try:
        with urllib.request.urlopen(f"{BASE}{p}", timeout=10) as r: return json.loads(r.read())
    except: return {}
def post(p, b):
    try:
        req=urllib.request.Request(f"{BASE}{p}", method="POST")
        req.add_header("Content-Type","application/json"); req.data=json.dumps(b).encode()
        with urllib.request.urlopen(req, timeout=10) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: return {"_err": e.code, "_body": e.read().decode()[:200]}
    except Exception as e: return {"_err": str(e)}
def chk(n,d,ok,detail=""):
    global PASS,FAIL
    if ok: print(f"{G}  [PASS]{N} Addition {n}: {d}"); PASS+=1
    else:  print(f"{R}  [FAIL]{N} Addition {n}: {d}"); FAIL+=1
    if detail: print(f"         {detail}")
def is_hb(u): return int(hashlib.md5(f"cinewave_holdback_v1:{u}".encode()).hexdigest()[:8],16)/0xFFFFFFFF<0.05
HB=next(u for u in range(1,500) if is_hb(u))
print(f"\n{'='*60}\n  CineWave v6 — Final verification\n{'='*60}\n  Holdback test user: {HB}\n")
r=get("/two_tower/status"); chk(1,"Two-tower",r.get("available") and (r.get("loaded") or r.get("has_embeddings")),f"loaded={r.get('loaded')} has_embeddings={r.get('has_embeddings')}")
r=get("/metrics/skew"); psi=r.get("psi_values",{}); c=sum(1 for v in psi.values() if v.get("psi") is not None); chk(2,"Skew PSI",c>=3,f"features={c}/6 max_psi={r.get('max_psi',0):.4f} status={r.get('status','?')}")
r=get("/metrics/slices"); chk(3,"Slice NDCG","global_ndcg" in r,f"global_ndcg={r.get('global_ndcg')} genre_slices={len(r.get('slices',{}).get('genre',{}))}")
r=get("/metrics/retention/2026-04-07"); chk(4,"Retention","cohort_date" in r,f"cohort_size={r.get('cohort_size',0)}")
r=post("/recommend",{"user_id":1,"k":3}); rnk=r.get("model_version",{}).get("ranker_model",""); chk(5,"Context features","ctx" in rnk,f"ranker={rnk}")
r=get("/metrics/drift"); pd=r.get("prediction_drift",{}); chk(6,"Drift monitoring",pd.get("status") in ("ok","warn","alert") and pd.get("n_scores",0)>0,f"status={pd.get('status')} n={pd.get('n_scores',0)} mean={pd.get('current_mean',0):.3f}")
r=post("/recommend",{"user_id":1,"k":3}); chk(7,"Holdback ML user→ml_full",r.get("experiment_group")=="ml_full",f"group={r.get('experiment_group')}")
time.sleep(1); r=post("/recommend",{"user_id":HB,"k":3}); grp=r.get("experiment_group"); items=len(r.get("items",[])); chk(7,f"Holdback user {HB}→holdback_popularity",grp=="holdback_popularity" and items>0,f"group={grp} items={items}" + (f" err={r.get('_err','')}" if r.get("_err") else ""))
if grp!="holdback_popularity" or items==0:
    print(f"\n  {Y}[DEBUG]{N}")
    try:
        from recsys.serving.context_and_additions import get_experiment_group,popularity_fallback
        print(f"    get_experiment_group({HB})={get_experiment_group(HB)!r}")
        from recsys.serving.app import CATALOG,ScoredItem
        items_raw=popularity_fallback(CATALOG,top_k=3)
        if items_raw:
            s=items_raw[0]; iid=s.get("item_id") or s.get("movieId"); pop=float(s.get("popularity") or 10)
            sc=round(min(pop/(pop+1.0),0.999),4); print(f"    item_id={iid} pop={pop:.1f} score={sc}")
            try: si=ScoredItem(item_id=int(iid),score=sc,als_score=0.0,ranker_score=0.0); print(f"    ScoredItem OK: {si}")
            except Exception as e: print(f"    ScoredItem FAIL: {e}")
    except Exception as e: print(f"    {e}")
r=get("/ab/analyse_cuped/cuped_demo_v6"); chk(8,"CUPED","cuped_analysis" in r or ("cuped" in str(r) and not r.get("error")),f"var_red={r.get('variance_reduction_pct','?')}% src={r.get('source','live')} raw_p={r.get('raw_analysis',{}).get('pvalue','?')}")
r=get("/clip/search?q=dark+psychological+thriller&top_k=3"); n=len(r.get("results",[])); chk(9,"Semantic search",n>0,f"n_results={n} method={r.get('method','?')}")
print(f"\n{'='*60}")
total=PASS+FAIL
if FAIL==0: print(f"{G}  RESULT: {PASS}/{total} — ALL 9 ADDITIONS COMPLETE ✓{N}")
else: print(f"{R}  RESULT: {PASS}/{total} PASS  |  {FAIL} FAIL{N}")
print(f"{'='*60}\n")
