"""
smoke_and_loadtest.py — boots the API in-process, exercises REST + GraphQL, and
measures /recommend latency so the policy gate has a real p95 to check.

Run: PYTHONPATH=src python3 scripts/smoke_and_loadtest.py
"""
from __future__ import annotations
import json, os, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("TMDB_API_KEY", "")

from fastapi.testclient import TestClient
from recsys.serving import app as A

client = TestClient(A.app)
FAILS = []

def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond: FAILS.append(name)

def gql(query: str):
    r = client.post("/graphql", json={"query": query})
    body = r.json()
    if "errors" in body:
        print("    GraphQL errors:", json.dumps(body["errors"])[:400])
    return body.get("data") or {}

print("\n== REST ==")
r = client.post("/recommend", json={"user_id": 1, "k": 5})
check("POST /recommend 200", r.status_code == 200, f"status={r.status_code}")
body = r.json()
items = body.get("items", [])
check("returns k items", len(items) == 5, f"got {len(items)}")
check("items are distinct", len({i["item_id"] for i in items}) == len(items))
check("als_score != ranker_score for at least one item",
      any(abs(i["als_score"] - i["ranker_score"]) > 1e-9 for i in items),
      "(the two stages must not be the same number)")

m = client.get("/metrics/pipeline").json()
check("/metrics/pipeline reports measured", m["live"]["status"] == "measured",
      f"ndcg={m['live'].get('ndcg_at_10')}")
check("baselines are present", len(m.get("baselines", {})) >= 3,
      f"{list(m.get('baselines', {}))}")

print("\n== GraphQL ==")
d = gql("""query{recommendations(userId:1,k:3){userId modelVersion latencyMs
          diversityScore items{itemId title primaryGenre year score alsScore
          rankerScore retrievalSource}}}""")
sl = d.get("recommendations") or {}
check("recommendations resolves", bool(sl.get("items")), f"{len(sl.get('items', []))} items")
if sl.get("items"):
    it = sl["items"][0]
    print(f"    -> {it['title']} ({it['year']}) {it['primaryGenre']}  "
          f"als={it['alsScore']:.4f} ranker={it['rankerScore']:.4f} "
          f"src={it['retrievalSource']}")
    check("titles are populated", bool(it["title"]))

d = gql("""query{modelMetrics{status modelVersion ndcgAt10 ndcgAt10AlsOnly
          ndcgLiftPctVsAls ipsNdcgAt10 bestBaseline bestBaselineNdcg
          baselines{ranker ndcgAt10} bootstrap{comparison deltaMean ci95Lo ci95Hi pDeltaGt0}}}""")
mm = d.get("modelMetrics") or {}
check("modelMetrics measured", mm.get("status") == "measured")
check("no invented defaults", mm.get("ndcgAt10") not in (0.1409, None),
      f"ndcg={mm.get('ndcgAt10')}")
print(f"    -> NDCG@10={mm.get('ndcgAt10')} vs ALS {mm.get('ndcgAt10AlsOnly')} "
      f"(+{mm.get('ndcgLiftPctVsAls')}%) | best baseline "
      f"{mm.get('bestBaseline')}={mm.get('bestBaselineNdcg')}")
for b in mm.get("bootstrap", []):
    print(f"       {b['comparison']:<34} delta={b['deltaMean']:+.4f} "
          f"CI95[{b['ci95Lo']:+.4f},{b['ci95Hi']:+.4f}] p={b['pDeltaGt0']}")

d = gql("""mutation{recordFeedback(userId:1,itemId:2858,event:"play",
          dwellSeconds:120.0){accepted message}}""")
check("recordFeedback mutation", (d.get("recordFeedback") or {}).get("accepted") is True)
d = gql("""mutation{recordFeedback(userId:1,itemId:2858,event:"teleport"){accepted message}}""")
check("mutation rejects bad event",
      (d.get("recordFeedback") or {}).get("accepted") is False)

d = gql("""query{search(q:"star",limit:3){itemId title year primaryGenre}}""")
check("search resolves", len(d.get("search") or []) > 0,
      f"{[m['title'] for m in (d.get('search') or [])]}")

print("\n== Load test: /recommend ==")
import random
rng = random.Random(42)
N_WARM, N = 50, 600
for _ in range(N_WARM):
    client.post("/recommend", json={"user_id": rng.randint(1, 6040), "k": 10})
lat = []
t_start = time.perf_counter()
for _ in range(N):
    uid = rng.randint(1, 6040)
    t0 = time.perf_counter()
    resp = client.post("/recommend", json={"user_id": uid, "k": 10})
    lat.append((time.perf_counter() - t0) * 1000.0)
    if resp.status_code != 200: FAILS.append("load-test non-200")
wall = time.perf_counter() - t_start
lat.sort()
def pct(p): return lat[min(int(len(lat) * p / 100), len(lat) - 1)]
print(f"  n={N}  wall={wall:.1f}s  throughput={N/wall:.0f} req/s (single process)")
print(f"  p50={pct(50):.2f}ms  p90={pct(90):.2f}ms  p95={pct(95):.2f}ms  "
      f"p99={pct(99):.2f}ms  max={lat[-1]:.2f}ms")

print("\n== Policy gate against measured latency ==")
d = gql("""query{policyGate{recommendation gatePassed summary nChecks nPassed
          nInconclusive checks{name value threshold comparison passed critical
          inconclusive}}}""")
pg = d.get("policyGate") or {}
print(f"  {pg.get('recommendation')} — {pg.get('summary')}")
for c in pg.get("checks", []):
    mark = "PASS " if c["passed"] else ("BLOCK" if c["critical"] else "WARN ")
    if c["inconclusive"]: mark = "INCON"
    print(f"   {mark} {c['name']:<38} {c['value']} {c['comparison']} {c['threshold']}")
check("gate ran the latency check on real samples",
      any(c["name"] == "p95_ms_recommend" and not c["inconclusive"]
          for c in pg.get("checks", [])))

lat_stats = A._recommend_latency()
print(f"\n  server-side ring buffer: n={lat_stats['n']} p95={lat_stats['p95_ms']}ms "
      f"p99={lat_stats['p99_ms']}ms route={lat_stats['route']}")

print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILURES: ' + str(FAILS)}")
sys.exit(1 if FAILS else 0)
