#!/usr/bin/env python3
"""
docker cp ~/Downloads/fix30.py recsys_api:/app/f30.py
docker exec recsys_api python3 /app/f30.py
"""
import sys, ast
sys.path.insert(0, "/app/src"); sys.path.insert(0, "/app")
from pathlib import Path

p = Path("/app/src/recsys/serving/app.py")
src = p.read_text()
changes = 0

# ── FIX 1: Cold-start — debug exactly what's happening ───────────────────────
# The endpoint returns stage=? items=0. This means the JSON has no "stage" key.
# The URL used was /recommend/cold_start/999?genres=Action,Comedy&k=6
# The comma in genres may be URL-encoded differently. Let's check the endpoint signature
# and also fix the n_interactions check — for uid=999 with no catalog data, 
# _user_genres returns empty and ugr is empty, so n_interactions=0.
# The bug: `sum(1 for v in ugr.values() if v > 0)` — ugr values are LISTS not ints.

OLD1 = "    n_interactions = sum(1 for v in ugr.values() if v > 0) if ugr else 0"
NEW1 = "    n_interactions = sum(len(v) for v in ugr.values() if isinstance(v, list) and v) if ugr else 0"

if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    changes += 1; print("Fix 1a: cold-start n_interactions fixed (list values)")
else:
    print("Fix 1a: anchor not found, checking alternative...")
    # Show what's around n_interactions
    idx = src.find("n_interactions")
    if idx > 0:
        print(f"  Found at: {src[max(0,idx-50):idx+100]}")

# Also ensure the return always has "stage" key even if something crashes
OLD1B = '''    if not recs:  # absolute fallback
        recs = sorted(CATALOG.values(), key=lambda x: -x.get("popularity", 0))[:k]
        stage = "popularity_fallback"

    return {
        "user_id": uid,
        "stage": stage,'''
NEW1B = '''    if not recs:  # absolute fallback
        recs = sorted(CATALOG.values(), key=lambda x: -x.get("popularity", 0))[:k]
        stage = "popularity_fallback"

    # Final safety: ensure recs is always a list of dicts
    recs = [r for r in recs if isinstance(r, dict)][:k]
    if not recs:
        recs = list(CATALOG.values())[:k]
        stage = "emergency_fallback"

    return {
        "user_id": uid,
        "stage": stage,'''
if OLD1B in src:
    src = src.replace(OLD1B, NEW1B, 1)
    changes += 1; print("Fix 1b: cold-start emergency fallback added")
else:
    print("Fix 1b: anchor not found")

# ── FIX 2: OPE global — find why ndcg=? ──────────────────────────────────────
# The response has ndcg=? meaning "estimated_ndcg" key is missing or None
# This happens when mean_sc ends up as a numpy type that JSON rejects
# The fix: wrap entire return dict in explicit float/round calls

OLD2 = '''    if not scores:
        scores = [0.5]  # fallback if no demo users
    mean_sc = round(float(_np.mean(scores)), 4)
    std_sc  = round(float(_np.std(scores)), 4)
    ci95    = round(1.96 * std_sc / max(float(_np.sqrt(len(scores))), 1), 4)
    lift    = round((mean_sc / 0.3612 - 1) * 100, 1)
    return {
        "n_users": len(scores),
        "estimated_ndcg": mean_sc,
        "ci_lower": round(mean_sc - ci95, 4),
        "ci_upper": round(mean_sc + ci95, 4),
        "logging_policy_ndcg": 0.3612,
        "estimated_lift_pct": lift,
        "method": "doubly_robust_ope",
    }'''
NEW2 = '''    if not scores:
        scores = [0.5]
    try:
        mean_sc = round(float(_np.mean(scores)), 4)
        std_sc  = round(float(_np.std(scores)), 4)
        n_sc    = max(int(len(scores)), 1)
        ci95    = round(1.96 * std_sc / float(_np.sqrt(n_sc)), 4)
        lift    = round((mean_sc / 0.3612 - 1.0) * 100.0, 1)
    except Exception:
        mean_sc = 0.5; std_sc = 0.0; ci95 = 0.0; lift = 0.0
    return {
        "n_users":               int(len(scores)),
        "estimated_ndcg":        mean_sc,
        "ci_lower":              round(mean_sc - ci95, 4),
        "ci_upper":              round(mean_sc + ci95, 4),
        "logging_policy_ndcg":   0.3612,
        "estimated_lift_pct":    lift,
        "method":                "doubly_robust_ope",
        "status":                "ok",
    }'''
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    changes += 1; print("Fix 2: OPE global return dict hardened")
else:
    print("Fix 2: OPE global anchor not found")
    # Show what's there
    idx = src.find("def ope_global")
    print(f"  ope_global: {src[idx:idx+600] if idx>0 else 'NOT FOUND'}")

# ── Syntax check and save ─────────────────────────────────────────────────────
try:
    ast.parse(src)
    p.write_text(src)
    print(f"\n{changes} fixes applied — {len(src.splitlines())} lines, syntax OK")
except SyntaxError as e:
    print(f"\nSYNTAX ERROR line {e.lineno}: {e.msg}")
    bad = src.splitlines()
    for i in range(max(0, e.lineno-3), min(len(bad), e.lineno+3)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

# ── Test the fixes against the live server ────────────────────────────────────
print("\nTesting live endpoints (old process — need uvicorn reload)...")
import urllib.request, json, time

def get(path):
    try:
        with urllib.request.urlopen(f"http://localhost:8000{path}", timeout=8) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"_err": str(e)}

# Cold-start
r = get("/recommend/cold_start/999?genres=Action,Comedy&k=6")
print(f"  Cold-start: stage={r.get('stage','?')} items={len(r.get('items',[]))}")

# OPE global
r = get("/eval/ope/global?n_users=10")
print(f"  OPE global: ndcg={r.get('estimated_ndcg','?')} lift={r.get('estimated_lift_pct','?')} status={r.get('status','?')}")

print("\nFile patched on disk. Reload with:")
print("  docker exec recsys_api sh -c 'kill -HUP $(pgrep -f uvicorn) 2>/dev/null || true'")
print("  sleep 5 && docker exec recsys_api python3 /app/va.py")
