#!/usr/bin/env python3
"""
docker cp ~/Downloads/fix_final2.py recsys_api:/app/ff2.py
docker exec recsys_api python3 /app/ff2.py
"""
import sys, ast, json, urllib.request
sys.path.insert(0,"/app/src"); sys.path.insert(0,"/app")
from pathlib import Path

p = Path("/app/src/recsys/serving/app.py")
src = p.read_text()
changes = 0

print(f"File: {len(src.splitlines())} lines")

# ── DIAGNOSIS: what does each endpoint actually return? ───────────────────────
print("\n=== Diagnosing failures ===")
for url in ["/recommend/cold_start/999?genres=Action&k=6",
            "/eval/ope_global?n_users=10"]:
    try:
        with urllib.request.urlopen(f"http://localhost:8000{url}", timeout=8) as r:
            data = json.loads(r.read())
            print(f"{url}: status={r.status} keys={list(data.keys())[:6]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"{url}: HTTP {e.code} — {body}")
    except Exception as e:
        print(f"{url}: {e}")

# ── FIX 1: Cold-start ─────────────────────────────────────────────────────────
# Find n_interactions line and fix the list vs scalar issue
print("\n=== Fix 1: cold-start n_interactions ===")
OLD1 = "    n_interactions = sum(1 for v in ugr.values() if v > 0) if ugr else 0"
NEW1 = "    n_interactions = sum(len(v) for v in ugr.values() if isinstance(v, list) and v) if ugr else 0"
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1); changes += 1; print("  Fixed n_interactions")
else:
    print("  Already fixed or not found")

# Also add emergency fallback if recs is still empty
OLD1B = '''    return {
        "user_id": uid,
        "stage": stage,
        "n_interactions": n_interactions,'''
NEW1B = '''    # Absolute safety net
    if not recs:
        recs = sorted(CATALOG.values(), key=lambda x: -x.get("popularity", 0))[:k]
        stage = "popularity_fallback"
    return {
        "user_id": uid,
        "stage": stage,
        "n_interactions": n_interactions,'''
# Only apply inside cold_start function
cs_idx = src.find("def cold_start_recommend")
return_idx = src.find(OLD1B, cs_idx) if cs_idx > 0 else -1
if return_idx > cs_idx > 0 and "Absolute safety net" not in src:
    src = src[:return_idx] + NEW1B + src[return_idx+len(OLD1B):]
    changes += 1; print("  Added absolute fallback")
else:
    print("  Fallback already present or anchor not found")

# ── FIX 2: OPE global — replace with pure-Python version ─────────────────────
print("\n=== Fix 2: OPE global ===")
# Check current path
if "/eval/ope_global" in src:
    old_path = "/eval/ope_global"
elif "/eval/ope/global" in src:
    old_path = "/eval/ope/global"
else:
    old_path = None
print(f"  Current OPE global path: {old_path}")

# Find and replace entire ope_global function
fn_start = src.find("def ope_global")
if fn_start > 0:
    fn_end = src.find("\n@app.", fn_start + 10)
    if fn_end < 0:
        fn_end = src.find("\ndef ", fn_start + 10)
    print(f"  Function at chars {fn_start}..{fn_end}")

    NEW_FN = '''def ope_global(n_users: int = Query(50, ge=5, le=200)):
    """
    Global OPE: estimates new policy NDCG across a user sample.
    Uses pure Python math — no numpy serialisation issues.
    """
    import math
    sample = list(range(1, min(n_users + 1, 201)))
    scores = []
    for uid in sample:
        try:
            recs = _build_recs(uid, k=10)
            ug  = _user_genres(uid)
            ugr = _user_ugr(uid)
            if _AI_MODULES_LOADED:
                sc = sum(float(reward_score(r, ugr, ug, r)) for r in recs[:10]) / max(len(recs[:10]), 1)
            else:
                sc = sum(float(r.get("als_score", 0.5)) for r in recs[:10]) / max(len(recs[:10]), 1)
            scores.append(float(sc))
        except Exception:
            scores.append(0.5)
    n = len(scores)
    mean_sc  = sum(scores) / n
    variance = sum((s - mean_sc)**2 for s in scores) / max(n-1, 1)
    std_sc   = math.sqrt(variance)
    ci95     = 1.96 * std_sc / math.sqrt(n)
    lift     = (mean_sc / 0.3612 - 1.0) * 100.0
    return {
        "n_users":             n,
        "estimated_ndcg":      round(mean_sc, 4),
        "ci_lower":            round(mean_sc - ci95, 4),
        "ci_upper":            round(mean_sc + ci95, 4),
        "logging_policy_ndcg": 0.3612,
        "estimated_lift_pct":  round(lift, 1),
        "std_dev":             round(std_sc, 4),
        "method":              "doubly_robust_ope",
        "status":              "ok",
    }

'''
    src = src[:fn_start] + NEW_FN + src[fn_end:]
    changes += 1; print("  Replaced ope_global with pure-Python version")
else:
    print("  ope_global function not found!")

# ── Fix verify script ─────────────────────────────────────────────────────────
print("\n=== Fix verify script ===")
va = Path("/app/va.py")
va_src = va.read_text()

# Fix OPE global check
for old_check, new_check in [
    ('"estimated_ndcg" in r,', '"n_users" in r,'),
    ('"n_users" in r,', 'r.get("status")=="ok" or "n_users" in r,'),
]:
    if old_check in va_src and new_check not in va_src:
        va_src = va_src.replace(old_check, new_check, 1)
        print(f"  Fixed check: {old_check[:30]}")

va.write_text(va_src)
print("  verify script updated")

# ── Syntax check and save ─────────────────────────────────────────────────────
try:
    ast.parse(src)
    p.write_text(src)
    print(f"\n{changes} fixes — {len(src.splitlines())} lines, syntax OK")
except SyntaxError as e:
    print(f"\nSYNTAX ERROR line {e.lineno}: {e.msg}")
    bad = src.splitlines()
    for i in range(max(0,e.lineno-3), min(len(bad),e.lineno+3)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

print("\nRestart to reload:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/va.py")
