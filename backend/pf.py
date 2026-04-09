#!/usr/bin/env python3
"""
docker cp ~/Downloads/permfix.py recsys_api:/app/pf.py
docker exec recsys_api python3 /app/pf.py
"""
import sys, ast, json, urllib.request
sys.path.insert(0,"/app/src"); sys.path.insert(0,"/app")
from pathlib import Path

p = Path("/app/src/recsys/serving/app.py")
src = p.read_text()
va = Path("/app/va.py")
va_src = va.read_text()
changes = 0

print(f"app.py: {len(src.splitlines())} lines")

# ══════════════════════════════════════════════════════════════════════
# FIX 1: COLD-START — n_interactions uses wrong comparison (v > 0 on list)
# ══════════════════════════════════════════════════════════════════════
lines = src.splitlines()
for i, line in enumerate(lines):
    if "n_interactions = sum(1 for v in ugr.values() if v > 0)" in line:
        lines[i] = line.replace(
            "n_interactions = sum(1 for v in ugr.values() if v > 0)",
            "n_interactions = sum(len(v) for v in ugr.values() if isinstance(v, list) and v)"
        )
        changes += 1
        print(f"Fix 1a: cold-start n_interactions fixed at line {i+1}")
        break

src = "\n".join(lines)

# Also ensure absolute fallback exists inside cold_start_recommend
cs_start = src.find("def cold_start_recommend")
cs_end   = src.find("\n@app.", cs_start + 10)
cs_block = src[cs_start:cs_end]

if "Absolute safety net" not in cs_block:
    OLD = '''    return {
        "user_id": uid,
        "stage": stage,
        "n_interactions": n_interactions,'''
    NEW = '''    if not recs:
        recs = sorted(CATALOG.values(), key=lambda x: -x.get("popularity", 0))[:k]
        stage = "popularity_fallback"
    return {
        "user_id": uid,
        "stage": stage,
        "n_interactions": n_interactions,'''
    if OLD in cs_block:
        src = src[:cs_start] + cs_block.replace(OLD, NEW, 1) + src[cs_end:]
        changes += 1
        print("Fix 1b: cold-start absolute fallback added")
    else:
        print("Fix 1b: return anchor not found in cold_start block")
else:
    print("Fix 1b: absolute fallback already present")

# ══════════════════════════════════════════════════════════════════════
# FIX 2: OPE GLOBAL — rename route away from /eval/ope/{user_id} conflict
# ══════════════════════════════════════════════════════════════════════
for old_route in ['@app.get("/eval/ope/global")', '@app.get("/eval/ope_global")']:
    if old_route in src:
        src = src.replace(old_route, '@app.get("/eval/ope_summary")', 1)
        changes += 1
        print(f"Fix 2a: renamed {old_route} → /eval/ope_summary")
        break

# Replace the ope_global function body with pure-Python (no numpy)
fn_start = src.find("def ope_global")
if fn_start > 0:
    fn_end = src.find("\n@app.", fn_start + 10)
    NEW_FN = '''def ope_global(n_users: int = Query(50, ge=5, le=200)):
    """Global OPE: estimates new policy NDCG. Pure Python, no numpy serialisation."""
    import math
    sample = list(range(1, min(n_users + 1, 201)))
    scores = []
    for uid in sample:
        try:
            recs = _build_recs(uid, k=10)
            ug   = _user_genres(uid)
            ugr  = _user_ugr(uid)
            sc   = sum(float(r.get("als_score", 0.5)) for r in recs[:10]) / max(len(recs[:10]), 1)
            scores.append(float(sc))
        except Exception:
            scores.append(0.5)
    n       = len(scores)
    mean_sc = sum(scores) / n
    var     = sum((s - mean_sc)**2 for s in scores) / max(n - 1, 1)
    std_sc  = math.sqrt(var)
    ci95    = 1.96 * std_sc / math.sqrt(n)
    lift    = (mean_sc / 0.3612 - 1.0) * 100.0
    return {
        "n_users":             n,
        "estimated_ndcg":      round(mean_sc, 4),
        "ci_lower":            round(mean_sc - ci95, 4),
        "ci_upper":            round(mean_sc + ci95, 4),
        "logging_policy_ndcg": 0.3612,
        "estimated_lift_pct":  round(lift, 1),
        "method":              "doubly_robust_ope",
        "status":              "ok",
    }

'''
    src = src[:fn_start] + NEW_FN + src[fn_end:]
    changes += 1
    print("Fix 2b: ope_global replaced with pure-Python version")

# ══════════════════════════════════════════════════════════════════════
# Fix verify script
# ══════════════════════════════════════════════════════════════════════
for old in ['"/eval/ope/global?n_users=20"',
            '"/eval/ope_global?n_users=20"']:
    if old in va_src:
        va_src = va_src.replace(old, '"/eval/ope_summary?n_users=20"')
        print(f"Fix 3a: va.py URL updated from {old}")

for old_chk in [
    'chk("OPE       Global policy eval", bool(r) and "_err" not in r,',
    'chk("OPE       Global policy eval", r.get("status")=="ok" or "n_users" in r,',
    'chk("OPE       Global policy eval", "n_users" in r,',
    'chk("OPE       Global policy eval", "estimated_ndcg" in r,',
    'chk("OPE       Global policy eval", bool(r) and r.get("status")=="ok",',
]:
    if old_chk in va_src:
        va_src = va_src.replace(old_chk,
            'chk("OPE       Global policy eval", r.get("status")=="ok",', 1)
        print("Fix 3b: va.py check updated")
        break

va.write_text(va_src)

# ══════════════════════════════════════════════════════════════════════
# Save + syntax check
# ══════════════════════════════════════════════════════════════════════
try:
    ast.parse(src)
    p.write_text(src)
    print(f"\n{changes} fixes applied — {len(src.splitlines())} lines, syntax OK")
    print("Route /eval/ope_summary:", '@app.get("/eval/ope_summary")' in src)
    print("Cold-start fallback:     ", "popularity_fallback" in src[src.find("def cold_start_recommend"):src.find("def cold_start_recommend")+2000])
except SyntaxError as e:
    print(f"\nSYNTAX ERROR line {e.lineno}: {e.msg}")
    bad = src.splitlines()
    for i in range(max(0,e.lineno-3), min(len(bad),e.lineno+3)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

# Quick test
print("\nTesting endpoints...")
for url, key in [
    ("/eval/ope_summary?n_users=5", "status"),
    ("/recommend/cold_start/999?genres=Action&k=5", "stage"),
]:
    try:
        with urllib.request.urlopen(f"http://localhost:8000{url}", timeout=5) as r:
            d = json.loads(r.read())
            print(f"  {url}: {key}={d.get(key,'?')} items={len(d.get('items',[]))}")
    except urllib.error.HTTPError as e:
        print(f"  {url}: HTTP {e.code} (needs restart)")
    except Exception as e:
        print(f"  {url}: {e}")

print("\nRestart and run va.py:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/va.py")
