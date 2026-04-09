#!/usr/bin/env python3
"""
Fix holdback 500 error.
The debug showed: 'int' object has no attribute 'get'
popularity_fallback() returns a list of ints (item IDs), not dicts.
The holdback block calls .get() on each item — crashes immediately.

Fix: look up each int in CATALOG to get the full item dict.

docker cp ~/Downloads/fix_final.py recsys_api:/app/ff.py
docker exec recsys_api python3 /app/ff.py
"""
import sys, ast
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

from pathlib import Path

app = Path("/app/src/recsys/serving/app.py")
src = app.read_text()
lines = src.splitlines(keepends=True)

# ── 1: Find the holdback block ────────────────────────────────────────────────
hb_idx = None
for i, line in enumerate(lines):
    if 'experiment_group == "holdback_popularity"' in line:
        hb_idx = i
        break

assert hb_idx is not None, "Holdback line not found"

# Find end (first line of normal flow after the if-block)
end_idx = None
for i in range(hb_idx + 1, hb_idx + 35):
    s = lines[i].strip()
    if s.startswith("wm = ") or s.startswith("features_snapshot_id"):
        end_idx = i
        break

assert end_idx is not None, "Block end not found"

print(f"Holdback block: lines {hb_idx+1}–{end_idx}")
print("Current block:")
for ln in lines[hb_idx:end_idx]:
    print("  " + ln, end="")

# ── 2: Check what popularity_fallback actually returns ────────────────────────
print("\nChecking popularity_fallback return type...")
try:
    from recsys.serving.context_and_additions import popularity_fallback
    from recsys.serving.app import CATALOG
    items = popularity_fallback(CATALOG, top_k=3)
    print(f"  Returns {len(items)} items, type of first: {type(items[0])}")
    print(f"  First item: {repr(items[0])[:120]}")
    is_int = isinstance(items[0], int)
    print(f"  is_int={is_int}")
except Exception as e:
    print(f"  Could not check: {e}")
    is_int = True  # assume int based on error message

# ── 3: Write the correct replacement ─────────────────────────────────────────
# Handles BOTH cases:
#   - popularity_fallback returns list of ints  (item IDs) → look up in CATALOG
#   - popularity_fallback returns list of dicts → use directly
REPLACEMENT = '''\
    if experiment_group == "holdback_popularity":
        _hb_raw = popularity_fallback(CATALOG, top_k=max(req.k, 10))
        _hb_scored = []
        for _hb_item in _hb_raw:
            # popularity_fallback may return ints (item IDs) or dicts
            if isinstance(_hb_item, int):
                _hb_dict = CATALOG.get(_hb_item, {})
                _iid = _hb_item
            else:
                _hb_dict = _hb_item
                _iid = _hb_dict.get("item_id") or _hb_dict.get("movieId")
            if not _iid:
                continue
            _pop = float(_hb_dict.get("popularity") or 10)
            _sc  = round(min(_pop / (_pop + 1.0), 0.999), 4)
            _hb_scored.append(ScoredItem(
                item_id=int(_iid), score=_sc,
                als_score=0.0, ranker_score=0.0,
                features_snapshot_id=request_id,
                policy_id="holdback_popularity",
            ))
        try:
            RETENTION.record_recommendation(
                uid, [s.item_id for s in _hb_scored[:10]]
            )
        except Exception:
            pass
        return RecommendResponse(
            user_id=uid, k=req.k,
            items=_hb_scored[:req.k],
            model_version=_MANIFEST,
            exploration_slots=0,
            diversity_score=0.0,
            freshness_watermark=_make_watermark(request_id),
            experiment_group="holdback_popularity",
        )
'''

new_lines = lines[:hb_idx] + [REPLACEMENT] + lines[end_idx:]
new_src = "".join(new_lines)

# ── 4: Syntax check ───────────────────────────────────────────────────────────
try:
    ast.parse(new_src)
    print("\nSyntax OK after patch")
except SyntaxError as e:
    print(f"\nSYNTAX ERROR at line {e.lineno}: {e.msg}")
    bad = new_src.splitlines()
    for i in range(max(0, e.lineno-3), min(len(bad), e.lineno+3)):
        mark = ">>>" if i+1 == e.lineno else "   "
        print(f"  {mark} {i+1}: {bad[i]}")
    sys.exit(1)

app.write_text(new_src)
print(f"Saved — {len(new_src.splitlines())} lines")

# ── 5: Test immediately against the live server ───────────────────────────────
print("\nTesting live server (old process — will fail until restart)...")
import urllib.request, json

def recommend(uid):
    req = urllib.request.Request(
        "http://localhost:8000/recommend", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.data = json.dumps({"user_id": uid, "k": 3}).encode()
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_err": e.code}
    except Exception as e:
        return {"_err": str(e)}

import hashlib
def is_hb(u): return int(hashlib.md5(f"cinewave_holdback_v1:{u}".encode()).hexdigest()[:8],16)/0xFFFFFFFF<0.05
HB = next(u for u in range(1,500) if is_hb(u))

r = recommend(HB)
grp = r.get("experiment_group"); items = len(r.get("items", []))
print(f"  user {HB}: group={grp!r}  items={items}  err={r.get('_err','none')}")

if grp == "holdback_popularity" and items > 0:
    print("\n  PASS — holdback working on live server (hot reload worked!)")
    print("  No restart needed.")
else:
    print("\n  Server needs restart to load patched code.")
    print("\n  Run now:")
    print("  docker restart recsys_api && sleep 40 && \\")
    print("  docker cp p.py recsys_api:/app/p.py && \\")
    print("  docker exec recsys_api python3 /app/p.py && sleep 5 && \\")
    print("  docker exec recsys_api python3 /app/fv.py")
