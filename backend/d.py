#!/usr/bin/env python3
"""
Diagnoses the holdback issue and fixes it in one shot.
Run: docker cp ~/Downloads/diagnose.py recsys_api:/app/d.py && docker exec recsys_api python3 /app/d.py
"""
import sys, ast
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

from pathlib import Path

print("=" * 50)
print("STEP 1: Check context_and_additions import")
print("=" * 50)

try:
    from recsys.serving.context_and_additions import (
        get_experiment_group, is_holdback_user, popularity_fallback
    )
    grp63 = get_experiment_group(63)
    hb63  = is_holdback_user(63)
    print(f"  get_experiment_group(63) = {grp63!r}")
    print(f"  is_holdback_user(63)     = {hb63!r}")
    IMPORT_OK = True
except Exception as e:
    print(f"  IMPORT FAILED: {e}")
    IMPORT_OK = False

print()
print("=" * 50)
print("STEP 2: Check app.py holdback block")
print("=" * 50)

app = Path("/app/src/recsys/serving/app.py")
src = app.read_text()
lines = src.splitlines()

hb_idx = None
for i, line in enumerate(lines):
    if 'experiment_group == "holdback_popularity"' in line:
        hb_idx = i
        break

if hb_idx is None:
    print("  ERROR: holdback block not found in app.py!")
else:
    print(f"  Found at line {hb_idx + 1}")
    print("  Block:")
    for ln in lines[hb_idx:hb_idx + 20]:
        print(f"    {ln}")

print()
print("=" * 50)
print("STEP 3: Fix the holdback block")
print("=" * 50)

if hb_idx is None:
    print("  Cannot fix - block not found")
    sys.exit(1)

# Find end of block
end_idx = None
for i in range(hb_idx + 1, min(hb_idx + 30, len(lines))):
    stripped = lines[i].strip()
    if stripped.startswith("wm = ") or stripped.startswith("features_snapshot_id"):
        end_idx = i
        break

if end_idx is None:
    print("  Cannot find block end - showing lines:")
    for i, ln in enumerate(lines[hb_idx:hb_idx+30], start=hb_idx):
        print(f"    {i+1}: {ln}")
    sys.exit(1)

print(f"  Block spans lines {hb_idx+1} to {end_idx} (exclusive)")

FIXED_BLOCK = [
    '    if experiment_group == "holdback_popularity":\n',
    '        holdback_items = popularity_fallback(CATALOG, top_k=max(req.k, 10))\n',
    '        try:\n',
    '            RETENTION.record_recommendation(uid, [i.get("item_id", 0) for i in holdback_items[:10]])\n',
    '        except Exception:\n',
    '            pass\n',
    '        _hb = []\n',
    '        for _hi in holdback_items:\n',
    '            _iid = _hi.get("item_id") or _hi.get("movieId")\n',
    '            if not _iid:\n',
    '                continue\n',
    '            _pop = float(_hi.get("popularity") or 10)\n',
    '            _sc  = round(min(_pop / (_pop + 1.0), 0.999), 4)\n',
    '            _hb.append(ScoredItem(\n',
    '                item_id=int(_iid), score=_sc,\n',
    '                als_score=0.0, ranker_score=0.0,\n',
    '                features_snapshot_id=request_id,\n',
    '                policy_id="holdback_popularity",\n',
    '            ))\n',
    '        return RecommendResponse(\n',
    '            user_id=uid, k=req.k,\n',
    '            items=_hb[:req.k],\n',
    '            model_version=_MANIFEST,\n',
    '            exploration_slots=0,\n',
    '            diversity_score=0.0,\n',
    '            freshness_watermark=_make_watermark(request_id),\n',
    '            experiment_group="holdback_popularity",\n',
    '        )\n',
]

new_lines = lines[:hb_idx]
new_lines = [l + "\n" for l in new_lines]
new_lines += FIXED_BLOCK
new_lines += [l + "\n" for l in lines[end_idx:]]
new_src = "".join(new_lines)

try:
    ast.parse(new_src)
    print("  Syntax OK after patch")
except SyntaxError as e:
    print(f"  SYNTAX ERROR at line {e.lineno}: {e.msg}")
    bad = new_src.splitlines()
    for i in range(max(0, e.lineno-3), min(len(bad), e.lineno+3)):
        mark = ">>>" if i+1 == e.lineno else "   "
        print(f"  {mark} {i+1}: {bad[i]}")
    sys.exit(1)

app.write_text(new_src)
print(f"  Saved — {len(new_src.splitlines())} lines")

print()
print("=" * 50)
print("STEP 4: Test live endpoint")
print("=" * 50)

import urllib.request, json
req = urllib.request.Request(
    "http://localhost:8000/recommend", method="POST"
)
req.add_header("Content-Type", "application/json")
req.data = json.dumps({"user_id": 63, "k": 3}).encode()
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read().decode()
        d = json.loads(body)
        grp   = d.get("experiment_group")
        items = len(d.get("items", []))
        print(f"  experiment_group={grp!r}  items={items}")
        if grp == "holdback_popularity" and items > 0:
            print("  PASS — holdback working!")
        else:
            print("  Still failing — the running server needs a restart")
            print("  Run: docker restart recsys_api")
            print("  Then: docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py")
except Exception as e:
    body = getattr(e, 'read', lambda: b'')()
    print(f"  HTTP error: {e}")
    print(f"  Body: {body[:200] if body else '(empty)'}")
    print("  NOTE: File was patched on disk. Restart the container to reload.")
    print("  Run: docker restart recsys_api")

print()
print("Done.")
