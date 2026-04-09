#!/usr/bin/env python3
"""
docker cp ~/Downloads/nuke_hb.py recsys_api:/app/nh.py
docker exec recsys_api python3 /app/nh.py
"""
import sys, ast
from pathlib import Path

app = Path("/app/src/recsys/serving/app.py")
src = app.read_text()
lines = src.splitlines()

print(f"File: {len(lines)} lines")

# Show every line that mentions holdback so we see the full mess
print("\nAll holdback-related lines:")
for i, l in enumerate(lines):
    if "holdback" in l.lower() or "popularity_fallback" in l.lower() or "_hb" in l:
        print(f"  {i+1:4d}: {l}")

# Find the FIRST holdback if-line
hb_start = next((i for i,l in enumerate(lines)
                 if 'experiment_group == "holdback_popularity"' in l), None)
assert hb_start is not None, "No holdback line found"

# Find where normal flow resumes (wm = or features_snapshot_id)
hb_end = None
for i in range(hb_start + 1, hb_start + 60):
    s = lines[i].strip()
    if s.startswith("wm = ") or s.startswith("features_snapshot_id"):
        hb_end = i
        break

assert hb_end is not None, f"Could not find block end after line {hb_start+1}"
print(f"\nWill replace lines {hb_start+1}..{hb_end} (exclusive)")
print("Current block:")
for l in lines[hb_start:hb_end]:
    print(f"  |{l}")

# Detect indent from the if-line itself
raw = lines[hb_start]
n = len(raw) - len(raw.lstrip())
SP  = " " * n        # 4 spaces (matches "    if experiment_group")
SP2 = " " * (n + 4)  # 8 spaces (one level inside the if)

print(f"\nIndent: base={n}, inner={n+4}")

# Build the clean replacement — plain string, no f-string nesting issues
BLOCK = (
    SP  + 'if experiment_group == "holdback_popularity":\n'
    + SP2 + '_hbr = popularity_fallback(CATALOG, top_k=max(req.k, 10))\n'
    + SP2 + '_hbs = []\n'
    + SP2 + 'for _x in _hbr:\n'
    + SP2 + '    if isinstance(_x, int):\n'
    + SP2 + '        _d = CATALOG.get(_x, {}); _id = _x\n'
    + SP2 + '    elif isinstance(_x, dict):\n'
    + SP2 + '        _d = _x; _id = _x.get("item_id") or _x.get("movieId")\n'
    + SP2 + '    else:\n'
    + SP2 + '        continue\n'
    + SP2 + '    if not _id:\n'
    + SP2 + '        continue\n'
    + SP2 + '    _p = float(_d.get("popularity") or 10)\n'
    + SP2 + '    _hbs.append(ScoredItem(\n'
    + SP2 + '        item_id=int(_id),\n'
    + SP2 + '        score=round(min(_p/(_p+1.0), 0.999), 4),\n'
    + SP2 + '        als_score=0.0, ranker_score=0.0,\n'
    + SP2 + '        features_snapshot_id=request_id,\n'
    + SP2 + '        policy_id="holdback_popularity",\n'
    + SP2 + '    ))\n'
    + SP2 + 'try:\n'
    + SP2 + '    RETENTION.record_recommendation(uid, [s.item_id for s in _hbs[:10]])\n'
    + SP2 + 'except Exception:\n'
    + SP2 + '    pass\n'
    + SP2 + 'return RecommendResponse(\n'
    + SP2 + '    user_id=uid, k=req.k, items=_hbs[:req.k],\n'
    + SP2 + '    model_version=_MANIFEST, exploration_slots=0,\n'
    + SP2 + '    diversity_score=0.0,\n'
    + SP2 + '    freshness_watermark=_make_watermark(request_id),\n'
    + SP2 + '    experiment_group="holdback_popularity",\n'
    + SP2 + ')\n'
)

before = "\n".join(lines[:hb_start])
after  = "\n".join(lines[hb_end:])
new_src = before + "\n" + BLOCK + after

try:
    ast.parse(new_src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    bad = new_src.splitlines()
    for i in range(max(0,e.lineno-4), min(len(bad),e.lineno+4)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

app.write_text(new_src)
n_lines = len(new_src.splitlines())
print(f"Saved — {n_lines} lines")

# Confirm holdback lines now look right
src2 = app.read_text(); lines2 = src2.splitlines()
print("\nNew holdback block:")
hb2 = next(i for i,l in enumerate(lines2) if 'experiment_group == "holdback_popularity"' in l)
for l in lines2[hb2:hb2+35]:
    print(f"  |{l}")
    if l.strip().startswith("wm = ") or l.strip().startswith("features_snapshot_id"):
        break

print("\nDone. Now restart:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/fv.py")
