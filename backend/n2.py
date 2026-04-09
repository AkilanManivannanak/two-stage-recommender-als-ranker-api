#!/usr/bin/env python3
"""
docker cp ~/Downloads/nuke2.py recsys_api:/app/n2.py
docker exec recsys_api python3 /app/n2.py
"""
import sys, ast
from pathlib import Path

app = Path("/app/src/recsys/serving/app.py")
lines = app.read_text().splitlines()

print(f"Total lines: {len(lines)}")

# Find start
hb_start = next(i for i,l in enumerate(lines)
                if 'experiment_group == "holdback_popularity"' in l)
n = len(lines[hb_start]) - len(lines[hb_start].lstrip())
SP  = " " * n
SP2 = " " * (n + 4)

print(f"Holdback starts at line {hb_start+1}, indent={n}")

# Print lines from hb_start to find the TRUE end
print("\nLines from holdback start:")
for i in range(hb_start, min(hb_start+50, len(lines))):
    print(f"  {i+1:4d}: |{lines[i]}")

# Find true end: first line at or below base indent that is NOT blank
# and is NOT part of the holdback block
hb_end = None
for i in range(hb_start + 1, hb_start + 60):
    line = lines[i]
    stripped = line.strip()
    if not stripped:
        continue  # skip blank lines
    line_indent = len(line) - len(line.lstrip())
    # End of block = line at same or lower indent than the if-line
    # i.e. it's a sibling statement (wm = ...) at indent n
    if line_indent <= n and stripped:
        hb_end = i
        print(f"\nTrue block end at line {hb_end+1}: {lines[hb_end]!r}")
        break

assert hb_end is not None, "Could not find end"

# Build replacement
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

new_src = "\n".join(lines[:hb_start]) + "\n" + BLOCK + "\n".join(lines[hb_end:])

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
print(f"Saved — {len(new_src.splitlines())} lines")
print("Restart now:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/fv.py")
