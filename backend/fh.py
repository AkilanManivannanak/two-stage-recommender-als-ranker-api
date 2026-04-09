#!/usr/bin/env python3
"""
docker cp ~/Downloads/fix_hb.py recsys_api:/app/fh.py
docker exec recsys_api python3 /app/fh.py
"""
import sys, ast
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")
from pathlib import Path

app = Path("/app/src/recsys/serving/app.py")
src = app.read_text()
lines = src.splitlines()

# Find holdback line and detect its exact indentation
hb_idx = next(i for i,l in enumerate(lines) if 'experiment_group == "holdback_popularity"' in l)
indent = len(lines[hb_idx]) - len(lines[hb_idx].lstrip())
I = " " * indent        # base indent (matches the if line)
I2 = " " * (indent+4)   # one level in

print(f"Holdback at line {hb_idx+1}, base indent={indent} spaces")

# Find end of block
end_idx = None
for i in range(hb_idx+1, hb_idx+40):
    s = lines[i].lstrip()
    if s.startswith("wm = ") or s.startswith("features_snapshot_id"):
        end_idx = i
        break

assert end_idx, "Could not find block end"
print(f"Block ends at line {end_idx+1}")

# Build replacement using exact detected indentation
block = f'''\
{I}if experiment_group == "holdback_popularity":
{I2}_hb_raw = popularity_fallback(CATALOG, top_k=max(req.k, 10))
{I2}_hb_scored = []
{I2}for _hbi in _hb_raw:
{I2}    if isinstance(_hbi, int):
{I2}        _hbd = CATALOG.get(_hbi, {{}}); _iid = _hbi
{I2}    elif isinstance(_hbi, dict):
{I2}        _hbd = _hbi; _iid = _hbi.get("item_id") or _hbi.get("movieId")
{I2}    else:
{I2}        continue
{I2}    if not _iid:
{I2}        continue
{I2}    _pop = float(_hbd.get("popularity") or 10)
{I2}    _sc  = round(min(_pop / (_pop + 1.0), 0.999), 4)
{I2}    _hb_scored.append(ScoredItem(
{I2}        item_id=int(_iid), score=_sc,
{I2}        als_score=0.0, ranker_score=0.0,
{I2}        features_snapshot_id=request_id,
{I2}        policy_id="holdback_popularity",
{I2}    ))
{I2}try:
{I2}    RETENTION.record_recommendation(uid, [s.item_id for s in _hb_scored[:10]])
{I2}except Exception:
{I2}    pass
{I2}return RecommendResponse(
{I2}    user_id=uid, k=req.k,
{I2}    items=_hb_scored[:req.k],
{I2}    model_version=_MANIFEST,
{I2}    exploration_slots=0,
{I2}    diversity_score=0.0,
{I2}    freshness_watermark=_make_watermark(request_id),
{I2}    experiment_group="holdback_popularity",
{I2})
'''

new_lines = lines[:hb_idx] + block.splitlines() + [""] + lines[end_idx:]
new_src = "\n".join(new_lines)

try:
    ast.parse(new_src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    bad = new_src.splitlines()
    for i in range(max(0,e.lineno-3), min(len(bad),e.lineno+3)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

app.write_text(new_src)
print(f"Saved — {len(new_src.splitlines())} lines")

# Quick sanity check
assert 'isinstance(_hbi, int)' in new_src
assert 'experiment_group="holdback_popularity"' in new_src
print("Assertions OK — restart container now:")
print("  docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/fv.py")
