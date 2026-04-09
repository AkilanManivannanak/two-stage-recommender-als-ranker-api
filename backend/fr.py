#!/usr/bin/env python3
"""
Fix the root cause: popularity_fallback() returns integers, not dicts.
Fix it in context_and_additions.py to return full item dicts.
Also add isinstance guard in app.py as defence-in-depth.

docker cp ~/Downloads/fix_root.py recsys_api:/app/fr.py
docker exec recsys_api python3 /app/fr.py
"""
import sys, ast
from pathlib import Path

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

# ── Step 1: Show what popularity_fallback currently does ──────────────────────
print("=== Step 1: Read current popularity_fallback ===")
ctx_path = Path("/app/src/recsys/serving/context_and_additions.py")
ctx_src = ctx_path.read_text()

idx = ctx_src.find("def popularity_fallback")
if idx >= 0:
    end = ctx_src.find("\ndef ", idx + 1)
    print(ctx_src[idx:end if end > 0 else idx+300])
else:
    print("popularity_fallback function not found — checking fallback stub")
    idx2 = ctx_src.find("popularity_fallback")
    print(ctx_src[max(0,idx2-50):idx2+200])

# ── Step 2: Show what it actually returns ─────────────────────────────────────
print("\n=== Step 2: Test return value ===")
try:
    from recsys.serving.context_and_additions import popularity_fallback
    from recsys.serving.app import CATALOG
    result = popularity_fallback(CATALOG, top_k=3)
    print(f"Returns {len(result)} items")
    print(f"Type of first: {type(result[0])}")
    print(f"First item: {repr(result[0])[:150]}")
except Exception as e:
    print(f"Error: {e}")

# ── Step 3: Fix popularity_fallback in context_and_additions.py ──────────────
print("\n=== Step 3: Fix popularity_fallback ===")

# The function should return list of dicts, not ints
OLD_STUB = "def popularity_fallback(catalog, top_k=30): return sorted(catalog.values(), key=lambda x: -x.get(\"popularity\", 0))[:top_k]"
NEW_STUB = """def popularity_fallback(catalog, top_k=30):
    items = sorted(catalog.values(), key=lambda x: -x.get("popularity", 0))[:top_k]
    # Always return dicts with item_id guaranteed
    result = []
    for item in items:
        if isinstance(item, dict):
            d = dict(item)
            if "item_id" not in d and "movieId" in d:
                d["item_id"] = d["movieId"]
            result.append(d)
        elif isinstance(item, int):
            d = dict(catalog.get(item, {"item_id": item}))
            if "item_id" not in d:
                d["item_id"] = item
            result.append(d)
    return result"""

if OLD_STUB in ctx_src:
    new_ctx = ctx_src.replace(OLD_STUB, NEW_STUB, 1)
    print("  Fixed stub version")
else:
    # Try the full function version
    if "def popularity_fallback" in ctx_src:
        lines = ctx_src.splitlines()
        fn_start = next(i for i,l in enumerate(lines) if "def popularity_fallback" in l)
        fn_end = next((i for i in range(fn_start+1, fn_start+20)
                       if lines[i].strip() and not lines[i].startswith(" ") and not lines[i].startswith("\t")),
                      fn_start + 5)
        new_lines = lines[:fn_start] + NEW_STUB.splitlines() + lines[fn_end:]
        new_ctx = "\n".join(new_lines)
        print(f"  Fixed full function at line {fn_start+1}")
    else:
        print("  Function not found — appending it")
        new_ctx = ctx_src + "\n\n" + NEW_STUB + "\n"

try:
    ast.parse(new_ctx)
    ctx_path.write_text(new_ctx)
    print(f"  Saved context_and_additions.py ({len(new_ctx.splitlines())} lines)")
except SyntaxError as e:
    print(f"  Syntax error: {e} — not saving")

# ── Step 4: Also fix the app.py holdback block with isinstance guard ──────────
print("\n=== Step 4: Fix app.py holdback to handle both int and dict ===")
app_path = Path("/app/src/recsys/serving/app.py")
app_src = app_path.read_text()
app_lines = app_src.splitlines()

hb_idx = next(i for i,l in enumerate(app_lines)
              if 'experiment_group == "holdback_popularity"' in l)
n = len(app_lines[hb_idx]) - len(app_lines[hb_idx].lstrip())
SP = " " * n
SP2 = " " * (n+4)

# Find true end (first non-blank line at indent <= n after hb_idx)
hb_end = next(i for i in range(hb_idx+1, hb_idx+60)
              if app_lines[i].strip() and
              len(app_lines[i]) - len(app_lines[i].lstrip()) <= n)

print(f"  Block: lines {hb_idx+1}–{hb_end}")

BLOCK = (
    SP  + 'if experiment_group == "holdback_popularity":\n'
    + SP2 + '_hbr = popularity_fallback(CATALOG, top_k=max(req.k, 10))\n'
    + SP2 + '_hbs = []\n'
    + SP2 + 'for _x in _hbr:\n'
    + SP2 + '    # Handle both int item IDs and full item dicts\n'
    + SP2 + '    if isinstance(_x, int):\n'
    + SP2 + '        _d = CATALOG.get(_x, {}); _id = _x\n'
    + SP2 + '    elif isinstance(_x, dict):\n'
    + SP2 + '        _d = _x\n'
    + SP2 + '        _id = _x.get("item_id") or _x.get("movieId")\n'
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
    + SP2 + '    RETENTION.record_recommendation(\n'
    + SP2 + '        uid, [s.item_id for s in _hbs[:10]])\n'
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

new_app = ("\n".join(app_lines[:hb_idx])
           + "\n" + BLOCK
           + "\n".join(app_lines[hb_end:]))

try:
    ast.parse(new_app)
    app_path.write_text(new_app)
    print(f"  Saved app.py ({len(new_app.splitlines())} lines, syntax OK)")
except SyntaxError as e:
    print(f"  Syntax error line {e.lineno}: {e.msg}")
    bad = new_app.splitlines()
    for i in range(max(0,e.lineno-3), min(len(bad),e.lineno+3)):
        print(f"  {'>>>' if i+1==e.lineno else '   '} {i+1}: {bad[i]}")
    sys.exit(1)

# ── Step 5: Verify the fix works locally ─────────────────────────────────────
print("\n=== Step 5: Local sanity check ===")
try:
    import importlib
    import recsys.serving.context_and_additions as _ctx
    importlib.reload(_ctx)
    pfb = _ctx.popularity_fallback
    from recsys.serving.app import CATALOG
    items = pfb(CATALOG, top_k=3)
    print(f"  popularity_fallback returns {len(items)} items")
    print(f"  Type: {type(items[0])}")
    if isinstance(items[0], dict):
        print(f"  item_id={items[0].get('item_id')}  popularity={items[0].get('popularity'):.1f}")
        print("  FIXED — returns dicts with item_id")
    else:
        print(f"  Still returning {type(items[0])} — check manually")
except Exception as e:
    print(f"  Could not verify: {e}")

print("\n=== Done ===")
print("Restart container:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/fv.py")
