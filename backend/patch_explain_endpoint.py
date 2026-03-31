#!/usr/bin/env python3
"""
patch_explain_endpoint.py
=========================
Replaces the hardcoded /explain endpoint in app.py with the
real OpenAI-powered smart_explain engine.

Run from the backend/ directory:
  python3 patch_explain_endpoint.py
"""
import re
import sys
from pathlib import Path

APP_PY = Path(__file__).parent / "src/recsys/serving/app.py"

if not APP_PY.exists():
    print(f"ERROR: {APP_PY} not found"); sys.exit(1)

content = APP_PY.read_text(encoding="utf-8")

# Already patched?
if "smart_explain" in content:
    print("app.py already has smart_explain — nothing to do."); sys.exit(0)

# Add import after catalog_patch import (or after last recsys import)
import_line = "from recsys.serving.smart_explain import get_explanations as _smart_explain\n"
if "catalog_patch" in content:
    content = content.replace(
        "from recsys.serving.catalog_patch import get_tmdb_catalog, reload_catalog\n",
        "from recsys.serving.catalog_patch import get_tmdb_catalog, reload_catalog\n" + import_line
    )
else:
    # Insert after last 'from recsys' import
    lines = content.split("\n")
    last_recsys = 0
    for i, line in enumerate(lines):
        if line.startswith("from recsys.") or line.startswith("import recsys"):
            last_recsys = i
    lines.insert(last_recsys + 1, import_line.strip())
    content = "\n".join(lines)

# Replace the /explain endpoint
# Matches the existing hardcoded endpoint pattern
NEW_EXPLAIN = '''
@app.post("/explain")
def explain(body: dict):
    """
    Real per-user per-movie explanation using GPT-4o + SHAP attribution.
    Cached in Redis (TTL 6h) — never in the hot ranking path.
    """
    user_id  = int(body.get("user_id", 1))
    item_ids = [int(i) for i in body.get("item_ids", [])]

    # Build catalog lookup for richer context
    try:
        catalog_items = get_tmdb_catalog(1200)
        catalog_map   = {int(c["item_id"]): c for c in catalog_items}
    except Exception:
        catalog_map = {}

    results = _smart_explain(
        user_id=user_id,
        item_ids=item_ids,
        catalog=catalog_map,
    )

    return {
        "user_id":      user_id,
        "explanations": [
            {
                "item_id":            r["item_id"],
                "reason":             r["reason"],
                "method":             r.get("method", "gpt4o_structured"),
                "attribution_method": "shap_gpt4o_hybrid",
            }
            for r in results
        ]
    }

'''

# Find and replace the existing /explain endpoint
pattern = re.compile(
    r'@app\.post\(["\']\/explain["\'][^\)]*\)\s*\n'
    r'(?:async\s+)?def\s+\w+[^:]*:.*?(?=\n@app\.|\Z)',
    re.DOTALL
)

match = pattern.search(content)
if match:
    content = content[:match.start()] + NEW_EXPLAIN + content[match.end():]
    print("Replaced /explain endpoint with smart_explain.")
else:
    # Append before the last line
    content = content.rstrip() + "\n" + NEW_EXPLAIN
    print("Appended new /explain endpoint (existing not found — manual check may be needed).")

APP_PY.write_text(content, encoding="utf-8")
print(f"Patched: {APP_PY}")
print("Now copy smart_explain.py into the container and restart the API.")
