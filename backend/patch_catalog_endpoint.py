#!/usr/bin/env python3
"""
patch_catalog_endpoint.py
==========================
Run this ONCE from the backend container or local environment to patch
app.py so /catalog/popular serves TMDB movies with real posters.

Usage:
  python backend/patch_catalog_endpoint.py

What it does:
  1. Reads backend/src/recsys/serving/app.py
  2. Finds the /catalog/popular route
  3. Replaces it with a version that calls catalog_patch.get_tmdb_catalog()
  4. Writes back

Safe to run multiple times — idempotent.
"""
import re
import sys
from pathlib import Path

APP_PY = Path(__file__).parent / "src/recsys/serving/app.py"

if not APP_PY.exists():
    print(f"ERROR: {APP_PY} not found")
    sys.exit(1)

content = APP_PY.read_text(encoding="utf-8")

# Check if already patched
if "catalog_patch" in content:
    print("app.py already patched — nothing to do.")
    sys.exit(0)

# Add import at top (after existing imports)
import_line = "from recsys.serving.catalog_patch import get_tmdb_catalog, reload_catalog\n"
# Insert after last 'from recsys' import line
content = re.sub(
    r'(from recsys\.[^\n]+\n)(?!from recsys)',
    r'\1' + import_line,
    content,
    count=1
)

# Replace /catalog/popular endpoint
# Pattern: matches both sync and async def, with various parameter names
old_pattern = re.compile(
    r'@app\.get\(["\']\/catalog\/popular["\'][^\)]*\)\s*\n'
    r'(?:async\s+)?def\s+\w+[^:]*:[^\n]*\n'
    r'(?:.*\n)*?(?=@app\.|^$|\Z)',
    re.MULTILINE
)

new_endpoint = '''@app.get("/catalog/popular")
def catalog_popular(k: int = 1200):
    """
    Serve TMDB-enriched catalog with real posters.
    Patched by patch_catalog_endpoint.py to use catalog_patch.get_tmdb_catalog().
    """
    return {"items": get_tmdb_catalog(k)}

'''

match = old_pattern.search(content)
if match:
    content = content[:match.start()] + new_endpoint + content[match.end():]
    print("Replaced /catalog/popular endpoint.")
else:
    # Fallback: append the endpoint if pattern not found
    print("WARNING: Could not find existing /catalog/popular — appending new endpoint.")
    content += "\n\n" + new_endpoint

APP_PY.write_text(content, encoding="utf-8")
print(f"Patched: {APP_PY}")
print("Restart the API: docker compose restart api")
