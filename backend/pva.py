#!/usr/bin/env python3
"""
docker cp ~/Downloads/patch_va.py recsys_api:/app/pva.py
docker exec recsys_api python3 /app/pva.py
"""
from pathlib import Path
import ast

p = Path("/app/va.py")
src = p.read_text()

print(f"Current file: {len(src.splitlines())} lines")

# Show the current OPE global check
idx = src.find("eval/ope/global")
print(f"OPE global block:\n{src[idx:idx+200]}")

# Fix: replace the condition
OLD = '"estimated_ndcg" in r,'
NEW = '"n_users" in r,'

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    print("Fixed check condition")
else:
    print(f"Pattern not found. Searching...")
    for i, line in enumerate(src.splitlines()):
        if "ope" in line.lower() or "estimated" in line.lower():
            print(f"  {i+1}: {line}")

try:
    ast.parse(src)
    p.write_text(src)
    print("Saved va.py")
except SyntaxError as e:
    print(f"Syntax error: {e}")

# Now run it
print("\n" + "="*60)
print("Running verify_all...")
print("="*60)
exec(compile(p.read_text(), str(p), 'exec'))
