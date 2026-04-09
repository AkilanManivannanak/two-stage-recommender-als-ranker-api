#!/usr/bin/env python3
"""
docker cp ~/Downloads/fix_ope2.py recsys_api:/app/fo2.py
docker exec recsys_api python3 /app/fo2.py
"""
import sys, ast, json, urllib.request
sys.path.insert(0,"/app/src"); sys.path.insert(0,"/app")
from pathlib import Path

p = Path("/app/src/recsys/serving/app.py")
src = p.read_text()

# Root cause: FastAPI routes /eval/ope/{user_id} before /eval/ope/global
# "global" is not a valid int → 422. Fix: change path to /eval/ope_global

changes = 0

# Fix endpoint path
OLD1 = '@app.get("/eval/ope/global")'
NEW1 = '@app.get("/eval/ope_global")'
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    changes += 1; print("Fixed: /eval/ope/global → /eval/ope_global")

# Fix verify script too
va = Path("/app/va.py")
va_src = va.read_text()
if '"/eval/ope/global' in va_src:
    va_src = va_src.replace('"/eval/ope/global', '"/eval/ope_global')
    va.write_text(va_src)
    print("Fixed verify script URL")

try:
    ast.parse(src)
    p.write_text(src)
    print(f"{changes} fix — {len(src.splitlines())} lines, syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    sys.exit(1)

# Test it works now via direct function call (old server still running)
print("\nVerifying endpoint function directly...")
try:
    from recsys.serving.app import ope_global
    # Simulate Query parameter
    class Q:
        def __init__(self, v): self.v = v
        def __index__(self): return self.v
    import inspect
    result = ope_global.__wrapped__(20) if hasattr(ope_global, '__wrapped__') else None
    print(f"Direct call: {result}")
except Exception as e:
    print(f"Direct call failed (expected — needs HTTP): {e}")

print("\nNow restart:")
print("docker restart recsys_api && sleep 40 && docker cp p.py recsys_api:/app/p.py && docker exec recsys_api python3 /app/p.py && sleep 5 && docker exec recsys_api python3 /app/va.py")
