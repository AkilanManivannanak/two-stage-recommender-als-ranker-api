"""
Script: package_bundle.py
==========================
Writes the CURRENT pointer in the model registry, making the latest
bundle the active serving version.

artifacts/registry/{env}/CURRENT  → bundle_id
"""
import os, time
from pathlib import Path

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR","artifacts"))
ENV           = os.environ.get("ENV","dev")
BUNDLE_ID     = os.environ.get("BUNDLE_ID","rec-bundle-v2.4.0")

registry_dir = ARTIFACTS_DIR/"registry"/ENV
registry_dir.mkdir(parents=True, exist_ok=True)

# Copy bundle to registry
import shutil
src = ARTIFACTS_DIR/f"bundle_{ENV}"
dst = registry_dir/BUNDLE_ID
if src.exists() and not dst.exists():
    shutil.copytree(src, dst)

# Write CURRENT pointer
current_file = registry_dir/"CURRENT"
current_file.write_text(BUNDLE_ID)

print(f"[package] Registry updated: {current_file}  →  {BUNDLE_ID}")
