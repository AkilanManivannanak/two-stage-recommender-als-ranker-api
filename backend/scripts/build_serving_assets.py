"""
Script: build_serving_assets.py
================================
Assembles the serving bundle from all trained artifacts.
The bundle is a single directory that the FastAPI serving layer loads on startup.

Bundle structure:
  artifacts/bundle_{env}/
    manifest.json
    feature_spec.json
    serving_config.json
    als/
      user_factors.npy
      item_factors.npy
      mappings.json
    ranker/
      model.txt  (or model.pkl)
    features/
      user_features.parquet
      item_features.parquet
      item_metadata.parquet
      user_recent_items.parquet
      cooccurrence_neighbors.parquet
      popularity.json
"""
import json, os, shutil, time
from pathlib import Path
import numpy as np

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR","artifacts"))
ENV           = os.environ.get("ENV","dev")
BUNDLE_ID     = os.environ.get("BUNDLE_ID", f"rec-bundle-v2.4.0")

t0 = time.time()
bundle_dir = ARTIFACTS_DIR / f"bundle_{ENV}"
bundle_dir.mkdir(parents=True, exist_ok=True)
print(f"[bundle] Assembling serving bundle → {bundle_dir}")

def copy_if_exists(src, dst):
    src=Path(src); dst=Path(dst)
    if src.exists():
        if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
        else: shutil.copy2(src, dst)
        print(f"  ✓ {src.name}")
    else:
        print(f"  ⚠ missing: {src}")

# ALS
als_dst = bundle_dir/"als"; als_dst.mkdir(exist_ok=True)
copy_if_exists(ARTIFACTS_DIR/f"als_{ENV}"/"user_factors.npy", als_dst/"user_factors.npy")
copy_if_exists(ARTIFACTS_DIR/f"als_{ENV}"/"item_factors.npy", als_dst/"item_factors.npy")
copy_if_exists(ARTIFACTS_DIR/f"als_{ENV}"/"mappings.json",    als_dst/"mappings.json")

# Ranker
ranker_dst = bundle_dir/"ranker"; ranker_dst.mkdir(exist_ok=True)
copy_if_exists(ARTIFACTS_DIR/f"ranker_{ENV}"/"model.txt", ranker_dst/"model.txt")
copy_if_exists(ARTIFACTS_DIR/f"ranker_{ENV}"/"model.pkl", ranker_dst/"model.pkl")

# Features
feat_dst = bundle_dir/"features"; feat_dst.mkdir(exist_ok=True)
feat_src = ARTIFACTS_DIR/f"features_{ENV}"
for fname in ["user_features.parquet","item_features.parquet","item_metadata.parquet",
              "user_recent_items.parquet","cooccurrence_neighbors.parquet","popularity.json"]:
    copy_if_exists(feat_src/fname, feat_dst/fname)

# Specs
copy_if_exists(ARTIFACTS_DIR/f"feature_spec_{ENV}.json",  bundle_dir/"feature_spec.json")
copy_if_exists(ARTIFACTS_DIR/f"serving_config_{ENV}.json", bundle_dir/"serving_config.json")

# Manifest
manifest = {
    "bundle_id":  BUNDLE_ID,
    "version":    "2.4.0",
    "env":        ENV,
    "als_model":  f"als_rank64_reg0.01_iter15",
    "ranker_model": "gbm_ranker_v2",
    "als_factors": 64, "als_iterations": 15, "als_regularization": 0.01,
    "ranker_estimators": 200, "ranker_max_depth": 6,
    "diversity_cap": 3, "mmr_lambda": 0.75, "exploration_ratio": 0.15,
    "created_at_utc": __import__("datetime").datetime.utcnow().isoformat(),
}
with open(bundle_dir/"manifest.json","w") as f:
    json.dump(manifest, f, indent=2)
print(f"  ✓ manifest.json  (bundle_id={BUNDLE_ID})")

print(f"[bundle] Done in {time.time()-t0:.1f}s  →  {bundle_dir}")
