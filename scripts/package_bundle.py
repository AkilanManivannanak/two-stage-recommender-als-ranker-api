from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "unknown"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    bundle = Path("artifacts/bundle")
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)

    # Choose “prod” artifacts:
    # - ALS trained on train_val => artifacts/als_test
    # - Ranker model => artifacts/ranker_lgbm/model.txt
    als_dir = Path("artifacts/als_test")
    ranker_model = Path("artifacts/ranker_lgbm/model.txt")
    user_feat = Path("artifacts/features/user_features.parquet")
    item_feat = Path("artifacts/features/item_features.parquet")
    pop = Path("artifacts/features/popularity.json")
    recent = Path("artifacts/features/user_recent_items.parquet")

    required = [als_dir, ranker_model, user_feat, item_feat, pop, recent]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required artifact(s): {missing}")

    # Copy artifacts into bundle
    copy_tree(als_dir, bundle / "als")
    (bundle / "ranker").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ranker_model, bundle / "ranker" / "model.txt")

    (bundle / "features").mkdir(parents=True, exist_ok=True)
    shutil.copy2(user_feat, bundle / "features" / "user_features.parquet")
    shutil.copy2(item_feat, bundle / "features" / "item_features.parquet")
    shutil.copy2(pop, bundle / "features" / "popularity.json")
    shutil.copy2(recent, bundle / "features" / "user_recent_items.parquet")

    # Feature spec for serving
    feature_cols = [
        "als_score",
        "user_cnt_total",
        "user_cnt_7d",
        "user_cnt_30d",
        "user_tenure_days",
        "user_recency_days",
        "item_cnt_total",
        "item_cnt_7d",
        "item_cnt_30d",
        "item_age_days",
        "item_recency_days",
    ]
    (bundle / "feature_spec.json").write_text(json.dumps({"feature_cols": feature_cols}, indent=2))

    # Serving config
    serving_cfg = {
        "topn_candidates": 500,
        "default_k": 10,
        "filter_recent_seen_n": 200,
    }
    (bundle / "serving_config.json").write_text(json.dumps(serving_cfg, indent=2))

    # Build manifest with hashes
    files: Dict[str, str] = {}
    for p in bundle.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(bundle))
            files[rel] = sha256_file(p)

    manifest = {
        "bundle_version": "v1",
        "git_commit": try_git_commit(),
        "created_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "paths": {
            "als": "als/",
            "ranker_model": "ranker/model.txt",
            "features": "features/",
            "feature_spec": "feature_spec.json",
            "serving_config": "serving_config.json",
        },
        "file_sha256": files,
        "notes": {
            "als_source": str(als_dir),
            "ranker_source": str(ranker_model),
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[OK] wrote artifacts/bundle/")
    print("[OK] wrote artifacts/bundle/manifest.json")


if __name__ == "__main__":
    main()
