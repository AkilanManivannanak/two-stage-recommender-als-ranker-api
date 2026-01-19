from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.features.build_ranker_dataset import build_aggregates, FeatureConfig


def main() -> None:
    Path("artifacts/features").mkdir(parents=True, exist_ok=True)

    split = json.loads(Path("data/processed/split_stats.json").read_text())
    test_cutoff = int(split["test_cutoff_ts"])

    # Use train_val as "latest history" before test window
    hist = pd.read_parquet("data/processed/train_val.parquet").sort_values(["user_id", "ts"])

    # 1) User/item aggregates (same logic as training features, leakage-safe)
    ufeat, ifeat = build_aggregates(hist, test_cutoff, FeatureConfig())
    ufeat.to_parquet("artifacts/features/user_features.parquet", index=False)
    ifeat.to_parquet("artifacts/features/item_features.parquet", index=False)

    # 2) Popularity list (fallback for unknown users / fill)
    pop = (
        hist.groupby("item_id")["weight"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(int)
        .tolist()
    )
    Path("artifacts/features/popularity.json").write_text(json.dumps({"items": pop[:5000]}, indent=2))

    # 3) Recent-items per user (filter previously seen items without loading all history)
    # Keep last N=200 items to filter; this is a practical serving compromise.
    N_RECENT = 200
    recent = hist.groupby("user_id")["item_id"].apply(lambda x: x.tail(N_RECENT).astype(int).tolist()).reset_index()
    recent.to_parquet("artifacts/features/user_recent_items.parquet", index=False)

    print("[OK] wrote artifacts/features/user_features.parquet")
    print("[OK] wrote artifacts/features/item_features.parquet")
    print("[OK] wrote artifacts/features/popularity.json")
    print("[OK] wrote artifacts/features/user_recent_items.parquet")


if __name__ == "__main__":
    main()
