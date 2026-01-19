from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.features.build_ranker_dataset import build_ranker_dataset


def main() -> None:
    split = json.loads(Path("data/processed/split_stats.json").read_text())
    val_cutoff = int(split["val_cutoff_ts"])
    test_cutoff = int(split["test_cutoff_ts"])

    # Build VAL ranker dataset (history=train, labels=val holdout)
    val_path = build_ranker_dataset(
        history_path="data/processed/train.parquet",
        candidates_path="data/processed/candidates_val.parquet",
        holdout_path="data/processed/holdout_targets_val.parquet",
        cutoff_ts=val_cutoff,
        out_path="data/processed/ranker_val_full.parquet",
    )
    print("[OK] wrote", val_path)

    # Split val users into train/val for ranker training
    dfv = pd.read_parquet(val_path)
    users = dfv["user_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(42)
    rng.shuffle(users)
    n_train = int(0.8 * len(users))
    train_users = set(users[:n_train].tolist())

    ranker_train = dfv[dfv["user_id"].isin(train_users)]
    ranker_val = dfv[~dfv["user_id"].isin(train_users)]

    ranker_train.to_parquet("data/processed/ranker_train.parquet", index=False)
    ranker_val.to_parquet("data/processed/ranker_val.parquet", index=False)
    print("[OK] wrote data/processed/ranker_train.parquet")
    print("[OK] wrote data/processed/ranker_val.parquet")

    # Build TEST ranker dataset (history=train_val, labels=test holdout)
    test_path = build_ranker_dataset(
        history_path="data/processed/train_val.parquet",
        candidates_path="data/processed/candidates_test.parquet",
        holdout_path="data/processed/holdout_targets_test.parquet",
        cutoff_ts=test_cutoff,
        out_path="data/processed/ranker_test.parquet",
    )
    print("[OK] wrote", test_path)


if __name__ == "__main__":
    main()
