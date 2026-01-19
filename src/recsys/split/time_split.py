from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplitConfig:
    test_frac: float = 0.10
    val_frac: float = 0.10
    test_cutoff_ts: Optional[int] = None
    val_cutoff_ts: Optional[int] = None


def _compute_cutoffs(ts: np.ndarray, cfg: TimeSplitConfig) -> Tuple[int, int]:
    if cfg.test_cutoff_ts is not None and cfg.val_cutoff_ts is not None:
        if cfg.val_cutoff_ts >= cfg.test_cutoff_ts:
            raise ValueError("val_cutoff_ts must be < test_cutoff_ts")
        return int(cfg.val_cutoff_ts), int(cfg.test_cutoff_ts)

    if not (0.0 < cfg.test_frac < 1.0):
        raise ValueError("test_frac must be in (0,1)")
    if not (0.0 <= cfg.val_frac < 1.0):
        raise ValueError("val_frac must be in [0,1)")
    if cfg.test_frac + cfg.val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be < 1")

    q_test = 1.0 - cfg.test_frac
    q_val = 1.0 - (cfg.test_frac + cfg.val_frac)

    test_cutoff = int(np.quantile(ts, q_test))
    val_cutoff = int(np.quantile(ts, q_val))
    if val_cutoff >= test_cutoff:
        val_cutoff = test_cutoff - 1
    return val_cutoff, test_cutoff


def time_split_interactions(
    interactions_path: str | Path,
    out_dir: str | Path,
    cfg: TimeSplitConfig,
) -> Dict[str, Path]:
    interactions_path = Path(interactions_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(interactions_path)
    required = {"user_id", "item_id", "ts", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ts = df["ts"].to_numpy()
    val_cutoff, test_cutoff = _compute_cutoffs(ts, cfg)

    train_df = df[df["ts"] <= val_cutoff].copy()
    val_df = df[(df["ts"] > val_cutoff) & (df["ts"] <= test_cutoff)].copy()
    test_df = df[df["ts"] > test_cutoff].copy()

    train_users = set(train_df["user_id"].unique().tolist())
    eligible_val_users = sorted(list(set(val_df["user_id"].unique().tolist()) & train_users))
    eligible_test_users = sorted(list(set(test_df["user_id"].unique().tolist()) & train_users))

    holdout_val = val_df[val_df["user_id"].isin(eligible_val_users)][["user_id", "item_id"]].drop_duplicates()
    holdout_test = test_df[test_df["user_id"].isin(eligible_test_users)][["user_id", "item_id"]].drop_duplicates()

    paths: Dict[str, Path] = {}
    paths["train"] = out_dir / "train.parquet"
    paths["val"] = out_dir / "val.parquet"
    paths["test"] = out_dir / "test.parquet"
    paths["holdout_val"] = out_dir / "holdout_targets_val.parquet"
    paths["holdout_test"] = out_dir / "holdout_targets_test.parquet"
    paths["eligible_val_users"] = out_dir / "eligible_users_val.parquet"
    paths["eligible_test_users"] = out_dir / "eligible_users_test.parquet"
    paths["split_stats"] = out_dir / "split_stats.json"

    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["val"], index=False)
    test_df.to_parquet(paths["test"], index=False)
    holdout_val.to_parquet(paths["holdout_val"], index=False)
    holdout_test.to_parquet(paths["holdout_test"], index=False)

    pd.DataFrame({"user_id": eligible_val_users}).to_parquet(paths["eligible_val_users"], index=False)
    pd.DataFrame({"user_id": eligible_test_users}).to_parquet(paths["eligible_test_users"], index=False)

    stats = {
        "val_cutoff_ts": int(val_cutoff),
        "test_cutoff_ts": int(test_cutoff),
        "counts": {
            "train_events": int(len(train_df)),
            "val_events": int(len(val_df)),
            "test_events": int(len(test_df)),
            "train_users": int(train_df["user_id"].nunique()),
            "val_users": int(val_df["user_id"].nunique()),
            "test_users": int(test_df["user_id"].nunique()),
            "eligible_val_users": int(len(eligible_val_users)),
            "eligible_test_users": int(len(eligible_test_users)),
            "holdout_val_pairs": int(len(holdout_val)),
            "holdout_test_pairs": int(len(holdout_test)),
        },
        "fractions": {"test_frac": cfg.test_frac, "val_frac": cfg.val_frac},
    }
    paths["split_stats"].write_text(json.dumps(stats, indent=2))
    return paths
