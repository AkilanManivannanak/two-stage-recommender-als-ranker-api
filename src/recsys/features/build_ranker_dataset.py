from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


SECONDS_IN_DAY = 86400.0


@dataclass(frozen=True)
class FeatureConfig:
    window_7d: int = 7
    window_30d: int = 30


def _counts_in_window(df: pd.DataFrame, cutoff_ts: int, days: int, key: str) -> pd.DataFrame:
    lo = cutoff_ts - int(days * SECONDS_IN_DAY)
    w = df[(df["ts"] >= lo) & (df["ts"] <= cutoff_ts)]
    out = w.groupby(key)["weight"].sum().reset_index().rename(columns={"weight": f"{key}_cnt_{days}d"})
    return out


def build_aggregates(history: pd.DataFrame, cutoff_ts: int, cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # user aggregates
    u_base = history.groupby("user_id").agg(
        user_cnt_total=("weight", "sum"),
        user_first_ts=("ts", "min"),
        user_last_ts=("ts", "max"),
    ).reset_index()

    u_7 = _counts_in_window(history, cutoff_ts, cfg.window_7d, "user_id")
    u_30 = _counts_in_window(history, cutoff_ts, cfg.window_30d, "user_id")

    u = u_base.merge(u_7, on="user_id", how="left").merge(u_30, on="user_id", how="left")
    u[f"user_id_cnt_{cfg.window_7d}d"] = u[f"user_id_cnt_{cfg.window_7d}d"].fillna(0.0)
    u[f"user_id_cnt_{cfg.window_30d}d"] = u[f"user_id_cnt_{cfg.window_30d}d"].fillna(0.0)

    u["user_tenure_days"] = (cutoff_ts - u["user_first_ts"]) / SECONDS_IN_DAY
    u["user_recency_days"] = (cutoff_ts - u["user_last_ts"]) / SECONDS_IN_DAY

    # item aggregates
    i_base = history.groupby("item_id").agg(
        item_cnt_total=("weight", "sum"),
        item_first_ts=("ts", "min"),
        item_last_ts=("ts", "max"),
    ).reset_index()

    i_7 = _counts_in_window(history, cutoff_ts, cfg.window_7d, "item_id")
    i_30 = _counts_in_window(history, cutoff_ts, cfg.window_30d, "item_id")

    it = i_base.merge(i_7, on="item_id", how="left").merge(i_30, on="item_id", how="left")
    it[f"item_id_cnt_{cfg.window_7d}d"] = it[f"item_id_cnt_{cfg.window_7d}d"].fillna(0.0)
    it[f"item_id_cnt_{cfg.window_30d}d"] = it[f"item_id_cnt_{cfg.window_30d}d"].fillna(0.0)

    it["item_age_days"] = (cutoff_ts - it["item_first_ts"]) / SECONDS_IN_DAY
    it["item_recency_days"] = (cutoff_ts - it["item_last_ts"]) / SECONDS_IN_DAY

    # rename window columns to clean names
    u = u.rename(columns={
        f"user_id_cnt_{cfg.window_7d}d": "user_cnt_7d",
        f"user_id_cnt_{cfg.window_30d}d": "user_cnt_30d",
    })
    it = it.rename(columns={
        f"item_id_cnt_{cfg.window_7d}d": "item_cnt_7d",
        f"item_id_cnt_{cfg.window_30d}d": "item_cnt_30d",
    })

    return u[["user_id", "user_cnt_total", "user_cnt_7d", "user_cnt_30d", "user_tenure_days", "user_recency_days"]], \
           it[["item_id", "item_cnt_total", "item_cnt_7d", "item_cnt_30d", "item_age_days", "item_recency_days"]]


def build_ranker_dataset(
    history_path: str,
    candidates_path: str,
    holdout_path: str,
    cutoff_ts: int,
    out_path: str,
) -> Path:
    hist = pd.read_parquet(history_path)
    cand = pd.read_parquet(candidates_path)
    hold = pd.read_parquet(holdout_path)[["user_id", "item_id"]].drop_duplicates()
    hold["label"] = 1

    # leakage-safe aggregates from HISTORY ONLY
    ufeat, ifeat = build_aggregates(hist, cutoff_ts, FeatureConfig())

    df = cand.merge(ufeat, on="user_id", how="left").merge(ifeat, on="item_id", how="left")
    df = df.merge(hold, on=["user_id", "item_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)

    # fill any missing numeric features (should be rare)
    for c in ["user_cnt_total","user_cnt_7d","user_cnt_30d","user_tenure_days","user_recency_days",
              "item_cnt_total","item_cnt_7d","item_cnt_30d","item_age_days","item_recency_days"]:
        df[c] = df[c].fillna(0.0).astype(np.float32)

    df["group_id"] = df["user_id"].astype(np.int64)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)
    return outp
