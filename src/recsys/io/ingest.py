from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IngestConfig:
    min_rating: float = 4.0
    keep_latest_per_user_item: bool = True
    output_name: str = "interactions.parquet"
    stats_name: str = "stats.json"


def _percentiles(x: np.ndarray, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def ingest_movielens_25m(raw_dir: str | Path, out_dir: str | Path, cfg: IngestConfig) -> Tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = raw_dir / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found at: {ratings_path}")

    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    df = pd.read_csv(ratings_path, dtype=dtypes)

    df = df[df["rating"] >= cfg.min_rating].copy()
    df.rename(columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"}, inplace=True)

    if cfg.keep_latest_per_user_item:
        df.sort_values(["user_id", "item_id", "ts"], inplace=True)
        df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    df["weight"] = 1.0
    df = df[["user_id", "item_id", "ts", "weight"]].sort_values(["user_id", "ts"])

    out_interactions = out_dir / cfg.output_name
    df.to_parquet(out_interactions, index=False)

    n_events = int(len(df))
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())

    events_per_user = df.groupby("user_id")["item_id"].count().to_numpy()
    events_per_item = df.groupby("item_id")["user_id"].count().to_numpy()

    denom = float(n_users) * float(n_items)
    density = float(n_events) / denom if denom > 0 else float("nan")
    sparsity = 1.0 - density if np.isfinite(density) else float("nan")

    ts_min = int(df["ts"].min()) if n_events > 0 else None
    ts_max = int(df["ts"].max()) if n_events > 0 else None

    stats = {
        "dataset": "movielens-25m",
        "min_rating": cfg.min_rating,
        "keep_latest_per_user_item": cfg.keep_latest_per_user_item,
        "n_events": n_events,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "sparsity": sparsity,
        "ts_min_epoch_sec": ts_min,
        "ts_max_epoch_sec": ts_max,
        "events_per_user": _percentiles(events_per_user),
        "events_per_item": _percentiles(events_per_item),
    }

    out_stats = out_dir / cfg.stats_name
    out_stats.write_text(json.dumps(stats, indent=2))

    return out_interactions, out_stats
