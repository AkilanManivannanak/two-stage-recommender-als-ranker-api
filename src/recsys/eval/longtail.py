# src/recsys/eval/longtail.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd


HistoryLike = Union[str, Path, pd.DataFrame]


def _load_history(history: HistoryLike) -> pd.DataFrame:
    if isinstance(history, (str, Path)):
        return pd.read_parquet(history)
    if isinstance(history, pd.DataFrame):
        return history
    raise TypeError(f"history must be a parquet path or DataFrame, got {type(history)}")


def compute_item_popularity(history: HistoryLike) -> Dict[int, float]:
    """
    Returns item popularity counts from HISTORY ONLY (leakage-safe).
    Popularity is defined as #interactions per item in the provided history window.
    """
    df = _load_history(history)
    if "item_id" not in df.columns:
        raise ValueError("history must have column: item_id")
    # Use interaction counts (not unique users) for a simple, stable definition.
    counts = df.groupby("item_id").size()
    return {int(item_id): float(cnt) for item_id, cnt in counts.items()}


def longtail_items(popularity: Dict[int, float], tail_quantile: float = 0.80) -> Set[int]:
    """
    Items whose popularity is <= quantile threshold are treated as "long-tail".
    Example: tail_quantile=0.80 => bottom 80% by popularity are long-tail.
    """
    if not popularity:
        return set()

    vals = np.array(list(popularity.values()), dtype=np.float64)
    thr = float(np.quantile(vals, tail_quantile))
    return {int(it) for it, c in popularity.items() if float(c) <= thr}


def longtail_at_k(user_rankings: Dict[int, List[int]], tail_set: Set[int], k: int = 10) -> float:
    """
    Fraction of top-k recommendations that are in long-tail (macro-averaged over users).
    """
    if not user_rankings:
        return 0.0

    fracs: List[float] = []
    for _, recs in user_rankings.items():
        topk = recs[:k]
        if not topk:
            continue
        fracs.append(sum(1 for it in topk if it in tail_set) / len(topk))

    return float(np.mean(fracs)) if fracs else 0.0
