from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from recsys.eval.metrics import ndcg_at_k, mrr_at_k, recall_at_k


@dataclass(frozen=True)
class MetricSample:
    ndcg10: float
    mrr10: float
    recall10: float
    recall50: float


def per_user_metrics(
    user_rankings: Dict[int, List[int]],
    user_truth: Dict[int, set[int]],
) -> Dict[int, MetricSample]:
    out: Dict[int, MetricSample] = {}
    for u, truth in user_truth.items():
        if u not in user_rankings or len(truth) == 0:
            continue
        ranked = user_rankings[u]
        out[u] = MetricSample(
            ndcg10=ndcg_at_k(ranked, truth, 10),
            mrr10=mrr_at_k(ranked, truth, 10),
            recall10=recall_at_k(ranked, truth, 10),
            recall50=recall_at_k(ranked, truth, 50),
        )
    return out


def bootstrap_ci(
    per_user: Dict[int, MetricSample],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    User-level bootstrap. Resample users with replacement.
    Returns mean and 95% CI for each metric.
    """
    users = np.array(list(per_user.keys()), dtype=np.int64)
    rng = np.random.default_rng(seed)

    ndcg = np.array([per_user[u].ndcg10 for u in users], dtype=np.float64)
    mrr = np.array([per_user[u].mrr10 for u in users], dtype=np.float64)
    r10 = np.array([per_user[u].recall10 for u in users], dtype=np.float64)
    r50 = np.array([per_user[u].recall50 for u in users], dtype=np.float64)

    def boot(arr: np.ndarray) -> Tuple[float, float, float]:
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        samples = arr[idx].mean(axis=1)
        mean = float(arr.mean())
        lo = float(np.quantile(samples, 0.025))
        hi = float(np.quantile(samples, 0.975))
        return mean, lo, hi

    out = {}
    for name, arr in [("ndcg10", ndcg), ("mrr10", mrr), ("recall10", r10), ("recall50", r50)]:
        mean, lo, hi = boot(arr)
        out[name] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}
    out["n_users"] = {"mean": float(len(users)), "ci95_lo": float(len(users)), "ci95_hi": float(len(users))}
    return out


def bootstrap_delta_ci(
    per_user_a: Dict[int, MetricSample],
    per_user_b: Dict[int, MetricSample],
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap CI for delta (B - A) using paired users intersection.
    """
    users = sorted(list(set(per_user_a.keys()) & set(per_user_b.keys())))
    rng = np.random.default_rng(seed)
    if not users:
        return {"delta_mean": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n_users": 0}

    a = np.array([getattr(per_user_a[u], metric) for u in users], dtype=np.float64)
    b = np.array([getattr(per_user_b[u], metric) for u in users], dtype=np.float64)
    d = b - a

    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    samples = d[idx].mean(axis=1)

    return {
        "delta_mean": float(d.mean()),
        "ci95_lo": float(np.quantile(samples, 0.025)),
        "ci95_hi": float(np.quantile(samples, 0.975)),
        "n_users": int(len(d)),
    }
