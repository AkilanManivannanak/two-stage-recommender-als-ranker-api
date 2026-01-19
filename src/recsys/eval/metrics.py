from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _dcg(rels: np.ndarray) -> float:
    # rels are binary relevance in ranked order
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(ranked_items: Sequence[int], true_items: set[int], k: int) -> float:
    topk = ranked_items[:k]
    rels = np.array([1.0 if it in true_items else 0.0 for it in topk], dtype=np.float64)
    dcg = _dcg(rels)
    ideal_rels = np.ones(min(k, len(true_items)), dtype=np.float64)
    idcg = _dcg(ideal_rels)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_items: Sequence[int], true_items: set[int], k: int) -> float:
    for i, it in enumerate(ranked_items[:k], start=1):
        if it in true_items:
            return 1.0 / float(i)
    return 0.0


def recall_at_k(ranked_items: Sequence[int], true_items: set[int], k: int) -> float:
    if not true_items:
        return 0.0
    hit = 0
    for it in ranked_items[:k]:
        if it in true_items:
            hit += 1
    return float(hit) / float(len(true_items))


@dataclass(frozen=True)
class EvalResult:
    ndcg10: float
    mrr10: float
    recall10: float
    recall50: float
    coverage10: float
    n_users: int


def evaluate_rankings(
    user_rankings: Dict[int, List[int]],
    user_truth: Dict[int, set[int]],
    k_cov: int = 10,
) -> EvalResult:
    users = [u for u in user_rankings.keys() if u in user_truth and len(user_truth[u]) > 0]
    if not users:
        return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    ndcg10 = []
    mrr10 = []
    r10 = []
    r50 = []

    all_topk = set()

    for u in users:
        ranked = user_rankings[u]
        truth = user_truth[u]
        ndcg10.append(ndcg_at_k(ranked, truth, 10))
        mrr10.append(mrr_at_k(ranked, truth, 10))
        r10.append(recall_at_k(ranked, truth, 10))
        r50.append(recall_at_k(ranked, truth, 50))
        all_topk.update(ranked[:k_cov])

    return EvalResult(
        ndcg10=float(np.mean(ndcg10)),
        mrr10=float(np.mean(mrr10)),
        recall10=float(np.mean(r10)),
        recall50=float(np.mean(r50)),
        coverage10=float(len(all_topk)),
        n_users=len(users),
    )
