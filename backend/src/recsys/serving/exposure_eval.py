"""
Exposure-Aware Evaluation  —  Fixes Naive NDCG
================================================
Problem with naive NDCG/Recall:
  Standard offline evaluation pretends items not interacted with
  are "not relevant." In reality they may simply not have been
  shown (exposure bias). This makes popular items look better
  than they are, and rare items look worse.

What this module adds:
  1. Impression logging — track what was shown vs what was clicked
  2. Exposure-corrected relevance — only evaluate on items that WERE shown
  3. IPS-corrected NDCG — weight by inverse propensity of being shown
  4. Point-in-time correctness — features must be from BEFORE the interaction
  5. Delayed-label handling — interactions logged up to 24h after recommendation

References:
  - Schnabel et al. "Recommendations as Treatments" (IPS for RecSys)
  - Saito "Unbiased Recommender Learning from Missing-Not-At-Random" (IPS-NDCG)
  - Netflix observability and title-launch monitoring
"""
from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ImpressionLog:
    user_id:     int
    item_ids:    list[int]   # items SHOWN (not just clicked)
    timestamp:   float = field(default_factory=time.time)
    model_version: str = ""
    row_name:    str  = ""
    propensities: list[float] = field(default_factory=list)  # P(shown | user, item)


@dataclass
class InteractionLog:
    user_id:  int
    item_id:  int
    event:    str    # play, like, dislike, abandon
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0
    label:    int   = 0   # 1 = positive engagement


class ImpressionStore:
    """
    Tracks what was shown to each user.
    Enables exposure-corrected evaluation and IPS-corrected NDCG.
    In production: backed by a time-series store (Kafka + Cassandra).
    """
    def __init__(self):
        self._impressions: dict[int, list[ImpressionLog]] = defaultdict(list)
        self._interactions: dict[int, list[InteractionLog]] = defaultdict(list)

    def log_impression(self, log: ImpressionLog):
        self._impressions[log.user_id].append(log)

    def log_interaction(self, log: InteractionLog):
        self._interactions[log.user_id].append(log)

    def get_shown_items(self, user_id: int,
                        since_ts: float | None = None) -> set[int]:
        """Items actually shown — exposure-correct the evaluation set."""
        shown = set()
        for imp in self._impressions.get(user_id, []):
            if since_ts is None or imp.timestamp >= since_ts:
                shown.update(imp.item_ids)
        return shown

    def get_positive_interactions(self, user_id: int,
                                   since_ts: float | None = None) -> set[int]:
        """Items user positively engaged with."""
        pos = set()
        for inter in self._interactions.get(user_id, []):
            if since_ts and inter.timestamp < since_ts:
                continue
            if inter.label == 1 or inter.event in ("play","like"):
                pos.add(inter.item_id)
        return pos


def ips_ndcg_at_k(*args, **kwargs):
    """
    Delegates to ope_eval.ips_ndcg_at_k.

    This module used to carry a second, independently-written implementation of
    IPS-NDCG. Two copies of an estimator is two chances to be subtly wrong and
    no way to tell which number a given caller produced, so this is now a thin
    forwarder to the one in ope_eval.
    """
    from recsys.serving.ope_eval import ips_ndcg_at_k as _canonical
    return _canonical(*args, **kwargs)


def slice_ndcg(
    recs_by_user:    dict[int, list[int]],
    positives_by_user: dict[int, set[int]],
    user_metadata:   dict[int, dict],
    slice_key:       str = "primary_genre",
    k:               int = 10,
) -> dict[str, float]:
    """
    Compute NDCG@k per slice (genre, cohort, tenure bucket, etc.).
    Enables slice-level regression detection — e.g. new model hurts
    Horror fans while improving overall NDCG.
    """
    slice_dcg:  dict[str, list[float]] = defaultdict(list)

    for uid, recs in recs_by_user.items():
        rel   = positives_by_user.get(uid, set())
        if not rel: continue
        meta  = user_metadata.get(uid, {})
        slice_val = str(meta.get(slice_key, "unknown"))
        dcg = sum(1/np.log2(i+2) for i,r in enumerate(recs[:k]) if r in rel)
        idcg= sum(1/np.log2(i+2) for i in range(min(len(rel),k)))
        ndcg = dcg/idcg if idcg > 0 else 0.0
        slice_dcg[slice_val].append(ndcg)

    return {k: round(float(np.mean(v)), 4)
            for k, v in slice_dcg.items() if v}


# Singleton
IMPRESSION_STORE = ImpressionStore()
