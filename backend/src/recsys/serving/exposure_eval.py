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


def ips_ndcg_at_k(
    recommendations: list[int],
    shown_items:     set[int],
    positive_items:  set[int],
    propensities:    dict[int, float],
    k:               int = 10,
) -> float:
    """
    IPS-corrected NDCG@k.
    Only evaluates on items that were shown (exposure-corrected).
    Weights each hit by 1/P(shown) to correct for popularity bias.

    Naive NDCG:  assumes all non-interacted items are irrelevant
    IPS-NDCG:    weights by inverse propensity of being shown

    Returns 0.0 if no items were shown (cannot evaluate).
    """
    # Filter to shown items only
    shown_recs  = [r for r in recommendations[:k] if r in shown_items]
    shown_rel   = positive_items & shown_items

    if not shown_recs or not shown_rel:
        return 0.0

    dcg  = sum(
        (1.0 / propensities.get(r, 0.1)) / np.log2(i + 2)
        for i, r in enumerate(shown_recs)
        if r in positive_items
    )
    # Ideal: sort shown_rel by propensity-weighted gain
    ideal_gains = sorted(
        [1.0 / propensities.get(r, 0.1) for r in shown_rel],
        reverse=True
    )
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains[:k]))

    return float(dcg / idcg) if idcg > 0 else 0.0


def point_in_time_check(
    feature_timestamp: float,
    interaction_timestamp: float,
    max_lag_seconds: float = 3600.0,
) -> bool:
    """
    Check that features were computed BEFORE the interaction.
    Training-serving skew happens when future information leaks into features.
    Returns True if the feature is temporally valid.
    """
    if feature_timestamp > interaction_timestamp:
        return False   # future leakage
    lag = interaction_timestamp - feature_timestamp
    if lag > max_lag_seconds:
        return False   # features too stale
    return True


def delayed_label_window(
    recommendation_ts: float,
    evaluation_ts:     float,
    window_hours:      float = 24.0,
) -> bool:
    """
    Check if we are within the delayed-label observation window.
    Interactions can arrive up to `window_hours` after recommendation.
    Evaluating too early misses positive labels and understates quality.
    """
    elapsed = (evaluation_ts - recommendation_ts) / 3600.0
    return elapsed >= window_hours


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
