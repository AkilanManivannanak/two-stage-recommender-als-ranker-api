"""
Off-Policy Evaluation (OPE)  —  Upgrade 4: Real Evaluation
============================================================
Addresses the core problem: NDCG@10 = 0.0 because evaluation was done
on synthetic data where the model never saw real patterns.

WHAT THIS ADDS:
  1. Inverse Propensity Scoring (IPS-NDCG):
     Standard NDCG treats all items equally. IPS-NDCG corrects for
     exposure bias: items that were rarely shown get up-weighted.
     This gives unbiased estimates of true recommendation quality
     even when training data is not uniformly distributed.

  2. Doubly-Robust (DR) Estimator:
     Combines IPS with a learned reward model for lower variance.
     DR is unbiased if EITHER the propensity model OR the reward
     model is correct — more robust than either alone.

  3. Counterfactual Policy Evaluation:
     Estimates what NDCG would have been under the new policy
     using logged data from the old policy. No A/B test required.

  4. Slice-level Regression Detection:
     Checks NDCG per slice (genre, user segment, cold/warm).
     A new model can improve overall NDCG but regress on cold users.
     This flags those regressions before deployment.

  5. Regret Monitoring:
     Tracks cumulative regret (optimal - actual reward) over time.
     A good system has sublinear regret growth.

  6. Interleaving Metric:
     For A/B proxy: given user interactions with a mixed list from
     model A and model B, which model won more clicks?

Reference:
  Joachims et al. — Unbiased Learning-to-Rank with Biased Feedback
  Dudik et al.    — Doubly Robust Policy Evaluation and Optimisation
  Netflix Tech Blog — "How We Evaluate Recommendations"
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── 1. IPS-NDCG ──────────────────────────────────────────────────────────

def dcg_at_k(scores: list[float], k: int) -> float:
    """Standard DCG@k (log2 denominator)."""
    return sum(
        s / math.log2(i + 2)
        for i, s in enumerate(scores[:k])
    )

def ips_ndcg_at_k(
    recommendations:  list[int],
    relevant_items:   set[int],
    propensity:       dict[int, float],
    k:                int = 10,
    clip:             float = 10.0,
) -> float:
    """
    IPS-corrected NDCG@k.

    For each item in the recommendation list that is relevant,
    weight its contribution by 1/propensity(item) to correct for
    the fact that popular items were over-shown in training.

    clip: maximum IPS weight (prevents high-variance estimates
    from rare items dominating the metric).
    """
    if not relevant_items:
        return 0.0

    gains = []
    for iid in recommendations[:k]:
        if iid in relevant_items:
            p     = propensity.get(iid, 0.1)
            w     = min(1.0 / max(p, 1e-6), clip)
            gains.append(w)
        else:
            gains.append(0.0)

    dcg = dcg_at_k(gains, k)

    # Ideal: top-k relevant items, IPS-weighted
    ideal_gains = sorted(
        [min(1.0 / max(propensity.get(i, 0.1), 1e-6), clip)
         for i in relevant_items],
        reverse=True,
    )
    idcg = dcg_at_k(ideal_gains, k)

    return dcg / idcg if idcg > 0 else 0.0


# ── 2. Doubly-Robust Estimator ───────────────────────────────────────────

def doubly_robust_reward(
    logged_action:  int,
    new_action:     int,
    observed_reward: float,
    propensity:     float,
    predicted_reward_logged: float,
    predicted_reward_new:    float,
    clip:           float = 5.0,
) -> float:
    """
    DR estimator for one (user, item) pair.

    Combines IPS correction with a direct reward model:
      DR = reward_model(new) +
           (1/propensity) * I(logged==new) * (observed - reward_model(logged))

    Unbiased if either propensity model or reward model is correct.
    """
    ips_weight = min(1.0 / max(propensity, 1e-4), clip)
    indicator  = float(logged_action == new_action)
    dr = (predicted_reward_new
          + ips_weight * indicator * (observed_reward - predicted_reward_logged))
    return dr


# ── 3. Counterfactual Policy Evaluator ───────────────────────────────────

@dataclass
class LoggedInteraction:
    user_id:    int
    item_id:    int
    position:   int         # position shown at (0-indexed)
    was_shown:  bool
    was_clicked: bool
    reward:     float       # 0.0 = skip, 0.5 = like, 1.0 = play
    propensity: float       # P(item shown | context) under logging policy


class CounterfactualEvaluator:
    """
    Evaluate a new ranking policy using logged data from an old policy.

    Usage:
      eval = CounterfactualEvaluator()
      eval.log(interaction)
      metrics = eval.evaluate_policy(new_policy_fn, k=10)
    """

    def __init__(self):
        self._logs: list[LoggedInteraction] = []

    def log(self, interaction: LoggedInteraction) -> None:
        self._logs.append(interaction)

    def evaluate_policy(
        self,
        policy_rankings: dict[int, list[int]],  # user_id -> ranked item_ids
        propensity_map:  dict[int, float],       # item_id -> propensity
        k:               int = 10,
        method:          str = "ips",            # "ips" | "dr"
        reward_model:    Optional[callable] = None,
    ) -> dict:
        """
        Estimate reward for a new policy from logs collected under an old one.

        method="ips" — inverse propensity scoring. Unbiased if the propensities
        are correct, but high variance when some were small.

        method="dr"  — doubly robust. Adds a reward model as a control variate:

            DR = rhat(new) + (1/p) * 1[logged == new] * (observed - rhat(logged))

        Unbiased if EITHER the propensity model OR the reward model is right,
        and lower variance than IPS whenever the reward model carries signal.

        Previously this method accepted `method` and then ignored it: the body
        only ever computed IPS and echoed the argument back in the result, while
        doubly_robust_reward() sat in this module with no callers anywhere in
        the repository. The README advertised doubly-robust evaluation on the
        strength of that dead function.

        reward_model: callable (user_id, item_id) -> predicted reward in [0, 1].
        Required for method="dr". A DR estimate cannot be formed without one, so
        this raises rather than silently degrading to IPS.
        """
        if method not in ("ips", "dr"):
            raise ValueError(f"method must be 'ips' or 'dr', got {method!r}")
        if method == "dr" and reward_model is None:
            raise ValueError(
                "method='dr' requires reward_model; refusing to silently "
                "compute IPS and label the result doubly-robust")

        per_user_scores: dict[int, list[float]] = defaultdict(list)
        n_matched = 0

        for log in self._logs:
            uid     = log.user_id
            ranking = policy_rankings.get(uid, [])
            if not ranking:
                continue

            in_slate = log.item_id in ranking[:k]
            propensity = max(log.propensity, 1e-4)
            if propensity_map:
                propensity = max(propensity_map.get(log.item_id, propensity), 1e-4)

            if method == "ips":
                # Only logged items the new policy would also have shown
                # contribute; everything else has an indicator of zero.
                if in_slate and log.was_clicked:
                    pos   = ranking.index(log.item_id)
                    gain  = log.reward / math.log2(pos + 2)
                    ips_w = min(1.0 / propensity, 10.0)
                    per_user_scores[uid].append(gain * ips_w)
                    n_matched += 1
                elif in_slate:
                    per_user_scores[uid].append(0.0)
            else:
                # Doubly robust. The direct-method term applies to every logged
                # record, so records the new policy would NOT have shown still
                # contribute their model-predicted reward — which is the whole
                # point of the control variate.
                new_item = ranking[0] if ranking else log.item_id
                dr = doubly_robust_reward(
                    logged_action=log.item_id,
                    new_action=(log.item_id if in_slate else new_item),
                    observed_reward=log.reward,
                    propensity=propensity,
                    predicted_reward_logged=float(reward_model(uid, log.item_id)),
                    predicted_reward_new=float(
                        reward_model(uid, log.item_id if in_slate else new_item)),
                )
                if in_slate:
                    pos = ranking.index(log.item_id)
                    dr /= math.log2(pos + 2)     # position discount
                    n_matched += 1
                per_user_scores[uid].append(dr)

        if not per_user_scores:
            return {"estimate": 0.0, "n_users": 0, "k": k, "method": method,
                    "n_matched": 0,
                    "note": "no logged record overlapped the new policy's slates"}

        per_user = {u: float(np.mean(v)) for u, v in per_user_scores.items()}
        vals = np.array(list(per_user.values()))
        return {
            "estimate":       float(vals.mean()),
            "median":         float(np.median(vals)),
            "std":            float(vals.std()),
            # A standard error makes the variance difference between IPS and DR
            # visible, which is the reason to prefer DR in the first place.
            "stderr":         float(vals.std() / max(np.sqrt(len(vals)), 1)),
            "n_users":        len(per_user),
            "n_logs":         len(self._logs),
            "n_matched":      n_matched,
            "k":              k,
            "method":         method,
        }


# ── 4. Slice-level Regression Detection ──────────────────────────────────

@dataclass
class SliceResult:
    slice_name:   str
    ndcg:         float
    n_users:      int
    regressed:    bool = False   # True if below threshold
    delta_vs_baseline: Optional[float] = None


def slice_ndcg(
    eval_pairs:        list[dict],
    recommend_fn,
    propensity:        dict[int, float],
    slice_key:         str = "segment",     # field in eval_pairs
    k:                 int = 10,
    regression_thresh: float = -0.02,       # flag if > 2% worse than baseline
    baseline_ndcg:     Optional[dict[str, float]] = None,
) -> list[SliceResult]:
    """
    Compute IPS-NDCG@k per slice and flag regressions.

    eval_pairs: [{user_id, positive_items, <slice_key>}]
    recommend_fn: user_id → list[item_id]
    baseline_ndcg: previous model's NDCG per slice (for regression check)
    """
    slice_scores: dict[str, list[float]] = defaultdict(list)

    for pair in eval_pairs:
        uid       = pair["user_id"]
        positives = set(pair["positive_items"])
        sl        = str(pair.get(slice_key, "all"))
        recs      = recommend_fn(uid)[:k]
        score     = ips_ndcg_at_k(recs, positives, propensity, k)
        slice_scores[sl].append(score)

    results = []
    for sl, scores in slice_scores.items():
        ndcg = float(np.mean(scores))
        baseline = (baseline_ndcg or {}).get(sl)
        delta    = (ndcg - baseline) if baseline is not None else None
        regressed = delta is not None and delta < regression_thresh
        results.append(SliceResult(
            slice_name=sl,
            ndcg=ndcg,
            n_users=len(scores),
            regressed=regressed,
            delta_vs_baseline=delta,
        ))

    return sorted(results, key=lambda r: r.ndcg)


# ── 5. Regret Monitor ────────────────────────────────────────────────────

class RegretMonitor:
    """
    Track cumulative regret over time.
    Regret = optimal_reward - actual_reward.
    A good bandit/recommender has sublinear cumulative regret.
    """

    def __init__(self):
        self._regrets:    list[float] = []
        self._cumulative: float       = 0.0
        self._steps:      int         = 0

    def record(self, optimal_reward: float, actual_reward: float) -> None:
        r = float(optimal_reward - actual_reward)
        self._regrets.append(r)
        self._cumulative += r
        self._steps += 1

    def stats(self) -> dict:
        if not self._regrets:
            return {"cumulative_regret": 0.0, "mean_regret": 0.0, "steps": 0}
        recent = self._regrets[-100:]
        return {
            "cumulative_regret": self._cumulative,
            "mean_regret":       float(np.mean(self._regrets)),
            "recent_mean_regret": float(np.mean(recent)),
            "steps":             self._steps,
            "is_sublinear":      (self._cumulative / max(self._steps, 1))
                                  < float(np.mean(self._regrets[:max(10, self._steps//10)]) + 0.01),
        }


# ── 6. Interleaving (proxy for online A/B) ───────────────────────────────

def interleaved_comparison(
    ranking_a: list[int],
    ranking_b: list[int],
    clicked_items: set[int],
    k: int = 20,
) -> dict:
    """
    Team-draft interleaving proxy.
    Alternately picks from A and B to build an interleaved list.
    Winner = whichever model's items got more clicks.

    This is a cheap proxy for proper online interleaving but captures
    the directional signal of which policy is better for this user.
    """
    interleaved = []
    source      = {}
    ai, bi, turn = 0, 0, 0
    while len(interleaved) < k:
        if turn % 2 == 0:
            while ai < len(ranking_a):
                item = ranking_a[ai]; ai += 1
                if item not in source:
                    interleaved.append(item); source[item] = "A"; break
        else:
            while bi < len(ranking_b):
                item = ranking_b[bi]; bi += 1
                if item not in source:
                    interleaved.append(item); source[item] = "B"; break
        turn += 1
        if ai >= len(ranking_a) and bi >= len(ranking_b):
            break

    a_clicks = sum(1 for i in clicked_items if source.get(i) == "A")
    b_clicks = sum(1 for i in clicked_items if source.get(i) == "B")
    winner   = "A" if a_clicks > b_clicks else ("B" if b_clicks > a_clicks else "tie")

    return {
        "winner":       winner,
        "a_clicks":     a_clicks,
        "b_clicks":     b_clicks,
        "interleaved_k": len(interleaved),
    }


# ── 7. Abandonment signal ────────────────────────────────────────────────

def compute_abandonment_rate(
    impressions: list[dict],
    interactions: list[dict],
    window_s: float = 300.0,
) -> dict:
    """
    Abandonment = impression shown but no interaction within window.
    High abandonment on a row indicates poor recommendation quality.
    """
    shown_items = {(d["user_id"], d["item_id"]) for d in impressions}
    clicked     = {(d["user_id"], d["item_id"]) for d in interactions}

    abandoned = shown_items - clicked
    rate = len(abandoned) / max(len(shown_items), 1)

    return {
        "abandonment_rate": rate,
        "n_shown":          len(shown_items),
        "n_clicked":        len(clicked),
        "n_abandoned":      len(abandoned),
        "quality_signal":   "poor" if rate > 0.85 else "ok" if rate > 0.70 else "good",
    }
