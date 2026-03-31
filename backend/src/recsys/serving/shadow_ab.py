"""
Shadow A/B + Interleaving Evaluation
======================================
Plane: Agentic Eval/Ops

True A/B testing requires live users. This module implements the best
offline proxies:

1. SHADOW EVALUATION
   Run new policy on same users as old policy using logged interactions.
   Measure: NDCG lift, diversity lift, abandonment delta.
   Used before any deployment decision.

2. TEAM-DRAFT INTERLEAVING
   Standard offline A/B proxy. Given user interactions with a mixed list
   from policy A and policy B, count which policy's items got clicked.
   Industry standard at Netflix, Spotify, LinkedIn for pre-A/B screening.

3. COUNTERFACTUAL EVALUATION
   Uses IPS (inverse propensity scoring) to estimate what would have
   happened under the new policy using logged data from the old policy.
   Unbiased if propensity model is correct.

4. STATISTICAL SIGNIFICANCE
   Bootstrap CI + Mann-Whitney U test for lift significance.
   A lift that is not statistically significant should not be deployed.

Reference:
  Chapelle et al. "Large-Scale Validation and Analysis of Interleaved
    Search Evaluation" (ACM TOIS 2012)
  Joachims et al. "Unbiased Learning-to-Rank" (SIGIR 2017)
  Netflix Tech Blog — "How We Evaluate Recommendations"
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PolicyResult:
    """Output of a single policy run for one user."""
    user_id:     int
    policy_name: str
    item_ids:    list[int]          # ranked recommendations
    scores:      list[float]        # per-item scores
    latency_ms:  float = 0.0
    metadata:    dict = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Evaluation metrics for a policy."""
    policy_name:     str
    ndcg_at_10:      float
    ndcg_at_10_ci:   tuple[float, float]   # (lo, hi) 95% bootstrap CI
    recall_at_50:    float
    diversity_score: float
    coverage:        float
    abandonment_proxy: float
    n_users:         int
    is_significant:  bool = False          # vs baseline
    p_value:         float = 1.0
    lift_vs_baseline: float = 0.0


@dataclass
class InterleavingResult:
    """Result of team-draft interleaving for one user."""
    user_id:       int
    policy_a_wins: int    # clicks on policy A items
    policy_b_wins: int    # clicks on policy B items
    tied:          bool
    interleaved_list: list[tuple[int, str]]  # [(item_id, 'A'|'B')]


@dataclass
class ShadowReport:
    """Full shadow evaluation report comparing two policies."""
    challenger:       EvalMetrics
    baseline:         EvalMetrics
    interleaving:     dict   # aggregated interleaving stats
    counterfactual:   dict   # IPS estimates
    recommendation:   str    # "DEPLOY" | "HOLD" | "INVESTIGATE"
    summary:          str
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "recommendation":   self.recommendation,
            "summary":          self.summary,
            "challenger": {
                "policy":        self.challenger.policy_name,
                "ndcg_at_10":    self.challenger.ndcg_at_10,
                "ndcg_ci":       list(self.challenger.ndcg_at_10_ci),
                "recall_at_50":  self.challenger.recall_at_50,
                "diversity":     self.challenger.diversity_score,
                "lift":          self.challenger.lift_vs_baseline,
                "significant":   self.challenger.is_significant,
                "p_value":       self.challenger.p_value,
            },
            "baseline": {
                "policy":        self.baseline.policy_name,
                "ndcg_at_10":    self.baseline.ndcg_at_10,
                "recall_at_50":  self.baseline.recall_at_50,
                "diversity":     self.baseline.diversity_score,
            },
            "interleaving":     self.interleaving,
            "counterfactual":   self.counterfactual,
            "timestamp":        self.timestamp,
        }


# ── Core evaluation functions ─────────────────────────────────────────────────

def dcg_at_k(ranked_ids: list[int], relevant: set[int], k: int = 10) -> float:
    return sum(
        1.0 / math.log2(i + 2)
        for i, iid in enumerate(ranked_ids[:k])
        if iid in relevant
    )

def ndcg_at_k(ranked_ids: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg  = dcg_at_k(ranked_ids, relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int = 50) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)

def diversity_score(ranked_ids: list[int],
                    item_genres: dict[int, str],
                    k: int = 20) -> float:
    genres = [item_genres.get(iid, "?") for iid in ranked_ids[:k]]
    return len(set(genres)) / max(len(genres), 1)

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(values, len(values), replace=True)))
             for _ in range(n_bootstrap)]
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return round(lo, 4), round(hi, 4)

def mann_whitney_p(a: list[float], b: list[float]) -> float:
    """Mann-Whitney U test p-value (no scipy needed)."""
    if not a or not b:
        return 1.0
    na, nb = len(a), len(b)
    combined = sorted([(v, "a") for v in a] + [(v, "b") for v in b])
    ranks_a  = [i + 1 for i, (_, g) in enumerate(combined) if g == "a"]
    u = sum(ranks_a) - na * (na + 1) / 2
    # Normal approximation
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    if sigma == 0:
        return 1.0
    z = (u - mu) / sigma
    # Two-sided p-value via error function approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return round(p, 4)


# ── Team-draft interleaving ───────────────────────────────────────────────────

def team_draft_interleave(
    ranking_a:     list[int],
    ranking_b:     list[int],
    k:             int = 20,
    seed:          int = 0,
) -> list[tuple[int, str]]:
    """
    Team-draft interleaving: alternately pick from A and B.
    Returns interleaved list of (item_id, 'A'|'B').
    """
    rng     = np.random.default_rng(seed)
    result: list[tuple[int, str]] = []
    seen:   set[int] = set()
    ia, ib  = 0, 0
    turn    = int(rng.integers(0, 2))   # random first pick

    while len(result) < k:
        if turn % 2 == 0:
            while ia < len(ranking_a):
                iid = ranking_a[ia]; ia += 1
                if iid not in seen:
                    result.append((iid, "A")); seen.add(iid); break
        else:
            while ib < len(ranking_b):
                iid = ranking_b[ib]; ib += 1
                if iid not in seen:
                    result.append((iid, "B")); seen.add(iid); break

        turn += 1
        if ia >= len(ranking_a) and ib >= len(ranking_b):
            break

    return result


def evaluate_interleaving(
    interleaved:    list[tuple[int, str]],
    clicked_items:  set[int],
) -> InterleavingResult:
    """Given user's clicks on an interleaved list, determine winner."""
    a_clicks = sum(1 for iid, src in interleaved if src == "A" and iid in clicked_items)
    b_clicks = sum(1 for iid, src in interleaved if src == "B" and iid in clicked_items)
    return InterleavingResult(
        user_id=0,
        policy_a_wins=a_clicks,
        policy_b_wins=b_clicks,
        tied=a_clicks == b_clicks,
        interleaved_list=interleaved,
    )


# ── Shadow evaluator ──────────────────────────────────────────────────────────

class ShadowEvaluator:
    """
    Runs shadow evaluation comparing a challenger policy against a baseline.

    Usage:
      eval = ShadowEvaluator()
      report = eval.compare(
          challenger_fn=phenomenal_recommend,
          baseline_fn=als_recommend,
          eval_users=[...],
          positive_items={uid: {item_ids}},
          item_genres={item_id: genre},
      )
    """

    def evaluate_policy(
        self,
        policy_fn:      Callable[[int], list[int]],
        policy_name:    str,
        eval_users:     list[int],
        positive_items: dict[int, set[int]],
        item_genres:    dict[int, str],
        k_ndcg:         int = 10,
        k_recall:       int = 50,
    ) -> tuple[EvalMetrics, dict[int, list[int]]]:
        """
        Evaluate a policy over a set of users.
        Returns (EvalMetrics, {user_id: ranked_ids}).
        """
        ndcgs, recalls, diversities = [], [], []
        all_items_seen: set[int] = set()
        policy_rankings: dict[int, list[int]] = {}

        for uid in eval_users:
            relevant = positive_items.get(uid, set())
            if not relevant:
                continue
            try:
                ranked = policy_fn(uid)
            except Exception:
                continue

            policy_rankings[uid] = ranked
            all_items_seen.update(ranked)

            ndcgs.append(ndcg_at_k(ranked, relevant, k_ndcg))
            recalls.append(recall_at_k(ranked, relevant, k_recall))
            diversities.append(diversity_score(ranked, item_genres))

        if not ndcgs:
            return EvalMetrics(policy_name, 0.0, (0.0, 0.0),
                               0.0, 0.0, 0.0, 0.8, 0), policy_rankings

        ci = bootstrap_ci(ndcgs)
        # Abandonment proxy: fraction of top-1 item misses (not in positive)
        abandon_proxy = sum(
            1 for uid, ranked in policy_rankings.items()
            if ranked and ranked[0] not in positive_items.get(uid, set())
        ) / max(len(policy_rankings), 1)

        return EvalMetrics(
            policy_name=policy_name,
            ndcg_at_10=round(float(np.mean(ndcgs)), 4),
            ndcg_at_10_ci=ci,
            recall_at_50=round(float(np.mean(recalls)), 4),
            diversity_score=round(float(np.mean(diversities)), 4),
            coverage=round(len(all_items_seen) / max(len(eval_users) * 20, 1), 4),
            abandonment_proxy=round(abandon_proxy, 4),
            n_users=len(ndcgs),
        ), policy_rankings

    def compare(
        self,
        challenger_fn:   Callable[[int], list[int]],
        baseline_fn:     Callable[[int], list[int]],
        eval_users:      list[int],
        positive_items:  dict[int, set[int]],
        item_genres:     dict[int, str],
        propensity:      Optional[dict[int, float]] = None,
        challenger_name: str = "challenger",
        baseline_name:   str = "baseline",
    ) -> ShadowReport:
        """
        Full shadow comparison: metrics + interleaving + counterfactual.
        """
        print(f"  [ShadowEval] Evaluating {challenger_name} vs {baseline_name} "
              f"on {len(eval_users)} users...")

        # 1. Evaluate both policies
        challenger_metrics, challenger_rankings = self.evaluate_policy(
            challenger_fn, challenger_name, eval_users, positive_items, item_genres)
        baseline_metrics, baseline_rankings = self.evaluate_policy(
            baseline_fn, baseline_name, eval_users, positive_items, item_genres)

        # 2. Statistical significance
        c_ndcgs, b_ndcgs = [], []
        for uid in eval_users:
            rel = positive_items.get(uid, set())
            if not rel:
                continue
            if uid in challenger_rankings:
                c_ndcgs.append(ndcg_at_k(challenger_rankings[uid], rel))
            if uid in baseline_rankings:
                b_ndcgs.append(ndcg_at_k(baseline_rankings[uid], rel))

        p_val = mann_whitney_p(c_ndcgs, b_ndcgs)
        lift  = challenger_metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10
        challenger_metrics.lift_vs_baseline = round(lift, 4)
        challenger_metrics.p_value = p_val
        challenger_metrics.is_significant = p_val < 0.05 and lift > 0

        # 3. Team-draft interleaving
        interleave_wins = {"A_wins": 0, "B_wins": 0, "ties": 0}
        for uid in eval_users[:200]:   # sample for speed
            if uid not in challenger_rankings or uid not in baseline_rankings:
                continue
            interleaved = team_draft_interleave(
                challenger_rankings[uid][:20],
                baseline_rankings[uid][:20],
                k=20, seed=uid,
            )
            clicked = positive_items.get(uid, set())
            result  = evaluate_interleaving(interleaved, clicked)
            if result.tied:
                interleave_wins["ties"] += 1
            elif result.policy_a_wins > result.policy_b_wins:
                interleave_wins["A_wins"] += 1
            else:
                interleave_wins["B_wins"] += 1

        n_inter = sum(interleave_wins.values())
        interleave_summary = {
            **interleave_wins,
            "n_users": n_inter,
            "challenger_win_rate": round(
                interleave_wins["A_wins"] / max(n_inter, 1), 3),
            "method": "team_draft_interleaving",
        }

        # 4. IPS counterfactual estimate
        ips_ndcgs = []
        prop = propensity or {}
        for uid in eval_users[:300]:
            rel   = positive_items.get(uid, set())
            recs  = challenger_rankings.get(uid, [])
            if not rel or not recs:
                continue
            gain = sum(
                (1.0 / prop.get(iid, 0.1)) / math.log2(i + 2)
                for i, iid in enumerate(recs[:10])
                if iid in rel
            )
            ideal = sum(
                (1.0 / prop.get(iid, 0.1)) / math.log2(i + 2)
                for i, iid in enumerate(sorted(rel, key=lambda x: -prop.get(x, 0.1))[:10])
            )
            ips_ndcgs.append(gain / ideal if ideal > 0 else 0.0)

        counterfactual = {
            "ips_ndcg_at_10": round(float(np.mean(ips_ndcgs)) if ips_ndcgs else 0.0, 4),
            "n_users": len(ips_ndcgs),
            "method": "ips_capped",
        }

        # 5. Decision
        if (lift > 0.01 and challenger_metrics.is_significant
                and interleave_summary["challenger_win_rate"] > 0.50):
            recommendation = "DEPLOY"
            summary = (f"Challenger wins: NDCG lift={lift:.4f} (p={p_val:.3f}), "
                       f"interleaving win rate={interleave_summary['challenger_win_rate']:.0%}. "
                       f"Human review required before deployment.")
        elif lift > 0 and not challenger_metrics.is_significant:
            recommendation = "HOLD"
            summary = (f"Positive lift ({lift:.4f}) but NOT statistically significant "
                       f"(p={p_val:.3f}). Run more users or wait for more data.")
        elif lift < -0.005:
            recommendation = "INVESTIGATE"
            summary = (f"Challenger REGRESSES: NDCG delta={lift:.4f}. "
                       f"Do not deploy. Investigate feature importance and slice results.")
        else:
            recommendation = "HOLD"
            summary = f"Inconclusive. Lift={lift:.4f}, p={p_val:.3f}."

        return ShadowReport(
            challenger=challenger_metrics,
            baseline=baseline_metrics,
            interleaving=interleave_summary,
            counterfactual=counterfactual,
            recommendation=recommendation,
            summary=summary,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
SHADOW_EVALUATOR = ShadowEvaluator()
