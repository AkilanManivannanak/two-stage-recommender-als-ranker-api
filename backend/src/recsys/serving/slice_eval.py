"""
slice_eval.py — CineWave
=========================
Slice-level NDCG evaluation and long-term retention tracking.

WHY SLICE EVALUATION MATTERS
------------------------------
A single global NDCG@10 = 0.3847 hides cohort-level failures. The model
could score well overall while badly underserving:
  - New users (cold start → low NDCG due to sparse history)
  - Niche genre fans (Documentary lovers in an action-heavy training set)
  - Inactive users (long-tail of low-activity users with sparse signals)
  - Mobile users (different interaction patterns vs desktop)

Slice evaluation surfaces these blindspots. Netflix engineers will
immediately ask: "What's NDCG for new vs returning users?" Having this
answer separates you from candidates who only report global metrics.

SLICES COMPUTED
----------------
1. Genre slice:        NDCG grouped by item's primary_genre
2. Activity decile:    Users split into 10 buckets by interaction count
3. User age bucket:    New (<7 days), recent (<30 days), established (30+ days)
4. Device type:        mobile / desktop / tablet / tv (from User-Agent)

WHY LONG-TERM RETENTION
-------------------------
Short-term metrics (CTR, add-to-list) can conflict with long-term retention.
A clickbait recommendation might get a click but the user abandons after
2 minutes, reduces trust, and doesn't return for 2 weeks.

We track 30-day cohort retention: did this user, who received a recommendation
on day 0, return and play something on any day within days 1-30?

7-day return rate is already in CineWave (A/B experiment 3: 61.2% → 68.4%).
30-day rate gives a longer-horizon view of whether we are building or
destroying long-term engagement.

USAGE
-----
    from recsys.serving.slice_eval import SliceEvaluator, RetentionTracker

    # Slice evaluation (call after each DuckDB IPS-NDCG run)
    evaluator = SliceEvaluator()
    report = evaluator.run_slice_eval(impressions_df, interactions_df, catalog_df)

    # Retention tracking (call at serving time)
    RETENTION.record_recommendation(user_id, item_ids, timestamp)

    # Retention reporting (call daily from Airflow)
    cohort_report = RETENTION.compute_30d_retention(cohort_date="2024-01-01")
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cinewave.slice_eval")

# ── IPS-NDCG helper (mirrors DuckDB computation in Python for slices) ─────────

def _ndcg_at_k(
    ranked_item_ids: List[int],
    relevant_item_ids: set,
    k: int = 10,
) -> float:
    """
    Standard NDCG@K (without IPS correction — applied at the slice level
    after propensity weighting in the caller).
    """
    dcg = 0.0
    for i, iid in enumerate(ranked_item_ids[:k]):
        if iid in relevant_item_ids:
            dcg += 1.0 / np.log2(i + 2)
    ideal_n = min(len(relevant_item_ids), k)
    idcg    = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def _ips_weight(position: int, propensity_by_position: Dict[int, float]) -> float:
    """
    IPS weight for an item shown at `position`.
    Weight = 1 / P(shown at position). Higher positions get lower weight.
    """
    prop = propensity_by_position.get(position, 0.01)
    return 1.0 / max(prop, 0.001)


# ── Slice definitions ─────────────────────────────────────────────────────────

ACTIVITY_DECILE_LABELS = [f"decile_{i}" for i in range(1, 11)]
USER_AGE_BUCKETS = {"new": (0, 7), "recent": (7, 30), "established": (30, 9999)}
GENRE_LIST = [
    "action", "comedy", "drama", "horror", "sci-fi", "romance",
    "thriller", "documentary", "animation", "crime", "adventure",
    "fantasy", "family", "war", "western", "music", "history", "mystery",
]


# ── Slice evaluator ───────────────────────────────────────────────────────────

class SliceEvaluator:
    """
    Computes IPS-NDCG@10 for each slice and writes a JSON report.

    Input data format (as dict lists, matching Parquet schema):
    impressions_df: list of {
        user_id, item_id, position, model_score, timestamp, policy_version
    }
    interactions_df: list of {
        user_id, item_id, event (play/add_to_list/skip), timestamp
    }
    catalog_df: list of {
        item_id, primary_genre, year, popularity_score
    }
    user_df: list of {
        user_id, n_interactions, first_interaction_date, device_type
    }
    """

    def __init__(self, output_path: str = "/app/artifacts/slice_eval/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    # ── Data preparation helpers ────────────────────────────────────────────

    def _build_propensity(self, impressions: List[Dict]) -> Dict[int, float]:
        """
        Estimate P(shown at position k) from the impression log.
        Positions near the top have higher propensity.
        """
        position_counts: Dict[int, int] = defaultdict(int)
        for imp in impressions:
            position_counts[imp["position"]] += 1
        total = sum(position_counts.values())
        return {pos: count / total for pos, count in position_counts.items()}

    def _build_relevant_items(
        self,
        interactions: List[Dict],
        positive_events: set = {"play", "add_to_list"},
    ) -> Dict[int, set]:
        """Map user_id → set of item_ids the user positively interacted with."""
        relevant: Dict[int, set] = defaultdict(set)
        for inter in interactions:
            if inter["event"] in positive_events:
                relevant[inter["user_id"]].add(inter["item_id"])
        return relevant

    def _build_ranked_lists(
        self,
        impressions: List[Dict],
    ) -> Dict[int, List[int]]:
        """Map user_id → list of item_ids sorted by model_score descending."""
        user_items: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for imp in impressions:
            user_items[imp["user_id"]].append((imp["model_score"], imp["item_id"]))
        return {
            uid: [iid for _, iid in sorted(items, reverse=True)]
            for uid, items in user_items.items()
        }

    # ── Slice NDCG computation ──────────────────────────────────────────────

    def _compute_slice_ndcg(
        self,
        user_ids: List[int],
        ranked_lists: Dict[int, List[int]],
        relevant_items: Dict[int, set],
        propensity: Dict[int, float],
        impressions_by_user: Dict[int, List[Dict]],
        k: int = 10,
    ) -> Dict:
        """
        Compute mean IPS-NDCG@K for a set of user_ids.
        Returns dict with mean, std, n_users, n_users_with_relevant.
        """
        ndcg_scores: List[float] = []
        for uid in user_ids:
            if uid not in ranked_lists or uid not in relevant_items:
                continue
            ranked  = ranked_lists[uid]
            relev   = relevant_items[uid]
            if not relev:
                continue

            # IPS-weighted NDCG
            dcg  = 0.0
            for i, iid in enumerate(ranked[:k]):
                if iid in relev:
                    # Find position this item was shown at for this user
                    pos = i + 1  # approximate: use rank as proxy for position
                    w   = _ips_weight(pos, propensity)
                    dcg += w / np.log2(i + 2)

            # Ideal DCG (with IPS weights for top positions)
            ideal_n = min(len(relev), k)
            idcg    = sum(_ips_weight(i + 1, propensity) / np.log2(i + 2) for i in range(ideal_n))
            ndcg    = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        if not ndcg_scores:
            return {"ndcg": None, "n_users": 0, "reason": "no_users_with_relevant_items"}

        return {
            "ndcg":                round(float(np.mean(ndcg_scores)), 4),
            "ndcg_std":            round(float(np.std(ndcg_scores)), 4),
            "n_users":             len(ndcg_scores),
            "n_users_with_relev":  len(ndcg_scores),
        }

    # ── Main evaluation entry point ─────────────────────────────────────────

    def run_slice_eval(
        self,
        impressions: List[Dict],
        interactions: List[Dict],
        catalog: List[Dict],
        users: List[Dict],
        k: int = 10,
    ) -> Dict:
        """
        Run full slice evaluation. Returns nested dict with NDCG per slice.
        """
        t0 = time.time()

        # Build indexes
        propensity        = self._build_propensity(impressions)
        relevant_items    = self._build_relevant_items(interactions)
        ranked_lists      = self._build_ranked_lists(impressions)
        imps_by_user: Dict[int, List[Dict]] = defaultdict(list)
        for imp in impressions:
            imps_by_user[imp["user_id"]].append(imp)

        # Catalog indexes
        item_genre: Dict[int, str] = {r["item_id"]: r["primary_genre"] for r in catalog}
        user_activity: Dict[int, int] = {u["user_id"]: u["n_interactions"] for u in users}
        user_first_date: Dict[int, str] = {u["user_id"]: u.get("first_interaction_date", "") for u in users}
        user_device: Dict[int, str] = {u["user_id"]: u.get("device_type", "unknown") for u in users}

        all_user_ids = list(set(uid for imp in impressions for uid in [imp["user_id"]]))

        report: Dict = {
            "evaluated_at":  datetime.utcnow().isoformat(),
            "n_impressions": len(impressions),
            "n_users":       len(all_user_ids),
            "global_ndcg":   None,
            "slices": {
                "genre":           {},
                "activity_decile": {},
                "user_age":        {},
                "device":          {},
            },
            "alerts": [],
        }

        # Global NDCG
        global_result = self._compute_slice_ndcg(
            all_user_ids, ranked_lists, relevant_items, propensity, imps_by_user, k
        )
        report["global_ndcg"] = global_result.get("ndcg")

        # ── Slice 1: Genre ──────────────────────────────────────────────────
        # Group users by the primary genre of the items they were served
        genre_users: Dict[str, set] = defaultdict(set)
        for imp in impressions:
            genre = item_genre.get(imp["item_id"], "unknown")
            genre_users[genre].add(imp["user_id"])

        for genre, uid_set in genre_users.items():
            result = self._compute_slice_ndcg(
                list(uid_set), ranked_lists, relevant_items,
                propensity, imps_by_user, k
            )
            report["slices"]["genre"][genre] = result

        # ── Slice 2: User activity decile ───────────────────────────────────
        if user_activity:
            activity_vals = np.array([user_activity.get(uid, 0) for uid in all_user_ids])
            decile_edges  = np.percentile(activity_vals, np.linspace(0, 100, 11))
            decile_labels = ACTIVITY_DECILE_LABELS

            for i, label in enumerate(decile_labels):
                lo = decile_edges[i]
                hi = decile_edges[i + 1]
                uid_set = [
                    uid for uid in all_user_ids
                    if lo <= user_activity.get(uid, 0) < hi
                ]
                result = self._compute_slice_ndcg(
                    uid_set, ranked_lists, relevant_items,
                    propensity, imps_by_user, k
                )
                report["slices"]["activity_decile"][label] = result

        # ── Slice 3: User age ────────────────────────────────────────────────
        now_str = datetime.utcnow().strftime("%Y-%m-%d")
        for bucket_name, (lo_days, hi_days) in USER_AGE_BUCKETS.items():
            uid_set = []
            for uid in all_user_ids:
                first = user_first_date.get(uid, "")
                if not first:
                    continue
                try:
                    first_dt = datetime.strptime(first[:10], "%Y-%m-%d")
                    age_days = (datetime.utcnow() - first_dt).days
                    if lo_days <= age_days < hi_days:
                        uid_set.append(uid)
                except Exception:
                    pass
            result = self._compute_slice_ndcg(
                uid_set, ranked_lists, relevant_items,
                propensity, imps_by_user, k
            )
            report["slices"]["user_age"][bucket_name] = result

        # ── Slice 4: Device type ────────────────────────────────────────────
        device_users: Dict[str, List[int]] = defaultdict(list)
        for uid in all_user_ids:
            device = user_device.get(uid, "unknown")
            device_users[device].append(uid)

        for device, uid_set in device_users.items():
            result = self._compute_slice_ndcg(
                uid_set, ranked_lists, relevant_items,
                propensity, imps_by_user, k
            )
            report["slices"]["device"][device] = result

        # ── Detect slice gaps ───────────────────────────────────────────────
        global_ndcg = report["global_ndcg"] or 0.0
        for slice_type, slice_data in report["slices"].items():
            for slice_name, slice_result in slice_data.items():
                slice_ndcg = slice_result.get("ndcg")
                if slice_ndcg is not None and global_ndcg > 0:
                    gap = (global_ndcg - slice_ndcg) / global_ndcg
                    if gap > 0.20:  # >20% below global = alert
                        report["alerts"].append({
                            "slice_type": slice_type,
                            "slice_name": slice_name,
                            "slice_ndcg": slice_ndcg,
                            "global_ndcg": global_ndcg,
                            "gap_pct": round(gap * 100, 1),
                            "severity": "high" if gap > 0.40 else "medium",
                        })

        elapsed = time.time() - t0
        report["eval_time_s"] = round(elapsed, 2)

        # Persist report
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M")
        out = self.output_path / f"slice_eval_{ts}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        n_alerts = len(report["alerts"])
        logger.info(
            f"[SliceEval] Done in {elapsed:.1f}s — global_ndcg={global_ndcg:.4f}  "
            f"alerts={n_alerts}  report={out}"
        )
        return report


# ── Long-term retention tracker ───────────────────────────────────────────────

class RetentionTracker:
    """
    Tracks 30-day cohort retention.

    METHODOLOGY
    -----------
    A "cohort" is all users who received their first recommendation on a
    given date. We track whether each cohort user returns and plays something
    within 30 days.

    30-day retention = (users who played something within 30 days) / (cohort size)

    This is a strictly stronger signal than 7-day return rate. A user might
    return within 7 days to browse but not engage. 30-day play rate measures
    genuine long-term engagement.

    DATA STORAGE
    -----------
    Each recommendation event writes to cohort_recommendations.jsonl.
    Each play event writes to cohort_plays.jsonl.
    compute_30d_retention() joins these two files for a given cohort date.

    In production, these would live in Postgres or a columnar store.
    JSONL is used here for simplicity and portability.
    """

    def __init__(self, data_path: str = "/app/artifacts/retention/"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._rec_log   = self.data_path / "cohort_recommendations.jsonl"
        self._play_log  = self.data_path / "cohort_plays.jsonl"
        self._lock = threading.Lock()

    def record_recommendation(
        self,
        user_id: int,
        item_ids: List[int],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Log that user_id received recommendations. Call from /recommend endpoint.
        This is the cohort entry point.
        """
        ts  = timestamp or time.time()
        rec = {
            "user_id":   user_id,
            "item_ids":  item_ids[:10],  # top-10 only
            "timestamp": ts,
            "date":      datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
        }
        with self._lock:
            with open(self._rec_log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def record_play(
        self,
        user_id: int,
        item_id: int,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Log that user_id played item_id. Call from /feedback endpoint on play events.
        """
        ts   = timestamp or time.time()
        play = {
            "user_id":   user_id,
            "item_id":   item_id,
            "timestamp": ts,
            "date":      datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
        }
        with self._lock:
            with open(self._play_log, "a") as f:
                f.write(json.dumps(play) + "\n")

    def compute_30d_retention(
        self,
        cohort_date: str,          # "YYYY-MM-DD"
        window_days: int = 30,
    ) -> Dict:
        """
        Compute 30-day retention for the cohort who received their first
        recommendation on cohort_date.

        Returns:
        {
          "cohort_date": "2024-01-01",
          "cohort_size": 1234,
          "returned_within_30d": 892,
          "retention_rate_30d": 0.723,
          "median_days_to_return": 4.2,
          "day_by_day": {"1": 0.55, "7": 0.64, "14": 0.68, "30": 0.72}
        }
        """
        # Load recommendation log for cohort date
        cohort_users: Dict[int, float] = {}  # user_id → first_seen_timestamp
        if self._rec_log.exists():
            with open(self._rec_log) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec["date"] == cohort_date:
                            uid = rec["user_id"]
                            if uid not in cohort_users or rec["timestamp"] < cohort_users[uid]:
                                cohort_users[uid] = rec["timestamp"]
                    except Exception:
                        continue

        if not cohort_users:
            return {
                "cohort_date":  cohort_date,
                "cohort_size":  0,
                "error":        "no_cohort_data",
            }

        cohort_start_ts = min(cohort_users.values())
        cohort_end_ts   = cohort_start_ts + window_days * 86400

        # Load play log for window
        plays_by_user: Dict[int, List[float]] = defaultdict(list)
        if self._play_log.exists():
            with open(self._play_log) as f:
                for line in f:
                    try:
                        play = json.loads(line)
                        uid  = play["user_id"]
                        ts   = play["timestamp"]
                        if uid in cohort_users and cohort_users[uid] < ts <= cohort_end_ts:
                            plays_by_user[uid].append(ts)
                    except Exception:
                        continue

        # Compute metrics
        cohort_size  = len(cohort_users)
        returned     = set(uid for uid, plays in plays_by_user.items() if plays)
        retention_30 = len(returned) / cohort_size if cohort_size > 0 else 0.0

        # Days to first return
        days_to_return = []
        for uid in returned:
            user_first_ts = cohort_users[uid]
            first_play_ts = min(plays_by_user[uid])
            days_to_return.append((first_play_ts - user_first_ts) / 86400)

        # Day-by-day retention curve
        day_buckets = [1, 3, 7, 14, 21, 30]
        day_by_day  = {}
        for day in day_buckets:
            end_ts   = cohort_start_ts + day * 86400
            returned_by_day = sum(
                1 for uid in cohort_users
                if any(ts <= end_ts for ts in plays_by_user.get(uid, []))
            )
            day_by_day[str(day)] = round(returned_by_day / cohort_size, 4) if cohort_size > 0 else 0.0

        report = {
            "cohort_date":            cohort_date,
            "cohort_size":            cohort_size,
            "returned_within_30d":    len(returned),
            "retention_rate_30d":     round(retention_30, 4),
            "median_days_to_return":  round(float(np.median(days_to_return)), 1) if days_to_return else None,
            "mean_days_to_return":    round(float(np.mean(days_to_return)), 1) if days_to_return else None,
            "day_by_day":             day_by_day,
            "computed_at":            datetime.utcnow().isoformat(),
        }

        logger.info(
            f"[RetentionTracker] Cohort {cohort_date}: "
            f"size={cohort_size}  30d_retention={retention_30:.1%}  "
            f"median_days={report['median_days_to_return']}"
        )

        # Persist
        out = self.data_path / f"retention_{cohort_date}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def compute_rolling_7d_vs_30d(self, n_cohorts: int = 7) -> Dict:
        """
        Compare 7-day and 30-day retention across the last n_cohorts cohort dates.
        Surfaces whether short-term metric aligns with long-term.
        """
        today = datetime.utcnow().date()
        results = []
        for i in range(n_cohorts, 0, -1):
            cohort_date = (today - timedelta(days=30 + i)).strftime("%Y-%m-%d")
            report = self.compute_30d_retention(cohort_date)
            if "error" not in report:
                results.append({
                    "date":          cohort_date,
                    "retention_7d":  report["day_by_day"].get("7"),
                    "retention_30d": report["retention_rate_30d"],
                    "cohort_size":   report["cohort_size"],
                })

        return {
            "cohorts": results,
            "avg_retention_7d":  round(np.mean([r["retention_7d"] for r in results if r["retention_7d"]]), 4) if results else None,
            "avg_retention_30d": round(np.mean([r["retention_30d"] for r in results]), 4) if results else None,
        }


# ── Singletons ────────────────────────────────────────────────────────────────

SLICE_EVALUATOR = SliceEvaluator(
    output_path=os.environ.get("SLICE_EVAL_PATH", "/app/artifacts/slice_eval/")
)

RETENTION = RetentionTracker(
    data_path=os.environ.get("RETENTION_PATH", "/app/artifacts/retention/")
)
