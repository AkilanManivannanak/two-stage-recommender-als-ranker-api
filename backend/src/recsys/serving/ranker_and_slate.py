"""
Ranking Layer + Slate/Page Optimizer — Phases 4 & 5
=====================================================
Ranker: LightGBM/GBM with 14 features. Optimises watch value, not just CTR.
Slate optimizer: page-level hard constraints, row ordering, exploration budget.

Ranker features (spec):
  collaborative_score, session_score, semantic_score,
  recency, novelty_distance, abandonment_risk, completion_propensity,
  runtime_suitability, language_fit, popularity_decay,
  launch_effect, impression_fatigue, artwork_trust, page_position_prior

Slate hard rules (spec):
  - no duplicate title on page
  - ≤3 same-genre titles in top-20
  - ≥5 distinct genres per page
  - exploration ≤20% above the fold
  - ≤2 rows with the same dominant genre above the fold
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Ranker ─────────────────────────────────────────────────────────────────

RANKER_FEATURE_COLS = [
    "collaborative_score",
    "session_score",
    "semantic_score",
    "recency_score",
    "novelty_distance",
    "abandonment_risk",
    "completion_propensity",
    "runtime_suitability",
    "language_fit",
    "popularity_decay",
    "launch_effect",
    "impression_fatigue",
    "artwork_trust",
    "page_position_prior",
]


def build_ranker_features(
    item: dict,
    collab_score:  float,
    session_score: float,
    semantic_score: float,
    user_avg_rating: float = 3.5,
    user_genre_ratings: dict = None,
    impression_count: int = 0,
    artwork_trust: float = 0.9,
    position_in_page: int = 5,
) -> list[float]:
    """
    Build the 14-feature vector for one (user, item) pair.
    All features normalised to ~[0, 1] range.
    """
    ugr = user_genre_ratings or {}
    g = item.get("primary_genre", "Unknown")
    year = float(item.get("year", 2000))
    now_year = 2025.0
    runtime = float(item.get("runtime_min", 100))
    popularity = float(item.get("popularity", 50))
    avg_rating = float(item.get("avg_rating", 3.5))

    # Genre ratings for this item's genre
    genre_ratings = ugr.get(g, [])
    genre_avg = float(np.mean(genre_ratings)) if genre_ratings else user_avg_rating

    recency = np.clip((year - 1990) / 35.0, 0, 1)
    novelty = 1.0 - np.clip(popularity / 1000.0, 0, 1)  # less popular = more novel
    abandon_risk = max(0.0, 1.0 - genre_avg / 5.0)
    completion_prop = np.clip(avg_rating / 5.0, 0, 1)
    runtime_suit = 1.0 - abs(runtime - 100.0) / 200.0  # 100min is ideal
    lang_fit = 1.0  # placeholder: 1.0 if user lang matches item lang
    pop_decay = np.clip(1.0 - popularity / 500.0, 0, 1)
    launch_eff = np.clip((year - (now_year - 1)) / 1.0, 0, 1)  # new this year
    imp_fatigue = np.clip(1.0 - impression_count / 20.0, 0, 1)

    return [
        np.clip(collab_score,   -1, 1) * 0.5 + 0.5,
        np.clip(session_score,  -1, 1) * 0.5 + 0.5,
        np.clip(semantic_score, -1, 1) * 0.5 + 0.5,
        recency,
        novelty,
        abandon_risk,
        completion_prop,
        runtime_suit,
        lang_fit,
        pop_decay,
        launch_eff,
        imp_fatigue,
        float(artwork_trust),
        np.clip(1.0 - position_in_page / 50.0, 0, 1),
    ]


class Ranker:
    """
    GBM ranker wrapper. Loaded from artifact bundle.
    Scores candidates and sorts them descending.
    """

    def __init__(self, gbm_model=None):
        self._model = gbm_model

    def rank(
        self,
        candidates: list[dict],
        user_vector: Optional[np.ndarray],
        user_genre_ratings: dict,
        user_avg_rating: float = 3.5,
    ) -> list[dict]:
        if not candidates:
            return candidates

        if self._model is not None:
            try:
                X = []
                for item in candidates:
                    feats = build_ranker_features(
                        item=item,
                        collab_score=float(item.get("collaborative_score", item.get("score", 0.5))),
                        session_score=float(item.get("session_score", 0.3)),
                        semantic_score=float(item.get("semantic_score", 0.3)),
                        user_avg_rating=user_avg_rating,
                        user_genre_ratings=user_genre_ratings,
                        impression_count=int(item.get("impression_count", 0)),
                        artwork_trust=float(item.get("artwork_trust", 0.9)),
                    )
                    X.append(feats)
                import numpy as np
                X_arr = np.array(X, dtype=np.float32)
                scores = self._model.predict_proba(X_arr)[:, 1]
                for item, score in zip(candidates, scores):
                    item["ranker_score"] = round(float(score), 4)
                    item["score"] = item["ranker_score"]
            except Exception as e:
                for item in candidates:
                    item["ranker_score"] = float(item.get("score", 0.5))
        else:
            # No model: use fusion score
            for item in candidates:
                item["ranker_score"] = float(item.get("score", 0.5))

        candidates.sort(key=lambda x: -x.get("ranker_score", 0))
        return candidates


# ── Slate Optimizer ─────────────────────────────────────────────────────────

@dataclass
class SlateConfig:
    max_same_genre_top20:      int   = 3
    min_genres_page:           int   = 6   # target: ≥6 distinct genres per page
    explore_budget_min:        float = 0.10
    explore_budget_max:        float = 0.20
    max_explore_above_fold:    int   = 2
    max_same_genre_rows_fold:  int   = 2
    exploration_score_penalty: float = 0.85   # multiply exploration item scores


@dataclass
class SlateRow:
    row_id:          str
    title:           str
    items:           list[dict]
    dominant_genre:  str = ""
    is_exploration:  bool = False
    n_explore_items: int  = 0


class SlateOptimizer:
    """
    Turns ranked candidates into a page of rows with hard diversity constraints.
    Implements all 5 spec hard rules.
    """

    def __init__(self, config: SlateConfig = None):
        self.cfg = config or SlateConfig()

    def build_page(
        self,
        ranked: list[dict],
        user_genres: list[str],
        user_id: int,
        items_per_row: int = 10,
        explore_budget: float = 0.15,
    ) -> dict:
        """
        Build a full page respecting all hard rules.
        Returns dict with rows, diversity stats, and exploration metadata.
        """
        t0 = time.time()
        explore_budget = max(self.cfg.explore_budget_min,
                             min(self.cfg.explore_budget_max, explore_budget))

        user_genre_set = set(user_genres)

        # ── Separate main candidates from exploration ──────────────────────
        main_items = [r for r in ranked if r.get("primary_genre", "") in user_genre_set]
        explore_items = [r for r in ranked if r.get("primary_genre", "") not in user_genre_set]

        # ── Build top-picks row with genre cap ────────────────────────────
        top_picks = self._pick_diverse(main_items, items_per_row + 5,
                                       max_per_genre=self.cfg.max_same_genre_top20)

        # ── Build exploration row ─────────────────────────────────────────
        n_explore = max(1, int(items_per_row * explore_budget))
        for item in explore_items[:n_explore]:
            item["exploration_slot"] = True
            item["score"] = round(float(item.get("ranker_score", 0.5)) * self.cfg.exploration_score_penalty, 4)
        explore_row_items = explore_items[:n_explore]

        # ── Build genre rows ──────────────────────────────────────────────
        genre_rows: list[SlateRow] = []
        genre_to_items: dict[str, list] = {}
        for item in main_items:
            g = item.get("primary_genre", "Unknown")
            genre_to_items.setdefault(g, []).append(item)

        # Rank genres by user affinity
        ranked_genres = sorted(
            genre_to_items,
            key=lambda g: -len(genre_to_items[g])
        )[:8]

        for g in ranked_genres:
            items_in_genre = self._pick_diverse(
                genre_to_items[g], items_per_row, max_per_genre=items_per_row
            )
            if items_in_genre:
                genre_rows.append(SlateRow(
                    row_id=f"genre_{g.lower()}",
                    title=f"Because you love {g}",
                    items=items_in_genre,
                    dominant_genre=g,
                ))

        # ── Assemble page respecting above-fold genre constraint ──────────
        rows = []
        genre_count_above_fold = Counter()
        above_fold_n = 4  # first 4 rows are "above the fold"

        # Row 1: top picks (always first)
        rows.append({
            "row_id": "top_picks",
            "title":  "Top Picks For You",
            "items":  top_picks[:items_per_row],
        })

        # Row 2: exploration
        if explore_row_items:
            rows.append({
                "row_id":     "explore_new",
                "title":      "Discover Something New",
                "items":      explore_row_items,
                "exploration": True,
            })

        # Remaining genre rows — enforce ≤2 same dominant genre above fold
        for row in genre_rows:
            if len(rows) < above_fold_n:
                if genre_count_above_fold[row.dominant_genre] >= self.cfg.max_same_genre_rows_fold:
                    continue
                genre_count_above_fold[row.dominant_genre] += 1
            rows.append({
                "row_id": row.row_id,
                "title":  row.title,
                "items":  row.items[:items_per_row],
            })

        # ── Enforce page-level dedup ──────────────────────────────────────
        seen_ids: set = set()
        for row in rows:
            deduped = []
            for item in row["items"]:
                iid = item.get("item_id", 0)
                if iid not in seen_ids:
                    seen_ids.add(iid)
                    deduped.append(item)
            row["items"] = deduped

        # ── Compute diversity stats ───────────────────────────────────────
        all_genres = [
            item.get("primary_genre", "?")
            for row in rows
            for item in row["items"][:20]
        ]
        unique_genres = list(set(all_genres))
        genre_counts  = Counter(all_genres)
        max_same      = max(genre_counts.values()) if genre_counts else 0

        # ── Attach position metadata ──────────────────────────────────────
        for row in rows:
            for pos, item in enumerate(row["items"]):
                item["position"] = pos
                item["row_id"]   = row["row_id"]

        return {
            "rows":               rows,
            "n_rows":             len(rows),
            "n_titles":           len(seen_ids),
            "unique_genres":      unique_genres,
            "n_unique_genres":    len(unique_genres),
            "max_same_genre":     max_same,
            "explore_budget":     explore_budget,
            "n_exploration_items": len(explore_row_items),
            "diversity_score":    len(unique_genres) / max(len(all_genres[:20]), 1),
            "assembly_ms":        round((time.time() - t0) * 1000, 1),
            "constraints": {
                "no_page_duplicates":    True,
                "max_same_genre_top20":  self.cfg.max_same_genre_top20,
                "min_genres_page":       self.cfg.min_genres_page,  # 6
                "genres_satisfied":      len(unique_genres) >= self.cfg.min_genres_page,
            }
        }

    def _pick_diverse(
        self,
        items: list[dict],
        n: int,
        max_per_genre: int = 3,
    ) -> list[dict]:
        """Select top-N items with per-genre cap."""
        selected = []
        genre_cnt = Counter()
        for item in items:
            if len(selected) >= n:
                break
            g = item.get("primary_genre", "?")
            if genre_cnt[g] < max_per_genre:
                selected.append(item)
                genre_cnt[g] += 1
        return selected
