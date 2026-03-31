"""
Online Serving Pipeline  —  v5  —  <80ms p99
=============================================
Plane: Core Recommendation

Wiring changes from v4:
  Layer 3  — feature_store_v2.REDIS_FEATURE_STORE replaces redis_feature_store.
              _get_user_features() calls get_user_profile() (v2 API).
  Layer 3b — retrieval_engine_v2.RetrievalEngine replaces retrieval_engine (v1).
              Constructor accepts item_factors, item_embeddings, redis_store, qdrant_client.
              retrieve() signature: user_id, user_vector, user_genre_ratings, session_events.
  Layer 4+5 — ranker_and_slate.SlateOptimizer replaces slate_optimizer_v2 (standalone).
              page() calls build_page() instead of assemble().
  Layer 2  — freshness_layer.FRESH_STORE replaces freshness_engine.FRESH_STORE in page().
              freshness_engine.DRIFT_DETECTOR kept as fallback for drift-aware explore budget.

Latency budget (unchanged):
  feature fetch:        5–10ms
  candidate generation: 10–20ms   (4 retrievers, fused)
  ranking:              5–15ms    (LightGBM / GBM)
  slate optimization:   5–10ms
  serialization:        5–10ms
  p99 target:           <80ms for page assembly
  p99 target:           <40ms for plain top-k

CRITICAL: No LLM in this path. All LLM calls are offline/cached.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Latency budget (ms) ──────────────────────────────────────────────────────
BUDGET = {
    "feature_fetch":     10,
    "candidate_gen":     20,
    "ranking":           15,
    "slate":             10,
    "serialization":     10,
    "total_page":        80,
    "total_topk":        40,
}


@dataclass
class ServingContext:
    """
    Point-in-time snapshot of features for a request.
    Once created, features are frozen — no mid-request updates.
    """
    request_id:           str
    user_id:              int
    k:                    int
    session_item_ids:     list[int]
    user_genres:          list[str]
    user_genre_ratings:   dict
    features_snapshot_id: str = ""
    snapshot_time:        float = field(default_factory=time.time)

    def __post_init__(self):
        self.features_snapshot_id = f"snap_{self.user_id}_{int(self.snapshot_time)}"


@dataclass
class ServingResult:
    """Result from the serving pipeline."""
    request_id:           str
    user_id:              int
    items:                list[dict]
    exploration_slots:    int = 0
    diversity_score:      float = 0.0
    freshness_watermark:  dict = field(default_factory=dict)
    latency_breakdown:    dict = field(default_factory=dict)
    total_latency_ms:     float = 0.0
    n_candidates:         int = 0
    retriever_stats:      dict = field(default_factory=dict)
    gate_passed:          bool = True


class ServingPipeline:
    """
    The online serving path. Wired together once at startup, called per request.

    Follows the spec's serving path:
      1. request starts
      2. snapshot hot features
      3. fetch candidates from 4 retrievers
      4. fuse and dedupe
      5. fast ranker
      6. slate optimizer
      7. attach cached explanations and row titles
      8. return
    """

    def __init__(self, catalog: dict[int, dict]):
        self.catalog = catalog
        self._retrieval_engine = None
        self._ranker = None
        self._slate_optimizer = None
        self._bundle_loaded = False
        self._latency_history: deque = deque(maxlen=10_000)
        self._init_components()

    def _init_components(self):
        """Initialise all pipeline components once."""
        # Retrieval engine — v2: four-retriever fusion
        try:
            from recsys.serving.retrieval_engine_v2 import RetrievalEngine
            import pickle
            bundle = Path("artifacts/bundle")
            item_factors = {}
            try:
                with open(bundle / "item_factors.pkl", "rb") as f:
                    item_factors = pickle.load(f)
            except Exception:
                pass
            self._retrieval_engine = RetrievalEngine(
                catalog=self.catalog,
                item_factors=item_factors,
                item_embeddings={},
                redis_store=None,    # wired after Redis connects
                qdrant_client=None,
            )
            print(f"  [Pipeline] RetrievalEngine v2 ready — {len(item_factors)} item factors")
        except Exception as e:
            print(f"  [Pipeline] Retrieval engine v2 init failed: {e}")

        # Slate optimizer — use ranker_and_slate.SlateOptimizer (v2, 5 hard rules)
        try:
            from recsys.serving.ranker_and_slate import SlateOptimizer
            self._slate_optimizer = SlateOptimizer()
            print("  [Pipeline] SlateOptimizer (ranker_and_slate) ready")
        except ImportError:
            try:
                from recsys.serving.page_optimizer import PageOptimizer
                self._slate_optimizer = PageOptimizer()
                print("  [Pipeline] Fallback: PageOptimizer loaded")
            except Exception:
                pass

        # Load ranker from bundle
        self._try_load_ranker()

    def _try_load_ranker(self):
        """Load GBM ranker from artifact bundle."""
        import pickle
        bundle = Path("artifacts/bundle")
        try:
            with open(bundle / "ranker.pkl", "rb") as f:
                self._ranker = pickle.load(f)
            self._bundle_loaded = True
        except Exception:
            pass

    def recommend(
        self,
        user_id:         int,
        k:               int = 20,
        session_item_ids: list[int] = None,
    ) -> list[dict]:
        """
        Fast recommendation path. Target: <40ms p99.
        Returns list of scored, diversity-reranked items.
        """
        t_total = time.time()
        request_id = str(uuid.uuid4())
        session_item_ids = session_item_ids or []

        # ── 1. Feature snapshot (<10ms) ──────────────────────────────
        t0 = time.time()
        user_genres, user_genre_ratings = self._get_user_features(user_id)
        ctx = ServingContext(
            request_id=request_id,
            user_id=user_id,
            k=k,
            session_item_ids=session_item_ids,
            user_genres=user_genres,
            user_genre_ratings=user_genre_ratings,
        )
        t_features = (time.time() - t0) * 1000

        # ── 2. Candidate generation (<20ms) ─────────────────────────
        t0 = time.time()
        candidates = self._get_candidates(ctx)
        t_candidates = (time.time() - t0) * 1000

        # ── 3. Ranking (<15ms) ──────────────────────────────────────
        t0 = time.time()
        ranked = self._rank(candidates, ctx)
        t_ranking = (time.time() - t0) * 1000

        # ── 4. Diversity rerank + exploration budget ─────────────────
        t0 = time.time()
        final = self._apply_diversity_and_exploration(ranked, ctx, k)
        t_diversity = (time.time() - t0) * 1000

        t_total_ms = (time.time() - t_total) * 1000
        self._latency_history.append(t_total_ms)

        # Attach freshness watermark to items
        for item in final:
            item["features_snapshot_id"] = ctx.features_snapshot_id
            item["policy_id"] = "v4.0.0"

        return final

    def page(
        self,
        user_id:         int,
        items_per_row:   int = 10,
        session_item_ids: list[int] = None,
    ) -> dict:
        """
        Full page assembly. Target: <80ms p99.
        Returns assembled rows with diversity constraints enforced.
        """
        t_total = time.time()
        session_item_ids = session_item_ids or []

        user_genres, user_genre_ratings = self._get_user_features(user_id)

        # Get candidates
        ctx = ServingContext(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            k=50,
            session_item_ids=session_item_ids,
            user_genres=user_genres,
            user_genre_ratings=user_genre_ratings,
        )
        candidates = self._get_candidates(ctx)
        ranked = self._rank(candidates, ctx)

        # Build row candidates
        row_cands = self._build_row_candidates(ranked, user_genres, user_id, items_per_row)

        # Assemble page with hard constraints
        # SlateOptimizer (ranker_and_slate) uses build_page(); PageOptimizer uses assemble()
        if self._slate_optimizer:
            try:
                page = self._slate_optimizer.build_page(
                    ranked=ranked,
                    user_genres=user_genres,
                    user_id=user_id,
                    items_per_row=items_per_row,
                )
            except AttributeError:
                # Fallback: PageOptimizer uses assemble()
                page = self._slate_optimizer.assemble(row_cands, user_genres, user_id)
        else:
            page = {"rows": [], "n_rows": 0, "n_titles": 0}

        page["latency_ms"] = round((time.time() - t_total) * 1000, 1)
        page["features_snapshot_id"] = ctx.features_snapshot_id

        # Freshness watermark — use freshness_layer (v2), fall back to freshness_engine
        try:
            from recsys.serving.freshness_layer import FRESH_STORE
            page["freshness_watermark"] = FRESH_STORE.staleness_report()
        except Exception:
            try:
                from recsys.serving.freshness_engine import FRESH_STORE
                page["freshness_watermark"] = FRESH_STORE.staleness_report()
            except Exception:
                page["freshness_watermark"] = {}

        return page

    def _get_user_features(self, user_id: int) -> tuple[list[str], dict]:
        """Fetch user features from store (target: <10ms).
        Layer 3: uses feature_store_v2.RedisFeatureStore as primary store."""
        # Try feature_store_v2 (v2 API: get_user_profile returns (profile, age, stale))
        try:
            from recsys.serving.feature_store_v2 import REDIS_FEATURE_STORE
            profile, age, stale = REDIS_FEATURE_STORE.get_user_profile(user_id)
            if profile and profile.get("genres"):
                return profile["genres"], profile.get("genre_ratings", {})
        except Exception:
            pass

        # Fallback: deterministic from user_id (same as app.py)
        try:
            from recsys.serving.app import _user_genres, _user_ugr
            return _user_genres(user_id), _user_ugr(user_id)
        except Exception:
            import numpy as np
            rng = np.random.default_rng(user_id * 137)
            genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi",
                      "Romance", "Thriller", "Documentary", "Animation", "Crime"]
            ug = list(rng.choice(genres, size=int(rng.integers(2, 5)), replace=False))
            return ug, {g: [float(rng.uniform(3.0, 5.0))] for g in ug}

    def _get_candidates(self, ctx: ServingContext) -> list[dict]:
        """Run all four retrievers and fuse (target: <20ms)."""
        if self._retrieval_engine is None:
            # Fallback to old _build_recs
            try:
                from recsys.serving.app import _build_recs
                return _build_recs(ctx.user_id, k=min(ctx.k * 5, 200),
                                   session_item_ids=ctx.session_item_ids)
            except Exception:
                return []

        fused = self._retrieval_engine.retrieve(
            user_id=ctx.user_id,
            user_vector=None,            # populated by embedding_worker in production
            user_genre_ratings=ctx.user_genre_ratings,
            session_events=[
                {"item_id": iid, "event_type": "play_start"}
                for iid in (ctx.session_item_ids or [])
            ],
        )
        return fused.items

    def _rank(self, candidates: list[dict], ctx: ServingContext) -> list[dict]:
        """Apply GBM ranker to candidates (target: <15ms)."""
        if not candidates:
            return candidates

        FEAT_COLS = ["als_score", "u_avg", "u_cnt", "item_pop",
                     "item_avg_rating", "item_year", "genre_affinity",
                     "runtime_min", "fused_score"]

        if self._ranker is not None:
            try:
                import numpy as np
                user_genres = set(ctx.user_genres)
                X = []
                for item in candidates:
                    feat = [
                        float(item.get("als_score", item.get("fused_score", 0.5))),
                        float(item.get("u_avg", 3.5)),
                        float(item.get("u_cnt", 50)),
                        float(item.get("popularity", item.get("item_pop", 50))),
                        float(item.get("avg_rating", item.get("item_avg_rating", 3.5))),
                        float(item.get("year", item.get("item_year", 2015))),
                        float(item.get("primary_genre", "") in user_genres),
                        float(item.get("runtime_min", 100)),
                        float(item.get("fused_score", 0.5)),
                    ]
                    X.append(feat)
                X_arr = np.array(X, dtype=np.float32)
                scores = self._ranker.predict_proba(X_arr)[:, 1]
                for item, score in zip(candidates, scores):
                    item["ranker_score"] = round(float(score), 4)
                    item["score"] = item["ranker_score"]
            except Exception:
                for item in candidates:
                    item["ranker_score"] = float(item.get("fused_score", 0.5))
                    item["score"] = item["ranker_score"]
        else:
            # No ranker: use fused score
            for item in candidates:
                item["ranker_score"] = float(item.get("fused_score", 0.5))
                item["score"] = item["ranker_score"]

        candidates.sort(key=lambda x: -x.get("ranker_score", 0.0))
        return candidates

    def _apply_diversity_and_exploration(
        self,
        ranked:    list[dict],
        ctx:       ServingContext,
        k:         int,
    ) -> list[dict]:
        """Diversity rerank + exploration budget."""
        from collections import Counter
        user_genres = set(ctx.user_genres)

        # Drift-aware exploration budget — try freshness_engine, safe fallback
        explore_budget = 0.15
        try:
            from recsys.serving.freshness_engine import DRIFT_DETECTOR
            boost = DRIFT_DETECTOR.drift_exploration_boost(ctx.user_id, ctx.user_genres)
            explore_budget = min(0.35, explore_budget + boost)
        except Exception:
            pass

        n_explore = max(1, int(k * explore_budget))
        n_main = k - n_explore

        # Main: diverse, genre-capped
        main = []
        genre_cnt: Counter = Counter()
        for item in ranked:
            if len(main) >= n_main:
                break
            g = item.get("primary_genre", "?")
            if genre_cnt[g] >= 3:
                continue
            if g not in user_genres:
                continue
            genre_cnt[g] += 1
            item = dict(item)
            item["exploration_slot"] = False
            main.append(item)

        # Exploration: genres outside history
        explore = []
        for item in ranked:
            if len(explore) >= n_explore:
                break
            g = item.get("primary_genre", "?")
            if g in user_genres:
                continue
            if any(m.get("item_id") == item.get("item_id") for m in main):
                continue
            item = dict(item)
            item["exploration_slot"] = True
            item["score"] = round(item.get("ranker_score", 0.5) * 0.85, 4)
            explore.append(item)

        # Attach als_score for backwards compat
        result = main + explore
        for item in result:
            if "als_score" not in item:
                item["als_score"] = item.get("fused_score", item.get("ranker_score", 0.5))

        return result[:k]

    def _build_row_candidates(
        self,
        ranked:     list[dict],
        user_genres: list[str],
        user_id:     int,
        items_per_row: int,
    ) -> dict[str, list[dict]]:
        """Build per-row candidate pools from ranked results."""
        user_genre_set = set(user_genres)

        try:
            from recsys.serving.realtime_engine import TRENDING, SESSION
            trending_raw = TRENDING.top_trending(items_per_row + 5)
            trending_items = []
            for mid, t_score in trending_raw:
                if mid in self.catalog:
                    item = dict(self.catalog[mid])
                    item["trending_score"] = round(t_score, 4)
                    item["ranker_score"] = round(t_score, 4)
                    trending_items.append(item)
            session_ids = SESSION.session_item_ids(user_id)
        except Exception:
            trending_items = []
            session_ids = []

        return {
            "top_picks": [r for r in ranked if not r.get("exploration_slot")][:items_per_row + 5],
            "explore_new_genres": [r for r in ranked if r.get("exploration_slot")][:items_per_row],
            "highly_rated": sorted(ranked, key=lambda x: -float(x.get("avg_rating", 0)))[:items_per_row + 5],
            "trending_now": trending_items[:items_per_row + 5] or ranked[5:15],
            "because_you_watched": [r for r in ranked if session_ids and
                                    r.get("primary_genre", "") in user_genre_set][:items_per_row + 5],
        }

    def latency_stats(self) -> dict:
        """Latency percentiles from recent requests."""
        if not self._latency_history:
            return {}
        s = sorted(self._latency_history)
        n = len(s)
        def p(pct): return round(float(s[min(int(n * pct // 100), n - 1)]), 1)
        return {
            "n": n, "p50_ms": p(50), "p95_ms": p(95), "p99_ms": p(99),
            "max_ms": round(max(s), 1),
            "budget_p99_ms": BUDGET["total_page"],
            "p99_within_budget": p(99) <= BUDGET["total_page"],
        }
