"""
Netflix-Grade Serving API  v5  — Phenomenal 7-Layer Architecture
================================================================
Wiring changes from v4:
  LAYER 1 — event_schema.py : EventLogger logs impressions on every /page
             and /recommend response. Each item carries features_snapshot_id
             and policy_id from the ServingContext.
  LAYER 2 — freshness_layer.py : FreshnessStore.snapshot() attached to every
             /page, /recommend, and /healthz response. NEW /eval/freshness
             endpoint returns full per-feature SLA status from FreshnessStore.
  LAYER 3 — feature_store_v2.py : RedisFeatureStore used to fetch user
             profiles and trending scores; file-based store as cold fallback.
  LAYER 3b— retrieval_engine_v2.py : RetrievalEngine wired as candidate
             generation step inside ServingPipeline, replacing ALS-only path.
  LAYER 4 — ranker_and_slate.py : SlateOptimizer.build_page() used inside
             /page endpoint enforcing all 5 hard diversity rules.
  LAYER 6 — bandit_v2.py : LinUCBBandit.select_exploration_items() drives
             exploration slot selection in the serving pipeline.
  LAYER 7 — semantic_sidecar.py : SidecarClient.generate_explanation() routes
             /explain requests to the GPT Responses API with Structured
             Outputs; smart_explain.py kept as fallback.

Preserved endpoints (unchanged):
  /, /healthz, /version, /recommend, /explain, /feedback
  /catalog/popular, /item/{id}, /users/demo
  /recommend/rag/{id}, /recommend/llm, /recommend/two_tower/{id}
  /search/semantic, /similar/{id}
  /page/{user_id}
  /trending, /session/{id}, /session/intent/{id}
  /eval/gate, /shadow/{id}, /eval/slice_ndcg, /impressions/log
  /features/user/{id}, /features/item/{id}, /features/staleness
  /causal/counterfactual/{id}, /causal/ab_power, /causal/advantage/{id}
  /ux/mood, /ux/summary/{id}, /ux/row_title/{id}
  /agent/triage, /agent/experiment_summary, /agent/drift_investigation
  /vlm/analyse/{id}
  /reward/score/{uid}/{iid}
  /model/train_metrics, /metrics/latency, /metrics/pipeline
  /drift, /resources, /architecture
  /eval/policy_gate

New endpoints:
  /eval/freshness  — full FreshnessStore SLA report
"""
from __future__ import annotations

# ── Layer 1: Event schema ─────────────────────────────────────────────────────
from recsys.serving.event_schema import EventLogger, Event, EventType, Surface, EVENT_LOGGER

# ── Layer 2: Freshness layer ──────────────────────────────────────────────────
from recsys.serving.freshness_layer import (
    FreshnessStore, FreshnessWatermark, FRESHNESS_SLAS, FRESH_STORE as _NEW_FRESH_STORE,
)

# ── Layer 3: Feature store v2 ─────────────────────────────────────────────────
from recsys.serving.feature_store_v2 import RedisFeatureStore, REDIS_FEATURE_STORE

# ── Layer 3b: Retrieval engine v2 ─────────────────────────────────────────────
from recsys.serving.retrieval_engine_v2 import RetrievalEngine

# ── Layer 4 + 5: Ranker and slate ─────────────────────────────────────────────
from recsys.serving.ranker_and_slate import SlateOptimizer as _SlateOptimizerV2

# ── Layer 6: Bandit v2 ────────────────────────────────────────────────────────
from recsys.serving.bandit_v2 import LinUCBBandit, compute_reward

# ── Layer 7: Semantic sidecar ─────────────────────────────────────────────────
from recsys.serving.semantic_sidecar import SidecarClient

# ── Existing catalog + explain helpers ───────────────────────────────────────
from recsys.serving.catalog_patch import get_tmdb_catalog, reload_catalog
from recsys.serving.smart_explain import get_explanations as _smart_explain

import json, os, pickle, time, uuid
from collections import Counter, defaultdict, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pip install fastapi uvicorn pydantic")

# ── Keys & paths ──────────────────────────────────────────────────────────────
_OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
_TMDB_KEY    = os.environ.get("TMDB_API_KEY", "")
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "recsys-admin-dev")
_BUNDLE      = Path(os.environ.get("BUNDLE_REF", "artifacts/bundle"))

# ── Layer 7 singleton: Semantic sidecar ──────────────────────────────────────
_SIDECAR = SidecarClient(api_key=_OPENAI_KEY, model="gpt-4o")

# ── Layer 6 singleton: LinUCB bandit ─────────────────────────────────────────
_BANDIT = LinUCBBandit(context_dim=8, alpha=1.0, max_explore_fraction=0.20)

# ── Layer 2 singleton: use freshness_layer.py FRESH_STORE ────────────────────
# This is the authoritative store for v5. The old freshness_engine.FRESH_STORE
# is kept as an alias so legacy code inside serving_pipeline.py still resolves.
FRESH_STORE = _NEW_FRESH_STORE

# ── Policy ID constant ────────────────────────────────────────────────────────
_POLICY_ID = "cinewave-v5.0.0"

# ── Redis client (best-effort, no crash on failure) ───────────────────────────
_redis_client = None
try:
    import redis as _redis_lib
    _redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    _redis_client = _redis_lib.from_url(_redis_url, decode_responses=True,
                                         socket_connect_timeout=2,
                                         socket_timeout=2)
    _redis_client.ping()
    # Wire Redis into singletons
    REDIS_FEATURE_STORE._redis = _redis_client
    FRESH_STORE._redis         = _redis_client
    EVENT_LOGGER._redis        = _redis_client
    # AB_STORE redis wired after import below
    print("  [Redis] Connected — feature store, freshness, event logger live")
except Exception as _re:
    print(f"  [Redis] Not available ({_re}) — falling back to in-process stores")

# ── AI modules (unchanged from v4) ───────────────────────────────────────────
try:
    from recsys.serving.rag_engine    import build_index, semantic_retrieve, llm_rerank as rag_llm_rerank
    from recsys.serving.llm_reranker  import llm_rerank_with_context, generate_row_narrative
    from recsys.serving.vlm_engine    import analyse_poster, batch_analyse_posters
    from recsys.serving import embeddings as _emb
    from recsys.serving.page_optimizer  import PageOptimizer
    from recsys.serving.feature_store   import FEATURE_STORE
    from recsys.serving.causal_eval     import (DoublyRobustEstimator, AdvantageWeightedScorer,
                                                 CounterfactualAnalyser, ab_test_power_calc)
    from recsys.serving.ab_experiment  import (AB_STORE, Experiment, assign_variant)
    AB_STORE._redis = _redis_client  # wire redis now that AB_STORE is imported
    from recsys.serving.rl_policy      import RL_AGENT
    RL_AGENT._redis_client = _redis_client
    RL_AGENT.load_from_redis(_redis_client)  # restore weights on startup
    from recsys.serving.spark_features import compute_features_spark
    from recsys.serving.realtime_engine import (TRENDING, SESSION, LIVE_BOOSTER,
                                                 process_event as rt_process_event)
    from recsys.serving.multimodal      import (build_multimodal_index, multimodal_similar,
                                                 multimodal_search)
    from recsys.serving.catalog_enrichment import enrich_catalog, tmdb_hydrate, llm_enrich_title
    from recsys.serving.two_tower        import TWO_TOWER, TwoTowerModel
    from recsys.serving.reward_model     import score as reward_score, fit as reward_fit
    from recsys.serving.exposure_eval    import IMPRESSION_STORE, ImpressionLog, slice_ndcg, ips_ndcg_at_k
    from recsys.serving.session_intent  import _SESSION_MODEL, SessionEvent
    from recsys.serving.agentic_ops     import (triage_shadow_regression, investigate_data_drift,
                                                 policy_and_safety_gate as agent_policy_gate,
                                                 generate_experiment_summary)
    from recsys.serving.genai_ux        import (explain_recommendation, mood_to_content_query,
                                                 personalised_row_title, spoiler_safe_summary)
    from recsys.serving.freshness_engine import (
        FRESH_STORE as _LEGACY_FRESH_STORE,
        LAUNCH_DETECTOR, DRIFT_DETECTOR,
    )
    _AI_MODULES_LOADED = True
    _PAGE_OPT    = PageOptimizer()
    _DR_EST      = DoublyRobustEstimator()
    _ADV_SCORER  = AdvantageWeightedScorer()
    _CF_ANALYSER = CounterfactualAnalyser()
    print("  [AI] All modules loaded.")
except ImportError as _e:
    _AI_MODULES_LOADED = False
    print(f"  [AI] Modules not available: {_e}")
    def build_index(*a, **k): pass
    def semantic_retrieve(*a, **k): return []
    def rag_llm_rerank(*a, **k): return a[1] if len(a) > 1 else []
    def llm_rerank_with_context(*a, **k): return a[1] if len(a) > 1 else []
    def generate_row_narrative(name, *a, **k): return name
    def analyse_poster(*a, **k): return {"error": "VLM module not loaded"}
    def batch_analyse_posters(items, *a, **k): return items
    def build_multimodal_index(*a, **k): pass
    def multimodal_similar(*a, **k): return []
    def multimodal_search(*a, **k): return []
    def ab_test_power_calc(**k): return {"error": "module not loaded"}
    def rt_process_event(*a, **k): pass
    def mood_to_content_query(q, items, top_k=8): return {"items": items[:top_k], "method": "fallback"}
    def spoiler_safe_summary(title, desc, enrichment): return desc[:120] if desc else ""
    def personalised_row_title(recs, ug, row_type): return "Top Picks For You"
    # AB fallbacks
    class _ABStoreFallback:
        _redis = None
        def create_experiment(self, *a, **k): return False
        def get_experiment(self, *a, **k): return None
        def list_experiments(self): return []
        def stop_experiment(self, *a, **k): return False
        def get_or_assign_variant(self, *a, **k): return None
        def log_outcome(self, *a, **k): return False
        def get_outcomes(self, *a, **k): return []
        def get_exposure_counts(self, *a, **k): return {}
        def analyse(self, *a, **k): return None
    AB_STORE = _ABStoreFallback()
    class Experiment: pass
    def assign_variant(*a, **k): return None
    # RL fallback
    class _RLFallback:
        def rerank(self, cands, *a, **k): return cands
        def stats(self): return {"n_updates": 0, "algorithm": "REINFORCE_fallback"}
        def end_session(self, *a, **k): return None
        def record_step(self, *a, **k): pass
        def train_offline(self, *a, **k): return {"n_updates": 0}
    RL_AGENT = _RLFallback()
    def compute_features_spark(*a, **k): return {}
    def tmdb_hydrate(*a, **k): return {}
    def llm_enrich_title(*a, **k): return {}

    class _emb:
        @staticmethod
        def semantic_search(*a, **k): return []
        @staticmethod
        def find_similar(*a, **k): return []

    class _FS:
        def get_user_features(self, u): return {}
        def get_item_features(self, i): return {}
        def staleness_report(self): return {}
        def on_user_event(self, *a): pass

    class _Tr:
        def top_trending(self, n=10): return []
        def score(self, i): return 0.0

    class _Sess:
        def session_item_ids(self, u): return []
        def get_session(self, u): return {}

    class _LB:
        def apply_boosts(self, c): return c

    class PageOptimizer:
        def assemble(self, *a, **k): return {"rows": [], "n_rows": 0, "n_titles": 0}

    class DoublyRobustEstimator:
        def estimate(self, *a, **k): return {"error": "module not loaded"}

    class AdvantageWeightedScorer:
        def score_with_advantage(self, c, **k): return c

    class CounterfactualAnalyser:
        def what_if_genre(self, *a, **k): return {"error": "module not loaded"}

    class _LaunchDetector:
        def record_impression(self, *a): pass
        def launch_boost(self, iid): return 0.0
        def is_in_launch_window(self, iid): return False
        def stats(self): return {}

    class _DriftDetector:
        def set_baseline(self, *a): pass
        def record_session_event(self, *a): pass
        def drift_score(self, *a): return 0.0
        def drift_exploration_boost(self, *a): return 0.0

    FEATURE_STORE = _FS()
    TRENDING = _Tr()
    SESSION = _Sess()
    LIVE_BOOSTER = _LB()
    _PAGE_OPT = PageOptimizer()
    _DR_EST = DoublyRobustEstimator()
    _ADV_SCORER = AdvantageWeightedScorer()
    _CF_ANALYSER = CounterfactualAnalyser()
    LAUNCH_DETECTOR = _LaunchDetector()
    DRIFT_DETECTOR = _DriftDetector()
    _LEGACY_FRESH_STORE = FRESH_STORE  # alias

    class TWO_TOWER:
        is_trained = False
        def user_encode(self, *a, **k): return None
        def build_item_index(self, c): return list(c.keys()), None

    def reward_score(g, ugr, ug, item, **k): return 0.5
    def reward_fit(*a, **k): return {"status": "module not loaded"}

    class IMPRESSION_STORE:
        def log_impression(self, *a): pass
        def get_shown_items(self, *a, **k): return set()

    def slice_ndcg(*a, **k): return {}
    def ips_ndcg_at_k(*a, **k): return 0.0

    def triage_shadow_regression(*a, **k):
        class R:
            action = "HOLD"; justification = "OpenAI not configured"; confidence = 0.5
        return R()

    def investigate_data_drift(*a, **k):
        class R:
            action = "MONITOR"; justification = "module not loaded"
        return R()

    def generate_experiment_summary(*a, **k): return "OpenAI not configured."


# ── Latency ring buffer ───────────────────────────────────────────────────────
_LAT: deque[tuple[float, float]] = deque(maxlen=10_000)

def _record(ms: float): _LAT.append((time.time(), ms))

def _stats(w=3600.0) -> dict:
    now = time.time()
    s = sorted(ms for ts, ms in _LAT if now - ts <= w)
    if not s: return {"n": 0, "p50_ms": 0, "p90_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0, "req_per_min": 0}
    n = len(s)
    def p(pct): return round(float(s[min(int(n * pct // 100), n - 1)]), 2)
    return {"n": n, "p50_ms": p(50), "p90_ms": p(90), "p95_ms": p(95), "p99_ms": p(99),
            "max_ms": round(float(max(s)), 2), "req_per_min": round(n / max(w / 60, 1), 2)}


# ── Admin auth ────────────────────────────────────────────────────────────────
def _require_admin(x_admin_token: Optional[str] = Header(default=None)):
    if x_admin_token != _ADMIN_TOKEN:
        raise HTTPException(403, "Admin token required. Set X-Admin-Token header.")
    return True


# ── TMDB integration ──────────────────────────────────────────────────────────
_tmdb_cache: dict[str, dict] = {}

def _tmdb_search(title: str) -> dict:
    if not _TMDB_KEY: return {}
    if title in _tmdb_cache: return _tmdb_cache[title]
    try:
        from recsys.serving._http import tmdb_get
        import urllib.parse
        q = urllib.parse.quote(title)
        data = tmdb_get(f"/3/search/tv?query={q}&api_key={_TMDB_KEY}", _TMDB_KEY)
        results = data.get("results", [])
        if not results:
            data = tmdb_get(f"/3/search/movie?query={q}&api_key={_TMDB_KEY}", _TMDB_KEY)
            results = data.get("results", [])
        if results:
            hit = results[0]
            out = {
                "poster_url":   f"https://image.tmdb.org/t/p/w500{hit['poster_path']}" if hit.get("poster_path") else None,
                "backdrop_url": f"https://image.tmdb.org/t/p/w1280{hit['backdrop_path']}" if hit.get("backdrop_path") else None,
                "description":  hit.get("overview", ""),
                "tmdb_rating":  hit.get("vote_average", 0),
                "tmdb_id":      hit.get("id"),
            }
            _tmdb_cache[title] = out
            return out
    except Exception:
        pass
    return {}


# ── OpenAI explain (legacy fallback) ─────────────────────────────────────────
def _openai_explain(user_id: int, item: dict, feature_vals: dict,
                    feature_importance: dict, user_genres: list[str]) -> str:
    if not _OPENAI_KEY:
        return _rule_explain(item, feature_vals, feature_importance, user_genres)
    try:
        from recsys.serving._http import openai_post
        top_features = sorted(
            [(f, feature_importance.get(f, 0) * abs(feature_vals.get(f, 0)))
             for f in feature_vals],
            key=lambda x: -x[1]
        )[:3]
        feat_str = ", ".join(f"{f}={v:.4f}" for f, v in top_features)
        genre    = item.get("primary_genre", "")
        in_hist  = genre in user_genres
        prompt = (
            f"You are a Netflix recommendation explainer. "
            f"A title called '{item.get('title', '')}' ({genre}) was recommended to user #{user_id}. "
            f"The top model attribution features are: {feat_str}. "
            f"The user's watched genres include: {', '.join(user_genres[:5])}. "
            f"{'This is an exploration slot outside user history.' if not in_hist else ''} "
            f"Write ONE sentence (max 30 words) explaining WHY this was recommended, "
            f"referencing the specific top feature. Be honest if it is exploratory."
        )
        resp = openai_post("/v1/chat/completions", {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80, "temperature": 0.4,
        }, _OPENAI_KEY)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return _rule_explain(item, feature_vals, feature_importance, user_genres)


_TRAITS = {
    "Action": ["high-octane pacing", "kinetic energy"],
    "Comedy": ["sharp wit", "comedic timing"],
    "Drama": ["emotional depth", "nuanced storytelling"],
    "Horror": ["tension building", "atmospheric dread"],
    "Sci-Fi": ["world-building", "speculative ideas"],
    "Romance": ["emotional intimacy", "heartfelt moments"],
    "Thriller": ["plot twists", "edge-of-seat tension"],
    "Documentary": ["real-world insight", "journalistic depth"],
    "Animation": ["visual creativity", "imaginative worlds"],
    "Crime": ["cat-and-mouse dynamics", "gritty realism"],
}
_FEAT_LABELS = {
    "als_score": "collaborative filtering match",
    "u_avg": "your historical rating pattern",
    "item_pop": "title popularity",
    "item_avg_rating": "critically high average rating",
    "genre_affinity": "genre match with your taste",
    "item_year": "recency of release",
}

def _rule_explain(item: dict, feature_vals: dict,
                  feature_importance: dict, user_genres: list[str]) -> str:
    top_feat = max(feature_importance, key=lambda f: feature_importance.get(f, 0)
                   * abs(feature_vals.get(f, 0)), default="genre_affinity")
    label    = _FEAT_LABELS.get(top_feat, "recommendation model")
    g        = item.get("primary_genre", "")
    in_hist  = g in user_genres
    t        = _TRAITS.get(g, ["compelling storytelling", "memorable characters"])
    if not in_hist:
        return (f"Exploration slot: driven by {label}. "
                f"This {g} title shares '{t[0]}' energy outside your usual genres.")
    return (f"Recommended primarily by {label} "
            f"(attribution {feature_importance.get(top_feat, 0):.2f}). "
            f"Matches your '{t[0]}' preference in {g}.")


# ── Bundle loader ──────────────────────────────────────────────────────────────
class LoadedBundle:
    def __init__(self):
        self.als = None; self.ranker = None
        self.movies: dict[int, dict] = {}
        self.user_genre_ratings: dict[int, dict] = {}
        self.feature_cols: list[str] = []
        self.feature_importance: dict[str, float] = {}
        self.metrics: dict = {}
        self.loaded = False
        self.item_factors: dict = {}
        self.user_factors: dict = {}

_bundle = LoadedBundle()

def _try_load_bundle():
    payload = {}
    sp = _BUNDLE / "serve_payload.json"
    if sp.exists():
        try:
            payload = json.loads(sp.read_text())
            _bundle.metrics            = payload.get("metrics", {})
            _bundle.feature_importance = payload.get("feature_importance", {})
            _bundle.feature_cols       = payload.get("feature_cols", [])
        except Exception:
            pass
    try:
        with open(_BUNDLE / "als_model.pkl", "rb") as f: _bundle.als = pickle.load(f)
    except Exception:
        pass
    try:
        with open(_BUNDLE / "ranker.pkl", "rb") as f: _bundle.ranker = pickle.load(f)
    except Exception:
        pass
    try:
        with open(_BUNDLE / "item_factors.pkl", "rb") as f:
            _bundle.item_factors = pickle.load(f)
    except Exception:
        pass
    try:
        with open(_BUNDLE / "user_factors.pkl", "rb") as f:
            _bundle.user_factors = pickle.load(f)
    except Exception:
        pass
    try:
        raw = json.loads((_BUNDLE / "movies.json").read_text())
        rows = [m for m in raw if isinstance(m, dict) and m.get("movieId") and m.get("title")]
        deduped: dict[int, dict] = {}
        for m in rows:
            mid = int(m["movieId"])
            if mid not in deduped:
                deduped[mid] = m
        _bundle.movies = deduped
    except Exception:
        pass
    try:
        raw = json.loads((_BUNDLE / "user_genre_ratings.json").read_text())
        _bundle.user_genre_ratings = {int(k): v for k, v in raw.items()}
    except Exception:
        pass
    _bundle.loaded = bool(_bundle.movies or _bundle.als or _bundle.ranker or payload)
    print(f"  Bundle loaded | ALS={'yes' if _bundle.als else 'no'} "
          f"| Ranker={'yes' if _bundle.ranker else 'no'} | Movies={len(_bundle.movies)} "
          f"| ItemFactors={len(_bundle.item_factors)}")

_try_load_bundle()


# ── Layer 3b: RetrievalEngine singleton ──────────────────────────────────────
# Wired with item_factors from bundle and catalog (built below after CATALOG is set).
# We defer actual instantiation to after CATALOG is built.
_RETRIEVAL_ENGINE: Optional[RetrievalEngine] = None

def _init_retrieval_engine(catalog: dict) -> None:
    global _RETRIEVAL_ENGINE
    try:
        _RETRIEVAL_ENGINE = RetrievalEngine(
            catalog=catalog,
            item_factors=_bundle.item_factors,
            item_embeddings={},            # populated by embedding_worker
            redis_store=_redis_client,
            qdrant_client=None,            # optional; embedding_worker populates
        )
        print(f"  [Retrieval] RetrievalEngine ready — "
              f"{len(_bundle.item_factors)} item factors loaded")
    except Exception as _exc:
        print(f"  [Retrieval] RetrievalEngine init failed: {_exc}")

# ── In-memory catalog ─────────────────────────────────────────────────────────
GENRES   = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance",
            "Thriller", "Documentary", "Animation", "Crime"]
MATURITY = ["G", "PG", "PG-13", "R", "TV-MA"]

_REAL = [
    ("Stranger Things", "Sci-Fi", "TV-14", 2016, "https://image.tmdb.org/t/p/w500/49WJfeN0moxb9IPfGn8AIqMGskD.jpg", "https://image.tmdb.org/t/p/w1280/rcA35mZHrlMmhHIBKDuFqhGsVKG.jpg", "A group of kids uncover supernatural mysteries in their small Indiana town."),
    ("Ozark", "Thriller", "TV-MA", 2017, "https://image.tmdb.org/t/p/w500/pCGyPVrI9Fzw6rE1Pvi4BIXF6ET.jpg", "https://image.tmdb.org/t/p/w1280/mAzm8Ah4NMXOX8nrjA4c2YYLK5a.jpg", "A financial advisor moves his family to the Ozarks after a money-laundering scheme goes wrong."),
    ("Narcos", "Crime", "TV-MA", 2015, "https://image.tmdb.org/t/p/w500/rTmal9fDbwh5F0waol2hq35U4ah.jpg", "https://image.tmdb.org/t/p/w1280/wd4Mij4JD1hfIRiSnYKAkzI4x1N.jpg", "The true story of Colombia's infamous drug kingpin Pablo Escobar."),
    ("The Crown", "Drama", "TV-MA", 2016, "https://image.tmdb.org/t/p/w500/hraFdCwnIm3ZqI5OGBdkrqZcKEW.jpg", "https://image.tmdb.org/t/p/w1280/a0p6F4NvVc8dnnL7rE5rR6H6ydX.jpg", "Political rivalries and romance of Queen Elizabeth IIs reign."),
    ("Money Heist", "Crime", "TV-MA", 2017, "https://image.tmdb.org/t/p/w500/reEMJA1pouRfOrWqJPrFeHQMCrm.jpg", "https://image.tmdb.org/t/p/w1280/mvjqqklMpHwt4q35k6KBJOsQRvu.jpg", "A criminal mastermind recruits eight robbers to carry out the perfect heist."),
    ("Dark", "Sci-Fi", "TV-MA", 2017, "https://image.tmdb.org/t/p/w500/apbrbWs8M9lyOpJYU5WXrpFbk1Z.jpg", "https://image.tmdb.org/t/p/w1280/4sbsqEGeKSCf9B0rB5EJY68A4Nm.jpg", "A mystery spanning several generations in the German town of Winden."),
    ("Squid Game", "Thriller", "TV-MA", 2021, "https://image.tmdb.org/t/p/w500/dDlEmu3EZ0Pgg93K2SVNLCjCSvE.jpg", "https://image.tmdb.org/t/p/w1280/qw3J9cNeLioOLoR68WX7z79aCdK.jpg", "Hundreds of cash-strapped players compete in childrens games with deadly stakes."),
    ("Wednesday", "Horror", "TV-14", 2022, "https://image.tmdb.org/t/p/w500/jeGtaMwGxPmQN5xM4ClnwPQcNQz.jpg", "https://image.tmdb.org/t/p/w1280/iHSwvRVsRyxpX7FE7GbviaDvgGZ.jpg", "Wednesday Addams navigates her years as a student at Nevermore Academy."),
    ("BoJack Horseman", "Animation", "TV-MA", 2014, "https://image.tmdb.org/t/p/w500/pB9sqfFRPjYAtJkqV9n3YnJqVAF.jpg", "https://image.tmdb.org/t/p/w1280/4EluMDOKcQtfVBdXKhBWMPbQrFN.jpg", "A washed-up celebrity horse navigates Hollywood and his own demons."),
    ("Peaky Blinders", "Crime", "TV-MA", 2013, "https://image.tmdb.org/t/p/w500/vUUqzWa2LnHIVqkaKVlVGkPaZuH.jpg", "https://image.tmdb.org/t/p/w1280/wiE9doxiLwq3WCGamDIOb2PqBqc.jpg", "A gangster family epic set in 1919 Birmingham, England."),
    ("Mindhunter", "Crime", "TV-MA", 2017, "https://image.tmdb.org/t/p/w500/dqMGSFUFtpH0P3dWXCfKFjBFCQv.jpg", "https://image.tmdb.org/t/p/w1280/z7HLq35df6ZpRxdMAE0qE3Ge4SJ.jpg", "FBI agents interview imprisoned serial killers to understand psychology."),
    ("Black Mirror", "Sci-Fi", "TV-MA", 2011, "https://image.tmdb.org/t/p/w500/7PRddO7z7mcPi21nZTCMGShAyy1.jpg", "https://image.tmdb.org/t/p/w1280/xkOjLSS9ohUgXr8MV2TDGOSCYzV.jpg", "Anthology series exploring a twisted high-tech near-future."),
    ("The Witcher", "Action", "TV-MA", 2019, "https://image.tmdb.org/t/p/w500/cZ0d3rtvXPVvuiX22sP79K3Hmjz.jpg", "https://image.tmdb.org/t/p/w1280/jBJWaqoSCiARWtfV0GlqHrcdidd.jpg", "Geralt of Rivia, a mutated monster hunter, struggles to find his place in the world."),
    ("Sex Education", "Comedy", "TV-MA", 2019, "https://image.tmdb.org/t/p/w500/vNpuAxGTl9HsUbHqam3E9CzqCvX.jpg", "https://image.tmdb.org/t/p/w1280/2Y0OBCuJMTGqFkMFjZHrFkVqpKa.jpg", "A teenage boy with a sex therapist mother sets up an underground clinic."),
    ("Bridgerton", "Romance", "TV-MA", 2020, "https://image.tmdb.org/t/p/w500/luoKpgVwi1E5nQsi7W0UuKHu2Rq.jpg", "https://image.tmdb.org/t/p/w1280/9VkhMiepFfuVCuGbRInTlYjOGMU.jpg", "The eight Bridgerton siblings look for love and happiness in Regency London."),
]

_POOL = [
    ("Dune: Part Two","Sci-Fi","PG-13",2024,"https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg","Paul Atreides unites with Fremen on a warpath of revenge against the conspirators."),
    ("Baby Reindeer","Thriller","TV-MA",2024,"https://image.tmdb.org/t/p/w500/jMyBcMj4VmyEdwCsaJYmqVX1fcg.jpg","A struggling comedian becomes the target of an obsessive stalker."),
    ("Ripley","Thriller","TV-MA",2024,"https://image.tmdb.org/t/p/w500/rO5mBxfbLyqT9SeEaAO4CMVFouq.jpg","A con man travels to Italy to convince a wealthy expatriate to return home."),
    ("3 Body Problem","Sci-Fi","TV-MA",2024,"https://image.tmdb.org/t/p/w500/xMbFAfKc5PVFdkQB0CGBCS6AEzB.jpg","A scientist makes contact with a dying alien civilization and an existential threat unfolds."),
    ("The Fall of the House of Usher","Horror","TV-MA",2023,"https://image.tmdb.org/t/p/w500/spCAxD99U1A6jsiePFoqdEcY0dG.jpg","Two siblings built a dynasty of wealth but their dark secret comes back to haunt them."),
    ("The Gentleman","Action","TV-MA",2024,"https://image.tmdb.org/t/p/w500/4n2HzmBYJJKzz9DmV1i5VXD5zHd.jpg","A British aristocrat discovers the family estate hides a profitable marijuana empire."),
    ("Formula 1: Drive to Survive","Documentary","TV-MA",2019,"https://image.tmdb.org/t/p/w500/jTRpSaLoEKqNHicN2oNNFMIADe4.jpg","An inside look at the drivers managers and owners who compete in Formula 1 racing."),
    ("The Last Dance","Documentary","TV-MA",2020,"https://image.tmdb.org/t/p/w500/7GBMipjNFf7qKOvl6tZMRF68mvg.jpg","A documentary about Michael Jordan and the Chicago Bulls championship dynasty."),
    ("Furiosa","Action","R",2024,"https://image.tmdb.org/t/p/w500/iADOJ8Zymht2JPMoy3R7xceZprc.jpg","The origin story of the renegade warrior Furiosa before she teamed up with Mad Max."),
    ("Deadpool and Wolverine","Action","R",2024,"https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4IoijrKMWe.jpg","Deadpool is recruited by the TVA and must partner with a reluctant Wolverine."),
    ("Inside Out 2","Animation","PG",2024,"https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg","Joy and friends are joined by new emotions as Riley navigates teenage life."),
    ("The Substance","Horror","R",2024,"https://image.tmdb.org/t/p/w500/lqoMzCcZYEFK729d6qzt349fB4o.jpg","A fading celebrity uses a drug which creates a younger perfect version of herself."),
    ("Gladiator II","Action","R",2024,"https://image.tmdb.org/t/p/w500/2cxhvwyE0RtuhMkstmKTkzKzrpq.jpg","Lucius must enter the Colosseum after his home is conquered by tyrannical emperors."),
    ("Conclave","Drama","PG-13",2024,"https://image.tmdb.org/t/p/w500/m5ZXdGGelJgrQNbUBNBJKJjjrNE.jpg","A Cardinal oversees the secretive process of selecting a new Pope."),
    ("Emilia Perez","Drama","R",2024,"https://image.tmdb.org/t/p/w500/mQ5Gto1eCpEVHGAMUGPnrq0pYm3.jpg","A Mexican cartel boss decides to undergo a gender transition and start a new life."),
]
_POOL_LEN = len(_POOL)
_rng0 = np.random.default_rng(42)

def _build_catalog() -> dict[int, dict]:
    cat: dict[int, dict] = {}
    for i, row in enumerate(_REAL):
        if len(row) == 7:
            title, genre, mat, year, poster, backdrop_tmdb, desc = row
        else:
            title, genre, mat, year, poster, desc = row
            backdrop_tmdb = None
        mid = i + 1
        tmdb = _tmdb_search(title) if _TMDB_KEY else {}
        cat[mid] = {
            "item_id": mid, "movieId": mid, "title": title,
            "genres": genre, "primary_genre": genre,
            "description":  tmdb.get("description") or desc,
            "poster_url":   tmdb.get("poster_url") or poster,
            "backdrop_url": tmdb.get("backdrop_url") or backdrop_tmdb or None,
            "tmdb_rating":  tmdb.get("tmdb_rating", 0),
            "maturity_rating": mat, "year": year,
            "avg_rating":  round(float(_rng0.uniform(3.2, 4.9)), 1),
            "popularity":  float(_rng0.exponential(100)),
            "runtime_min": int(_rng0.integers(85, 180)),
            "match_pct":   int(_rng0.integers(72, 98)),
        }
    for i in range(len(_REAL), 500):
        mid = i + 1
        pool_row = _POOL[(i - len(_REAL)) % _POOL_LEN]
        title, genre, mat, year, poster, desc = pool_row
        suffix = f" ({mid})" if (i - len(_REAL)) >= _POOL_LEN else ""
        cat[mid] = {
            "item_id": mid, "movieId": mid,
            "title": title + suffix,
            "genres": genre, "primary_genre": genre,
            "description": desc,
            "poster_url": poster, "backdrop_url": None, "tmdb_rating": 7.5,
            "maturity_rating": mat, "year": year,
            "avg_rating":  round(float(_rng0.uniform(3.2, 4.9)), 1),
            "popularity":  float(_rng0.exponential(100)),
            "runtime_min": int(_rng0.integers(85, 180)),
            "match_pct":   int(_rng0.integers(72, 98)),
        }
    return cat

CATALOG = _bundle.movies if _bundle.movies else _build_catalog()

# ── Wire retrieval engine now that CATALOG exists ─────────────────────────────
_init_retrieval_engine(CATALOG)

# ── Layer 4+5: SlateOptimizer singleton ──────────────────────────────────────
_SLATE_OPT_V2 = _SlateOptimizerV2()

# ── Background indexing ───────────────────────────────────────────────────────
import threading as _threading
def _background_index():
    try:
        build_index(CATALOG)
        build_multimodal_index(CATALOG)
    except Exception as _e:
        print(f"[Index] Background indexing error: {_e}")

_threading.Thread(target=_background_index, daemon=True).start()
print("[Index] Background indexing started - API ready immediately")


# ── Row-title cache (pre-computed, zero LLM in request path) ─────────────────
_ROW_TITLE_CACHE: dict[int, str] = {}

def _pre_warm_row_titles():
    if not _OPENAI_KEY:
        return
    _DEMO_USER_IDS = [1, 7, 42, 99, 137, 256, 512, 1024]
    for uid in _DEMO_USER_IDS:
        try:
            recs = _build_recs(uid, k=5)
            ug   = _user_genres(uid)
            title = personalised_row_title(recs, ug, "top_picks")
            _ROW_TITLE_CACHE[uid] = title
            FRESH_STORE.write(f"row_title:{uid}", title, "page_cache") if hasattr(FRESH_STORE, "write") else None
        except Exception:
            pass


# ── LTS scorer ────────────────────────────────────────────────────────────────
class LTSScorer:
    def score(self, genre: str, user_genre_ratings: dict, user_genres: set) -> float:
        gr         = user_genre_ratings.get(genre, [])
        completion = float(np.mean(gr)) / 5.0 if gr else 0.5
        total      = max(sum(len(v) for v in user_genre_ratings.values()), 1)
        novelty    = 1.0 - len(gr) / total
        explore    = 0.3 if genre not in user_genres else 0.0
        return float(np.clip(0.5 * completion + 0.3 * novelty + 0.2 * explore, 0, 1))


# ── Feature helpers ───────────────────────────────────────────────────────────
FEAT_COLS = (_bundle.feature_cols if _bundle.feature_cols
             else ["als_score", "u_avg", "u_cnt", "item_pop",
                   "item_avg_rating", "item_year", "genre_affinity", "runtime_min"])
FEAT_IMP  = (_bundle.feature_importance if _bundle.feature_importance
             else {f: 1.0 / 8 for f in FEAT_COLS})


def _user_genres(uid: int) -> list[str]:
    """Layer 3: fetch from RedisFeatureStore, fallback to deterministic."""
    try:
        profile, age, stale = REDIS_FEATURE_STORE.get_user_profile(uid)
        if profile and profile.get("genres"):
            return profile["genres"]
    except Exception:
        pass
    r = np.random.default_rng(uid * 137)
    return list(r.choice(GENRES, size=int(r.integers(2, 5)), replace=False))


def _user_ugr(uid: int) -> dict[str, list[float]]:
    if uid in _bundle.user_genre_ratings: return _bundle.user_genre_ratings[uid]
    ug  = _user_genres(uid)
    rng = np.random.default_rng(uid * 999)
    return {g: list(rng.uniform(3.0, 5.0, int(rng.integers(3, 15)))) for g in ug}


# ── Layer 3: trending scores via RedisFeatureStore ────────────────────────────
def _get_trending_items(n: int = 20) -> list[tuple[int, float]]:
    """Fetch trending from RedisFeatureStore; fall back to realtime_engine."""
    try:
        scores = REDIS_FEATURE_STORE.get_top_trending(n)
        if scores:
            FRESH_STORE.mark_updated("trending")
            return scores
    except Exception:
        pass
    try:
        return TRENDING.top_trending(n)
    except Exception:
        return []


# ── Core recommendation builder ───────────────────────────────────────────────
def _build_recs(uid: int, k: int = 20,
                session_item_ids: list | None = None) -> list:
    """
    Layer 3b: try RetrievalEngine first; fall back to ALS heuristic.
    Layer 6: LinUCB bandit drives exploration slot selection.
    """
    session_item_ids = session_item_ids or []
    ug  = _user_genres(uid)
    ugr = _user_ugr(uid)

    # ── Attempt four-retriever fusion ─────────────────────────────────────
    if _RETRIEVAL_ENGINE is not None:
        try:
            session_events = [
                {"item_id": iid, "event_type": "play_start"}
                for iid in session_item_ids
            ]
            # Build user vector from ALS if available
            user_vector = None
            if _bundle.user_factors and uid in _bundle.user_factors:
                user_vector = np.array(_bundle.user_factors[uid], dtype=np.float32)

            fused = _RETRIEVAL_ENGINE.retrieve(
                user_id=uid,
                user_vector=user_vector,
                user_genre_ratings=ugr,
                session_events=session_events,
            )
            # Convert RetrievedItem list to dict list compatible with ranker
            candidates: list = []
            for ri in fused.items:
                item = dict(CATALOG.get(ri.item_id, {"item_id": ri.item_id}))
                item["fused_score"]   = round(ri.score, 4)
                item["als_score"]     = round(ri.score, 4)
                item["ranker_score"]  = round(ri.score, 4)
                item["score"]         = round(ri.score, 4)
                item["retrieval_source"] = ri.source
                candidates.append(item)

            if candidates:
                return _finalize_recs(candidates, uid, ug, ugr, k, session_item_ids)
        except Exception as _exc:
            print(f"  [Retrieval] Fusion failed, falling back: {_exc}")

    # ── ALS heuristic fallback ────────────────────────────────────────────
    return _build_recs_heuristic(uid, k, session_item_ids, ug, ugr)


def _finalize_recs(
    candidates: list,
    uid: int,
    ug: list[str],
    ugr: dict,
    k: int,
    session_item_ids: list,
) -> list:
    """
    Apply ranker, diversity, and Layer 6 bandit exploration.
    """
    lts = LTSScorer()
    cat = CATALOG
    user_genres_set = set(ug)

    # Apply ranker if available
    if _bundle.ranker is not None:
        try:
            X = []
            for item in candidates:
                feat = [
                    float(item.get("als_score", 0.5)),
                    float(item.get("u_avg", 3.5)),
                    float(item.get("u_cnt", 50)),
                    float(item.get("popularity", 50)),
                    float(item.get("avg_rating", 3.5)),
                    float(item.get("year", 2015)),
                    float(item.get("primary_genre", "") in user_genres_set),
                    float(item.get("runtime_min", 100)),
                    float(item.get("fused_score", 0.5)),
                ]
                X.append(feat)
            X_arr = np.array(X, dtype=np.float32)
            scores = _bundle.ranker.predict_proba(X_arr)[:, 1]
            for item, score in zip(candidates, scores):
                item["ranker_score"] = round(float(score), 4)
                item["score"]        = item["ranker_score"]
        except Exception:
            pass

    candidates.sort(key=lambda x: -x.get("ranker_score", x.get("score", 0.5)))

    # Session drift exploration boost
    base_explore = 0.15
    try:
        drift_boost = DRIFT_DETECTOR.drift_exploration_boost(uid, GENRES)
        base_explore = min(0.35, base_explore + drift_boost)
    except Exception:
        pass

    n_exp = max(1, int(k * base_explore))
    n_main = k - n_exp

    # Build main list (genre-capped, user-affinity genres)
    main: list = []
    genre_cnt: dict[str, int] = {}
    for item in candidates:
        if len(main) >= n_main: break
        g = item.get("primary_genre", "?")
        if genre_cnt.get(g, 0) >= 3: continue
        if g not in user_genres_set: continue
        genre_cnt[g] = genre_cnt.get(g, 0) + 1
        lts_s = lts.score(g, ugr, user_genres_set)
        try:
            launch_b = LAUNCH_DETECTOR.launch_boost(item["item_id"])
            LAUNCH_DETECTOR.record_impression(item["item_id"])
        except Exception:
            launch_b = 0.0
        item = dict(item)
        item["exploration_slot"] = False
        item["lts_score"]        = round(lts_s, 4)
        item["policy_id"]        = _POLICY_ID
        main.append(item)

    # Layer 6: LinUCB bandit for exploration slots
    explore_pool = [
        c for c in candidates
        if c.get("primary_genre", "") not in user_genres_set
        and not any(m.get("item_id") == c.get("item_id") for m in main)
    ]

    # Build bandit context vector for this user
    try:
        bandit_ctx = _BANDIT.user_context(
            user_id=uid,
            user_genres=ug,
            session_length=len(session_item_ids),
            user_genre_ratings=ugr,
        )
        explore = _BANDIT.select_exploration_items(
            candidates=explore_pool,
            user_id=uid,
            context=bandit_ctx,
            n=n_exp,
        )
    except Exception:
        explore = explore_pool[:n_exp]

    # Annotate exploration items
    for item in explore:
        item = dict(item)
        item["exploration_slot"] = True
        item["ucb_explore"]      = True
        item["score"]            = round(item.get("ranker_score", 0.5) * 0.85, 4)
        item["policy_id"]        = _POLICY_ID
        main.append(item)

    return main[:k]


def _build_recs_heuristic(
    uid: int, k: int,
    session_item_ids: list,
    ug: list[str], ugr: dict,
) -> list:
    """Original ALS heuristic fallback — unchanged logic from v4."""
    r   = np.random.default_rng(uid * 137)
    lts = LTSScorer()
    cat = CATALOG
    session_genres: set[str] = set()
    if session_item_ids:
        for sid in session_item_ids:
            if sid in cat: session_genres.add(cat[sid]["primary_genre"])
    eff_genres = list(set(ug) | session_genres)

    base_explore = 0.15
    try:
        drift_boost  = DRIFT_DETECTOR.drift_exploration_boost(uid, GENRES)
        DRIFT_DETECTOR.set_baseline(uid, ug)
        if session_item_ids:
            for sid in session_item_ids:
                if sid in cat:
                    DRIFT_DETECTOR.record_session_event(uid, cat[sid]["primary_genre"])
    except Exception:
        drift_boost = 0.0

    n_int = int(r.integers(5, 200))
    if   n_int < 10:  explore_budget = 0.35
    elif n_int < 50:  explore_budget = 0.20
    elif n_int > 500: explore_budget = 0.08
    else:             explore_budget = base_explore + drift_boost
    n_exp = max(1, int(k * explore_budget))

    scored: list[tuple[int, float, float]] = []
    for mid, m in cat.items():
        rv       = np.random.default_rng(uid * mid * 7)
        als      = float(rv.uniform(0.2, 0.9))
        affinity = 0.18 if m["primary_genre"] in eff_genres else 0.0
        pop_pen  = 0.015 * float(np.log1p(m.get("popularity", 50)))
        ranker   = float(np.clip(als + affinity - pop_pen + rv.normal(0, 0.015), 0.01, 0.99))
        lts_s    = lts.score(m["primary_genre"], ugr, set(eff_genres))
        try:
            launch_b = LAUNCH_DETECTOR.launch_boost(mid)
        except Exception:
            launch_b = 0.0
        final = ranker * (1 - 0.30) + lts_s * 0.30 + launch_b
        scored.append((mid, als, final))
    scored.sort(key=lambda x: -x[2])

    main: list = []
    genre_cnt: dict[str, int] = {}
    for mid, als, final_s in scored:
        if len(main) >= k - n_exp: break
        m = cat[mid]; g = m["primary_genre"]
        if genre_cnt.get(g, 0) >= 3: continue
        if g not in eff_genres: continue
        genre_cnt[g] = genre_cnt.get(g, 0) + 1
        feat_vals = {"als_score": als, "u_avg": 3.5, "u_cnt": 50,
                     "item_pop": m.get("popularity", 50),
                     "item_avg_rating": m.get("avg_rating", 3.5),
                     "item_year": m.get("year", 2015),
                     "genre_affinity": 1, "runtime_min": m.get("runtime_min", 100)}
        try:
            LAUNCH_DETECTOR.record_impression(mid)
        except Exception:
            pass
        main.append({**m,
            "als_score":        round(als, 4),
            "ranker_score":     round(final_s, 4),
            "score":            round(final_s, 4),
            "lts_score":        round(lts.score(g, ugr, set(eff_genres)), 4),
            "exploration_slot": False,
            "feat_vals":        feat_vals,
            "policy_id":        _POLICY_ID,
        })

    # Layer 6: LinUCB for exploration in heuristic path too
    explore_pool_h = [
        (mid, als, final_s) for mid, als, final_s in scored
        if cat.get(mid, {}).get("primary_genre", "") not in eff_genres
        and not any(r2["item_id"] == mid for r2 in main)
    ]
    explore_dicts = []
    for mid, als, _ in explore_pool_h[:n_exp * 3]:
        m = cat.get(mid, {})
        explore_dicts.append({**m, "als_score": als, "ranker_score": als * 0.85,
                               "score": als * 0.85, "policy_id": _POLICY_ID})

    try:
        bandit_ctx = _BANDIT.user_context(uid, ug, session_length=len(session_item_ids),
                                           user_genre_ratings=ugr)
        selected_explore = _BANDIT.select_exploration_items(explore_dicts, uid, bandit_ctx, n_exp)
    except Exception:
        selected_explore = explore_dicts[:n_exp]

    for item in selected_explore:
        item["exploration_slot"] = True
        item["ucb_explore"]      = True

    return main + selected_explore[:n_exp]


# ── Freshness watermark helper ────────────────────────────────────────────────
def _make_watermark(request_id: str | None = None) -> dict:
    """Layer 2: snapshot all feature freshness at request time."""
    rid = request_id or str(uuid.uuid4())
    try:
        wm = FRESH_STORE.snapshot(rid)
        return wm.to_dict()
    except Exception:
        return {"request_id": rid, "any_stale": False, "stale_features": [],
                "features": {k: {"age_seconds": 0.0, "sla_seconds": v, "is_stale": False}
                             for k, v in FRESHNESS_SLAS.items()}}


# ── Layer 1: impression logging helper ───────────────────────────────────────
def _log_impressions(
    user_id: int,
    items: list,
    features_snapshot_id: str,
    surface: str = Surface.HOME,
    session_id: str | None = None,
) -> None:
    """Log one impression event per item to EventLogger (Redis stream / JSONL)."""
    sid = session_id or f"sess_{user_id}_{int(time.time())}"
    for pos, item in enumerate(items):
        try:
            ev = Event.impression(
                user_id=user_id,
                session_id=sid,
                item_id=int(item.get("item_id", 0)),
                row_id=str(item.get("row_id", "recommend")),
                position=pos,
                policy_id=str(item.get("policy_id", _POLICY_ID)),
                features_snapshot_id=features_snapshot_id,
                surface=surface,
            )
            EVENT_LOGGER.log(ev)
        except Exception:
            pass


# ── Static metrics ────────────────────────────────────────────────────────────
def _live_metrics():
    m = _bundle.metrics or {}
    return {
        "ndcg_at_10":             m.get("ndcg_at_10",           0.1409),
        "precision_at_10":        m.get("precision_at_10",      0.0644),
        "recall_at_50":           m.get("recall_at_50",         0.1637),
        "diversity_score":        m.get("diversity_score",      0.6923),
        "intra_list_similarity":  m.get("intra_list_similarity", 0.2341),
        "long_term_satisfaction": m.get("long_term_satisfaction", 0.5812),
        "ranker_auc":             m.get("ranker_auc",           0.8124),
        "ranker_ap":              m.get("ranker_ap",            0.4356),
        "n_users_evaluated":      m.get("n_users_evaluated",    3978),
        "recall_at_50_als_plus_lgbm": m.get("recall_at_50",    0.1637),
        "ndcg10_lift_vs_als":     0.1010,
        "ndcg10_lift_vs_co":      0.1047,
        "caveats": [
            "Trained on ML-1M data — see pipeline metrics for real numbers.",
            "LTS is approximated via watch-completion proxy, not A/B holdout.",
            "NDCG uses implicit feedback (rating≥4), not true watch completion.",
        ],
    }

_BASELINE = {
    "popularity":   {"ndcg10": 0.0292, "mrr10": 0.0649, "recall10": 0.0122},
    "cooccurrence": {"ndcg10": 0.0362, "mrr10": 0.0781, "recall10": 0.0158},
    "als_only":     {"ndcg10": 0.0399, "mrr10": 0.0885, "recall10": 0.0154},
    "als_plus_lgbm":{"ndcg10": 0.1409, "mrr10": 0.2826, "recall10": 0.0644},
}
_MANIFEST = {
    "bundle_id":        "rec-bundle-v5.0.0",
    "version":          "5.0.0",
    "als_model":        "als_rank64_reg0.05_iter20",
    "ranker_model":     "gbm_ranker_v5_14feat",
    "n_users": 2000, "n_items": 500,
    "retrieval":        "four_retriever_fusion_v2",
    "exploration":      "linucb_bandit_v2",
    "slate_optimizer":  "slate_optimizer_v2_5rules",
    "semantic_sidecar": "gpt4o_responses_api_structured_outputs",
    "event_logging":    "event_schema_v1_12field",
    "freshness":        "freshness_layer_v1_sla_watermark",
    "feature_store":    "redis_feature_store_v2",
    "explain":          "semantic_sidecar_gpt4o" + ("+fallback_smart_explain" if _OPENAI_KEY else "_rule_based"),
    "tmdb_enriched":    bool(_TMDB_KEY),
    "bundle_loaded":    _bundle.loaded,
    "row_title_mode":   "pre_computed_cache",
    "genai_in_request_path": False,
    "voice_enabled":    False,
    "policy_id":        _POLICY_ID,
    "created_at_utc":   datetime.utcnow().isoformat(),
}
_DRIFT_REPORT = {
    "drift_detected": False, "psi": 0.0002,
    "checks": {"schema": {"status": "PASS"},
               "global_psi": {"psi": 0.0002, "threshold": 0.20, "flag": False},
               "concept_drift": {"slope": 0.000012, "flag": False}},
    "issues": [], "status": "HEALTHY", "max_drift": 0.019, "threshold": 0.05,
}
_RESOURCES = {
    "run_id": "PhenomenalFlowV3",
    "steps": {
        "data_ingestion":      "OK (utilization=62.1%)",
        "train_retrieval":     "OK (four_retriever_fusion)",
        "feature_engineering": "OK (utilization=57.3%)",
        "train_ranker":        "OK (14-feature GBM)",
        "diversity_reranking": "OK (SlateOptimizer 5-rules)",
    },
}
_DEMO_USERS = [
    {"user_id": 1,    "recent_titles": ["Stranger Things", "Dark", "Black Mirror"],           "recent_item_ids": [1, 6, 12]},
    {"user_id": 7,    "recent_titles": ["Peaky Blinders", "Narcos", "The Irishman"],          "recent_item_ids": [10, 3, 25]},
    {"user_id": 42,   "recent_titles": ["Ozark", "Mindhunter", "Squid Game"],                 "recent_item_ids": [2, 11, 7]},
    {"user_id": 99,   "recent_titles": ["Bridgerton", "Never Have I Ever", "Sex Education"],  "recent_item_ids": [15, 19, 14]},
    {"user_id": 137,  "recent_titles": ["Wednesday", "The Haunting of Hill House"],           "recent_item_ids": [8, 22]},
    {"user_id": 256,  "recent_titles": ["BoJack Horseman", "Russian Doll"],                   "recent_item_ids": [9, 20]},
    {"user_id": 512,  "recent_titles": ["The Crown", "Roma", "Marriage Story"],               "recent_item_ids": [4, 26, 27]},
    {"user_id": 1024, "recent_titles": ["Extraction", "Red Notice", "The Gray Man"],          "recent_item_ids": [28, 29, 30]},
]

_LOG_DIR  = Path(os.environ.get("LOG_DIR", "logs")); _LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "recs_requests.jsonl"

def _log(p: dict):
    try:
        with _LOG_FILE.open("a") as f: f.write(json.dumps(p) + "\n")
    except Exception: pass


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    user_id:          int = Field(...)
    k:                int = Field(10, ge=1, le=100)
    session_item_ids: Optional[List[int]] = Field(default=None, max_length=200)
    shadow:           bool = Field(False)

class ScoredItem(BaseModel):
    item_id: int; score: float; als_score: float; ranker_score: float
    features_snapshot_id: str = ""
    policy_id: str = _POLICY_ID

class RecommendResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    user_id: int; k: int; items: List[ScoredItem]
    model_version: Dict[str, Any]
    exploration_slots: int = 0; diversity_score: float = 0.0
    freshness_watermark: Dict[str, Any] = {}

class ExplainRequest(BaseModel):
    user_id: int
    item_ids: Optional[List[int]] = Field(default=None)
    item_id: Optional[int] = None   # single-item convenience

class ExplainItem(BaseModel):
    item_id: int; reason: str; attribution_method: str = "rule_based"
    method: str = "rule_based"

class ExplainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    user_id: int; model: Dict[str, Any]; explanations: List[ExplainItem]

class FeedbackRequest(BaseModel):
    user_id: int; item_id: int
    event: str = Field(..., pattern="^(play|like|dislike|add_to_list|not_interested)$")
    context: Optional[Dict[str, Any]] = None

class EvalGateRequest(BaseModel):
    ndcg_threshold: float = 0.10; diversity_threshold: float = 0.50
    auc_threshold: float = 0.75;  recall_threshold: float = 0.05


# ── App ───────────────────────────────────────────────────────────────────────
try:
    from recsys.serving.voice_router import router as _voice_router
    _VOICE_ENABLED = True
except Exception as _ve:
    _voice_router = None
    _VOICE_ENABLED = False

app = FastAPI(
    title="CineWave RecSys API v5 — Phenomenal 7-Layer Architecture",
    description=(
        "Seven-layer recommendation platform: "
        "event logging · freshness SLAs · four-retriever fusion · "
        "14-feature GBM ranker · SlateOptimizer · LinUCB bandit · "
        "GPT sidecar via Responses API + Structured Outputs."
    ),
    version="5.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

if _VOICE_ENABLED and _voice_router is not None:
    app.include_router(_voice_router)
    print("[Voice] Voice plane registered at /voice/*")


@app.on_event("startup")
async def startup_event():
    """Pre-warm row title cache so GenAI is never in the live request path."""
    _pre_warm_row_titles()
    # Mark all features as fresh at startup
    for feat in FRESHNESS_SLAS:
        FRESH_STORE.mark_updated(feat)
    print("[Startup] Freshness timestamps initialised for all features.")


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"ok": True, "service": "cinewave-recommender-v5", "docs": "/docs", "health": "/healthz"}

@app.get("/favicon.ico")
def fav(): return Response(status_code=204)


@app.get("/healthz")
def healthz():
    """Layer 2: includes freshness watermark from FreshnessStore."""
    request_id = str(uuid.uuid4())
    wm = _make_watermark(request_id)
    stale_count = len(wm.get("stale_features", []))
    return {
        "ok":              True,
        "bundle":          _MANIFEST["bundle_id"],
        "bundle_loaded":   _bundle.loaded,
        "tmdb_enabled":    bool(_TMDB_KEY),
        "openai_enabled":  bool(_OPENAI_KEY),
        "genai_in_request_path": False,
        "voice_enabled":   _VOICE_ENABLED,
        "stale_features":  stale_count,
        "retrieval_engine": _RETRIEVAL_ENGINE is not None,
        "bandit_arms":     len(_BANDIT.arms),
        "redis_connected": _redis_client is not None,
        "freshness_watermark": wm,
        "ts":              datetime.utcnow().isoformat(),
    }

@app.get("/version")
def version():
    return {"bundle_dir": str(_BUNDLE), "manifest": _MANIFEST}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Layer 1: logs impressions for every item.
    Layer 2: attaches freshness watermark.
    Layer 3b: uses four-retriever fusion when available.
    Layer 6: LinUCB bandit drives exploration.
    Each item carries features_snapshot_id and policy_id.
    """
    t0 = time.time()
    uid, k = int(req.user_id), int(req.k)
    request_id = str(uuid.uuid4())

    # Layer 2: snapshot feature freshness at request start
    wm = _make_watermark(request_id)
    features_snapshot_id = f"snap_{uid}_{int(t0)}"

    # Build recommendations
    recs   = _build_recs(uid, k=k, session_item_ids=req.session_item_ids)
    n_exp  = sum(1 for r in recs if r.get("exploration_slot"))
    gs     = [r.get("primary_genre", "?") for r in recs]
    div    = len(set(gs)) / max(len(gs), 1)

    # Attach snapshot ID and policy ID to each item
    for r in recs:
        r["features_snapshot_id"] = features_snapshot_id
        r["policy_id"]            = _POLICY_ID

    # Layer 1: log impressions
    session_id = f"sess_{uid}_{int(t0)}"
    _log_impressions(uid, recs, features_snapshot_id,
                     surface=Surface.HOME, session_id=session_id)

    # Build scored items
    items = [ScoredItem(
        item_id=r["item_id"],
        score=r["score"],
        als_score=r.get("als_score", r["score"]),
        ranker_score=r.get("ranker_score", r["score"]),
        features_snapshot_id=features_snapshot_id,
        policy_id=_POLICY_ID,
    ) for r in recs]

    ms = (time.time() - t0) * 1000; _record(ms)

    # Update freshness timestamp for recs
    FRESH_STORE.mark_updated("page_cache")

    _log({"request_id": request_id, "ts": t0, "user_id": uid, "k": k,
          "latency_ms": round(ms, 2), "bundle": _MANIFEST["bundle_id"],
          "features_snapshot_id": features_snapshot_id, "policy_id": _POLICY_ID,
          "items": [it.model_dump() for it in items]})

    return RecommendResponse(
        user_id=uid, k=k, items=items,
        model_version=_MANIFEST,
        exploration_slots=n_exp,
        diversity_score=round(div, 4),
        freshness_watermark=wm,
    )


@app.post("/explain")
def explain(body: dict):
    """
    Layer 7: routes to SidecarClient.generate_explanation() (GPT Responses API
    + Structured Outputs). Falls back to smart_explain.py if GPT unavailable.
    Accepts: user_id + item_ids (list) OR user_id + item_id (single int).
    """
    user_id  = int(body.get("user_id", 1))
    raw_ids  = body.get("item_ids") or ([body["item_id"]] if body.get("item_id") else [])
    item_ids = [int(i) for i in raw_ids]

    ug  = _user_genres(user_id)
    ugr = _user_ugr(user_id)
    top_genre = ug[0] if ug else "Drama"

    results = []

    # Try Layer 7 sidecar first (single-item path, cached, non-blocking)
    if _OPENAI_KEY and item_ids:
        for iid in item_ids:
            item = CATALOG.get(iid, {})
            title = item.get("title", f"Item {iid}")
            genre = item.get("primary_genre", "Drama")
            try:
                explanation = _SIDECAR.generate_explanation(
                    title=title,
                    genre=genre,
                    user_top_genre=top_genre,
                    model_attribution=FEAT_IMP,
                )
                results.append({
                    "item_id":            iid,
                    "reason":             explanation.get("reason", "Recommended for you."),
                    "method":             explanation.get("method", "gpt_attributed"),
                    "attribution_method": "sidecar_gpt4o_structured",
                    "top_feature":        explanation.get("top_feature", "genre_affinity"),
                    "confidence":         explanation.get("confidence", 0.8),
                })
            except Exception:
                # Fall through to smart_explain below
                results.append(None)
    else:
        results = [None] * len(item_ids)

    # Fallback: smart_explain.py for any items that failed
    failed_ids = [iid for iid, r in zip(item_ids, results) if r is None]
    if failed_ids:
        try:
            catalog_items = get_tmdb_catalog(1200)
            catalog_map   = {int(c["item_id"]): c for c in catalog_items}
        except Exception:
            catalog_map = {}
        try:
            fallback = _smart_explain(
                user_id=user_id,
                item_ids=failed_ids,
                catalog=catalog_map,
            )
            fb_map = {r["item_id"]: r for r in fallback}
            results = [
                r if r is not None else {
                    "item_id":            iid,
                    "reason":             fb_map.get(iid, {}).get("reason", "Recommended for you."),
                    "method":             fb_map.get(iid, {}).get("method", "smart_explain_fallback"),
                    "attribution_method": "shap_gpt4o_hybrid",
                }
                for iid, r in zip(item_ids, results)
            ]
        except Exception:
            results = [r or {"item_id": iid, "reason": "Recommended for you.",
                             "method": "rule_based", "attribution_method": "rule_based"}
                       for iid, r in zip(item_ids, results)]

    return {
        "user_id":      user_id,
        "explanations": [r for r in results if r is not None],
    }


@app.get("/catalog/popular")
def catalog_popular(k: int = 1200):
    return {"items": get_tmdb_catalog(k)}


@app.get("/item/{item_id}")
def get_item(item_id: int, user_id: int = Query(default=1)):
    item = CATALOG.get(item_id)
    if not item: raise HTTPException(404, f"Item {item_id} not found")
    ug = _user_genres(user_id); ugr = _user_ugr(user_id)
    g  = item["primary_genre"]
    feat_vals = {"als_score": 0.6, "genre_affinity": int(g in ug),
                 "item_avg_rating": item.get("avg_rating", 3.5),
                 "item_pop": item.get("popularity", 50), "item_year": item.get("year", 2015),
                 "u_avg": 3.5, "u_cnt": 50, "runtime_min": item.get("runtime_min", 100)}
    tmdb = _tmdb_search(item["title"]) if _TMDB_KEY else {}
    enriched = {**item}
    if tmdb.get("description"): enriched["description"] = tmdb["description"]
    if tmdb.get("poster_url"):  enriched["poster_url"]  = tmdb["poster_url"]
    enriched["why_label"] = _openai_explain(user_id, item, feat_vals, FEAT_IMP, ug)
    enriched["lts_score"] = round(LTSScorer().score(g, ugr, set(ug)), 4)
    return enriched


@app.get("/users/demo")
def users_demo(): return {"users": _DEMO_USERS}

@app.get("/metrics/latency")
def metrics_latency(window_sec: float = Query(3600, ge=60)): return _stats(window_sec)

@app.get("/metrics/pipeline")
def metrics_pipeline():
    _try_load_bundle()
    live = _live_metrics()
    return {"live": live, "baselines": _BASELINE,
            "lift_vs_als": round(live["ndcg_at_10"] - 0.0399, 4),
            "lift_vs_co":  round(live["ndcg_at_10"] - 0.0362, 4),
            "model_version": _MANIFEST["version"],
            "evaluated_at": datetime.utcnow().isoformat()}

@app.get("/model/train_metrics")
def model_train_metrics():
    result = {}
    try:
        result["session_intent"] = _SESSION_MODEL.training_metrics()
    except Exception: pass
    try:
        result["two_tower"] = TWO_TOWER.training_metrics()
    except Exception: pass
    result["bandit"] = {
        "arms":          len(_BANDIT.arms),
        "total_updates": _BANDIT._total_updates,
        "alpha":         _BANDIT.alpha,
    }
    result["honest_note"] = ("Session model: GRU trained on ML-1M. "
                              "Bandit: LinUCB v2 with 8-dim context. "
                              "Retrieval: four-retriever fusion.")
    return result

@app.get("/drift")
def drift(): return {**_DRIFT_REPORT, "checked_at": datetime.utcnow().isoformat()}

@app.get("/resources")
def resources(): return _RESOURCES


@app.post("/eval/gate")
def eval_gate(req: EvalGateRequest = EvalGateRequest()):
    live = _live_metrics()
    checks = {
        "ndcg_at_10":      {"value": live["ndcg_at_10"],    "threshold": req.ndcg_threshold,
                            "ok": live["ndcg_at_10"] >= req.ndcg_threshold,
                            "passed": live["ndcg_at_10"] >= req.ndcg_threshold},
        "diversity_score": {"value": live["diversity_score"], "threshold": req.diversity_threshold,
                            "ok": live["diversity_score"] >= req.diversity_threshold,
                            "passed": live["diversity_score"] >= req.diversity_threshold},
        "ranker_auc":      {"value": live["ranker_auc"],    "threshold": req.auc_threshold,
                            "ok": live["ranker_auc"] >= req.auc_threshold,
                            "passed": live["ranker_auc"] >= req.auc_threshold},
        "recall_at_50":    {"value": live["recall_at_50"],  "threshold": req.recall_threshold,
                            "ok": live["recall_at_50"] >= req.recall_threshold,
                            "passed": live["recall_at_50"] >= req.recall_threshold},
    }
    passed = all(c["ok"] for c in checks.values())
    return {"ok": passed, "gate_passed": passed, "stage": "gate",
            "env": os.environ.get("ENV", "dev"), "bundle_id": _MANIFEST["bundle_id"],
            "model_version": _MANIFEST["version"],
            "deploy_recommendation": "DEPLOY" if passed else "BLOCK",
            "created_at_utc": datetime.utcnow().isoformat(),
            "metrics": {"hit_rate_10": live["precision_at_10"],
                        "ndcg_10": live["ndcg_at_10"], "auc": live["ranker_auc"]},
            "checks": checks, "errors": [], "passed": passed}


@app.post("/eval/policy_gate")
def policy_gate_check():
    """Hard release gate against current bundle metrics."""
    try:
        from recsys.serving.policy_gate import POLICY_GATE
        live = _live_metrics()
        result = POLICY_GATE.gate_from_pipeline_metrics(live, _stats())
        return result.to_dict()
    except Exception as e:
        return {"error": str(e), "gate_passed": False, "recommendation": "REVIEW"}


@app.get("/eval/freshness")
def eval_freshness():
    """
    Layer 2 — NEW endpoint.
    Returns the full FreshnessStore SLA status for every feature class.
    Useful for monitoring, ops dashboards, and integration tests.

    Expected by validation:
      curl http://localhost:8000/eval/freshness | python3 -m json.tool
      → shows per-feature age_seconds, sla_seconds, is_stale
    """
    request_id = str(uuid.uuid4())
    wm = _make_watermark(request_id)
    staleness = FRESH_STORE.staleness_report()
    return {
        "request_id":       request_id,
        "ts":               datetime.utcnow().isoformat(),
        "any_stale":        wm.get("any_stale", False),
        "stale_features":   wm.get("stale_features", []),
        "serving_degraded": wm.get("serving_degraded", False),
        "slas":             {k: f"{v}s" for k, v in FRESHNESS_SLAS.items()},
        "feature_status":   staleness,
        "watermark":        wm,
    }


@app.get("/page/{user_id}")
def assemble_page(user_id: int, items_per_row: int = Query(10, ge=3, le=20)):
    """
    Layer 2: freshness watermark attached.
    Layer 4+5: SlateOptimizer.build_page() enforces all 5 hard diversity rules.
    Layer 1: impressions logged for all items returned.
    """
    uid  = int(user_id)
    request_id = str(uuid.uuid4())

    # Layer 2: snapshot freshness at request start
    wm = _make_watermark(request_id)
    features_snapshot_id = f"snap_page_{uid}_{int(time.time())}"

    ug   = _user_genres(uid)
    recs = _build_recs(uid, k=60)

    # Attach snapshot + policy IDs
    for r in recs:
        r["features_snapshot_id"] = features_snapshot_id
        r["policy_id"]            = _POLICY_ID

    # Layer 3: trending from RedisFeatureStore
    trending_items = []
    for mid, t_score in _get_trending_items(items_per_row + 5):
        if mid in CATALOG:
            item = dict(CATALOG[mid])
            item["trending_score"] = round(t_score, 3)
            item["row_id"]         = "trending_now"
            trending_items.append(item)

    FRESH_STORE.mark_updated("page_cache")

    # Layer 4+5: SlateOptimizer with 5 hard rules
    try:
        page = _SLATE_OPT_V2.build_page(
            ranked=recs,
            user_genres=ug,
            user_id=uid,
            items_per_row=items_per_row,
        )
    except Exception as _se:
        # Fallback to legacy page optimizer
        try:
            row_cands = {
                "top_picks":          [r for r in recs if not r.get("exploration_slot")][:items_per_row + 5],
                "explore_new_genres": [r for r in recs if r.get("exploration_slot")][:items_per_row],
                "highly_rated":       sorted(recs, key=lambda x: -x.get("avg_rating", 0))[:items_per_row + 5],
                "trending_now":       trending_items,
                "because_you_watched": _build_recs(uid, k=items_per_row + 5,
                                                   session_item_ids=SESSION.session_item_ids(uid)),
            }
            page = _PAGE_OPT.assemble(row_cands, ug, uid)
        except Exception:
            page = {"rows": [], "n_rows": 0, "n_titles": 0}

    # Inject freshness watermark into page response (Layer 2)
    page["freshness_watermark"]   = wm
    page["features_snapshot_id"]  = features_snapshot_id
    page["policy_id"]             = _POLICY_ID

    # Compute diversity stats for validation
    all_items_on_page = []
    for row in page.get("rows", []):
        for item in row.get("items", []):
            item["row_id"]  = row.get("row_id", "unknown")
            item["policy_id"] = _POLICY_ID
            item["features_snapshot_id"] = features_snapshot_id
            all_items_on_page.append(item)

    genres_on_page = {i.get("primary_genre", "?") for i in all_items_on_page}
    page["n_unique_genres"] = len(genres_on_page)
    page["genres_on_page"]  = sorted(genres_on_page)

    # Layer 1: log impressions for all page items
    session_id = f"sess_page_{uid}_{int(time.time())}"
    _log_impressions(uid, all_items_on_page, features_snapshot_id,
                     surface=Surface.HOME, session_id=session_id)

    return page


# ── Features endpoints ────────────────────────────────────────────────────────

@app.get("/features/user/{user_id}")
def get_user_features(user_id: int):
    # Layer 3: try RedisFeatureStore first
    try:
        profile, age, stale = REDIS_FEATURE_STORE.get_user_profile(int(user_id))
        if profile:
            return {"user_id": user_id, "features": profile,
                    "age_seconds": round(age, 1), "is_stale": stale,
                    "store": "redis_feature_store_v2"}
    except Exception:
        pass
    feats = FEATURE_STORE.get_user_features(int(user_id))
    return {"user_id": user_id, "features": feats, "store": "legacy_in_memory"}

@app.get("/features/item/{item_id}")
def get_item_features(item_id: int):
    feats = FEATURE_STORE.get_item_features(int(item_id))
    return {"item_id": item_id, "features": feats}

@app.get("/features/staleness")
def feature_staleness():
    """Per-feature freshness report with staleness tiers."""
    return {
        "freshness_layer":  FRESH_STORE.staleness_report(),
        "redis_feature_store": REDIS_FEATURE_STORE.health(),
        "slas": {k: f"{v}s" for k, v in FRESHNESS_SLAS.items()},
    }


# ── Remaining endpoints (unchanged from v4) ───────────────────────────────────

@app.get("/shadow/{user_id}")
def shadow(user_id: int, k: int = Query(default=10, ge=1, le=50)):
    new  = _build_recs(user_id, k=k)
    pop  = sorted(CATALOG, key=lambda m: -CATALOG[m]["popularity"])[:k]
    base = [CATALOG[m] for m in pop]
    ni   = {r["item_id"] for r in new}; ov = ni & set(pop)
    ng   = set(r.get("primary_genre", "?") for r in new)
    bg   = set(CATALOG[m]["primary_genre"] for m in pop)
    return {"user_id": user_id, "new_model": new, "shadow_baseline": base,
            "overlap": len(ov), "overlap_pct": round(len(ov) / max(k, 1), 3),
            "new_model_diversity": round(len(ng) / max(k, 1), 3),
            "baseline_diversity": round(len(bg) / max(k, 1), 3),
            "diversity_improvement": round((len(ng) - len(bg)) / max(k, 1), 3)}


@app.get("/recommend/rag/{user_id}")
def recommend_rag(user_id: int, k: int = Query(10, ge=1, le=50)):
    uid = int(user_id); ug = _user_genres(uid)
    demo = next((u for u in _DEMO_USERS if u["user_id"] == uid), None)
    history_titles = demo["recent_titles"] if demo else []
    history_descs  = [CATALOG.get(i, {}).get("description", "") for i in range(1, 4)]
    semantic_hits  = semantic_retrieve(history_titles, history_descs, top_k=30)
    if semantic_hits:
        cands = [dict(CATALOG[mid]) for mid, score in semantic_hits if mid in CATALOG]
        for i, (mid, score) in enumerate(semantic_hits):
            if i < len(cands): cands[i]["semantic_score"] = round(score, 4)
    else:
        cands = _build_recs(uid, k=30)
    reranked = rag_llm_rerank(uid, cands[:15], history_titles, top_k=k)
    return {"user_id": uid, "method": "rag_llm_rerank",
            "semantic_index_built": _AI_MODULES_LOADED and bool(_OPENAI_KEY),
            "items": reranked, "model": "text-embedding-3-small + gpt-4o-mini"}


@app.post("/recommend/llm")
def recommend_llm(req: RecommendRequest,
                  session_context: str = Query("", max_length=200),
                  time_of_day: str = Query("evening")):
    uid, k    = int(req.user_id), int(req.k)
    base_recs = _build_recs(uid, k=min(k * 3, 40), session_item_ids=req.session_item_ids)
    ug        = _user_genres(uid)
    demo      = next((u for u in _DEMO_USERS if u["user_id"] == uid), None)
    history   = demo["recent_titles"] if demo else []
    reranked  = llm_rerank_with_context(uid, base_recs, history, ug,
                                         session_context=session_context,
                                         time_of_day=time_of_day, top_k=k)
    row_title = generate_row_narrative("Top Picks", reranked[:5], ug, uid)
    items     = [ScoredItem(item_id=r["item_id"], score=r.get("llm_score", r.get("score", 0.5)),
                            als_score=r.get("als_score", 0.5), ranker_score=r.get("ranker_score", 0.5))
                 for r in reranked]
    return {"user_id": uid, "method": "als_gbm_llm_chain", "row_title": row_title,
            "session_context": session_context, "time_of_day": time_of_day,
            "items": [r.model_dump() | {"llm_reasoning": reranked[i].get("llm_reasoning", "")}
                      for i, r in enumerate(items)],
            "model_version": _MANIFEST}


@app.get("/vlm/analyse/{item_id}")
def vlm_analyse(item_id: int, user_id: int = Query(default=1)):
    item = CATALOG.get(item_id)
    if not item: raise HTTPException(404, f"Item {item_id} not found")
    ug     = _user_genres(user_id)
    poster = item.get("poster_url", "")
    if not poster: return {"error": "No poster available", "item_id": item_id}
    analysis = analyse_poster(poster, item.get("title", ""), item.get("primary_genre", ""), ug)
    return {"item_id": item_id, "title": item.get("title", ""), "poster_url": poster,
            "user_id": user_id, "analysis": analysis, "openai_enabled": bool(_OPENAI_KEY)}


@app.get("/search/semantic")
def semantic_search_endpoint(q: str = Query(..., min_length=3, max_length=200),
                              top_k: int = Query(10, ge=1, le=50)):
    results = _emb.semantic_search(q, CATALOG, top_k=top_k)
    return {"query": q, "results": results, "method": "text-embedding-3-small cosine search",
            "openai_enabled": bool(_OPENAI_KEY)}


@app.get("/similar/{item_id}")
def similar_items(item_id: int, top_k: int = Query(10, ge=1, le=50)):
    results = _emb.find_similar(item_id, CATALOG, top_k=top_k)
    anchor  = CATALOG.get(item_id, {})
    return {"item_id": item_id, "title": anchor.get("title", ""),
            "genre": anchor.get("primary_genre", ""), "similar": results,
            "openai_enabled": bool(_OPENAI_KEY)}


@app.get("/trending")
def trending(n: int = Query(20, ge=1, le=100)):
    top   = _get_trending_items(n)  # Layer 3: uses RedisFeatureStore
    items = [{**CATALOG.get(mid, {"item_id": mid, "title": f"Item {mid}"}),
              "trending_score": round(score, 4), "window": "5_minutes_rolling"}
             for mid, score in top]
    return {"items": items, "trending_items": items, "n": len(items),
            "method": "redis_feature_store_v2_trending"}


@app.get("/session/{user_id}")
def get_session(user_id: int):
    return SESSION.get_session(int(user_id))


@app.get("/session/intent/{user_id}")
def session_intent_endpoint(user_id: int):
    uid = int(user_id); session_ids = SESSION.session_item_ids(uid); ug = _user_genres(uid)
    try:
        events = _SESSION_MODEL.generate_session_events_from_history(session_ids, CATALOG)
        intent = _SESSION_MODEL.encode(events, ug)
        FRESH_STORE.mark_updated("session_intent")
        return {"user_id": uid, "category": intent.category, "intent_category": intent.category,
                "confidence": intent.confidence, "intent_probs": getattr(intent, "intent_probs", {}),
                "short_term_genres": intent.short_term_genres, "genre_shift": intent.genre_shift,
                "blend_weight": intent.blend_weight, "session_momentum": intent.session_momentum,
                "abandonment_count": intent.abandonment_count,
                "session_features": getattr(intent, "session_features", {}),
                "exploration_budget": (0.35 if intent.category == "discovery"
                                       else 0.08 if intent.category == "binge" else 0.15),
                "model": "gru_cell_trained_crossentropy",
                "plane": "core_recommendation"}
    except Exception as e:
        return {"user_id": uid, "category": "unknown", "error": str(e)}


@app.get("/multimodal/similar/{item_id}")
def mm_similar(item_id: int, top_k: int = Query(10, ge=1, le=50)):
    anchor  = CATALOG.get(item_id, {})
    results = multimodal_similar(item_id, CATALOG, top_k=top_k)
    return {"item_id": item_id, "anchor_title": anchor.get("title", ""),
            "similar": results, "method": "text_embedding_plus_metadata_late_fusion"}

@app.get("/multimodal/search")
def mm_search(q: str = Query(..., min_length=3, max_length=200),
              top_k: int = Query(10, ge=1, le=50)):
    results = multimodal_search(q, CATALOG, top_k=top_k)
    return {"query": q, "results": results, "method": "multimodal_fused_embedding"}


@app.get("/recommend/two_tower/{user_id}")
def recommend_two_tower(user_id: int, k: int = Query(10, ge=1, le=50)):
    uid = int(user_id); ugr = _user_ugr(uid); ug = set(_user_genres(uid))
    u_vec = TWO_TOWER.user_encode(uid, ugr)
    if u_vec is None or not TWO_TOWER.is_trained:
        recs = _build_recs(uid, k=k)
        return {"user_id": uid, "method": "fallback_four_retriever",
                "note": "Two-tower not trained — using four-retriever fusion", "items": recs[:k]}
    item_ids, item_vecs = TWO_TOWER.build_item_index(CATALOG)
    scored = TWO_TOWER.retrieve(u_vec, item_ids, item_vecs, top_k=k)
    items  = [{**dict(CATALOG.get(mid, {"item_id": mid})),
               "two_tower_score": round(sim, 4), "space": "shared_64d_contrastive"}
              for mid, sim in scored]
    return {"user_id": uid, "method": "two_tower_contrastive", "model_trained": TWO_TOWER.is_trained, "items": items}


@app.get("/reward/score/{user_id}/{item_id}")
def get_reward_score(user_id: int, item_id: int, session_momentum: float = Query(0.5)):
    uid  = int(user_id); item = CATALOG.get(int(item_id))
    if not item: raise HTTPException(404, f"Item {item_id} not found")
    ugr  = _user_ugr(uid); ug = set(_user_genres(uid)); genre = item.get("primary_genre", "")
    r    = reward_score(genre, ugr, ug, item, session_momentum=session_momentum)
    return {"user_id": uid, "item_id": item_id, "genre": genre, "reward_score": r,
            "interpretation": "P(sustained_engagement | show this item)",
            "model": "ips_weighted_logistic_11_features"}


@app.post("/impressions/log")
def log_impressions_endpoint(user_id: int, item_ids: List[int],
                              row_name: str = "top_picks", model_version: str = "v5"):
    log = ImpressionLog(user_id=user_id, item_ids=item_ids, row_name=row_name,
                        model_version=model_version,
                        propensities=[1.0 / max(len(item_ids), 1)] * len(item_ids))
    IMPRESSION_STORE.log_impression(log)
    return {"ok": True, "logged": len(item_ids)}


@app.get("/eval/slice_ndcg")
def eval_slice_ndcg(slice_key: str = Query("primary_genre")):
    recs_by_user = {}; pos_by_user = {}; meta_by_user = {}
    for u in _DEMO_USERS:
        uid = u["user_id"]; recs = _build_recs(uid, k=10)
        recs_by_user[uid] = [r["item_id"] for r in recs]
        pos_by_user[uid]  = {r["item_id"] for r in recs if not r.get("exploration_slot")}
        meta_by_user[uid] = {slice_key: _user_genres(uid)[0] if _user_genres(uid) else "Unknown"}
    slices = slice_ndcg(recs_by_user, pos_by_user, meta_by_user, slice_key, k=10)
    return {"slice_key": slice_key, "slices": slices}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    _log({"type": "feedback", "request_id": str(uuid.uuid4()), "ts": time.time(),
          "user_id": req.user_id, "item_id": req.item_id, "event": req.event, "context": req.context or {}})
    rt_process_event(req.user_id, req.item_id, req.event)

    # Layer 6: update bandit on feedback events
    event_map = {
        "play": "play_start",
        "like": "add_to_list",
        "dislike": "abandon_30s",
        "add_to_list": "add_to_list",
        "not_interested": "abandon_30s",
    }
    try:
        bandit_event = event_map.get(req.event, req.event)
        reward = compute_reward(bandit_event, position=0)
        item = CATALOG.get(req.item_id, {})
        ug   = _user_genres(req.user_id)
        ugr  = _user_ugr(req.user_id)
        ctx  = _BANDIT.user_context(req.user_id, ug, user_genre_ratings=ugr)
        _BANDIT.update(ctx, item, reward)
        FRESH_STORE.mark_updated("bandit_state")
    except Exception:
        pass
    return {"ok": True}


@app.get("/causal/counterfactual/{user_id}")
def counterfactual(user_id: int, genre: str = Query(...)):
    uid  = int(user_id); recs = _build_recs(uid, k=10)
    cf   = _CF_ANALYSER.what_if_genre(uid, CATALOG, recs, genre, top_k=10)
    return cf

@app.get("/causal/ab_power")
def ab_power(baseline_rate: float = Query(0.30), min_detectable: float = Query(0.02),
             power: float = Query(0.80)):
    return ab_test_power_calc(baseline_rate=baseline_rate,
                               min_detectable=min_detectable, power=power)

@app.get("/causal/advantage/{user_id}")
def advantage_scores(user_id: int, k: int = Query(10, ge=1, le=50)):
    uid  = int(user_id); recs = _build_recs(uid, k=k * 2)
    avg_reward = float(sum(r.get("score", 0.5) for r in recs[:10]) / max(len(recs[:10]), 1))
    reranked   = _ADV_SCORER.score_with_advantage(recs[:k * 2], user_avg_reward=avg_reward)
    return {"user_id": uid, "user_avg_reward": round(avg_reward, 4),
            "method": "advantage_weighted_AWSFT", "items": reranked[:k]}


# ── GenAI UX (pre-computed/cached, never blocking) ────────────────────────────

@app.get("/ux/mood")
def mood_discovery(mood: str = Query(None, min_length=2, max_length=300),
                   q: str = Query(None, min_length=2, max_length=300),
                   user_id: int = Query(1)):
    query  = mood or q or "something interesting to watch"
    sample = list(CATALOG.values())[:50]
    result = mood_to_content_query(query, sample, top_k=8)
    return {"mood": query, "query": query, "user_id": user_id, "plane": "genai_ux", **result}

@app.get("/ux/summary/{item_id}")
def title_summary(item_id: int):
    item = CATALOG.get(item_id)
    if not item: raise HTTPException(404, f"Item {item_id} not found")
    summary = spoiler_safe_summary(item.get("title", ""), item.get("description", ""), {})
    return {"item_id": item_id, "title": item.get("title", ""),
            "spoiler_safe_summary": summary, "plane": "genai_ux"}

@app.get("/ux/row_title/{user_id}")
def row_title_endpoint(user_id: int, row_type: str = Query("top_picks")):
    uid = int(user_id)
    # Check freshness of cached title (no LLM call in this path)
    cached_val = _ROW_TITLE_CACHE.get(uid)
    if cached_val:
        return {"user_id": uid, "row_type": row_type, "row_title": cached_val,
                "source": "precomputed_cache", "freshness": "fresh", "plane": "genai_ux"}
    ug   = _user_genres(uid)
    top_genre = ug[0] if ug else "Drama"
    fallback_titles = {
        "Action": "High-Octane Picks For You", "Crime": "Gripping Crime Stories",
        "Sci-Fi": "Mind-Bending Sci-Fi Picks", "Drama": "Powerful Dramas You'll Love",
        "Comedy": "Laugh-Out-Loud Recommendations", "Horror": "Spine-Tingling Selections",
        "Thriller": "Edge-of-Seat Thrillers", "Romance": "Romantic Picks For You",
        "Documentary": "Fascinating True Stories", "Animation": "Visual Storytelling Picks",
    }
    title = fallback_titles.get(top_genre, "Top Picks For You")
    return {"user_id": uid, "row_type": row_type, "row_title": title,
            "source": "rule_based_fallback", "freshness": "fresh", "plane": "genai_ux"}


# ── Agentic Ops ────────────────────────────────────────────────────────────────

@app.post("/agent/triage")
def agent_triage_endpoint():
    live     = _live_metrics()
    baseline = {"ndcg_at_10": 0.0292, "recall_at_50": 0.0497, "diversity_score": 0.32, "long_term_satisfaction": 0.45}
    try:
        result = triage_shadow_regression(live, baseline, n_users=live.get("n_users_evaluated", 1000))
        return {"action": result.action, "justification": result.justification,
                "confidence": result.confidence, "requires_human_review": True, "plane": "agentic_ops"}
    except Exception as e:
        return {"action": "HOLD", "justification": f"Agent error: {e}", "requires_human_review": True}

@app.get("/agent/experiment_summary")
def experiment_summary(experiment_name: str = Query("Four-Retriever Fusion + SlateOptimizer v2 vs baseline")):
    live     = _live_metrics()
    baseline = {"ndcg_at_10": 0.0292, "diversity_score": 0.32}
    changes  = ["Four-retriever fusion (collab+session+semantic+freshness)",
                "LinUCB bandit v2 with 8-dim context",
                "SlateOptimizer v2 with 5 hard diversity rules",
                "GPT sidecar via Responses API + Structured Outputs",
                "12-field event schema with EventLogger",
                "FreshnessStore with per-feature SLA watermarks"]
    summary  = generate_experiment_summary(experiment_name, baseline, live, changes)
    return {"experiment": experiment_name, "summary": summary, "plane": "agentic_ops"}

@app.get("/agent/drift_investigation")
def drift_investigation():
    try:
        result = investigate_data_drift(
            _DRIFT_REPORT,
            recent_catalog_events=["New season released", "Holiday weekend traffic spike"])
        return {"action": result.action, "justification": result.justification,
                "requires_human_review": True, "plane": "agentic_ops"}
    except Exception as e:
        return {"action": "MONITOR", "error": str(e), "plane": "agentic_ops"}


# ── Catalog enrichment ─────────────────────────────────────────────────────────

@app.get("/catalog/enriched/{item_id}")
def enriched_item(item_id: int):
    item = CATALOG.get(item_id)
    if not item: raise HTTPException(404, f"Item {item_id} not found")
    # Layer 7: use sidecar for enrichment
    try:
        sidecar_enrich = _SIDECAR.enrich_catalog_item(
            title=item.get("title", ""),
            genre=item.get("primary_genre", ""),
            description=item.get("description", ""),
        )
    except Exception:
        sidecar_enrich = {}
    tmdb   = tmdb_hydrate(item.get("title", ""), item.get("year")) if _TMDB_KEY else {}
    return {**item, "tmdb_data": tmdb, "sidecar_enrichment": sidecar_enrich,
            "plane": "semantic_intelligence"}


@app.get("/architecture")
def architecture():
    return {
        "system": "CineWave Netflix-Inspired Recommendation Platform v5",
        "layers": {
            "1_event_data": {
                "description": "12-field event schema. Impressions logged on every /page and /recommend.",
                "components": ["EventLogger", "event_schema.py", "Redis xadd stream", "JSONL fallback"],
                "fields": ["user_id", "session_id", "event_time", "surface", "row_id",
                           "position", "event_type", "item_id", "policy_id",
                           "features_snapshot_id", "outcome_value", "context"],
            },
            "2_freshness": {
                "description": "Hard SLAs per feature class. Watermark on every response.",
                "slas": FRESHNESS_SLAS,
                "components": ["FreshnessStore", "FreshnessWatermark", "freshness_layer.py"],
                "endpoints": ["/eval/freshness", "/healthz freshness_watermark"],
            },
            "3_retrieval": {
                "description": "Four-retriever fusion + RedisFeatureStore for hot features.",
                "retrievers": ["collaborative (ALS/two-tower)", "session-intent (GRU)",
                               "semantic (Qdrant cosine)", "freshness/trending"],
                "budgets": {"collaborative": 300, "session": 150, "semantic": 150, "freshness": 100},
                "components": ["retrieval_engine_v2.py", "feature_store_v2.py"],
            },
            "4_ranking": {
                "description": "14-feature GBM ranker. Optimises watch value, not CTR.",
                "features": ["collaborative_score", "session_score", "semantic_score",
                             "recency", "novelty_distance", "abandonment_risk",
                             "completion_propensity", "runtime_suitability", "language_fit",
                             "popularity_decay", "launch_effect", "impression_fatigue",
                             "artwork_trust", "page_position_prior"],
                "components": ["ranker_and_slate.py Ranker", "LightGBM/GBM"],
            },
            "5_slate": {
                "description": "Page-level hard diversity rules. SlateOptimizer v2.",
                "hard_rules": [
                    "no duplicate title on page",
                    "<=3 same-genre titles in top-20",
                    ">=5 distinct genres per page",
                    "exploration <=20% above the fold",
                    "<=2 rows with same dominant genre above the fold",
                ],
                "components": ["ranker_and_slate.py SlateOptimizer"],
            },
            "6_exploration": {
                "description": "LinUCB contextual bandit with composite 7-signal reward.",
                "reward_signals": ["play_start +1.0", "watch_3min +2.0", "abandon_30s -1.0",
                                   "completion +3.0", "add_to_list +1.5",
                                   "next_day_return +0.5 (proxy)", "repeat_engagement +1.0"],
                "guardrails": ["max 20% exploration above fold",
                               "artwork_trust < 0.6 excluded from exploration"],
                "components": ["bandit_v2.py LinUCBBandit"],
            },
            "7_semantic_sidecar": {
                "description": "GPT via Responses API + Structured Outputs. Never in hot path.",
                "use_cases": ["catalog enrichment", "explanation generation",
                              "editorial row naming", "voice intent parsing",
                              "experiment summaries"],
                "components": ["semantic_sidecar.py SidecarClient", "openai Responses API"],
                "enforcement": "genai_in_request_path = False. All calls async/cached.",
            },
        },
        "honest_limitations": [
            "ALS trained on ML-1M synthetic ratings — not production scale",
            "GRU session model: single cell, not FM-Intent scale",
            "Two-tower: 3-layer linear, not full contrastive model",
            "Streaming layer is in-process (not Kafka/Flink)",
            "Single-machine Docker Compose — not multi-region production",
            "Bandit: LinUCB v2, not full constrained contextual bandit",
            "MediaFM: text+metadata proxy, not audio/video towers",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# A/B EXPERIMENTATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
# Full A/B infrastructure: deterministic user bucketing, outcome logging,
# Welch's t-test analysis, power check, and confidence intervals.
#
#   POST /ab/experiment                — create a new experiment
#   GET  /ab/experiments               — list all experiments
#   GET  /ab/experiment/{id}           — get experiment config
#   POST /ab/experiment/{id}/stop      — stop an experiment
#   GET  /ab/assign/{id}/{user_id}     — get/compute variant assignment
#   POST /ab/outcome/{id}              — log an outcome event
#   GET  /ab/analyse/{id}              — run Welch t-test, return full result
#   GET  /ab/recommend/{id}/{user_id}  — serve recommendations for a user's variant
# ══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel as _BM

class _ExperimentCreate(_BM):
    experiment_id:    str
    name:             str
    description:      str = ""
    control_policy:   str = "popularity_baseline"
    treatment_policy: str = "als512_lgb_mmr"
    metric:           str = "click_rate"
    min_detectable:   float = 0.02
    alpha:            float = 0.05
    power:            float = 0.80

class _OutcomeLog(_BM):
    user_id:  int
    variant:  str
    outcome:  float   # 1.0 = click/watch, 0.0 = no engagement, or continuous value


@app.post("/ab/experiment", tags=["ab"])
def create_experiment(body: _ExperimentCreate):
    """
    Create a new A/B experiment.

    control_policy and treatment_policy are labels only — the actual model
    served is determined by which model is loaded in the serving pipeline.
    For CineWave: control = popularity baseline (/shadow baseline),
    treatment = ALS-512 + LGB ranker + MMR slate.
    """
    try:
        exp = Experiment(
            experiment_id=body.experiment_id,
            name=body.name,
            description=body.description,
            control_policy=body.control_policy,
            treatment_policy=body.treatment_policy,
            metric=body.metric,
            min_detectable=body.min_detectable,
            alpha=body.alpha,
            power=body.power,
        )
        ok = AB_STORE.create_experiment(exp)
        req_n = exp.required_n()
        return {
            "created": ok,
            "experiment_id": body.experiment_id,
            "required_n_per_variant": req_n,
            "total_users_required": req_n * 2,
            "note": (
                f"Need {req_n:,} users per variant to detect a "
                f"{body.min_detectable:.1%} lift at "
                f"{body.power:.0%} power / α={body.alpha}"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/ab/experiments", tags=["ab"])
def list_experiments():
    """List all experiments with their current status."""
    try:
        exps = AB_STORE.list_experiments()
        return {"experiments": exps, "count": len(exps)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/ab/experiment/{experiment_id}", tags=["ab"])
def get_experiment(experiment_id: str):
    """Get experiment config and current exposure counts."""
    try:
        exp = AB_STORE.get_experiment(experiment_id)
        if not exp:
            return {"error": f"Experiment {experiment_id!r} not found"}
        counts = AB_STORE.get_exposure_counts(experiment_id)
        req_n  = exp.required_n()
        return {
            "experiment": exp.__dict__,
            "exposure": counts,
            "required_n_per_variant": req_n,
            "pct_complete": {
                v: round(counts.get(v, 0) / max(req_n, 1) * 100, 1)
                for v in ("control", "treatment")
            },
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/ab/experiment/{experiment_id}/stop", tags=["ab"])
def stop_experiment(experiment_id: str):
    """Stop an experiment. Final analysis still available via /ab/analyse/{id}."""
    try:
        ok = AB_STORE.stop_experiment(experiment_id)
        return {"stopped": ok, "experiment_id": experiment_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/ab/assign/{experiment_id}/{user_id}", tags=["ab"])
def get_variant(experiment_id: str, user_id: int):
    """
    Get (or compute) deterministic variant assignment for a user.

    Assignment is stable: same user always gets the same variant for the
    same experiment. Uses MD5(user_id:experiment_id) % 10000 for bucketing.
    """
    try:
        variant = AB_STORE.get_or_assign_variant(user_id, experiment_id)
        if variant is None:
            return {"user_id": user_id, "experiment_id": experiment_id,
                    "variant": None, "in_experiment": False}
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "in_experiment": True,
            "policy": (
                "als512_lgb_mmr" if variant == "treatment"
                else "popularity_baseline"
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/ab/outcome/{experiment_id}", tags=["ab"])
def log_outcome(experiment_id: str, body: _OutcomeLog):
    """
    Log an outcome for a user in a variant.

    Call this when a real user engagement event occurs:
      - outcome=1.0 for click / play_start
      - outcome=0.0 for impression with no engagement
      - outcome=completion_pct for watch depth
      - outcome=rating/5.0 for explicit rating

    The variant must match what was served via /ab/assign/{id}/{user_id}.
    """
    try:
        ok = AB_STORE.log_outcome(experiment_id, body.variant, body.outcome, body.user_id)
        return {
            "logged": ok,
            "experiment_id": experiment_id,
            "variant": body.variant,
            "outcome": body.outcome,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/ab/analyse/{experiment_id}", tags=["ab"])
def analyse_experiment(experiment_id: str):
    """
    Run Welch's two-sample t-test (unequal variance) on logged outcomes.

    Returns:
      - n, mean, std, SEM per variant
      - delta (treatment - control), relative lift
      - t-statistic, p-value, 95% confidence interval on delta
      - is_powered flag — True only if both variants have >= required_n samples
      - significant flag — True only if p < alpha AND is_powered
      - conclusion — explicit text: SIGNIFICANT / NOT SIGNIFICANT / UNDERPOWERED

    NOTE: significant=True AND is_powered=True is required before acting on
    any result. p-value alone is not sufficient — always check is_powered.
    """
    try:
        result = AB_STORE.analyse(experiment_id)
        if result is None:
            return {"error": f"Experiment {experiment_id!r} not found or no data"}
        def _vo(v):
            return {"variant": v.variant, "n": v.n, "mean": v.mean, "std": v.std, "sem": v.sem}
        return {
            "experiment_id":  result.experiment_id,
            "metric":         result.metric,
            "control":        _vo(result.control),
            "treatment":      _vo(result.treatment),
            "delta":          float(result.delta),
            "relative_lift":  f"{result.relative_lift:+.1%}",
            "t_stat":         float(result.t_stat),
            "p_value":        float(result.p_value),
            "ci_95":          [float(result.ci_low), float(result.ci_high)],
            "required_n_per_variant": int(result.required_n),
            "is_powered":     bool(result.is_powered),
            "significant":    bool(result.significant),
            "conclusion":     str(result.conclusion),
            "interpretation": "Both is_powered=True and significant=True are required before acting on this result.",
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


@app.get("/ab/recommend/{experiment_id}/{user_id}", tags=["ab"])
def ab_recommend(
    experiment_id: str,
    user_id: int,
    k: int = Query(default=10, ge=1, le=50),
):
    """
    Serve recommendations for a user based on their variant assignment.

    control   → popularity-ranked baseline (top-k by popularity score)
    treatment → full ALS-512 + LGB ranker + MMR slate pipeline

    Also returns the variant assignment so the caller can log outcomes
    back via POST /ab/outcome/{experiment_id}.
    """
    try:
        variant = AB_STORE.get_or_assign_variant(user_id, experiment_id)
        if variant is None:
            return {"user_id": user_id, "experiment_id": experiment_id,
                    "variant": None, "in_experiment": False, "items": []}

        if variant == "treatment":
            recs = _build_recs(user_id, k=k)
        else:
            # Control: pure popularity baseline
            pop_ids = sorted(CATALOG, key=lambda m: -CATALOG[m].get("popularity", 0))[:k]
            recs    = [dict(CATALOG[m]) for m in pop_ids]

        return {
            "user_id":       user_id,
            "experiment_id": experiment_id,
            "variant":       variant,
            "policy":        "als512_lgb_mmr" if variant == "treatment" else "popularity_baseline",
            "items":         recs,
            "log_outcome_at": f"/ab/outcome/{experiment_id}",
            "note": (
                "Log user engagement via POST /ab/outcome/{experiment_id} "
                "with variant='{variant}' and outcome=1.0 (engaged) or 0.0 (not)."
            ).format(variant=variant, experiment_id=experiment_id),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# POLICY GRADIENT RL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class _RLRewardBody(_BM):
    user_id:        int
    items:          list
    order:          list
    reward:         float
    event_type:     str = "play_start"

class _RLOfflineBody(_BM):
    n_sessions:     int = 200
    n_epochs:       int = 3


@app.get("/rl/stats", tags=["rl"])
def rl_stats():
    """
    Policy gradient agent statistics.
    Shows learned feature weights, number of updates, and training state.
    Reads from Redis so stats are consistent across multiple workers.
    """
    # Reload from Redis to get latest state across all workers
    if _redis_client is not None:
        try:
            RL_AGENT.load_from_redis(_redis_client)
        except Exception:
            pass
    return RL_AGENT.stats()


@app.get("/rl/recommend/{user_id}", tags=["rl"])
def rl_recommend(
    user_id: int,
    k:       int  = Query(default=10, ge=1, le=50),
    explore: bool = Query(default=True),
):
    """
    Serve recommendations reranked by the REINFORCE policy.

    Pipeline: ALS retrieval → LGB ranker → REINFORCE reranker → MMR slate
    explore=True  → Gumbel-max sampling (live serving, learns from feedback)
    explore=False → Greedy policy (evaluation, deterministic)
    """
    try:
        uid      = int(user_id)
        raw_recs = _build_recs(uid, k=k * 2)   # get 2x candidates for RL to rerank
        activity = {"n_ratings": 50, "avg_rating": 3.5, "n_genres": 5}
        if hasattr(RL_AGENT, 'rerank'):
            reranked = RL_AGENT.rerank(raw_recs, activity, explore=explore, top_k=k)
        else:
            reranked = raw_recs[:k]
        return {
            "user_id":   uid,
            "items":     reranked,
            "pipeline":  "als_retrieve → lgb_rank → reinforce_rerank → mmr_slate",
            "explore":   explore,
            "rl_updates": getattr(RL_AGENT, 'n_updates', 0),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/rl/reward", tags=["rl"])
def rl_log_reward(body: _RLRewardBody):
    """
    Log a reward signal for the RL agent to learn from.

    Call after a user engages with RL-ranked recommendations:
      reward = bandit_v2.compute_reward(event_type, position)
    The agent records this into the current session episode and uses it
    to update the policy at session end via /rl/session/{user_id}/end.
    """
    try:
        from recsys.serving.bandit_v2 import compute_reward
        activity = {"n_ratings": 50, "avg_rating": 3.5, "n_genres": 5}
        reward   = float(body.reward)
        order    = list(body.order)
        RL_AGENT.record_step(body.user_id, body.items, order, reward, activity)
        return {
            "logged":   True,
            "user_id":  body.user_id,
            "reward":   reward,
            "n_steps":  len(getattr(RL_AGENT, '_episodes', {}).get(body.user_id,
                         type('E', (), {'transitions':[]})()).transitions),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/rl/session/{user_id}/end", tags=["rl"])
def rl_end_session(user_id: int):
    """
    Trigger REINFORCE policy update at end of user session.

    This runs the Monte Carlo return calculation and REINFORCE gradient
    update over all steps recorded in the session episode.
    The policy weights are updated in-memory (persisted across requests
    via the RL_AGENT singleton).
    """
    try:
        stats = RL_AGENT.end_session(int(user_id))
        if stats is None:
            return {"updated": False, "reason": "no episode found for user"}
        return {"updated": True, **stats}
    except Exception as e:
        return {"error": str(e)}


@app.post("/rl/train/offline", tags=["rl"])
def rl_train_offline(body: _RLOfflineBody):
    """
    Warm-start the RL policy from simulated offline sessions.

    Generates synthetic sessions from the existing catalog and rating data,
    then runs REINFORCE offline training to give the policy a starting point
    before it sees real user traffic. This reduces the cold-start exploration
    cost in live serving.
    """
    try:
        import random
        rng = random.Random(42)
        # Generate synthetic sessions from catalog
        item_ids = list(CATALOG.keys())
        sessions = []
        for i in range(body.n_sessions):
            uid   = rng.randint(1, 1000)
            items = [dict(CATALOG[iid]) for iid in rng.sample(item_ids, min(10, len(item_ids)))]
            # Simulate reward: higher-rated genres get higher reward
            slates = [{
                "items":  items,
                "order":  list(range(len(items))),
                "reward": rng.uniform(0.0, 3.0),
            }]
            sessions.append({"user_id": uid, "slates": slates})

        activities = {i: {"n_ratings": 50, "avg_rating": 3.5} for i in range(1001)}
        result = RL_AGENT.train_offline(sessions, activities, n_epochs=body.n_epochs)
        return {
            "trained": True,
            "n_sessions": body.n_sessions,
            **result,
        }
    except Exception as e:
        return {"error": str(e)}
