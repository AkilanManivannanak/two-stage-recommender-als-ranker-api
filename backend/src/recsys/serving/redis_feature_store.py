"""
Redis Feature Store  —  Hot Feature Serving
=============================================
Plane: Core Recommendation (online feature plane)

Replaces the in-memory FeatureStore with a Redis-backed version
for serving hot features at <5ms latency.

Features stored in Redis (with TTLs per spec):
  session_intent:      300s  — user's current session intent
  trending_score:      60s   — rolling trending score per item
  user_genre_history:  3600s — aggregated genre affinities
  exploration_budget:  300s  — contextual bandit exploration rate
  page_cache:          120s  — assembled page cache

Redis key schema:
  user:{uid}:session_intent    → JSON
  user:{uid}:genres            → JSON list
  user:{uid}:exploration_budget → float
  item:{iid}:trending_score    → float
  page:{uid}:assembled         → JSON page

Fallback: if Redis unavailable, falls back to in-memory FeatureStore.
Redis loss is recoverable: warmup.py rebuilds from Postgres on restart.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Feature TTLs (seconds) — from spec
FEATURE_TTLS = {
    "session_intent":     300,     # fresh < 60s, warn 60-180s, stale > 180s
    "trending_score":     15,      # fresh < 15s, warn 15-60s, stale > 60s
    "item_embeddings":    86400,   # fresh < 24h
    "page_cache":         30,      # fresh < 30s, stale > 90s
    "user_genres":        3600,    # 1 hour
    "exploration_budget": 300,     # 5 minutes
}


class RedisFeatureStore:
    """
    Redis-backed hot feature store.
    Graceful fallback to in-memory store if Redis unavailable.
    """

    def __init__(self, redis_url: str = _REDIS_URL):
        self._url = redis_url
        self._redis = None
        self._available = False
        self._connect()

        # In-memory fallback
        self._memory: dict[str, tuple[Any, float]] = {}  # key → (value, written_at)

    def _connect(self):
        try:
            import redis as r
            self._redis = r.from_url(self._url, decode_responses=True,
                                     socket_connect_timeout=1,
                                     socket_timeout=0.5)
            self._redis.ping()
            self._available = True
        except Exception as e:
            self._available = False

    def _safe_get(self, key: str) -> Optional[str]:
        if self._available and self._redis:
            try:
                return self._redis.get(key)
            except Exception:
                self._available = False
        # Fallback memory
        entry = self._memory.get(key)
        if entry:
            val, written_at = entry
            return json.dumps(val) if not isinstance(val, str) else val
        return None

    def _safe_set(self, key: str, value: str, ttl: int):
        if self._available and self._redis:
            try:
                self._redis.setex(key, ttl, value)
                return
            except Exception:
                self._available = False
        # Fallback memory
        self._memory[key] = (value, time.time())

    # ── User features ───────────────────────────────────────────────

    def get_user_features(self, user_id: int) -> dict:
        key = f"user:{user_id}:features"
        raw = self._safe_get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def set_user_features(self, user_id: int, features: dict, ttl: int = 3600):
        key = f"user:{user_id}:features"
        self._safe_set(key, json.dumps(features), ttl)

    def get_user_session_intent(self, user_id: int) -> dict:
        key = f"user:{user_id}:session_intent"
        raw = self._safe_get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {"intent": "unknown", "blend_weight": 0.3, "genres": []}

    def set_user_session_intent(self, user_id: int, intent: dict,
                                 ttl: int = FEATURE_TTLS["session_intent"]):
        key = f"user:{user_id}:session_intent"
        self._safe_set(key, json.dumps(intent), ttl)

    def get_user_exploration_budget(self, user_id: int) -> float:
        key = f"user:{user_id}:exploration_budget"
        raw = self._safe_get(key)
        if raw:
            try:
                return float(raw)
            except Exception:
                pass
        return 0.15  # default

    def set_user_exploration_budget(self, user_id: int, budget: float,
                                     ttl: int = FEATURE_TTLS["exploration_budget"]):
        key = f"user:{user_id}:exploration_budget"
        self._safe_set(key, str(budget), ttl)

    # ── Item features ────────────────────────────────────────────────

    def get_item_trending_score(self, item_id: int) -> float:
        key = f"item:{item_id}:trending"
        raw = self._safe_get(key)
        if raw:
            try:
                return float(raw)
            except Exception:
                pass
        return 0.0

    def set_item_trending_score(self, item_id: int, score: float,
                                 ttl: int = FEATURE_TTLS["trending_score"]):
        key = f"item:{item_id}:trending"
        self._safe_set(key, str(score), ttl)

    # ── Page cache ───────────────────────────────────────────────────

    def get_page_cache(self, user_id: int) -> Optional[dict]:
        key = f"page:{user_id}:assembled"
        raw = self._safe_get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    def set_page_cache(self, user_id: int, page: dict,
                       ttl: int = FEATURE_TTLS["page_cache"]):
        key = f"page:{user_id}:assembled"
        try:
            self._safe_set(key, json.dumps(page), ttl)
        except Exception:
            pass  # page may be too large to cache

    # ── Batch warm-up (called on startup) ────────────────────────────

    def warm_demo_users(self, demo_users: list[dict]):
        """Pre-warm hot features for demo users on startup."""
        if not self._available:
            return
        for u in demo_users:
            uid = u["user_id"]
            self.set_user_session_intent(uid, {
                "intent": "unknown",
                "blend_weight": 0.3,
                "genres": [],
            })
            self.set_user_exploration_budget(uid, 0.15)
            self.set_user_features(uid, {
                "user_id": uid,
                "genres": [],
                "genre_ratings": {},
            })

    def status(self) -> dict:
        return {
            "redis_available": self._available,
            "redis_url": self._url.split("@")[-1],  # hide credentials
            "memory_fallback_keys": len(self._memory),
            "feature_ttls": FEATURE_TTLS,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
REDIS_FEATURE_STORE = RedisFeatureStore()
