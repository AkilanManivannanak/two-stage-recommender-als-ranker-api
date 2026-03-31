"""
Redis Feature Store — Phase 2
==============================
Hot features served from Redis. Cold fallback to file-based store.

Feature classes:
  user_profile      : long-term genre/language/runtime preferences
  session_state     : last N actions in current session (GRU input)
  trending_scores   : item trending scores updated every 15s
  bandit_state      : per-arm LinUCB parameters
  page_cache        : assembled page for user (TTL 30s)

Every feature read returns (value, age_seconds, is_stale).
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional, Tuple


class RedisFeatureStore:
    """
    Two-tier: Redis (hot path, <2ms) → in-process dict (fallback).
    Never throws — always returns a value, possibly stale.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: dict[str, tuple[Any, float]] = {}  # key → (value, written_at)

    # ── User Profile ──────────────────────────────────────────────────────────

    def get_user_profile(self, user_id: int) -> Tuple[dict, float, bool]:
        """Returns (profile_dict, age_seconds, is_stale). SLA = 300s."""
        key = f"user:profile:{user_id}"
        return self._get_json(key, sla=300)

    def set_user_profile(self, user_id: int, profile: dict) -> None:
        key = f"user:profile:{user_id}"
        self._set_json(key, profile, ttl=600)

    # ── Session State ─────────────────────────────────────────────────────────

    def get_session_state(self, session_id: str) -> Tuple[dict, float, bool]:
        """Returns (state_dict, age_seconds, is_stale). SLA = 60s."""
        key = f"session:{session_id}"
        return self._get_json(key, sla=60)

    def push_session_event(self, session_id: str, event: dict) -> None:
        """Append to session event list. Trim to last 50 events."""
        key = f"session:events:{session_id}"
        if self._redis:
            try:
                self._redis.lpush(key, json.dumps(event))
                self._redis.ltrim(key, 0, 49)
                self._redis.expire(key, 3600)
                return
            except Exception:
                pass
        # Local fallback
        events = self._local.get(key, ([], time.time()))[0]
        events = [event] + events[:49]
        self._local[key] = (events, time.time())

    def get_session_events(self, session_id: str, n: int = 20) -> list:
        key = f"session:events:{session_id}"
        if self._redis:
            try:
                raw = self._redis.lrange(key, 0, n - 1)
                return [json.loads(r) for r in raw]
            except Exception:
                pass
        local = self._local.get(key, ([], time.time()))[0]
        return local[:n]

    # ── Trending Scores ───────────────────────────────────────────────────────

    def get_trending_scores(self, n: int = 100) -> Tuple[list, float, bool]:
        """Returns ([(item_id, score)], age_seconds, is_stale). SLA = 15s."""
        key = "trending:scores"
        value, age, stale = self._get_json(key, sla=15)
        if value is None:
            return [], age, True
        return value, age, stale

    def set_trending_scores(self, scores: list[tuple]) -> None:
        """scores: [(item_id, score), ...]"""
        key = "trending:scores"
        self._set_json(key, [(int(i), float(s)) for i, s in scores], ttl=30)

    def get_top_trending(self, n: int = 100) -> list:
        scores, _, _ = self.get_trending_scores(n)
        return sorted(scores, key=lambda x: -x[1])[:n] if scores else []

    # ── Bandit State ──────────────────────────────────────────────────────────

    def get_bandit_state(self) -> Tuple[dict, float, bool]:
        key = "bandit:state"
        return self._get_json(key, sla=3600)

    def set_bandit_state(self, state: dict) -> None:
        key = "bandit:state"
        self._set_json(key, state, ttl=7200)

    # ── Page Cache ────────────────────────────────────────────────────────────

    def get_page_cache(self, user_id: int) -> Tuple[Optional[dict], float, bool]:
        key = f"page:cache:{user_id}"
        return self._get_json(key, sla=30)

    def set_page_cache(self, user_id: int, page: dict) -> None:
        key = f"page:cache:{user_id}"
        self._set_json(key, page, ttl=30)

    def invalidate_page_cache(self, user_id: int) -> None:
        key = f"page:cache:{user_id}"
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._local.pop(key, None)

    # ── Exploration Budget ─────────────────────────────────────────────────────

    def get_exploration_budget(self, user_id: int) -> float:
        key = f"explore:budget:{user_id}"
        val, _, _ = self._get_json(key, sla=3600)
        return float(val) if val is not None else 0.15

    def set_exploration_budget(self, user_id: int, budget: float) -> None:
        key = f"explore:budget:{user_id}"
        self._set_json(key, budget, ttl=3600)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_json(self, key: str, sla: float) -> Tuple[Any, float, bool]:
        """Returns (value, age_seconds, is_stale). Never raises."""
        now = time.time()

        # Try Redis
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    obj = json.loads(raw)
                    # Redis TTL does not tell us age; use a companion timestamp key
                    ts_raw = self._redis.get(f"ts:{key}")
                    ts = float(ts_raw) if ts_raw else now
                    age = now - ts
                    return obj["v"], age, age > sla
            except Exception:
                pass

        # Local fallback
        if key in self._local:
            val, written_at = self._local[key]
            age = now - written_at
            return val, age, age > sla

        return None, now, True   # no value found → stale

    def _set_json(self, key: str, value: Any, ttl: int = 300) -> None:
        now = time.time()
        self._local[key] = (value, now)
        if self._redis:
            try:
                payload = json.dumps({"v": value})
                self._redis.set(key, payload, ex=ttl)
                self._redis.set(f"ts:{key}", now, ex=ttl)
            except Exception:
                pass

    def health(self) -> dict:
        if self._redis:
            try:
                self._redis.ping()
                return {"redis": "ok", "fallback": "in-process"}
            except Exception:
                pass
        return {"redis": "unavailable", "fallback": "in-process"}


# Module-level singleton
REDIS_FEATURE_STORE = RedisFeatureStore()
