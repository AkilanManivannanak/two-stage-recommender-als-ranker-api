"""
Real-Time Event & Freshness Engine
====================================
Plane: Core Recommendation (online state — must update in seconds)

HONEST DESCRIPTION:
  This implements the conceptual interface of Netflix's real-time freshness
  infrastructure (TimeSeries abstraction, session fold-in, staleness SLAs)
  using in-process Python structures. Production-grade freshness requires:
    - Streaming event ingestion (Kafka/Flink)
    - Low-latency feature propagation (Redis with sub-second writes)
    - Per-feature staleness SLAs with watermarking
    - State circuit breakers (fall back to stale if fresh unavailable)

  This implementation uses the same data model and SLA framework but stores
  state in-process. Drop-in replacement: wire TRENDING/SESSION/LIVE_BOOSTER
  to Redis-backed equivalents for production deployment.

  Staleness SLAs (enforced in staleness_report()):
    session_intent:      <5min   (stale if not updated in 300s)
    trending_score:      <1min   (stale if not updated in 60s)
    user_genre_history:  <1hr    (stale if not updated in 3600s)
    item_embeddings:     <24hr   (stale if not rebuilt in 86400s)

  Circuit breaker: if a feature is stale beyond 2x its SLA, the
  get_with_fallback() method returns the fallback value and logs a staleness
  alert rather than silently serving stale data.

Reference:
  Netflix TimeSeries Data Abstraction Layer
  Netflix Real-Time Recommendations for Live Events
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ── Staleness SLAs (seconds) ─────────────────────────────────────────
STALENESS_SLAS = {
    "session_intent":     300,    # 5 min
    "trending_score":     60,     # 1 min
    "user_genre_history": 3600,   # 1 hr
    "item_embeddings":    86400,  # 24 hr
    "exploration_budget": 300,    # 5 min
    "page_cache":         120,    # 2 min
}
CIRCUIT_BREAK_MULTIPLIER = 2.0   # circuit-break at 2x SLA


@dataclass
class FeatureValue:
    value:      Any
    written_at: float = field(default_factory=time.time)
    feature_type: str = "unknown"

    def age_seconds(self) -> float:
        return time.time() - self.written_at

    def is_stale(self) -> bool:
        sla = STALENESS_SLAS.get(self.feature_type, 3600)
        return self.age_seconds() > sla

    def is_circuit_broken(self) -> bool:
        sla = STALENESS_SLAS.get(self.feature_type, 3600)
        return self.age_seconds() > sla * CIRCUIT_BREAK_MULTIPLIER


class StalenessMonitor:
    """
    Per-feature staleness tracking with circuit-breaker pattern.
    In production: integrated with Redis key TTL monitoring + alerting.
    """
    def __init__(self):
        self._store: dict[str, FeatureValue] = {}
        self._alerts: list[dict] = []

    def write(self, key: str, value: Any, feature_type: str = "unknown"):
        self._store[key] = FeatureValue(value=value, feature_type=feature_type)

    def get_with_fallback(self, key: str, fallback: Any = None) -> tuple[Any, bool]:
        """Returns (value, is_stale). Uses fallback if circuit-broken."""
        fv = self._store.get(key)
        if fv is None:
            return fallback, True
        if fv.is_circuit_broken():
            self._alerts.append({
                "key": key, "age_s": round(fv.age_seconds(),1),
                "sla_s": STALENESS_SLAS.get(fv.feature_type,3600),
                "action": "CIRCUIT_BREAK — serving fallback"
            })
            return fallback, True
        return fv.value, fv.is_stale()

    def staleness_report(self) -> dict:
        now = time.time()
        report = {"features": {}, "alerts": self._alerts[-10:], "slas": STALENESS_SLAS}
        stale_count = 0
        for key, fv in self._store.items():
            stale = fv.is_stale(); broken = fv.is_circuit_broken()
            if stale: stale_count += 1
            report["features"][key] = {
                "age_s":     round(fv.age_seconds(), 1),
                "stale":     stale,
                "broken":    broken,
                "sla_s":     STALENESS_SLAS.get(fv.feature_type, 3600),
            }
        report["summary"] = {
            "total": len(self._store),
            "stale": stale_count,
            "staleness_pct": round(stale_count / max(len(self._store),1), 3),
        }
        self._alerts.clear()
        return report


# ── Event types ───────────────────────────────────────────────────────
EVENT_PLAY       = "play"
EVENT_LIKE       = "like"
EVENT_DISLIKE    = "dislike"
EVENT_IMPRESSION = "impression"
EVENT_ABANDON    = "abandon"
EVENT_SEARCH     = "search"

EVENT_WEIGHTS = {
    EVENT_PLAY:       1.0,
    EVENT_LIKE:       0.8,
    EVENT_IMPRESSION: 0.1,
    EVENT_ABANDON:   -0.3,
    EVENT_DISLIKE:   -1.0,
}


class TrendingWindow:
    """
    Rolling-window trending score per item.
    Score = exponentially decayed event weight sum over last `window_seconds`.
    Decay half-life = window/4 so recent events dominate.
    Staleness SLA: 60 seconds (trending score must be < 1 min old).
    """
    def __init__(self, window_seconds: int = 300):
        self.window = window_seconds
        self._events: dict[int, deque[tuple[float,float]]] = defaultdict(deque)
        self._last_update: dict[int, float] = {}

    def record(self, item_id: int, event: str):
        w  = EVENT_WEIGHTS.get(event, 0.1)
        ts = time.time()
        self._events[item_id].append((ts, w))
        self._last_update[item_id] = ts

    def score(self, item_id: int) -> float:
        now    = time.time()
        cutoff = now - self.window
        buf    = self._events[item_id]
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        if not buf: return 0.0
        total = 0.0
        for ts, w in buf:
            decay  = 2.0 ** (-(now-ts) / (self.window/4))
            total += w * decay
        return float(total)

    def is_stale(self, item_id: int) -> bool:
        last = self._last_update.get(item_id)
        if last is None: return True
        return time.time() - last > STALENESS_SLAS["trending_score"]

    def top_trending(self, n: int = 50) -> list[tuple[int, float]]:
        scores = [(iid, self.score(iid)) for iid in self._events]
        scores.sort(key=lambda x: -x[1])
        return scores[:n]


class SessionState:
    """
    Per-user session state with staleness watermarking.
    SLA: session_intent must be updated within 5 minutes.
    Production: Redis hash per session with 30-min TTL.
    """
    def __init__(self, timeout: int = 1800):
        self.timeout = timeout
        self._sessions: dict[int, dict] = {}
        self._monitor = StalenessMonitor()

    def record_event(self, user_id: int, item_id: int, event: str):
        now = time.time()
        if user_id not in self._sessions:
            self._sessions[user_id] = {
                "started_at": now, "last_active": now,
                "items_seen": [], "items_played": [],
                "genres_seen": [], "search_queries": [],
            }
        s = self._sessions[user_id]
        s["last_active"] = now
        if item_id and item_id not in s["items_seen"]:
            s["items_seen"].append(item_id)
        if event == EVENT_PLAY:
            s["items_played"].append(item_id)
        # Write intent feature with SLA watermark
        self._monitor.write(
            f"session:{user_id}:last_event", event, "session_intent")

    def get_session(self, user_id: int) -> dict:
        s = self._sessions.get(user_id)
        if s is None:
            return {"items_seen":[],"items_played":[],"is_new_session":True}
        if time.time() - s["last_active"] > self.timeout:
            del self._sessions[user_id]
            return {"items_seen":[],"items_played":[],"is_new_session":True}
        return {**s, "is_new_session": False,
                "session_length_s": round(time.time()-s["started_at"],1)}

    def session_item_ids(self, user_id: int) -> list[int]:
        return self.get_session(user_id).get("items_seen", [])

    def staleness_report(self) -> dict:
        return self._monitor.staleness_report()


class LiveEventBooster:
    """
    Urgency boost for live/premiering content.
    Boost decays: >24hr away=1.0, 1-24hr=1.2, <1hr=2.0, during=1.6, ended=1.0.
    SLA: live event state must be < 60s stale (real-time critical).
    Reference: Netflix real-time live event recommendations.
    """
    def __init__(self):
        self._events: dict[int, dict] = {}

    def register(self, item_id: int, start_ts: float,
                 kind: str = "premiere", boost: float = 2.0):
        self._events[item_id] = {"start_ts":start_ts,"kind":kind,"boost":boost}

    def get_boost(self, item_id: int) -> float:
        ev = self._events.get(item_id)
        if ev is None: return 1.0
        delta = ev["start_ts"] - time.time()
        if delta > 86400:  return 1.0
        if delta > 3600:   return 1.2
        if delta > 0:      return ev["boost"]
        if delta > -7200:  return ev["boost"] * 0.8
        return 1.0

    def apply_boosts(self, candidates: list[dict]) -> list[dict]:
        result = []
        for c in candidates:
            mid   = c.get("item_id", c.get("movieId",0))
            boost = self.get_boost(mid)
            if boost != 1.0:
                c = dict(c)
                c["live_boost"]      = round(boost,2)
                c["final_score"]     = round(
                    c.get("final_score",c.get("ranker_score",0.5))*boost, 4)
                c["is_live_event"]   = True
            result.append(c)
        result.sort(key=lambda x: -x.get("final_score",x.get("ranker_score",0)))
        return result


# ── Singleton instances ───────────────────────────────────────────────
TRENDING     = TrendingWindow(window_seconds=300)
SESSION      = SessionState(timeout=1800)
LIVE_BOOSTER = LiveEventBooster()
_STALE_MON   = StalenessMonitor()


def process_event(user_id: int, item_id: int, event: str, metadata: dict = None):
    """Single entry point for all incoming events. Updates all state."""
    TRENDING.record(item_id, event)
    SESSION.record_event(user_id, item_id, event)
    _STALE_MON.write(f"user:{user_id}:last_event", event, "session_intent")
    try:
        from recsys.serving.feature_store import FEATURE_STORE
        FEATURE_STORE.on_user_event(user_id, item_id, event)
    except Exception:
        pass


def get_staleness_report() -> dict:
    """Unified staleness report across all real-time components."""
    return {
        "session_store":     SESSION.staleness_report(),
        "staleness_monitor": _STALE_MON.staleness_report(),
        "trending_sampled": {
            iid: {"score": round(TRENDING.score(iid),4),
                  "stale": TRENDING.is_stale(iid)}
            for iid in list(TRENDING._events.keys())[:10]
        },
        "honest_note": (
            "In-process state. Production requires: Redis with sub-second writes, "
            "Kafka event ingestion, per-feature TTL enforcement, "
            "circuit breakers with fallback to stale-but-available features."
        ),
    }
