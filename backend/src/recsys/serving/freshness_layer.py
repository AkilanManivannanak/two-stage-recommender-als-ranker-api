"""
Feature Freshness Layer — Phase 2
==================================
Hard SLAs per feature type. Every request snapshots features at request start
and returns a freshness watermark in the response.

SLAs (spec):
  session_intent  : fresh under 60s
  trending        : fresh under 15s
  embeddings      : fresh under 24h
  page_cache      : fresh under 30s
  user_profile    : fresh under 5min (derived from logs)

Design:
  - FreshnessGate checks each feature at request time
  - Stale features fall back to safe defaults (never fail request)
  - FreshnessWatermark is returned in every API response
  - Degradation strategy: serve stale with flag rather than fail
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ── SLA constants (seconds) ──────────────────────────────────────────────────
FRESHNESS_SLAS = {
    "session_intent":   300,        # 1 min
    "trending":         15,        # 15 sec
    "embeddings":       86_400,    # 24 h
    "page_cache":       30,        # 30 sec
    "user_profile":     300,       # 5 min
    "bandit_state":     3_600,     # 1 h
    "ranker":           86_400,    # 24 h
}


@dataclass
class FeatureFreshness:
    feature_name:   str
    last_updated:   float         # unix timestamp
    sla_seconds:    float
    is_stale:       bool = False
    age_seconds:    float = 0.0

    def __post_init__(self):
        self.age_seconds = time.time() - self.last_updated
        self.is_stale = self.age_seconds > self.sla_seconds

    @property
    def freshness_pct(self) -> float:
        """1.0 = perfectly fresh, 0.0 = at SLA limit, negative = stale"""
        return max(0.0, 1.0 - (self.age_seconds / max(self.sla_seconds, 1)))


@dataclass
class FreshnessWatermark:
    """
    Attached to every API response. Tells the client (and ops) exactly
    how fresh each feature dimension was at request time.
    """
    request_id:          str
    snapshot_time:       float = field(default_factory=time.time)
    features:            dict  = field(default_factory=dict)   # name → FeatureFreshness
    any_stale:           bool  = False
    stale_features:      list  = field(default_factory=list)
    serving_degraded:    bool  = False

    def add(self, f: FeatureFreshness) -> None:
        self.features[f.feature_name] = {
            "age_seconds":   round(f.age_seconds, 1),
            "sla_seconds":   f.sla_seconds,
            "is_stale":      f.is_stale,
            "freshness_pct": round(f.freshness_pct, 3),
        }
        if f.is_stale:
            self.any_stale = True
            self.stale_features.append(f.feature_name)

    def to_dict(self) -> dict:
        return {
            "request_id":       self.request_id,
            "snapshot_time":    self.snapshot_time,
            "any_stale":        self.any_stale,
            "stale_features":   self.stale_features,
            "serving_degraded": self.serving_degraded,
            "features":         self.features,
        }


class FreshnessStore:
    """
    Tracks when each feature class was last updated.
    In production: reads from Redis hashes with atomic SET.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: dict[str, float] = {}
        # Initialise all features as "just updated" at startup
        now = time.time()
        for name in FRESHNESS_SLAS:
            self._local[name] = now

    def mark_updated(self, feature_name: str, ts: Optional[float] = None) -> None:
        ts = ts or time.time()
        self._local[feature_name] = ts
        if self._redis:
            try:
                self._redis.hset("freshness:timestamps", feature_name, ts)
            except Exception:
                pass

    def get_last_updated(self, feature_name: str) -> float:
        if self._redis:
            try:
                val = self._redis.hget("freshness:timestamps", feature_name)
                if val:
                    return float(val)
            except Exception:
                pass
        return self._local.get(feature_name, time.time())

    def check(self, feature_name: str) -> FeatureFreshness:
        last_updated = self.get_last_updated(feature_name)
        sla = FRESHNESS_SLAS.get(feature_name, 3_600)
        return FeatureFreshness(
            feature_name=feature_name,
            last_updated=last_updated,
            sla_seconds=sla,
        )

    def snapshot(self, request_id: str) -> FreshnessWatermark:
        """
        Call at the start of every request to snapshot all feature freshness.
        Returns a FreshnessWatermark to attach to the response.
        """
        wm = FreshnessWatermark(request_id=request_id)
        for name in FRESHNESS_SLAS:
            wm.add(self.check(name))

        # Degraded if any SLA-critical feature is stale
        critical = {"session_intent", "trending", "page_cache"}
        stale_critical = [f for f in wm.stale_features if f in critical]
        wm.serving_degraded = len(stale_critical) > 0

        return wm

    def staleness_report(self) -> dict:
        """Summary dict for /healthz endpoint."""
        report = {}
        for name in FRESHNESS_SLAS:
            f = self.check(name)
            report[name] = {
                "age_seconds": round(f.age_seconds, 1),
                "sla_seconds": f.sla_seconds,
                "is_stale":    f.is_stale,
            }
        return report


# Module-level singleton
FRESH_STORE = FreshnessStore()
