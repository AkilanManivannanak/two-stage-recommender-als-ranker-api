"""
ML Feature Store  (Fact Store pattern)
=======================================
Missing piece #2 from review: feature-store / fact-store.

Netflix's ML Fact Store paper describes a system that:
  - Stores pre-computed features for users and items
  - Serves them at low latency to the recommendation model
  - Supports both batch (offline training) and online (real-time serving) reads
  - Tracks feature lineage and staleness

Reference: https://netflixtechblog.com/evolution-of-ml-fact-store-5941d3231762

What this implements:
  - In-memory key-value store with TTL (staleness tracking)
  - User features: interaction counts, avg rating, genre affinities, recency
  - Item features: popularity, avg rating, age, trending score
  - Real-time update method (called when new events arrive)
  - Staleness monitor: flags features older than threshold
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


FEATURE_TTL_SECONDS = {
    "user_realtime":   300,      # 5 min — interaction counts, session
    "user_daily":      86400,    # 1 day — genre affinities, avg rating
    "item_realtime":   60,       # 1 min — trending score, impression count
    "item_daily":      86400,    # 1 day — avg rating, popularity
}


@dataclass
class FeatureRecord:
    key:        str
    features:   dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    feature_type: str = "unknown"

    def age_seconds(self) -> float:
        return time.time() - self.updated_at

    def is_stale(self) -> bool:
        ttl = FEATURE_TTL_SECONDS.get(self.feature_type, 3600)
        return self.age_seconds() > ttl


class FeatureStore:
    """
    Lightweight in-memory feature store.
    Production replacement: Redis + Apache Kafka for real-time updates,
    or Netflix's internal Fact Store backed by Cassandra + Flink.
    """

    def __init__(self):
        self._store: dict[str, FeatureRecord] = {}
        self._staleness_alerts: list[str]     = []

    # ── Write ──────────────────────────────────────────────────────
    def write_user_features(self, user_id: int, features: dict, realtime: bool = False):
        key  = f"user:{user_id}"
        ftype= "user_realtime" if realtime else "user_daily"
        rec  = self._store.get(key)
        if rec:
            rec.features.update(features)
            rec.updated_at    = time.time()
            rec.feature_type  = ftype
        else:
            self._store[key] = FeatureRecord(key, features, feature_type=ftype)

    def write_item_features(self, item_id: int, features: dict, realtime: bool = False):
        key  = f"item:{item_id}"
        ftype= "item_realtime" if realtime else "item_daily"
        rec  = self._store.get(key)
        if rec:
            rec.features.update(features)
            rec.updated_at   = time.time()
            rec.feature_type = ftype
        else:
            self._store[key] = FeatureRecord(key, features, feature_type=ftype)

    # ── Read ───────────────────────────────────────────────────────
    def get_user_features(self, user_id: int) -> dict[str, Any]:
        rec = self._store.get(f"user:{user_id}")
        if rec is None:
            return self._default_user_features(user_id)
        if rec.is_stale():
            self._staleness_alerts.append(f"user:{user_id} stale by {rec.age_seconds():.0f}s")
        return rec.features

    def get_item_features(self, item_id: int) -> dict[str, Any]:
        rec = self._store.get(f"item:{item_id}")
        if rec is None:
            return self._default_item_features(item_id)
        if rec.is_stale():
            self._staleness_alerts.append(f"item:{item_id} stale by {rec.age_seconds():.0f}s")
        return rec.features

    # ── Real-time event update ─────────────────────────────────────
    def on_user_event(self, user_id: int, item_id: int, event: str):
        """
        Update user features in real-time when a play/like/dislike event arrives.
        In production: consumed from a Kafka topic.
        """
        feats = self.get_user_features(user_id)
        feats["last_event"]       = event
        feats["last_event_item"]  = item_id
        feats["last_active_ts"]   = time.time()
        feats["session_cnt"]      = feats.get("session_cnt", 0) + 1
        if event == "play":
            feats["play_cnt"]     = feats.get("play_cnt", 0) + 1
        elif event == "like":
            feats["like_cnt"]     = feats.get("like_cnt", 0) + 1
        self.write_user_features(user_id, feats, realtime=True)

        # Update item impression count
        item_feats = self.get_item_features(item_id)
        item_feats["impression_cnt"] = item_feats.get("impression_cnt", 0) + 1
        if event == "play":
            item_feats["play_cnt"]   = item_feats.get("play_cnt", 0) + 1
            item_feats["trending_score"] = item_feats.get("trending_score", 0) + 1.0
        self.write_item_features(item_id, item_feats, realtime=True)

    # ── Staleness report ───────────────────────────────────────────
    def staleness_report(self) -> dict:
        stale_users = sum(1 for k,v in self._store.items()
                          if k.startswith("user:") and v.is_stale())
        stale_items = sum(1 for k,v in self._store.items()
                          if k.startswith("item:") and v.is_stale())
        total       = len(self._store)
        alerts      = self._staleness_alerts[-10:]  # last 10
        self._staleness_alerts.clear()
        return {
            "total_records":    total,
            "stale_user_recs":  stale_users,
            "stale_item_recs":  stale_items,
            "staleness_pct":    round((stale_users+stale_items)/max(total,1),3),
            "recent_alerts":    alerts,
            "ttls":             FEATURE_TTL_SECONDS,
        }

    def _default_user_features(self, user_id: int) -> dict:
        import numpy as np
        rng = np.random.default_rng(user_id*17)
        return {
            "user_cnt_total":    int(rng.integers(5, 500)),
            "user_cnt_7d":       int(rng.integers(0, 30)),
            "user_cnt_30d":      int(rng.integers(0, 100)),
            "user_avg_rating":   round(float(rng.uniform(2.8,4.8)),2),
            "user_tenure_days":  int(rng.integers(1,1000)),
            "user_recency_days": int(rng.integers(0,30)),
            "session_cnt":       0,
            "play_cnt":          0,
            "like_cnt":          0,
            "is_cold_start":     True,
        }

    def _default_item_features(self, item_id: int) -> dict:
        import numpy as np
        rng = np.random.default_rng(item_id*41)
        return {
            "item_cnt_total":    int(rng.integers(10,5000)),
            "item_cnt_7d":       int(rng.integers(0,200)),
            "item_cnt_30d":      int(rng.integers(0,500)),
            "item_avg_rating":   round(float(rng.uniform(2.5,5.0)),2),
            "item_age_days":     int(rng.integers(30,3000)),
            "item_recency_days": int(rng.integers(0,60)),
            "impression_cnt":    0,
            "play_cnt":          0,
            "trending_score":    0.0,
        }

# Singleton feature store (shared across requests)
FEATURE_STORE = FeatureStore()
