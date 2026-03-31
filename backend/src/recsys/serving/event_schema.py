"""
Event Schema — Phase 1: Real Logged Behavior
=============================================
Full 12-field event schema for honest OPE, page-level analysis, and debugging.

Fields:
  user_id              — stable user identifier
  session_id           — browser/app session (reset on 30min inactivity)
  event_time           — UTC unix timestamp (ms precision)
  surface              — home | search | browse | detail | trailer
  row_id               — row label (e.g. "top_picks", "because_you_watched_X")
  position             — 0-indexed position within the row (for position bias)
  event_type           — impression | click | trailer_start | play_start |
                         abandon_30s | watch_3min | completion | add_to_list |
                         search_query | voice_query | remove_from_list
  item_id              — catalog item identifier
  policy_id            — model version that generated this recommendation
  features_snapshot_id — pointer to the feature state at request time
  outcome_value        — float: completion_pct, dwell_ms, rating, etc.
  context              — JSON blob: device, country, time_of_day, etc.

Without impressions + positions you cannot do honest IPS-NDCG or page simulation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import json


class EventType(str, Enum):
    IMPRESSION       = "impression"
    CLICK            = "click"
    TRAILER_START    = "trailer_start"
    PLAY_START       = "play_start"
    ABANDON_30S      = "abandon_30s"
    WATCH_3MIN       = "watch_3min"
    COMPLETION       = "completion"
    ADD_TO_LIST      = "add_to_list"
    REMOVE_FROM_LIST = "remove_from_list"
    SEARCH_QUERY     = "search_query"
    VOICE_QUERY      = "voice_query"
    PAGE_VIEW        = "page_view"
    SCROLL           = "scroll"


class Surface(str, Enum):
    HOME    = "home"
    SEARCH  = "search"
    BROWSE  = "browse"
    DETAIL  = "detail"
    TRAILER = "trailer"
    VOICE   = "voice"


@dataclass
class Event:
    """
    Canonical event record. One row in your event store.
    Satisfies the spec: user_id, session_id, event_time, surface, row_id,
    position, event_type, item_id, policy_id, features_snapshot_id, outcome_value.
    """
    user_id:              int
    session_id:           str
    event_time:           float          = field(default_factory=time.time)
    surface:              str            = Surface.HOME
    row_id:               str            = "unknown"
    position:             int            = -1          # -1 = unknown
    event_type:           str            = EventType.IMPRESSION
    item_id:              int            = 0
    policy_id:            str            = "unknown"
    features_snapshot_id: str            = ""
    outcome_value:        float          = 0.0
    context:              dict           = field(default_factory=dict)
    event_id:             str            = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def impression(
        cls,
        user_id: int,
        session_id: str,
        item_id: int,
        row_id: str,
        position: int,
        policy_id: str,
        features_snapshot_id: str,
        surface: str = Surface.HOME,
    ) -> "Event":
        return cls(
            user_id=user_id,
            session_id=session_id,
            surface=surface,
            row_id=row_id,
            position=position,
            event_type=EventType.IMPRESSION,
            item_id=item_id,
            policy_id=policy_id,
            features_snapshot_id=features_snapshot_id,
        )

    @classmethod
    def play_start(
        cls,
        user_id: int,
        session_id: str,
        item_id: int,
        policy_id: str,
        features_snapshot_id: str,
    ) -> "Event":
        return cls(
            user_id=user_id,
            session_id=session_id,
            event_type=EventType.PLAY_START,
            item_id=item_id,
            policy_id=policy_id,
            features_snapshot_id=features_snapshot_id,
        )


class EventLogger:
    """
    Thin event logger. Writes to Redis stream (hot path) and JSONL file (fallback).
    In production: replace with Kafka producer.
    """

    def __init__(self, redis_client=None, fallback_path: str = "logs/events.jsonl"):
        self._redis = redis_client
        self._fallback_path = fallback_path
        self._buffer: list[dict] = []
        self._buffer_limit = 500
        self._file_handle = None

    def log(self, event: Event) -> None:
        d = event.to_dict()
        written = False

        # Try Redis stream first
        if self._redis:
            try:
                self._redis.xadd("events:stream", d, maxlen=100_000, approximate=True)
                written = True
            except Exception:
                pass

        # File fallback
        if not written:
            self._buffer.append(d)
            if len(self._buffer) >= self._buffer_limit:
                self._flush_to_file()

    def log_impression_batch(
        self,
        user_id: int,
        session_id: str,
        items: list[dict],
        policy_id: str,
        features_snapshot_id: str,
        surface: str = Surface.HOME,
    ) -> None:
        """Log one impression event per item in a page/row response."""
        for item in items:
            ev = Event.impression(
                user_id=user_id,
                session_id=session_id,
                item_id=item.get("item_id", 0),
                row_id=item.get("row_id", "unknown"),
                position=item.get("position", -1),
                policy_id=policy_id,
                features_snapshot_id=features_snapshot_id,
                surface=surface,
            )
            self.log(ev)

    def _flush_to_file(self) -> None:
        import os
        os.makedirs(os.path.dirname(self._fallback_path), exist_ok=True)
        with open(self._fallback_path, "a") as f:
            for d in self._buffer:
                f.write(json.dumps(d) + "\n")
        self._buffer.clear()

    def __del__(self):
        if self._buffer:
            try:
                self._flush_to_file()
            except Exception:
                pass


# Module-level singleton — replace with DI in production
EVENT_LOGGER = EventLogger()
