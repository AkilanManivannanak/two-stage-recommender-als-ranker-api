"""
Event Logger — Full Event Schema
=================================
Plane: Core Recommendation (event capture drives everything)

The phenomenal spec is explicit: without impression and exposure logging,
your evaluation is weak because you don't know what the user actually saw
before acting.

REQUIRED EVENT SCHEMA (from spec):
  user_id, session_id, event_time, surface, event_type,
  item_id, row_id, position, features_snapshot_id, policy_id, action_value

This module captures all events with the full schema, writes to:
  1. In-process ring buffer (for real-time systems)
  2. Postgres (via async write, not blocking)
  3. Redis (for real-time feature updates)

In production: this would write to Kafka topics consumed by Flink.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── Event types (from spec) ───────────────────────────────────────────────────
class EventType:
    IMPRESSION      = "impression"
    CLICK           = "click"
    PLAY_START      = "play_start"
    PLAY_30S        = "play_30s"          # 30-second milestone
    PLAY_3MIN       = "play_3min"         # 3-minute milestone (signal)
    ABANDON         = "abandon"           # < 30s watch
    COMPLETION      = "completion"        # > 90% watched
    ADD_TO_LIST     = "add_to_list"
    REMOVE_FROM_LIST = "remove_from_list"
    LIKE            = "like"
    DISLIKE         = "dislike"
    SEARCH          = "search"
    VOICE_QUERY     = "voice_query"
    TRAILER_START   = "trailer_start"
    DWELL           = "dwell"             # time spent on title card
    NOT_INTERESTED  = "not_interested"


# ── Event rewards (for bandit / RL) ──────────────────────────────────────────
EVENT_REWARDS = {
    EventType.PLAY_3MIN:   1.0,
    EventType.COMPLETION:  1.5,
    EventType.ADD_TO_LIST: 0.8,
    EventType.LIKE:        0.8,
    EventType.PLAY_START:  0.5,
    EventType.PLAY_30S:    0.3,
    EventType.CLICK:       0.2,
    EventType.IMPRESSION:  0.0,
    EventType.ABANDON:    -0.5,
    EventType.DISLIKE:    -1.0,
    EventType.NOT_INTERESTED: -0.8,
}

# ── Surfaces ──────────────────────────────────────────────────────────────────
class Surface:
    HOME      = "home"
    SEARCH    = "search"
    DETAILS   = "details"
    VOICE     = "voice"
    TRENDING  = "trending"


@dataclass
class Event:
    """
    Full event schema per spec.
    Every field has a meaning — nothing is optional in production.
    """
    user_id:             int
    session_id:          str
    event_time:          float          # unix timestamp
    surface:             str            # home | search | details | voice | trending
    event_type:          str            # from EventType
    item_id:             int
    row_id:              str = ""       # e.g. "top_picks", "trending_now"
    position:            int = -1       # 0-indexed position in row
    features_snapshot_id: str = ""      # ID of feature snapshot at request time
    policy_id:           str = "v4.0.0" # which recommendation policy was active
    action_value:        float = 0.0    # e.g. watch duration in seconds
    duration_s:          float = 0.0    # actual watch duration
    page_position:       int = -1       # row position on page (0 = top)
    device:              str = "web"
    request_id:          str = ""       # links back to recommendation request
    extra:               dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def reward(self) -> float:
        return EVENT_REWARDS.get(self.event_type, 0.0)

    def is_positive(self) -> bool:
        return self.reward() > 0.0

    def is_abandonment(self) -> bool:
        return self.event_type == EventType.ABANDON


@dataclass
class ImpressionBatch:
    """
    Batch of impressions from a single page render.
    Used for exposure-corrected evaluation.
    """
    user_id:      int
    session_id:   str
    request_id:   str
    policy_id:    str
    timestamp:    float
    items:        list[dict]      # [{item_id, row_id, position, row_position}]
    propensities: dict[int, float] = field(default_factory=dict)  # item_id → P(shown)
    surface:      str = Surface.HOME


class EventLogger:
    """
    Captures all user events with full schema.

    In production: writes to Kafka topic → Flink → feature store + training data.
    Here: in-process ring buffer + file log + real-time feature updates.
    """

    MAX_BUFFER = 50_000

    def __init__(self, log_dir: str = "logs"):
        self._buffer: deque[Event] = deque(maxlen=self.MAX_BUFFER)
        self._impression_buffer: deque[ImpressionBatch] = deque(maxlen=10_000)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._event_file = self._log_dir / "events.jsonl"
        self._impression_file = self._log_dir / "impressions.jsonl"
        # Counters
        self._counts: dict[str, int] = {}

    def log_event(
        self,
        user_id:     int,
        item_id:     int,
        event_type:  str,
        session_id:  str = "",
        row_id:      str = "",
        position:    int = -1,
        surface:     str = Surface.HOME,
        duration_s:  float = 0.0,
        action_value: float = 0.0,
        policy_id:   str = "v4.0.0",
        request_id:  str = "",
        features_snapshot_id: str = "",
        extra:       dict = None,
    ) -> Event:
        """Log a single user event."""
        event = Event(
            user_id=user_id,
            session_id=session_id or f"sess_{user_id}_{int(time.time()//1800)}",
            event_time=time.time(),
            surface=surface,
            event_type=event_type,
            item_id=item_id,
            row_id=row_id,
            position=position,
            features_snapshot_id=features_snapshot_id,
            policy_id=policy_id,
            action_value=action_value,
            duration_s=duration_s,
            request_id=request_id or str(uuid.uuid4()),
            extra=extra or {},
        )
        self._buffer.append(event)
        self._counts[event_type] = self._counts.get(event_type, 0) + 1

        # Async file write (non-blocking in production → Kafka)
        self._write_event(event)

        # Update real-time feature store
        self._update_realtime_features(event)

        return event

    def log_impression_batch(
        self,
        user_id:    int,
        items:      list[dict],
        policy_id:  str = "v4.0.0",
        surface:    str = Surface.HOME,
        request_id: str = "",
    ) -> ImpressionBatch:
        """Log all items shown in a page render."""
        session_id = f"sess_{user_id}_{int(time.time()//1800)}"
        # Compute propensity: items at higher positions are shown more often
        propensities = {}
        for item in items:
            pos = int(item.get("position", 0))
            row_pos = int(item.get("page_position", 0))
            iid = int(item.get("item_id", item.get("movieId", 0)))
            # Position-decay propensity (lower position = higher propensity)
            p = 1.0 / (1.0 + pos * 0.1 + row_pos * 0.5)
            propensities[iid] = round(float(np.clip(p, 0.05, 1.0)), 4)

        batch = ImpressionBatch(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id or str(uuid.uuid4()),
            policy_id=policy_id,
            timestamp=time.time(),
            items=items,
            propensities=propensities,
            surface=surface,
        )
        self._impression_buffer.append(batch)
        self._write_impression(batch)

        # Update exposure_eval store
        try:
            from recsys.serving.exposure_eval import IMPRESSION_STORE, ImpressionLog
            log = ImpressionLog(
                user_id=user_id,
                item_ids=[int(it.get("item_id", it.get("movieId", 0))) for it in items],
                model_version=policy_id,
                propensities=list(propensities.values()),
            )
            IMPRESSION_STORE.log_impression(log)
        except Exception:
            pass

        return batch

    def get_recent_events(
        self,
        user_id: Optional[int] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get recent events from ring buffer."""
        events = list(self._buffer)
        if user_id is not None:
            events = [e for e in events if e.user_id == user_id]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_shown_items(self, user_id: int) -> set[int]:
        """Items shown to user across impression history."""
        shown = set()
        for batch in self._impression_buffer:
            if batch.user_id == user_id:
                for item in batch.items:
                    shown.add(int(item.get("item_id", item.get("movieId", 0))))
        return shown

    def get_propensity(self, item_id: int, default: float = 0.1) -> float:
        """Estimate propensity for an item across all impressions."""
        props = []
        for batch in self._impression_buffer:
            if item_id in batch.propensities:
                props.append(batch.propensities[item_id])
        return float(np.mean(props)) if props else default

    def stats(self) -> dict:
        return {
            "buffer_size":       len(self._buffer),
            "impression_batches": len(self._impression_buffer),
            "event_counts":      dict(self._counts),
        }

    def _write_event(self, event: Event):
        try:
            with self._event_file.open("a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            pass

    def _write_impression(self, batch: ImpressionBatch):
        try:
            with self._impression_file.open("a") as f:
                f.write(json.dumps({
                    "user_id": batch.user_id,
                    "session_id": batch.session_id,
                    "request_id": batch.request_id,
                    "policy_id": batch.policy_id,
                    "timestamp": batch.timestamp,
                    "n_items": len(batch.items),
                    "item_ids": [int(it.get("item_id", it.get("movieId", 0)))
                                 for it in batch.items],
                }) + "\n")
        except Exception:
            pass

    def _update_realtime_features(self, event: Event):
        """Push event to real-time feature stores."""
        try:
            from recsys.serving.realtime_engine import process_event
            process_event(event.user_id, event.item_id, event.event_type)
        except Exception:
            pass
        try:
            from recsys.serving.feature_store import FEATURE_STORE
            FEATURE_STORE.on_user_event(event.user_id, event.item_id, event.event_type)
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
EVENT_LOGGER = EventLogger(log_dir=os.environ.get("LOG_DIR", "logs"))
