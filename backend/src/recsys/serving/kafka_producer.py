"""
Kafka Event Producer  —  Real-Time Event Streaming
====================================================
Plane: Core Recommendation (event ingestion)

Replaces the file-based event logger with a Kafka producer.
Events are streamed to Kafka topics, consumed by Flink jobs,
and written to Postgres + Redis in real-time.

TOPICS:
  recsys.events          — all user interaction events
  recsys.impressions     — page impression batches
  recsys.feature_updates — real-time feature updates

HONEST NOTE:
  Kafka requires the docker-compose-kafka.yml overlay.
  Start with: docker compose -f docker-compose.yml -f docker-compose-kafka.yml up -d
  On Mac M1/M2: use confluentinc/cp-kafka:7.6.0 (ARM-compatible).
  Flink job runs in the recsys_flink container.

Production:
  Replace localhost:9092 with your Kafka cluster endpoint.
  Add schema registry for Avro serialisation.
  Add authentication (SASL/SSL) for production clusters.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Kafka availability flag
_KAFKA_AVAILABLE = False
_KAFKA_PRODUCER  = None

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TOPICS = {
    "events":          "recsys.events",
    "impressions":     "recsys.impressions",
    "feature_updates": "recsys.feature_updates",
}


def _init_kafka():
    """Initialise Kafka producer. Silent fail if Kafka not running."""
    global _KAFKA_AVAILABLE, _KAFKA_PRODUCER
    if _KAFKA_AVAILABLE:
        return True
    try:
        from kafka import KafkaProducer
        _KAFKA_PRODUCER = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k else None,
            acks="all",             # wait for all replicas
            retries=3,
            max_block_ms=2000,      # don't block serving for > 2s
            request_timeout_ms=5000,
        )
        _KAFKA_AVAILABLE = True
        print("  [Kafka] Producer connected to", KAFKA_BOOTSTRAP)
        return True
    except Exception as e:
        print(f"  [Kafka] Not available ({e}) — using file fallback")
        return False


class KafkaEventProducer:
    """
    Produces events to Kafka topics.
    Falls back to file logging if Kafka is unavailable.

    In production: Kafka → Flink consumer → Postgres + Redis updates.
    Locally: Kafka in Docker or file fallback.
    """

    def __init__(self):
        self._kafka_ok = _init_kafka()
        self._fallback_log = Path("logs/kafka_fallback.jsonl")
        self._fallback_log.parent.mkdir(parents=True, exist_ok=True)

    def send_event(
        self,
        user_id:    int,
        item_id:    int,
        event_type: str,
        session_id: str = "",
        row_id:     str = "",
        position:   int = -1,
        surface:    str = "home",
        duration_s: float = 0.0,
        policy_id:  str = "v4.0.0",
        features_snapshot_id: str = "",
        extra:      dict = None,
    ) -> bool:
        """Send a single event. Returns True if delivered."""
        payload = {
            "event_id":           str(uuid.uuid4()),
            "user_id":            user_id,
            "item_id":            item_id,
            "event_type":         event_type,
            "session_id":         session_id or f"sess_{user_id}_{int(time.time()//1800)}",
            "event_time":         time.time(),
            "surface":            surface,
            "row_id":             row_id,
            "position":           position,
            "duration_s":         duration_s,
            "policy_id":          policy_id,
            "features_snapshot_id": features_snapshot_id,
            **(extra or {}),
        }

        if self._kafka_ok and _KAFKA_PRODUCER:
            try:
                _KAFKA_PRODUCER.send(
                    TOPICS["events"],
                    key=str(user_id),
                    value=payload,
                )
                return True
            except Exception as e:
                print(f"  [Kafka] Send failed: {e} — falling back to file")
                self._kafka_ok = False

        # File fallback
        self._write_fallback(payload)
        return False

    def send_impression_batch(
        self,
        user_id:    int,
        items:      list[dict],
        policy_id:  str = "v4.0.0",
        surface:    str = "home",
    ) -> bool:
        """Send a batch of impressions (one per page render)."""
        payload = {
            "batch_id":   str(uuid.uuid4()),
            "user_id":    user_id,
            "timestamp":  time.time(),
            "policy_id":  policy_id,
            "surface":    surface,
            "n_items":    len(items),
            "item_ids":   [int(it.get("item_id", it.get("movieId", 0))) for it in items],
            "positions":  list(range(len(items))),
        }

        if self._kafka_ok and _KAFKA_PRODUCER:
            try:
                _KAFKA_PRODUCER.send(
                    TOPICS["impressions"],
                    key=str(user_id),
                    value=payload,
                )
                return True
            except Exception:
                self._kafka_ok = False

        self._write_fallback(payload, "impressions")
        return False

    def send_feature_update(
        self,
        entity_type: str,   # "user" | "item"
        entity_id:   int,
        features:    dict,
    ) -> bool:
        """Push real-time feature update for consumption by Flink → Redis."""
        payload = {
            "update_id":   str(uuid.uuid4()),
            "entity_type": entity_type,
            "entity_id":   entity_id,
            "timestamp":   time.time(),
            "features":    features,
        }

        if self._kafka_ok and _KAFKA_PRODUCER:
            try:
                _KAFKA_PRODUCER.send(
                    TOPICS["feature_updates"],
                    key=f"{entity_type}:{entity_id}",
                    value=payload,
                )
                return True
            except Exception:
                self._kafka_ok = False

        self._write_fallback(payload, "feature_updates")
        return False

    def _write_fallback(self, payload: dict, topic: str = "events"):
        try:
            with self._fallback_log.open("a") as f:
                f.write(json.dumps({"topic": topic, **payload}) + "\n")
        except Exception:
            pass

    def flush(self):
        """Flush pending Kafka messages."""
        if self._kafka_ok and _KAFKA_PRODUCER:
            try:
                _KAFKA_PRODUCER.flush(timeout=5)
            except Exception:
                pass

    def status(self) -> dict:
        return {
            "kafka_available": self._kafka_ok,
            "bootstrap_servers": KAFKA_BOOTSTRAP,
            "topics": TOPICS,
            "fallback_log": str(self._fallback_log),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
KAFKA_PRODUCER = KafkaEventProducer()
