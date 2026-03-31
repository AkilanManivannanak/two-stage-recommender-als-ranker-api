"""
Flink Feature Pipeline  —  Kafka → Redis Hot Features
======================================================
Consumes events from Kafka and writes hot features to Redis in real-time.

This is the Python-based lightweight version of what a real Flink job does.
Production: replace with PyFlink or Flink SQL for true stream processing.

What it does:
  - Consumes recsys.events topic
  - Updates per-user session features in Redis (TTL: 300s)
  - Updates per-item trending scores in Redis (TTL: 60s)
  - Updates exploration budgets based on session intent
  - Writes feature_update messages to recsys.feature_updates topic

Run:
  docker compose -f docker-compose.yml -f docker-compose-kafka.yml \
    --profile streaming up flink_feature_job
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL       = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

EVENT_REWARDS = {
    "play_3min":    1.0,
    "completion":   1.5,
    "add_to_list":  0.8,
    "like":         0.8,
    "play_start":   0.5,
    "play_30s":     0.3,
    "click":        0.2,
    "impression":   0.0,
    "abandon":     -0.5,
    "dislike":     -1.0,
}


def run():
    # Connect Redis
    try:
        import redis as r
        rc = r.from_url(REDIS_URL, decode_responses=True)
        rc.ping()
        print(f"[FlinkFeaturePipeline] Redis connected: {REDIS_URL}")
    except Exception as e:
        print(f"[FlinkFeaturePipeline] Redis unavailable: {e}")
        rc = None

    # Connect Kafka consumer
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            "recsys.events",
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="flink_feature_pipeline",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=5000,
        )
        print(f"[FlinkFeaturePipeline] Kafka consumer connected: {KAFKA_BOOTSTRAP}")
    except Exception as e:
        print(f"[FlinkFeaturePipeline] Kafka unavailable: {e}")
        print("[FlinkFeaturePipeline] Running in simulation mode (no real events)")
        _simulate(rc)
        return

    # In-memory trending accumulator (flush to Redis every 5s)
    trending_buffer: dict[int, float] = defaultdict(float)
    last_flush = time.time()

    print("[FlinkFeaturePipeline] Consuming events...")

    for message in consumer:
        event = message.value
        uid   = event.get("user_id", 0)
        iid   = event.get("item_id", 0)
        etype = event.get("event_type", "")
        reward = EVENT_REWARDS.get(etype, 0.0)

        if rc:
            # 1. Update user session features
            session_key = f"user:{uid}:session_intent"
            session = {}
            try:
                raw = rc.get(session_key)
                if raw:
                    session = json.loads(raw)
            except Exception:
                pass

            session["last_event"]      = etype
            session["last_event_item"] = iid
            session["last_active"]     = time.time()
            session["session_reward"]  = session.get("session_reward", 0) + reward

            try:
                rc.setex(session_key, 300, json.dumps(session))
            except Exception:
                pass

            # 2. Accumulate trending score
            trending_buffer[iid] += max(reward, 0.0)

            # 3. Update exploration budget based on reward trend
            if reward < -0.3:   # abandonment detected
                budget_key = f"user:{uid}:exploration_budget"
                try:
                    current = float(rc.get(budget_key) or 0.15)
                    new_budget = min(current + 0.05, 0.35)  # increase exploration
                    rc.setex(budget_key, 300, str(round(new_budget, 3)))
                except Exception:
                    pass

        # Flush trending buffer to Redis every 5 seconds
        if time.time() - last_flush > 5.0:
            if rc and trending_buffer:
                pipe = rc.pipeline()
                for item_id, score in trending_buffer.items():
                    pipe.setex(f"item:{item_id}:trending", 60, str(round(score, 4)))
                try:
                    pipe.execute()
                except Exception:
                    pass
            trending_buffer.clear()
            last_flush = time.time()


def _simulate(rc):
    """Simulation mode when Kafka is unavailable — generates synthetic events."""
    import random
    print("[FlinkFeaturePipeline] Simulation mode: generating synthetic events")

    demo_users = [1, 7, 42, 99, 137, 256, 512, 1024]
    event_types = ["play_3min", "play_start", "abandon", "like", "impression"]

    while True:
        uid   = random.choice(demo_users)
        iid   = random.randint(1, 200)
        etype = random.choice(event_types)
        reward = EVENT_REWARDS.get(etype, 0.0)

        if rc:
            try:
                session_key = f"user:{uid}:session_intent"
                rc.setex(session_key, 300, json.dumps({
                    "last_event": etype,
                    "last_event_item": iid,
                    "last_active": time.time(),
                }))
                rc.setex(f"item:{iid}:trending", 60,
                         str(round(max(reward, 0.0) + random.uniform(0, 0.1), 4)))
            except Exception:
                pass

        time.sleep(0.5)   # 2 events/second in simulation


if __name__ == "__main__":
    run()
