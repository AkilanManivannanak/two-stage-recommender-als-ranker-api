"""
metaflow_integration.py — CineWave
=====================================
Connects the serving layer to Metaflow artifacts produced by
phenomenal_flow_v3.py.  Also wires Kafka event streaming into
the /feedback and /impressions/log endpoints.

WHY THIS FILE EXISTS
--------------------
The existing app.py loads bundle artifacts via _try_load_bundle()
which reads pickle files from disk.  This module adds two things:

  1. MetaflowArtifactLoader  — fetches the LATEST successful
     phenomenal_flow_v3 run from Metaflow's datastore and
     refreshes the in-memory bundle WITHOUT a container restart.
     Called at startup and via the new /metaflow/refresh endpoint.

  2. KafkaEventBridge — wraps the existing KafkaEventProducer so
     every /feedback event and every impression batch is also
     streamed to Kafka topics in real time.  Falls back silently
     to the existing file logger if Kafka is down.

HOW IT WIRES IN
---------------
In app.py, after _try_load_bundle(), add:

    from recsys.serving.metaflow_integration import (
        MetaflowArtifactLoader, KafkaEventBridge,
        METAFLOW_LOADER, KAFKA_BRIDGE,
    )
    METAFLOW_LOADER.try_refresh_from_latest_run(_bundle)
    KAFKA_BRIDGE.start()

And in the /feedback endpoint body, add:

    KAFKA_BRIDGE.send_feedback(req.user_id, req.item_id, req.event)

And in _log_impressions(), add:

    KAFKA_BRIDGE.send_impressions(uid, items)

DATA FLOW DIAGRAM
-----------------

  User browser
       │
       ▼
  FastAPI /feedback, /page, /recommend
       │                        │
       ▼                        ▼
  KafkaEventBridge         MetaflowArtifactLoader
  (this file)              (this file)
       │                        │
       ▼                        ▼
  Kafka topics             Metaflow datastore
  recsys.events            (artifacts/bundle/)
  recsys.impressions            │
  recsys.feature_updates        ▼
       │                   _bundle (in-memory)
       ▼                   CATALOG, ALS, Ranker
  Flink consumer
  → Postgres (event log)
  → Redis (real-time features)
  → Qdrant (embedding updates)
"""
from __future__ import annotations

import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ── Metaflow availability ────────────────────────────────────────────────────

def _has_metaflow() -> bool:
    try:
        import metaflow  # noqa
        return True
    except ImportError:
        return False


class MetaflowArtifactLoader:
    """
    Loads artifacts from the latest successful Metaflow run of
    phenomenal_flow_v3.PhenomenalRecSysFlow.

    Usage:
        loader = MetaflowArtifactLoader()
        loader.try_refresh_from_latest_run(bundle)

    What it does:
        1. Queries Metaflow for all runs of PhenomenalRecSysFlow
        2. Finds the most recent run where step 'end' succeeded
        3. Pulls these artifacts from that run:
             - movies (catalog list)
             - user_genre_ratings (user profile features)
             - item_factors / user_factors (ALS embeddings)
             - ranker model pickle
             - serve_payload (metrics, feature importance)
        4. Writes them to disk at artifacts/bundle/ so _try_load_bundle()
           picks them up on next API restart
        5. Also hot-patches the in-memory _bundle directly so the
           new model is live immediately without restart

    This is the standard Netflix pattern: Metaflow produces a new
    model artifact every 24h (triggered by Airflow), the serving
    layer detects it and hot-swaps without downtime.
    """

    FLOW_NAME = "PhenomenalRecSysFlow"

    def __init__(self):
        self._last_refresh:    float = 0.0
        self._last_run_id:     str   = ""
        self._refresh_lock             = threading.Lock()
        self._bundle_dir               = Path(os.environ.get("BUNDLE_REF", "artifacts/bundle"))
        self.enabled                   = _has_metaflow()
        if not self.enabled:
            print("  [Metaflow] metaflow package not installed — artifact loading disabled")

    def try_refresh_from_latest_run(self, bundle: Any) -> bool:
        """
        Hot-patch `bundle` with artifacts from the latest Metaflow run.
        Returns True if the bundle was updated.
        """
        if not self.enabled:
            return False
        with self._refresh_lock:
            return self._do_refresh(bundle)

    def _do_refresh(self, bundle: Any) -> bool:
        try:
            from metaflow import Flow, namespace
            namespace(None)  # search all namespaces

            flow = Flow(self.FLOW_NAME)
            latest_run = None
            for run in flow.runs():
                if run.successful:
                    latest_run = run
                    break

            if latest_run is None:
                print(f"  [Metaflow] No successful {self.FLOW_NAME} run found")
                return False

            run_id = latest_run.id
            if run_id == self._last_run_id:
                print(f"  [Metaflow] Bundle already up-to-date (run {run_id})")
                return False

            end_step = latest_run["end"]

            # ── Pull catalog / movies ─────────────────────────────────────
            if hasattr(end_step.task.data, "movies") and end_step.task.data.movies:
                movies = end_step.task.data.movies
                bundle.movies = movies
                print(f"  [Metaflow] Loaded {len(movies)} movies from run {run_id}")

            # ── Pull user genre ratings (user profile features) ──────────
            if hasattr(end_step.task.data, "user_genre_ratings"):
                bundle.user_genre_ratings = {
                    int(k): v for k, v in end_step.task.data.user_genre_ratings.items()
                }

            # ── Pull ALS item/user factors ───────────────────────────────
            if hasattr(end_step.task.data, "item_factors"):
                bundle.item_factors = end_step.task.data.item_factors
            if hasattr(end_step.task.data, "user_factors"):
                bundle.user_factors = end_step.task.data.user_factors

            # ── Pull LightGBM ranker ─────────────────────────────────────
            if hasattr(end_step.task.data, "ranker"):
                bundle.ranker = end_step.task.data.ranker

            # ── Pull metrics + feature importance ────────────────────────
            if hasattr(end_step.task.data, "feature_importance"):
                bundle.feature_importance = end_step.task.data.feature_importance
            if hasattr(end_step.task.data, "metrics"):
                bundle.metrics = end_step.task.data.metrics

            # ── Write artifacts to disk for next cold-start ──────────────
            self._write_to_disk(bundle, run_id)

            bundle.loaded     = True
            self._last_run_id = run_id
            self._last_refresh = time.time()

            print(f"  [Metaflow] ✓ Bundle hot-swapped from run {run_id}")
            return True

        except Exception as e:
            print(f"  [Metaflow] Refresh failed: {e}")
            return False

    def _write_to_disk(self, bundle: Any, run_id: str):
        """Persist Metaflow artifacts to disk so cold-start loads them."""
        try:
            self._bundle_dir.mkdir(parents=True, exist_ok=True)

            # Write movies / catalog
            if bundle.movies:
                with open(self._bundle_dir / "movies.json", "w") as f:
                    json.dump(bundle.movies, f)

            # Write ALS factors
            if bundle.item_factors:
                with open(self._bundle_dir / "item_factors.pkl", "wb") as f:
                    pickle.dump(bundle.item_factors, f)
            if bundle.user_factors:
                with open(self._bundle_dir / "user_factors.pkl", "wb") as f:
                    pickle.dump(bundle.user_factors, f)

            # Write ranker
            if bundle.ranker:
                with open(self._bundle_dir / "ranker.pkl", "wb") as f:
                    pickle.dump(bundle.ranker, f)

            # Write serve payload (metrics, feature importance)
            payload = {
                "metrics":            bundle.metrics,
                "feature_importance": bundle.feature_importance,
                "feature_cols":       bundle.feature_cols,
                "run_id":             run_id,
                "refreshed_at":       time.time(),
            }
            with open(self._bundle_dir / "serve_payload.json", "w") as f:
                json.dump(payload, f, default=str)

            print(f"  [Metaflow] Artifacts written to {self._bundle_dir}")
        except Exception as e:
            print(f"  [Metaflow] Disk write failed: {e}")

    def status(self) -> dict:
        return {
            "enabled":          self.enabled,
            "flow_name":        self.FLOW_NAME,
            "last_run_id":      self._last_run_id,
            "last_refresh_ago": round(time.time() - self._last_refresh, 1) if self._last_refresh else None,
        }


# ── Kafka Event Bridge ────────────────────────────────────────────────────────

class KafkaEventBridge:
    """
    Bridges FastAPI endpoint events → Kafka topics.

    Wraps the existing KafkaEventProducer and adds:
      - Background flusher thread (flushes every 5s)
      - send_feedback()   — called from /feedback endpoint
      - send_impressions() — called from _log_impressions()
      - send_voice_event() — called from /voice/assist endpoint

    Data flow:
      FastAPI endpoint
          │
          ▼
      KafkaEventBridge.send_feedback()
          │
          ├─► Kafka topic: recsys.events       (if Kafka up)
          │
          └─► logs/kafka_fallback.jsonl        (if Kafka down)
                  │
                  ▼
              Replayed to Kafka on next restart via
              KafkaEventBridge.replay_fallback()

    Kafka consumer side (Flink job in recsys_flink container):
      recsys.events      → Postgres events table + Redis session cache
      recsys.impressions → IMPRESSION_STORE (IPS-NDCG evaluation)
      recsys.feature_updates → Redis feature store (real-time signals)
    """

    def __init__(self):
        self._producer   = None
        self._running    = False
        self._flush_thread: Optional[threading.Thread] = None
        self._fallback   = Path("logs/kafka_fallback.jsonl")
        self._fallback.parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Initialise Kafka producer and start background flusher."""
        try:
            from recsys.serving.kafka_producer import KAFKA_PRODUCER
            self._producer = KAFKA_PRODUCER
            print(f"  [KafkaBridge] Producer ready — Kafka={self._producer.status()['kafka_available']}")
        except Exception as e:
            print(f"  [KafkaBridge] Could not load kafka_producer: {e}")

        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="kafka-flusher"
        )
        self._flush_thread.start()

    def stop(self):
        self._running = False

    def send_feedback(
        self,
        user_id:   int,
        item_id:   int,
        event:     str,
        policy_id: str = "cinewave-v5.0.0",
    ):
        """
        Stream a user feedback event to Kafka.
        Called from the /feedback FastAPI endpoint.

        event is one of: play, like, dislike, add_to_list, not_interested
        """
        if self._producer:
            self._producer.send_event(
                user_id    = user_id,
                item_id    = item_id,
                event_type = event,
                surface    = "home",
                policy_id  = policy_id,
            )
        else:
            self._write_fallback({
                "type": "feedback", "user_id": user_id,
                "item_id": item_id, "event": event,
                "ts": time.time(),
            })

    def send_impressions(
        self,
        user_id:   int,
        items:     list,
        surface:   str = "home",
        policy_id: str = "cinewave-v5.0.0",
    ):
        """
        Stream a batch of impressions to Kafka.
        Called from _log_impressions() in app.py.

        This powers the IPS-NDCG off-policy evaluation:
          Kafka → Flink → Postgres impressions table
          → DuckDB offline eval reads impressions + clicks
          → computes IPS-corrected NDCG per policy version
        """
        if self._producer:
            self._producer.send_impression_batch(
                user_id   = user_id,
                items     = items,
                policy_id = policy_id,
                surface   = surface,
            )
        else:
            self._write_fallback({
                "type":     "impressions",
                "user_id":  user_id,
                "item_ids": [i.get("item_id") for i in items],
                "surface":  surface,
                "ts":       time.time(),
            })

    def send_voice_event(
        self,
        user_id:    int,
        transcript: str,
        n_results:  int,
        genres:     list,
        latency_ms: float,
    ):
        """
        Stream a voice search event to Kafka.
        Useful for analysing voice query patterns and result quality.
        """
        payload = {
            "type":        "voice_search",
            "user_id":     user_id,
            "transcript":  transcript[:200],
            "n_results":   n_results,
            "genres":      genres,
            "latency_ms":  round(latency_ms, 1),
            "ts":          time.time(),
        }
        if self._producer:
            self._producer.send_event(
                user_id    = user_id,
                item_id    = 0,
                event_type = "voice_search",
                extra      = payload,
            )
        else:
            self._write_fallback(payload)

    def replay_fallback(self) -> int:
        """
        Replay events from the fallback JSONL file to Kafka.
        Called on startup if Kafka is now available.
        Returns number of events replayed.
        """
        if not self._fallback.exists() or not self._producer:
            return 0
        replayed = 0
        remaining = []
        try:
            with open(self._fallback) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        t  = ev.get("type", "feedback")
                        if t == "impressions":
                            self._producer.send_impression_batch(
                                user_id = ev.get("user_id", 0),
                                items   = [{"item_id": i} for i in ev.get("item_ids", [])],
                            )
                        else:
                            self._producer.send_event(
                                user_id    = ev.get("user_id", 0),
                                item_id    = ev.get("item_id", 0),
                                event_type = ev.get("event", "replay"),
                            )
                        replayed += 1
                    except Exception:
                        remaining.append(line)
        except Exception:
            pass

        # Rewrite with only failed events
        with open(self._fallback, "w") as f:
            for line in remaining:
                f.write(line + "\n")

        if replayed:
            print(f"  [KafkaBridge] Replayed {replayed} fallback events to Kafka")
        return replayed

    def _flush_loop(self):
        """Background thread: flush Kafka every 5 seconds."""
        while self._running:
            time.sleep(5)
            if self._producer:
                self._producer.flush()

    def _write_fallback(self, payload: dict):
        try:
            with open(self._fallback, "a") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            pass

    def status(self) -> dict:
        kafka_ok = False
        if self._producer:
            kafka_ok = self._producer.status().get("kafka_available", False)
        return {
            "kafka_available":    kafka_ok,
            "flusher_running":    self._running,
            "fallback_log":       str(self._fallback),
            "fallback_size_kb":   round(self._fallback.stat().st_size / 1024, 1)
                                  if self._fallback.exists() else 0,
        }


# ── Singletons ─────────────────────────────────────────────────────────────────

METAFLOW_LOADER = MetaflowArtifactLoader()
KAFKA_BRIDGE    = KafkaEventBridge()
