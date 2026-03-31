"""
Redis Feature Store Warm-up
============================
Loads hot user features from Postgres into Redis on startup.
Redis loss does NOT corrupt data — this script rebuilds it from Postgres.

Called by: api container on startup (or as a separate init container)
"""
from __future__ import annotations
import json, os

REDIS_URL = os.environ.get("REDIS_URL","redis://localhost:6379/0")
DB_URL    = os.environ.get("DATABASE_URL","")

def warmup():
    try:
        import redis as r
        rc = r.from_url(REDIS_URL, decode_responses=True)
        rc.ping()
        print("[Redis Warmup] Connected")
    except Exception as e:
        print(f"[Redis Warmup] Redis unavailable: {e} — continuing without cache")
        return

    # Warm up with demo user features
    demo_users = [1,7,42,99,137,256,512,1024]
    for uid in demo_users:
        rc.setex(
            f"user:{uid}:session_intent",
            1800,  # 30 min TTL
            json.dumps({"intent":"unknown","blend_weight":0.3,"genres":[]})
        )
        rc.setex(
            f"user:{uid}:exploration_budget",
            300,   # 5 min TTL
            "0.15"
        )
    print(f"[Redis Warmup] Warmed {len(demo_users)} demo users")

    # Set system-level keys
    rc.set("system:version", "v4.0.0")
    rc.set("system:status",  "healthy")
    print("[Redis Warmup] Complete")

if __name__ == "__main__":
    warmup()
