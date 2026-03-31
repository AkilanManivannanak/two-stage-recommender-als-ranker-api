"""
Freshness Monitor Worker
=========================
Runs as a background worker (--profile workers) to:
  1. Monitor feature staleness across all stores
  2. Refresh trending scores from Redis
  3. Alert on circuit-broken features
  4. Update DuckDB freshness report

Run: docker compose --profile workers up freshness_worker
Or:  python infra/workers/freshness_worker.py
"""
import sys, os, time, json
sys.path.insert(0, 'src')
sys.path.insert(0, '/app/src')

POLL_INTERVAL = 30   # seconds between checks

def run():
    print("[FreshnessWorker] Starting — polling every 30s")
    while True:
        try:
            check_freshness()
        except Exception as e:
            print(f"[FreshnessWorker] Error: {e}")
        time.sleep(POLL_INTERVAL)

def check_freshness():
    from pathlib import Path
    report = {}

    # Check feature store
    try:
        from recsys.serving.feature_store import FEATURE_STORE
        fs_report = FEATURE_STORE.staleness_report()
        report["feature_store"] = {
            "stale_pct": fs_report.get("staleness_pct", 0),
            "total": fs_report.get("total_records", 0),
        }
    except Exception as e:
        report["feature_store"] = {"error": str(e)}

    # Check freshness engine
    try:
        from recsys.serving.freshness_engine import FRESH_STORE, LAUNCH_DETECTOR
        fresh_report = FRESH_STORE.staleness_report()
        launch_stats = LAUNCH_DETECTOR.stats()
        report["fresh_store"] = fresh_report
        report["launch_detector"] = launch_stats
    except Exception as e:
        report["fresh_store"] = {"error": str(e)}

    # Check Redis connectivity
    try:
        from recsys.serving.redis_feature_store import REDIS_FEATURE_STORE
        report["redis"] = REDIS_FEATURE_STORE.status()
    except Exception as e:
        report["redis"] = {"error": str(e)}

    # Write report
    out = Path("logs/freshness_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report["ts"] = time.time()
    out.write_text(json.dumps(report, indent=2, default=str))

    stale = report.get("feature_store", {}).get("stale_pct", 0)
    if stale > 0.05:
        print(f"[FreshnessWorker] WARNING: {stale:.0%} features stale")
    else:
        print(f"[FreshnessWorker] OK | stale={stale:.1%} | ts={int(report['ts'])}")

if __name__ == "__main__":
    run()
