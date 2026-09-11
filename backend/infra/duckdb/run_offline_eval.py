"""
run_offline_eval.py — periodic offline evaluation against the measured bundle.

Runs on a schedule (Airflow / cron). Reads the metrics the training pipeline
measured, compares the candidate model against the currently-promoted incumbent,
and asks the policy gate whether the candidate may be promoted.

This file used to carry the evaluation results as literals — `"ndcg_at_10":
0.1409` appeared seven times, and the SQL selected `0.1409 AS ndcg10`. It now
reads artifacts/bundle/metrics.json and exits non-zero if that file is absent,
so a scheduled run cannot report a result the pipeline never produced.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
BUNDLE   = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
STATE    = Path(os.environ.get("OPE_STATE_DIR", ROOT / "artifacts" / "ope"))
METRICS  = BUNDLE / "metrics.json"
STATE.mkdir(parents=True, exist_ok=True)

if not METRICS.exists():
    sys.exit(f"[ope] FAILED — {METRICS} not found. Run scripts/evaluate.py first.")

payload  = json.loads(METRICS.read_text())
headline = payload["headline"]
baselines = payload["baselines"]

if not headline.get("ndcg_at_10"):
    sys.exit("[ope] FAILED — metrics.json carries no ndcg_at_10")

# ── Incumbent ─────────────────────────────────────────────────────────────────
inc_path = STATE / "incumbent.json"
incumbent = json.loads(inc_path.read_text()) if inc_path.exists() else None
if incumbent is None:
    print("[ope] no incumbent on record — this run establishes the baseline")

# ── DuckDB analysis over the measured per-baseline table ──────────────────────
try:
    import duckdb
    con = duckdb.connect()
    rows = [(name, b["ndcg@10"], b["mrr@10"], b["recall@10"],
             b["ips_ndcg@10"], b["diversity_score"], b["coverage"], b["n_users"])
            for name, b in baselines.items()]
    con.execute("""CREATE TABLE eval(
        ranker VARCHAR, ndcg10 DOUBLE, mrr10 DOUBLE, recall10 DOUBLE,
        ips_ndcg10 DOUBLE, diversity DOUBLE, coverage DOUBLE, n_users BIGINT)""")
    con.executemany("INSERT INTO eval VALUES (?,?,?,?,?,?,?,?)", rows)
    table = con.execute("""
        SELECT ranker, ndcg10, ips_ndcg10, diversity, coverage,
               ndcg10 - (SELECT ndcg10 FROM eval WHERE ranker='als_only') AS lift_vs_als
        FROM eval ORDER BY ndcg10 DESC""").fetchall()
    print("\n[ope] measured ranking quality (test split)")
    print(f"  {'ranker':<18}{'NDCG@10':>10}{'IPS-NDCG':>11}{'div':>8}{'cov':>8}{'vs ALS':>10}")
    for r in table:
        print(f"  {r[0]:<18}{r[1]:>10.4f}{r[2]:>11.4f}{r[3]:>8.3f}{r[4]:>8.3f}{r[5]:>+10.4f}")
    con.close()
except ImportError:
    print("[ope] duckdb not installed — skipping the SQL summary, gate still runs")

# ── Policy gate ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))
from recsys.serving.policy_gate import PolicyGate

gate_input = dict(headline)
gate_input["ndcg_at_10_incumbent"] = (incumbent or {}).get("ndcg_at_10")
gate_input["slices"] = payload.get("slices_by_user_activity", {})
result = PolicyGate().gate_from_measured_metrics(gate_input)

print(f"\n[ope] gate: {result.recommendation} — {result.summary}")
for c in result.checks:
    mark = "PASS" if c.passed else ("BLOCK" if c.critical else "WARN")
    print(f"   {mark:<6}{c.name:<34}{c.value!r} {c.comparison} {c.threshold!r}")

record = {
    "ran_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_version": payload.get("dataset", {}).get("prepared_at"),
    "headline":     headline,
    "gate":         {"recommendation": result.recommendation,
                     "passed": result.gate_passed,
                     "blocking": result.blocking_checks,
                     "warnings": result.warnings},
}
(STATE / f"ope_{int(time.time())}.json").write_text(json.dumps(record, indent=2))

if result.recommendation == "DEPLOY":
    inc_path.write_text(json.dumps(headline, indent=2))
    print(f"[ope] promoted — incumbent updated to NDCG@10={headline['ndcg_at_10']}")
    sys.exit(0)
if result.recommendation == "REVIEW":
    print("[ope] held for review — critical gates passed, warnings present")
    sys.exit(0)
print("[ope] BLOCKED — incumbent left in place")
sys.exit(1)
