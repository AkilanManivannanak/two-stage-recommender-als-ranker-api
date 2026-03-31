"""
Airflow DAG  —  Daily RecSys Pipeline
=======================================
Schedules when the Metaflow pipeline runs.
Airflow = scheduler + SLA monitor
Metaflow = ML workflow + artifact management

DAG structure:
  daily_trigger
    → data_quality_check
    → run_phenomenal_flow
      → gate_check
        → notify_on_fail     (on failure)
        → update_serving     (on success)
          → warm_redis_cache
          → run_duckdb_report

SLA: pipeline must complete within 4 hours of start.
If gate fails: alert + hold (human must review before any deploy action).
"""
from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.utils.trigger_rule import TriggerRule
    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False
    # Stubs so the file is importable without Airflow
    class DAG:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    class BashOperator:
        def __init__(self, *a, **k): pass
    class PythonOperator:
        def __init__(self, *a, **k): pass
    class BranchPythonOperator:
        def __init__(self, *a, **k): pass
    class TriggerRule:
        ALL_DONE = "all_done"
        ONE_FAILED = "one_failed"
        ONE_SUCCESS = "one_success"


DEFAULT_ARGS = {
    "owner":            "recsys-platform",
    "depends_on_past":  False,
    "start_date":       datetime(2025, 1, 1),
    "email_on_failure": False,   # Set True + configure email in production
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "sla":              timedelta(hours=4),   # pipeline SLA: 4 hours
}

# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="recsys_daily_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily phenomenal recommendation pipeline",
    schedule_interval="0 2 * * *",   # 2am UTC daily
    catchup=False,
    max_active_runs=1,               # prevent concurrent pipeline runs
    tags=["recsys", "ml", "daily"],
) as dag:

    # ── 1. Data quality pre-check ─────────────────────────────────────
    data_quality_check = BashOperator(
        task_id="data_quality_check",
        bash_command="""
            cd /opt/airflow/recsys &&
            python -c "
import json, sys
from pathlib import Path
p = Path('artifacts/bundle/serve_payload.json')
if not p.exists():
    print('No prior bundle — first run, proceeding')
    sys.exit(0)
payload = json.loads(p.read_text())
dq = payload.get('data_quality', {})
fails = [k for k,v in dq.items() if not v.get('pass', True)]
if fails:
    print(f'DATA QUALITY FAILURES: {fails}')
    sys.exit(1)
print('Data quality OK')
"
        """,
    )

    # ── 2. Run Metaflow phenomenal flow ───────────────────────────────
    run_phenomenal_flow = BashOperator(
        task_id="run_phenomenal_flow",
        bash_command="""
            cd /opt/airflow/recsys &&
            python flows/phenomenal_flow.py run
                --use_real_data True
                --no_pylint
        """,
        execution_timeout=timedelta(hours=3),   # hard timeout
    )

    # ── 3. Check gate result ──────────────────────────────────────────
    def check_gate_result(**context):
        """Read gate result from bundle, return branch."""
        import json
        from pathlib import Path
        try:
            payload = json.loads(
                Path("/opt/airflow/recsys/artifacts/bundle/serve_payload.json").read_text())
            gate = payload.get("gate_result", {})
            rec  = gate.get("recommendation", "BLOCK")
            if rec == "DEPLOY":
                return "update_serving"
            elif rec == "REVIEW":
                return "notify_human_review"
            else:
                return "notify_gate_failure"
        except Exception as e:
            print(f"Gate check error: {e}")
            return "notify_gate_failure"

    gate_branch = BranchPythonOperator(
        task_id="gate_check",
        python_callable=check_gate_result,
        provide_context=True,
    )

    # ── 4a. Notify gate failure (BLOCK) ───────────────────────────────
    notify_gate_failure = BashOperator(
        task_id="notify_gate_failure",
        bash_command="""
            echo "GATE BLOCKED — pipeline output does not meet deployment thresholds"
            echo "Review artifacts/bundle/serve_payload.json gate_result section"
            cat /opt/airflow/recsys/artifacts/bundle/serve_payload.json | python -c "
import json,sys
p=json.load(sys.stdin)
g=p.get('gate_result',{})
print('Recommendation:', g.get('recommendation'))
print('Blocking checks:', g.get('blocking_checks', []))
print('Warnings:', g.get('warnings', []))
"
        """,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── 4b. Notify for human review (REVIEW) ─────────────────────────
    notify_human_review = BashOperator(
        task_id="notify_human_review",
        bash_command="""
            echo "GATE REVIEW REQUIRED — agent triage recommends human review"
            echo "Warnings found but no critical failures — human must approve before deploy"
            cat /opt/airflow/recsys/artifacts/bundle/serve_payload.json | python -c "
import json,sys
p=json.load(sys.stdin)
g=p.get('gate_result',{})
print('Recommendation:', g.get('recommendation'))
print('Warnings:', g.get('warnings', []))
print('Agent triage:', p.get('agent_triage', {}).get('action'), '-', p.get('agent_triage', {}).get('justification'))
"
        """,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── 4c. Update serving bundle (DEPLOY) ───────────────────────────
    update_serving = BashOperator(
        task_id="update_serving",
        bash_command="""
            echo "Gate PASSED — updating serving bundle"
            cd /opt/airflow/recsys
            echo "Bundle ready for serving at artifacts/bundle/"
            ls -la artifacts/bundle/
        """,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── 5. Warm Redis cache ───────────────────────────────────────────
    warm_redis = BashOperator(
        task_id="warm_redis_cache",
        bash_command="""
            cd /opt/airflow/recsys &&
            python infra/redis/warmup.py
        """,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── 6. Run DuckDB offline eval report ─────────────────────────────
    duckdb_report = BashOperator(
        task_id="duckdb_report",
        bash_command="""
            cd /opt/airflow/recsys &&
            python infra/duckdb/run_offline_eval.py
        """,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── Pool: protect TMDB and OpenAI from over-parallelism ───────────
    run_phenomenal_flow.pool = "external_api_pool"   # create in Airflow: 2 slots

    # ── Dependencies ─────────────────────────────────────────────────
    data_quality_check >> run_phenomenal_flow >> gate_branch
    gate_branch >> [update_serving, notify_gate_failure, notify_human_review]
    update_serving >> warm_redis >> duckdb_report
