"""
Netflix-Inspired RecSys — Apache Airflow DAG
=============================================
Orchestrates the full offline recommendation pipeline as a scheduled DAG.

Airflow vs Metaflow:
  Metaflow  = step-level orchestration, artifact tracking, compute scaling
  Airflow   = time-based scheduling, dependency management, monitoring,
              cross-system coordination, SLA enforcement

They complement each other:
  Airflow schedules WHEN things run and monitors SLAs.
  Metaflow manages HOW training steps run and tracks artifacts.

This DAG:
  - Runs daily at 2am UTC
  - Coordinates: TMDB refresh → LLM enrichment → embedding rebuild →
    Metaflow training pipeline → DuckDB evaluation → Qdrant index update →
    Redis feature warmup → API health check → Slack/alert notification
  - Each task has retry policy and SLA monitoring
  - Failed tasks trigger agentic triage via the ops API

Schedule: daily at 02:00 UTC
Owner:    recsys-platform-team
SLA:      pipeline must complete within 4 hours
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Default task args ─────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "recsys-platform",
    "depends_on_past":  False,
    "email_on_failure": False,   # use Slack alert task instead
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
}

SLA_4HR = timedelta(hours=4)

with DAG(
    dag_id="recsys_daily_pipeline",
    default_args=DEFAULT_ARGS,
    description=(
        "Daily offline recommendation pipeline: "
        "TMDB refresh → LLM enrichment → embedding rebuild → "
        "Metaflow training → DuckDB eval → Qdrant update → Redis warmup"
    ),
    schedule="0 2 * * *",          # 02:00 UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    sla_miss_callback=None,        # wire to PagerDuty/Slack in production
    tags=["recsys","ml","daily"],
    doc_md=__doc__,
) as dag:

    # ── Start ──────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start", sla=SLA_4HR)

    # ── Step 1: TMDB catalog refresh ─────────────────────────────────
    # Fetches new/updated titles, posters, descriptions from TMDB API
    # Plane 2 — Semantic Intelligence
    tmdb_refresh = BashOperator(
        task_id="tmdb_catalog_refresh",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python -c \""
            "import sys; sys.path.insert(0,'src'); "
            "from recsys.serving.catalog_enrichment import tmdb_hydrate; "
            "from recsys.serving.app import CATALOG; "
            "[tmdb_hydrate(item['title'],item.get('year')) "
            " for item in list(CATALOG.values())[:500]]; "
            "print('[TMDB] Refresh complete')\""
        ),
        sla=timedelta(minutes=30),
        doc_md="Hydrate catalog from TMDB API using Bearer token auth.",
    )

    # ── Step 2: LLM semantic enrichment ──────────────────────────────
    # Runs structured GPT-4o-mini enrichment for new/stale titles
    # Plane 2 — Semantic Intelligence (OFFLINE sidecar)
    llm_enrichment = BashOperator(
        task_id="llm_semantic_enrichment",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/workers/llm_enrichment_worker.py"
        ),
        sla=timedelta(minutes=45),
        doc_md="LLM enrichment: themes, moods, tags, summaries (OpenAI json_object format).",
    )

    # ── Step 3: VLM artwork audit ─────────────────────────────────────
    # Runs GPT-4o vision audit on new posters
    # Plane 2 — Semantic Intelligence
    vlm_audit = BashOperator(
        task_id="vlm_artwork_audit",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/workers/vlm_audit_worker.py"
        ),
        sla=timedelta(minutes=30),
        doc_md=(
            "VLM poster audit: trust score, bait-and-switch detection. "
            "Items with trust_score < 0.6 flagged for editorial review. "
            "Runs OFFLINE — never in request path."
        ),
    )

    # ── Step 4: Embedding rebuild ─────────────────────────────────────
    # Rebuilds fused multimodal embeddings for enriched items
    # Plane 2 — Semantic Intelligence
    embedding_rebuild = BashOperator(
        task_id="multimodal_embedding_rebuild",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/workers/embedding_worker.py"
        ),
        sla=timedelta(hours=1),
        doc_md=(
            "Rebuild fused text+metadata embeddings. "
            "Text: text-embedding-3-small → 1536-dim → W_TEXT projection → 64-dim. "
            "Metadata: one-hot+continuous → 22-dim → W_META projection → 64-dim. "
            "Fusion: 0.70*text + 0.30*meta, L2-normalised."
        ),
    )

    # ── Step 5: Metaflow training pipeline ────────────────────────────
    # Runs the full 18-step Metaflow pipeline
    # All 4 planes
    metaflow_run = BashOperator(
        task_id="metaflow_training_pipeline",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "USERNAME=airflow "
            "METAFLOW_DATASTORE_SYSROOT_LOCAL=artifacts "
            "python flows/two_stage_recsys_flow_v2.py run "
            "--n_users 2000 --n_movies 500 --n_ratings 80000 "
            "--use_llm True --use_tmdb True"
        ),
        sla=timedelta(hours=2),
        doc_md=(
            "Full 18-step Metaflow pipeline: "
            "catalog_ingestion → content_preprocessing → "
            "catalog_semantic_enrichment → multimodal_embedding_build → "
            "semantic_index_build → behavior_model_train → "
            "session_intent_modeling → candidate_fusion → "
            "multimodal_feature_join → rank_and_slate_optimize → "
            "artwork_grounding_check → genai_explanation_build → "
            "shadow_eval_and_release_gate → agentic_eval_triage → "
            "policy_and_safety_gate → bundle_serve_payload → end"
        ),
    )

    # ── Step 6: DuckDB offline evaluation ────────────────────────────
    # Reads Metaflow bundle and generates reports for frontend
    duckdb_eval = BashOperator(
        task_id="duckdb_offline_evaluation",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "pip install duckdb pandas pyarrow numpy --quiet && "
            "ARTIFACTS_DIR=artifacts REPORTS_DIR=../frontend/public/reports "
            "python infra/duckdb/run_offline_eval.py"
        ),
        sla=timedelta(minutes=15),
        doc_md="DuckDB: NDCG, diversity, bootstrap CIs, baseline comparison, report JSONs.",
    )

    # ── Step 7: Quality gate check ────────────────────────────────────
    def _quality_gate_check(**context):
        """
        Check metrics against thresholds.
        Returns 'metrics_ok' or 'metrics_fail' for branching.
        """
        import json
        from pathlib import Path
        p = Path("/opt/airflow/recsys/backend/artifacts/bundle/serve_payload.json")
        if not p.exists():
            return "metrics_fail"
        metrics = json.loads(p.read_text()).get("metrics", {})
        ndcg = metrics.get("ndcg_at_10", 0)
        div  = metrics.get("diversity_score", 0)
        if ndcg >= 0.08 and div >= 0.50:
            return "metrics_ok"
        return "metrics_fail"

    gate_check = BranchPythonOperator(
        task_id="quality_gate_check",
        python_callable=_quality_gate_check,
        doc_md="Gate: NDCG>=0.08 AND diversity>=0.50 → proceed. Else → alert.",
    )

    metrics_ok   = EmptyOperator(task_id="metrics_ok")
    metrics_fail = BashOperator(
        task_id="metrics_fail",
        bash_command=(
            "echo 'QUALITY GATE FAILED — alerting ops team' && "
            "curl -s -X POST http://api:8000/agent/triage || true"
        ),
        doc_md="On gate failure: trigger agent triage. Human reviews before any deploy.",
    )

    # ── Step 8: Qdrant index update ───────────────────────────────────
    # Pushes new embeddings into Qdrant vector store
    qdrant_update = BashOperator(
        task_id="qdrant_index_update",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/qdrant/init_collections.py"
        ),
        sla=timedelta(minutes=20),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md=(
            "Update Qdrant vector collections with new embeddings. "
            "Qdrant can be fully rebuilt from MinIO artifacts if storage is lost."
        ),
    )

    # ── Step 9: Redis feature warmup ─────────────────────────────────
    # Rebuilds hot feature cache from Postgres
    # Redis loss never corrupts data — this rebuilds from source of truth
    redis_warmup = BashOperator(
        task_id="redis_feature_warmup",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/redis/warmup.py"
        ),
        sla=timedelta(minutes=5),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md=(
            "Warm Redis from Postgres. Redis = online hot features. "
            "Postgres = source of truth. Redis loss never corrupts data."
        ),
    )

    # ── Step 10: API health check ─────────────────────────────────────
    api_health = BashOperator(
        task_id="api_health_check",
        bash_command=(
            "curl -f http://api:8000/healthz && "
            "curl -f http://api:8000/metrics/pipeline && "
            "echo 'API healthy'"
        ),
        sla=timedelta(minutes=2),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md="Verify API is healthy and serving updated pipeline metrics.",
    )

    # ── Step 11: Agent ops triage ─────────────────────────────────────
    # Runs agentic summary — DOES NOT DEPLOY
    agent_triage = BashOperator(
        task_id="agent_ops_triage",
        bash_command=(
            "cd /opt/airflow/recsys/backend && "
            "PYTHONPATH=src python infra/workers/agent_ops_worker.py"
        ),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md=(
            "Agent triage: summarise regressions, recommend DEPLOY/HOLD/INVESTIGATE. "
            "DOES NOT DEPLOY. Human reviews output before any release decision."
        ),
    )

    # ── End ───────────────────────────────────────────────────────────
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ── DAG dependencies ──────────────────────────────────────────────
    (
        start
        >> tmdb_refresh
        >> [llm_enrichment, vlm_audit]
        >> embedding_rebuild
        >> metaflow_run
        >> duckdb_eval
        >> gate_check
        >> [metrics_ok, metrics_fail]
        >> qdrant_update
        >> redis_warmup
        >> api_health
        >> agent_triage
        >> end
    )
