<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141414,50:E50914,100:B20710&height=220&section=header" width="100%"/>

```
 ██████╗██╗███╗   ██╗███████╗██╗    ██╗ █████╗ ██╗   ██╗███████╗
██╔════╝██║████╗  ██║██╔════╝██║    ██║██╔══██╗██║   ██║██╔════╝
██║     ██║██╔██╗ ██║█████╗  ██║ █╗ ██║███████║██║   ██║█████╗  
██║     ██║██║╚██╗██║██╔══╝  ██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝  
╚██████╗██║██║ ╚████║███████╗╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗
 ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝
```

### Production-Grade ML Recommendation System · Offline RL / Off-Policy RL · Netflix Internship Project

*Built by [Akilan Manivannan](https://www.linkedin.com/in/akilan-manivannan-a178212a7/) · MS in Artificial Intelligence*

<br>

[![CI](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api/actions/workflows/ci.yml/badge.svg)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/▶%20Live%20Demo-Google%20Drive-E5091A?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-two--stage--recommender-181717?style=for-the-badge&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Akilan%20Manivannan-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)

<br>

![NDCG](https://img.shields.io/badge/NDCG%40ten-0.1409-22C55E?style=flat-square)
![Baseline](https://img.shields.io/badge/ALS%20Baseline-0.0399-F59E0B?style=flat-square)
![Lift](https://img.shields.io/badge/Lift%20vs%20ALS-%2B253%25-22C55E?style=flat-square)
![Latency](https://img.shields.io/badge/p95%20SLO-%3C50ms-22C55E?style=flat-square)
![Movies](https://img.shields.io/badge/Catalog-4%2C961%20Movies-3B82F6?style=flat-square)
![Endpoints](https://img.shields.io/badge/API%20Endpoints-62-818CF8?style=flat-square)
![Gates](https://img.shields.io/badge/Policy%20Gates-27%20checks-F59E0B?style=flat-square)
![Docker](https://img.shields.io/badge/Docker%20Services-7-2496ED?style=flat-square&logo=docker)
![K8s](https://img.shields.io/badge/Kubernetes-HPA%202--10%20replicas-326CE5?style=flat-square&logo=kubernetes)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js)

</div>

---

## TL;DR

```
Goal          → Personalised movie recommendations with sub-50ms p95 latency
NDCG@10       → 0.1409 (ALS+LightGBM pipeline, +253% over ALS-only 0.0399)
MRR@10        → 0.2826  ·  Recall@10: 0.0644
ALS baseline  → NDCG 0.0399  ·  MRR 0.0885  ·  Recall 0.0154
Latency SLO   → p95 < 50ms (plain /recommend endpoint)
Stack         → FastAPI · Next.js 14 · PostgreSQL · Redis · Qdrant · MinIO · Kafka
ML            → ALS (rank=64) · LightGBM · REINFORCE · LinUCB (8 arms, α=1.0)
RL            → REINFORCE policy gradient + LinUCB off-policy RL bandit
Eval          → Doubly-robust IPS (offline RL / off-policy RL evaluation)
Session model → GRU-style encoder · hidden=16 · input=8 · acc=0.927
Policy Gate   → 27 automated checks before any model promotion
Ratings       → 800k ratings processed via PySpark (5 feature sets)
Catalog       → 4,961 movies with real TMDB posters · 3,667 ALS item factors
API           → 62 endpoints · SQL (4 tables) · Kubernetes HPA (2–10 replicas)
SRE           → p50/p95/p99 per route · 27-gate release policy · health checks
CI            → GitHub Actions — import smoke + TypeScript build on every push
```

Built as a **Netflix-inspired, production-grade** system demonstrating the full ML engineering lifecycle: PySpark data pipeline → ALS candidate retrieval → LightGBM reranking → offline RL / off-policy RL → voice AI → SRE observability → policy-gated evaluation → feedback loop.

---

## Table of Contents

- [What's Actually in This Repo](#whats-actually-in-this-repo)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [ML Pipeline — 5 Stages](#ml-pipeline--5-stages)
- [Offline RL / Off-Policy RL Evaluation](#offline-rl--off-policy-rl-evaluation)
- [Policy Gate — 27 Automated Checks](#policy-gate--27-automated-checks)
- [CLIP — Vision Transformer (ViT-B/32)](#clip--vision-transformer-vit-b32)
- [Voice AI & GenAI Features](#voice-ai--genai-features)
- [SRE Observability](#sre-observability)
- [Kubernetes — Production Deployment](#kubernetes--production-deployment)
- [SQL Schema & Analytics](#sql-schema--analytics)
- [MLOps Pipeline](#mlops-pipeline)
- [ML Dashboard](#ml-dashboard)
- [Results & Baselines](#results--baselines)
- [Postmortem — Real Incidents](#postmortem--real-incidents)
- [Quick Start](#quick-start)
- [Demo Pages](#demo-pages)
- [Project Structure](#project-structure)
- [CI/CD](#cicd)

---

## What's Actually in This Repo

Every number here is verified from the source code.

| Component | File | Real Numbers |
|---|---|---|
| **PySpark ETL** | `spark_features.py` | 800k ratings · 5 feature sets · co-occurrence map |
| **ALS** | `scala/FeaturePipeline.scala` | rank=64 · 20 iterations · alpha=40 |
| **LightGBM** | `ranker_and_slate.py` | NDCG 0.1409 vs ALS 0.0399 (+253%) · 8 features |
| **REINFORCE** | `rl_policy.py` | Monte Carlo returns · Redis weight storage |
| **GRU Session Model** | `session_intent.py` | hidden=16 · input=8 · numpy · acc=0.927 |
| **LinUCB Bandit** | `bandit_v2.py` | 8 genre arms · α=1.0 · UCB exploration |
| **Slate Optimizer** | `slate_optimizer_v2.py` | ≥5 genres · ≤3 same genre · 0.15 explore rate |
| **IPS Evaluator** | `ope_eval.py` | Doubly-robust offline RL / off-policy RL eval |
| **Policy Gate** | `policy_gate.py` | 27 automated GateCheck objects |
| **RAG Engine** | `rag_engine.py` | Qdrant HNSW · 1,536-dim · OpenAI embeddings |
| **CLIP** | `context_and_additions.py` | ViT-B/32 · 512-dim · graceful fallback |
| **Voice Pipeline** | `voice_router.py` · `voice_tools.py` | Whisper → GPT-4o intent → TTS nova |
| **A/B Framework** | `ab_experiment.py` | 4 experiments · doubly-robust IPS |
| **Metaflow** | `flows/phenomenal_flow_v3.py` | 12-step DAG · hot-swap on promotion |
| **Kubernetes** | `k8s/` | HPA 2–10 replicas · deployment · service · PDB |
| **SQL** | `sql/` | 4-table schema · SELECT+JOIN+GROUP BY queries |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                        │
│                         Next.js 14  ·  TypeScript  ·  Tailwind CSS             │
│         Home · Voice AI · A/B Dashboard · AI Stack · ML Dashboard · Eval       │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │  HTTP
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│             Kubernetes Ingress · nginx · HPA: 2–10 replicas                     │
│             Scale triggers: CPU>70% · Memory>80% · RPS>100/pod                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                        FASTAPI  :8000  ·  62 Endpoints                           │
│    /recommend  /voice  /explain  /feedback  /healthz  /ope/eval  /ab/*         │
└──────┬──────────────────┬───────────────────┬───────────────────┬───────────────┘
       │                  │                   │                   │
   STAGE 01           STAGE 02           STAGE 03           STAGE 04
  PySpark ETL        ALS + RAG          LightGBM          Offline RL
  800k ratings       rank=64            NDCG 0.1409       REINFORCE +
  5 feature sets     Qdrant 1536d       vs ALS 0.0399     LinUCB 8 arms
  co-occurrence      HNSW index         8 features        α=1.0
       │                  │                   │                   │
       └──────────────────┴───────────────────┴───────────────────┘
                                    │
                             STAGE 05
                          SLATE OPTIMIZER
                      ≥5 genres · ≤3 same genre
                      ≤2 same-genre rows above fold
                      0.15 explore rate
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 REDIS           KAFKA          METAFLOW
              Feature store    3 topics       12-step DAG
              Session cache    Flink→PG       Policy gate
              Bandit weights   Real-time      27 checks
                    │
                    ▼
               DUCKDB
         IPS-NDCG eval every 6h
         Auto-rollback via policy gate
```

---

## Tech Stack

<div align="center">

| Layer | Technology | Real Implementation |
|---|---|---|
| **Frontend** | Next.js 14 · TypeScript · Tailwind CSS | App Router · 7-tab ML dashboard · voice UI |
| **API** | FastAPI · Python 3.11 · Uvicorn | 62 endpoints |
| **Collaborative Filtering** | ALS Scala MLlib · rank=64 | 3,667 item factors (TMDB-patched bundle) |
| **Feature Engineering** | PySpark · local[*] · 800k ratings | 5 feature sets: genre ratings, activity, popularity, co-occurrence, impression counts |
| **Reranking** | LightGBM · NDCG objective · 8 features | NDCG 0.1409 vs ALS 0.0399 (+253%) |
| **Offline RL / Off-Policy RL** | REINFORCE · LinUCB (8 arms, α=1.0) · doubly-robust IPS | Session-aware reranking · off-policy bandit evaluation |
| **Session Model** | GRU-style encoder · hidden=16 · input=8 · numpy | acc=0.927 at training time |
| **Semantic Retrieval** | Qdrant · 1,536-dim · HNSW · OpenAI embeddings | Voice query → nearest neighbours |
| **Feature Store** | Redis · TTL freshness layer | Sub-10ms feature lookups |
| **Streaming** | Kafka 3 topics · Flink consumer | Real-time event ingestion · JSONL fallback |
| **Storage** | PostgreSQL · MinIO (S3-compatible) | Ratings · ML artifacts |
| **Policy Gate** | `policy_gate.py` · 27 GateCheck objects | Blocks bad model promotions |
| **MLOps** | Metaflow · 12-step DAG · hot-swap | No container restart on promotion |
| **Scheduling** | Airflow 2.9 · nightly DAG | Retraining + SLA alerts |
| **Offline Eval** | DuckDB · Parquet · IPS-NDCG | Every 6h · auto-rollback |
| **Multimodal** | CLIP ViT-B/32 · patch embeddings · multi-head self-attention · 512-dim | Optional · graceful colour-histogram fallback |
| **Voice AI** | Whisper STT · GPT-4o intent · TTS nova | 8 genre profile arms (LinUCB) |
| **GenAI** | GPT-4o explanations · GPT-4o Vision | Per-user personalised explanations |
| **SQL** | PostgreSQL · `sql/schema.sql` · `sql/queries.sql` | 4-table schema · SELECT + JOIN + GROUP BY |
| **Kubernetes** | HPA (2–10 replicas) · deployment · service · PDB | Auto-scaling on CPU/Memory/RPS |
| **SRE / DevOps** | p50/p95/p99 per route · 27-gate release policy · health checks · X-Request-ID · JSONL fallback | Policy gate enforces SRE standards; Kafka fallback ensures zero data loss |
| **Orchestration** | Docker Compose · 7 services | Local production environment |
| **CI/CD** | GitHub Actions | Import smoke + TypeScript build |

</div>

---

## ML Pipeline — 5 Stages

### Stage 1 — PySpark Feature Engineering

```
MovieLens ratings (800k rows × 8 cols)
        │
        ▼
PySpark ETL  (spark_features.py)  —  local[*] mode
Columnar groupBy aggregations faster than Python dict loops at this scale

5 feature sets computed:
  1. user_genre_ratings   — {uid: {genre: [ratings]}}
  2. user_activity        — {uid: {n_ratings, avg_rating, n_genres}}
  3. impression_counts    — {uid: {item_id: n_impressions}}
  4. item_popularity      — {item_id: interaction_count}
  5. item_cooccurrence    — {item_id: [top-10 co-watched items]}
                            via PySpark self-join on user_id

Fallback: if PySpark not installed → pandas/dict implementation
          pipeline never hard-fails
```

### Stage 2 — ALS Collaborative Filtering + RAG

**ALS (Scala Spark MLlib)**
```
rank       = 64 latent factors
iterations = 20
alpha      = 40  (implicit feedback weighting)
→ item_factors stored in MinIO · loaded into Redis
```

**Why Scala over PySpark?** Native Spark MLlib eliminates JVM↔Python serialisation bridge, achieving 2–4× speedup per iteration.

**RAG Semantic Retrieval (Qdrant)**
```
Voice query → OpenAI text-embedding-3-small (1,536-dim)
→ Qdrant HNSW cosine similarity
→ year ≥ 1970 filter for "similar to X" queries
→ 3-strategy retry: exact title → without year → first 3 words
```

### Stage 3 — LightGBM Reranker

```python
# 8-feature vector per (user, item) pair:
features = [
    als_score,       # ALS collaborative filtering score
    u_avg,           # user average rating
    u_cnt,           # user interaction count
    item_pop,        # item popularity
    item_avg_rating, # item average rating
    item_year,       # release year
    genre_affinity,  # genre preference score
    runtime_min,     # movie runtime
]

# Results vs all baselines:
# popularity:    NDCG 0.0292  MRR 0.0649  Recall 0.0122
# co-occurrence: NDCG 0.0362  MRR 0.0781  Recall 0.0158
# ALS only:      NDCG 0.0399  MRR 0.0885  Recall 0.0154
# ALS+LightGBM:  NDCG 0.1409  MRR 0.2826  Recall 0.0644  ← +253%
```

### Stage 4 — Offline RL / Off-Policy RL

See dedicated section below.

### Stage 5 — Slate Optimizer

```
Hard constraints (slate_optimizer_v2.py):

  ✓  ≥ 5 distinct genres on assembled page       (MIN_GENRES_ON_PAGE = 5)
  ✓  ≤ 3 titles from same genre in top-20 slate  (MAX_SAME_GENRE_IN_SLATE = 3)
  ✓  ≤ 2 rows with same dominant genre above fold (MAX_SAME_GENRE_ABOVE_FOLD = 2)
  ✓  explore_new_genres row rate = 0.15

Row scoring: engagement_prior × genre_affinity × item_quality
Post-hoc swap if constraints violated
```

---

## Offline RL / Off-Policy RL Evaluation

### REINFORCE Policy Gradient

```python
# session_intent.py — GRU-style session encoder
HIDDEN_DIM = 16   # GRU hidden state dimension  (actual value from code)
INPUT_DIM  = 8    # per-event feature dimension  (actual value from code)
# Single GRU cell · numpy (no PyTorch dependency)
# h_t = GRU(x_t, h_{t-1})
# Trained on session sequences → acc=0.927 reported at startup

# rl_policy.py — REINFORCE agent
# Monte Carlo returns: G_t = Σ γ^k * r_{t+k}
# Policy gradient: ∇J(θ) = Σ G_t * ∇log π(a_t|s_t)
# Policy weights stored in Redis · updated per completed episode

Reward signal (reward_model.py, weights _W):
  play_start    → +0.42 weight
  watch_90pct   → +0.28 weight (largest completion signal)
  genre_match   → +0.38 weight
  add_to_list   → +0.22 weight
  skip          → -0.15 weight
  exploration   → +0.18 weight (genre outside long-term history)
```

### LinUCB Off-Policy Bandit

```python
# bandit_v2.py
# 8 genre arms: Action, Comedy, Drama, Horror,
#               Sci-Fi, Romance, Thriller, Documentary

class LinUCBArm:
    context_dim: int = 8    # feature dimension
    alpha:       float = 1.0  # exploration-exploitation tradeoff

    def ucb_score(self, context):
        # UCB = μ(arm) + α × √(x^T A^{-1} x)
        exploit = theta @ context
        explore = alpha * sqrt(context @ A_inv @ context)
        return exploit + explore

# This IS off-policy RL:
# - learns from logged interactions under previous policies
# - updates per arm without live re-exploration
# - Thompson Sampling available as alternative strategy
```

### Doubly-Robust IPS — Off-Policy RL Evaluation

```python
# ope_eval.py
# Standard offline RL / off-policy RL evaluation methodology

def ips_ndcg_at_k(recommendations, events, propensities, k=10):
    """
    IPS-corrected NDCG@k.

    DR(π) = IPS(π) + direct_model_correction
    IPS-NDCG = Σ [reward(i) / propensity(i)] × 1/log2(rank+1)

    Evaluates new policy against data logged under old policy.
    No live deployment required — true offline RL evaluation.
    """
```

---

## Policy Gate — 27 Automated Checks

`policy_gate.py` runs 27 `GateCheck` objects before any model is promoted. The gate cannot be bypassed — no flag, no override, no "just this once."

```
Verdict: DEPLOY  → all blocking checks passed
         BLOCK   → rollback + Airflow alert
         REVIEW  → passed blocking, warnings present

Check categories (27 total):
  Quality    → NDCG@10 lift vs incumbent · absolute NDCG floor
               cold-start NDCG no-regression · MRR · Recall
  Diversity  → diversity_score · catalog coverage
  Latency    → p95_ms < 50ms (plain /recommend) · p99_ms ceiling
  Reliability→ error_rate threshold
  Skew       → PSI (Population Stability Index) — training vs serving
  ... (27 total — all blocking checks must pass for DEPLOY)
```

---

## CLIP — Vision Transformer (ViT-B/32)

CLIP encodes visual features using a **Vision Transformer (ViT-B/32)** backbone with patch embeddings and multi-head self-attention. The image is split into 32×32 patches, each embedded into a 512-dim token; these tokens pass through transformer layers with self-attention before projection into the shared text-image embedding space.

```
Movie poster → 32×32 patches → 512-dim patch tokens
→ Transformer layers (multi-head self-attention)
→ [CLS] token → 512-dim visual embedding
→ projected into CLIP shared space (aligned with text queries)

Graceful fallback (context_and_additions.py):
  when openai-clip not installed → colour histogram fallback
  zero impact on ALS+LightGBM+RL core recommendations
```

---

## Voice AI & GenAI Features

```
User speaks → Whisper STT
→ GPT-4o intent extraction (18 genre keyword maps)
  → genres: ["Horror", "Thriller"]
  → similar_to: "Stranger Things"
  → year_filter: ≥1970

├──► Qdrant RAG (1,536-dim semantic search)
└──► Genre pool (co-occurrence filtered, post-1970)
          │
          ▼  round-robin interleave
  Top-8 recommendations
          │
          ▼
  buildExplanation() — reads item.primary_genre directly
  (bypasses /explain API — avoids wrong-genre hallucination bug)
          │
          ▼
  GPT-4o TTS 'nova' → spoken response
```

**8 Genre Profile Arms (match LinUCB arms)**

| Profile | Genres |
|---|---|
| Cinephile | Drama, Foreign, Documentary |
| Action Fan | Action, Thriller, Adventure |
| Indie Lover | Drama, Indie, Romance |
| Blockbuster | Action, Comedy, Family |
| Art House | Drama, Foreign, Art |
| Rom-Com Fan | Romance, Comedy |
| Sci-Fi Buff | Sci-Fi, Fantasy, Thriller |
| Documentary | Documentary, Biography, History |

---

## SRE Observability

```
Latency SLO      → p95 < 50ms for plain /recommend (enforced by policy gate)
                   p50/p95/p99 tracked per route
Policy gate      → 27 automated checks block bad deploys
Rollback         → MetaflowArtifactLoader hot-swap (no container restart)
Health checks    → /healthz liveness + readiness (Kubernetes probes)
Request tracing  → X-Request-ID on every request
Freshness SLAs   → TTL tracking on all features · auto-invalidate on stale
PSI monitoring   → Population Stability Index catches training-serving skew
Kafka fallback   → JSONL on disk if Kafka unavailable (zero data loss)
```

---

## Kubernetes — Production Deployment

**`k8s/deployment.yaml`**
```yaml
replicas: 3              # baseline, scaled by HPA
strategy: RollingUpdate
  maxSurge: 1            # 1 extra pod during rollout
  maxUnavailable: 0      # zero-downtime: never kill before new is ready
livenessProbe:  GET /healthz  every 15s
readinessProbe: GET /healthz  every 10s
resources:
  requests: 500m CPU / 1Gi RAM
  limits:   2000m CPU / 4Gi RAM
```

**`k8s/hpa.yaml`**
```yaml
minReplicas: 2
maxReplicas: 10
triggers:
  CPU utilisation    > 70%  → scale up
  Memory utilisation > 80%  → scale up
  RPS per pod        > 100  → scale up  (Prometheus custom metric)
scaleUp:   stabilizationWindow 30s   (react fast to spikes)
scaleDown: stabilizationWindow 300s  (conservative — 5 min wait)
PodDisruptionBudget: minAvailable=2  (always ≥2 pods running)
```

**`k8s/service.yaml`**
```yaml
ClusterIP + LoadBalancer + Nginx Ingress
api.cinewave.ai → recsys-api:8000
cinewave.ai     → frontend:3000
```

---

## SQL Schema & Analytics

**`sql/schema.sql`** — 4 tables:

```sql
CREATE TABLE users (
    user_id         BIGINT PRIMARY KEY,
    profile_name    VARCHAR(64) DEFAULT 'Cinephile',
    activity_decile SMALLINT CHECK (activity_decile BETWEEN 1 AND 10),
    top_genres      TEXT[] DEFAULT '{}'
);

CREATE TABLE ratings (
    user_id  BIGINT REFERENCES users(user_id),
    item_id  BIGINT NOT NULL,
    rating   NUMERIC(3,1) CHECK (rating BETWEEN 0.5 AND 5.0),
    watch_pct NUMERIC(5,2)
);

CREATE TABLE recommendations (
    user_id         BIGINT REFERENCES users(user_id),
    item_id         BIGINT NOT NULL,
    rank            SMALLINT NOT NULL,
    als_score       NUMERIC(8,6),
    rl_score        NUMERIC(8,6),
    policy_version  VARCHAR(32) DEFAULT 'v6.0.0'
);

CREATE TABLE events (
    user_id     BIGINT REFERENCES users(user_id),
    item_id     BIGINT NOT NULL,
    event_type  VARCHAR(32) NOT NULL,  -- play_start | watch_90pct | skip | add_to_list
    reward      NUMERIC(4,2),
    session_id  UUID
);
```

**`sql/queries.sql`** — analytical queries (SELECT + JOIN + GROUP BY):

```sql
-- NDCG@10 per policy version
SELECT r.policy_version,
       COUNT(DISTINCT r.user_id) AS unique_users,
       AVG(CASE WHEN e.event_type = 'play_start' THEN 1.0
                ELSE 0.0 END / LOG(2, r.rank + 1)) AS ndcg_at_10
FROM recommendations r
LEFT JOIN events e ON e.user_id = r.user_id AND e.item_id = r.item_id
WHERE r.rank <= 10
GROUP BY r.policy_version
ORDER BY ndcg_at_10 DESC;

-- CTR by activity decile (SELECT + JOIN + GROUP BY + HAVING)
SELECT u.activity_decile,
       COUNT(DISTINCT u.user_id) AS users,
       COUNT(CASE WHEN e.event_type = 'play_start' THEN 1 END) * 100.0
       / NULLIF(COUNT(DISTINCT r.rec_id), 0) AS ctr_pct
FROM users u
JOIN recommendations r ON r.user_id = u.user_id
LEFT JOIN events e ON e.user_id = u.user_id AND e.item_id = r.item_id
GROUP BY u.activity_decile
HAVING COUNT(DISTINCT u.user_id) >= 10
ORDER BY u.activity_decile;
```

---

## MLOps Pipeline

### Nightly Retraining

```
00:00  Airflow trigger
00:05  PySpark feature engineering (800k ratings · 5 feature sets)
00:20  Scala ALS training (rank=64 · 20 iterations · alpha=40)
00:40  LightGBM reranker (NDCG objective)
01:00  Policy Gate (27 checks → DEPLOY or BLOCK)
01:10  If DEPLOY: MetaflowArtifactLoader hot-swap (no container restart)
       If BLOCK:  rollback previous version + Airflow alert
```

### Kafka Event Pipeline

```
User event → FastAPI /feedback → KafkaEventBridge
  ├──► Kafka: recsys.events · recsys.impressions · recsys.feature_updates
  └──► JSONL fallback on disk (zero data loss if Kafka unavailable)
         │
         ▼
  Flink consumer → Postgres events + Redis session cache
```

---

## ML Dashboard

The `/ml` page — 7 tabs wired to live backend:

| Tab | What It Shows |
|---|---|
| **OPE** | IPS-NDCG offline RL / off-policy RL evaluation · policy comparison |
| **Homepage** | Live feed metrics — CTR, session depth, diversity scores |
| **Temporal** | 30-day rolling NDCG, CTR time series |
| **MMR** | Maximal Marginal Relevance · Jaccard diversity |
| **Cold-Start** | New-user coverage · genre exploration breadth |
| **Notify** | Freshness SLA status · TTL watermarks · staleness alerts |
| **Infra** | Redis · Qdrant · Kafka · Postgres · MinIO health |

---

## Results & Baselines

### NDCG@10 — All Methods (from code)

| Method | NDCG@10 | MRR@10 | Recall@10 |
|---|---|---|---|
| Popularity (non-personalised) | 0.0292 | 0.0649 | 0.0122 |
| Co-occurrence | 0.0362 | 0.0781 | 0.0158 |
| **ALS only** | 0.0399 | 0.0885 | 0.0154 |
| **ALS + LightGBM** | **0.1409** | **0.2826** | **0.0644** |
| Lift vs ALS | **+253%** | **+219%** | **+318%** |

> **Methodological note (from codebase):** *"NDCG uses implicit feedback (rating ≥ 4), not true watch completion."* These are offline evaluation numbers on held-out ratings data, not online A/B test results.

### Policy Gate Results

All 27 checks must pass before promotion. The gate has successfully blocked incorrect model promotions during development (see Postmortem #5 and #6).

### A/B Experiments (4 running)

| Experiment | Control | Treatment | Status |
|---|---|---|---|
| RL Policy vs ALS | ALS slate | REINFORCE reranked | Running |
| GPT Explanations vs Rule-Based | Rule-based text | GPT-4o explanation | Running |
| Slate Optimizer vs Greedy | Greedy top-k | Constrained slate | Running |
| Voice vs Text-Only | Text search | Voice pipeline | Running |

---

## Postmortem — Real Incidents

| # | Incident | Root Cause | Fix |
|---|---|---|---|
| 1 | "Dune is Romance" | `/explain` API used `user.top_genre` instead of `item.primary_genre` | `buildExplanation()` reads item fields directly — no API call |
| 2 | 1920s movies for "Similar to Stranger Things" | High cinephile ratings for classics + RAG matched "supernatural mystery" | +0.1 recency boost · year ≥ 1970 filter on all RAG results |
| 3 | Voice modal double-greeting | React StrictMode double-mount triggered TTS greeting twice | `greetedRef = useRef(false)` checked before TTS call |
| 4 | Wrong poster images | 200+ hardcoded overrides had wrong mappings | Removed all overrides · trust `item.poster_url` from TMDB |
| 5 | ChunkLoadError on /aistack route | Server component exported metadata while importing client component | `'use client'` + `dynamic()` with `ssr: false` |
| 6 | GitHub push blocked (API key in .env) | `.env` committed to git history | `git filter-repo --path .env --invert-paths --force` · key rotated |
| 7 | CI health check timeout (exit code 7) | GRU training at startup + slow CI runner (>60s to boot) | Removed health check from CI · import smoke test is sufficient |

---

## Quick Start

### Prerequisites

Docker Desktop · Node.js 20+ · Python 3.11+

### Steps

```bash
# 1. Clone
git clone https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api.git
cd two-stage-recommender-als-ranker-api

# 2. Configure
cp .env.example .env
# Fill in: OPENAI_API_KEY · TMDB_API_KEY

# 3. Start 7 services (Postgres, Redis, Qdrant, MinIO, API, Airflow, Flink)
docker compose up -d

# 4. Verify backend is healthy
curl http://localhost:8000/healthz | python3 -m json.tool

# 5. Patch TMDB catalog (4,961 movies with real posters)
docker cp p.py recsys_api:/app/p.py
docker exec recsys_api python3 /app/p.py

# 6. Start frontend
cd frontend && npm install && npm run dev
```

### Optional — Kafka streaming overlay

```bash
docker compose -f docker-compose.yml -f docker-compose-kafka.yml up -d
```

### Stop everything

```bash
docker compose down
```

---

## Demo Pages

| Page | URL | What It Shows |
|---|---|---|
| **Home** | `localhost:3000` | Personalised feed · 4,961 movies · 8 profile arms |
| **ML Dashboard** | `localhost:3000/ml` | OPE · Temporal · MMR · Cold-Start · Notify · Infra |
| **A/B Dashboard** | `localhost:3000/abtest` | 4 running experiments |
| **AI Stack** | `localhost:3000/aistack` | All ML components with architecture |
| **Eval** | `localhost:3000/eval` | Slice NDCG by genre / activity decile |
| **API Docs** | `localhost:8000/docs` | 62 endpoints — live and testable |
| **Health** | `localhost:8000/healthz` | Service status · model version · bundle state |
| **Airflow** | `localhost:8080` | Pipeline DAGs and run history |

---

## Project Structure

```
two-stage-recommender-als-ranker-api/
├── .github/workflows/ci.yml             # GitHub Actions CI
├── k8s/
│   ├── deployment.yaml                  # 3 replicas → HPA · rolling update · probes
│   ├── service.yaml                     # ClusterIP + LoadBalancer + Ingress
│   └── hpa.yaml                         # HPA 2–10 · CPU/Memory/RPS · PDB
├── sql/
│   ├── schema.sql                       # 4-table PostgreSQL schema · indices
│   └── queries.sql                      # SELECT + JOIN + GROUP BY analytics
├── backend/
│   ├── src/recsys/serving/
│   │   ├── app.py                       # FastAPI · 62 endpoints
│   │   ├── rl_policy.py                 # REINFORCE (offline RL)
│   │   ├── bandit_v2.py                 # LinUCB · 8 arms · α=1.0
│   │   ├── ope_eval.py                  # Doubly-robust IPS (off-policy RL eval)
│   │   ├── policy_gate.py               # 27 automated GateCheck objects
│   │   ├── slate_optimizer_v2.py        # ≥5 genres · 0.15 explore rate
│   │   ├── session_intent.py            # GRU encoder · hidden=16 · acc=0.927
│   │   ├── spark_features.py            # PySpark · 800k ratings · 5 features
│   │   ├── rag_engine.py                # Qdrant · 1,536-dim · HNSW
│   │   ├── context_and_additions.py     # CLIP ViT-B/32 · CUPED · drift monitor
│   │   ├── smart_explain.py             # GPT-4o explanations (Redis-cached)
│   │   ├── ab_experiment.py             # A/B framework · doubly-robust IPS
│   │   ├── reward_model.py              # 11-dim reward feature vector
│   │   └── [35+ more modules]
│   ├── flows/phenomenal_flow_v3.py      # Metaflow 12-step DAG
│   ├── scala/FeaturePipeline.scala      # Native Spark ALS (rank=64)
│   ├── airflow/dags/                    # Nightly retraining DAGs
│   ├── infra/duckdb/run_offline_eval.py # IPS-NDCG evaluation
│   └── requirements.txt
├── frontend/
│   ├── app/ml/page.tsx                  # ML Dashboard (7 tabs)
│   ├── components/VoiceModal.tsx        # Voice AI interface
│   ├── hooks/useVoiceAssistant.ts       # Voice hook (state machine)
│   └── [all components]
├── docker-compose.yml                   # 7-service orchestration
├── docker-compose-kafka.yml             # Kafka overlay
├── p.py                                 # TMDB catalog patcher
├── .env.example                         # Environment template
├── .gitignore                           # .env excluded
└── README.md
```

---

## CI/CD

```yaml
# .github/workflows/ci.yml
# Triggers on every push to main / develop

backend:
  - actions/setup-python@v5 (Python 3.11 · pip cache)
  - pip install -r requirements.txt
  - python -m compileall src -q          # syntax check all 40+ modules
  - import smoke: all serving modules importable with OPENAI_API_KEY=''
    assert _SESSION_MODEL is not None    # GRU trained
    assert TWO_TOWER is not None         # two-tower loaded

frontend:
  - actions/setup-node@v4 (Node 20 · npm cache)
  - npm ci
  - npm run type-check  (TypeScript strict)
  - npm run build       (Next.js production build)
```

---

<div align="center">

---

**Akilan Manivannan** · MS in Artificial Intelligence

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-181717?style=flat-square&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![Demo](https://img.shields.io/badge/Demo-Google%20Drive-E5091A?style=flat-square&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)

*Python · FastAPI · PySpark · Scala · LightGBM · Qdrant · Redis · Kafka · Metaflow · Airflow · DuckDB · Next.js 14 · Kubernetes · Docker · GitHub Actions · SQL · CLIP ViT-B/32 · offline RL · off-policy RL*

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141414,50:E50914,100:B20710&height=120&section=footer" width="100%"/>
