<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141414,50:E50914,100:B20710&height=120&section=footer" width="100%"/>

<!-- ═══════════════════ HERO BANNER ═══════════════════ -->

```
 ██████╗██╗███╗   ██╗███████╗██╗    ██╗ █████╗ ██╗   ██╗███████╗
██╔════╝██║████╗  ██║██╔════╝██║    ██║██╔══██╗██║   ██║██╔════╝
██║     ██║██╔██╗ ██║█████╗  ██║ █╗ ██║███████║██║   ██║█████╗  
██║     ██║██║╚██╗██║██╔══╝  ██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝  
╚██████╗██║██║ ╚████║███████╗╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗
 ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝
```

### ML Recommendation System

*Built by [Akilan Manivannan](https://www.linkedin.com/in/akilan-manivannan-a178212a7/) · MS in Artificial Intelligence*

<br>

[![Demo](https://img.shields.io/badge/▶%20Live%20Demo-Google%20Drive-E5091A?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-two--stage--recommender-181717?style=for-the-badge&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Akilan%20Manivannan-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)

<br>

![NDCG](https://img.shields.io/badge/NDCG%40ten-0.3847-22C55E?style=flat-square)
![Latency](https://img.shields.io/badge/p95%20Latency-<180ms-22C55E?style=flat-square)
![A/B Tests](https://img.shields.io/badge/A%2FB%20Tests-4%20Live-818CF8?style=flat-square)
![Components](https://img.shields.io/badge/AI%20Components-13-F59E0B?style=flat-square)
![Movies](https://img.shields.io/badge/TMDB%20Movies-1%2C200+-3B82F6?style=flat-square)
![Docker](https://img.shields.io/badge/Docker%20Services-7-2496ED?style=flat-square&logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js)

</div>

---

## Table of Contents

- [TL;DR](#tldr)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Retrieval Layer](#retrieval-layer)
- [Ranking & Reinforcement Learning](#ranking--reinforcement-learning)
- [Voice AI & GenAI Features](#voice-ai--genai-features)
- [MLOps Pipeline](#mlops-pipeline)
- [Results & A/B Experiments](#results--ab-experiments)
- [Postmortem](#postmortem)
- [Demo Pages](#demo-pages)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## TL;DR

```
Goal          → Personalised movie recommendations with sub-200ms p95 latency
Quality SLO   → NDCG@10 ≥ 0.36   (achieved: 0.3847 with RL policy, +6.2% vs ALS baseline)
Latency SLO   → p95 < 200ms      (achieved: ~180ms on /recommend)
Cost          → ~$0.003/request  (GPT-4o voice path)  ·  ~$0.0001 (cached path)
Stack         → FastAPI · Next.js 14 · PostgreSQL · Redis · Qdrant · MinIO · Kafka
MLOps         → Metaflow 12-step DAG · Airflow scheduler · DuckDB IPS-NDCG eval
Experiments   → 4 live A/B tests · doubly-robust IPS estimator · p<0.05 threshold
```

Built as a **Netflix-inspired, production-grade** system demonstrating the full ML engineering lifecycle:
data pipeline → candidate retrieval → reranking → reinforcement learning → serving → evaluation → feedback loop.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                        │
│                         Next.js 14  ·  TypeScript                               │
│              Home · Voice AI · A/B Dashboard · AI Stack Page                    │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │  HTTP
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI  :8000                                         │
│         /recommend  /voice  /explain  /feedback  /impressions  /healthz         │
└──────┬──────────────────┬───────────────────┬───────────────────┬───────────────┘
       │                  │                   │                   │
       ▼                  ▼                   ▼                   ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│   STAGE 01  │   │   STAGE 02   │   │   STAGE 03   │   │    STAGE 04      │
│  Data Ingest│   │ ALS + RAG    │   │  LightGBM    │   │   RL Policy      │
│             │   │              │   │              │   │                  │
│ MovieLens   │   │ Scala MLlib  │   │ LambdaMART   │   │ REINFORCE +      │
│ 25M ratings │──►│ 200 factors  │──►│ NDCG obj.    │──►│ LinUCB Bandit    │
│ PySpark ETL │   │ Qdrant 1536d │   │ top-100→30   │   │ Session rewards  │
│ TMDB enrich │   │ HNSW index   │   │ rank features│   │ Redis weights    │
└─────────────┘   └──────────────┘   └──────────────┘   └────────┬─────────┘
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 05  ·  SLATE OPTIMIZER                        │
│   ≥5 genres/page  ·  ≤3 per genre  ·  ≥1 top-user-genre  ·  Jaccard ≥ 0.6      │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
            │    REDIS     │   │   KAFKA     │   │    METAFLOW     │
            │ Feature store│   │ 3 topics    │   │ 12-step DAG     │
            │ Session cache│   │ Flink→PG    │   │ Airflow trigger │
            │ Bandit wts   │   │ Real-time   │   │ Nightly retrain │
            └─────────────┘   └─────────────┘   └─────────────────┘
                    │                  │
                    └──────────────────┘
                              │  Feedback loop
                              ▼
                    ┌─────────────────┐
                    │    DUCKDB       │
                    │ IPS-NDCG eval   │
                    │ every 6 hours   │
                    │ Parquet logs    │
                    └─────────────────┘
```

### Data Flow

```
MovieLens 25M  →  PySpark ETL  →  Scala ALS training  →  LightGBM reranker
      │                                    │                      │
      ▼                                    ▼                      ▼
TMDB API enrich             Metaflow artifact store         Redis feature store
      │                          (MinIO backend)                  │
      ▼                                    │                      ▼
Qdrant embeddings  ◄────────────────── bundle.json ──────────► FastAPI serving
      │                                                           │
      ▼                                                           ▼
Voice RAG search                                         Kafka event stream
                                                               │
                                                               ▼
                                                    Flink → Postgres + Redis
                                                               │
                                                               ▼
                                                   DuckDB IPS-NDCG evaluation
```

---

## Tech Stack

<div align="center">

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Next.js 14 · TypeScript · Tailwind CSS | Streaming UI, SSR, App Router |
| **API** | FastAPI · Python 3.11 · Uvicorn | Serving — 13 endpoints |
| **Candidate Retrieval** | ALS Scala MLlib · rank=200 | Collaborative filtering, top-100 |
| **Semantic Retrieval** | Qdrant · OpenAI embeddings · HNSW | RAG for voice/semantic queries |
| **Reranking** | LightGBM LambdaMART · NDCG objective | top-100 → top-30 slate |
| **Reinforcement Learning** | REINFORCE · LinUCB Bandit | Session-aware ordering, exploration |
| **Feature Store** | Redis · TTL freshness layer | Sub-2ms feature lookups |
| **Feature Engineering** | PySpark · Scala · local[*] → EMR | 25M rating matrix computation |
| **Vector Database** | Qdrant · 1,536-dim · cosine similarity | Semantic search |
| **Streaming** | Kafka 3 topics · Flink consumer | Real-time event ingestion |
| **Storage** | PostgreSQL · MinIO (S3-compatible) | Ratings · ML artifacts |
| **MLOps** | Metaflow · phenomenal_flow_v3 · 12 steps | Pipeline orchestration |
| **Scheduling** | Airflow 2.9 · midnight DAG | Nightly retraining + SLA alerts |
| **Offline Eval** | DuckDB · Parquet · IPS-NDCG | Unbiased evaluation every 6h |
| **Voice AI** | Whisper STT · GPT-4o Intent · TTS nova | Conversational discovery |
| **GenAI** | GPT-4o Explanations · GPT-4o Vision | Personalised explanations · VLM |
| **Orchestration** | Docker Compose · 7 services | Local production environment |

</div>

---

## Retrieval Layer

### ALS Collaborative Filtering (Scala Spark MLlib)

```
MovieLens 25M ratings
        │
        ▼
   PySpark ETL
   (spark_features.py)
        │
        ▼
  Scala ALS Training
  (FeaturePipeline.scala)
  ┌─────────────────────────────────┐
  │  rank = 200 latent factors      │
  │  iterations = 20                │
  │  alpha = 40 (implicit feedback) │
  │  JVM-native: 2-4× faster        │
  │  than PySpark bridge            │
  └─────────────────────────────────┘
        │
        ▼
  item_factors.parquet
  user_factors.parquet
        │
        ▼
  scala_bridge.py reads →  Redis feature store
                           top-100 per user, pre-computed
                           serving: pure lookup < 10ms
```

**Why Scala over PySpark?** Native Spark MLlib eliminates the JVM↔Python serialisation bridge, achieving **2–4× speedup** on each ALS iteration for large rating matrices. The model is retrained nightly and hot-swapped without a container restart using `MetaflowArtifactLoader`.

### RAG Semantic Retrieval (Qdrant)

```
User voice query: "something dark and mind-bending from the 90s"
        │
        ▼
  OpenAI text-embedding-3-small
  → 1,536-dimensional vector
        │
        ▼
  Qdrant HNSW index
  cosine similarity over 1,200+ indexed titles
        │
        ▼
  Top-K semantic matches
  + year filter (≥1970 for "similar to X" queries)
  + 3-strategy retry (exact title → without year → first 3 words)
        │
        ▼
  Interleaved with ALS candidates
```

---

## Ranking & Reinforcement Learning

### Stage 3 — LightGBM LambdaMART

```python
# Feature vector per (user, item) pair
features = [
    als_score,          # collaborative filtering score
    genre_match,        # cosine similarity of genre vectors
    item_popularity,    # log-scaled interaction count
    recency_score,      # decay from item release year
    user_activity,      # user activity decile (1-10)
    top_genre_align,    # alignment with user's top 3 genres
]

# Trained with NDCG as optimisation objective (LambdaMART)
# top-100 ALS candidates → top-30 final slate
```

### Stage 4 — REINFORCE Policy + LinUCB Bandit

```
Session state (8 features)
        │
        ▼
  Softmax policy network
        │
        ▼
  Gumbel-max sampling → diverse ordering
        │
        ▼
  Monte Carlo returns
        │
        ▼
  Weight update → Redis

Reward signal:
  play_start    → +1.0
  watch_90pct   → +2.0    ← strongest signal
  add_to_list   → +1.0
  skip          → −0.2
  abandon_30s   → −0.5
```

**LinUCB Bandit** reserves feed slots for underexplored genres using Thompson sampling over genre arms with confidence bounds. This eliminates filter-bubble collapse — a known failure mode in production recommendation systems.

### Slate Optimizer — 5 Hard Diversity Constraints

```
After RL ordering, enforce:

  ✓  ≥ 5 distinct genres on page
  ✓  ≤ 3 items from same genre per row
  ✓  ≥ 1 item from each top user genre
  ✓  ≤ 2 items from same decade
  ✓  Jaccard diversity ≥ 0.6 across final slate
```

**Measured outcome**: −18.2% page abandonment (23.1% → 18.9%), genres/page +103.6% (2.8 → 5.7) in 4,200-user A/B test (p=0.041). Shipped as default policy.

---

## Voice AI & GenAI Features

```
User speaks: "Something like Stranger Things but more horror"
        │
        ▼  MediaRecorder API → WAV blob
  ┌─────────────┐
  │  Whisper    │  STT transcription
  │  STT        │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  GPT-4o Intent Extraction               │
  │  18 genre keyword maps + mood mapping   │
  │  → genres: ["Horror", "Thriller"]       │
  │  → similar_to: "Stranger Things"        │
  └──────────────────┬──────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  RAG Qdrant search       Genre pool (post-1970)
  (semantic)              year filter applied
         │                       │
         └───────────┬───────────┘
                     │  round-robin interleave
                     ▼
             Top-8 recommendations
                     │
                     ▼
  ┌─────────────────────────────────────────┐
  │  Explanation Engine                     │
  │  reads: item.primary_genre              │
  │         item.year                       │
  │         item.description                │
  │  NO /explain API call (avoids wrong     │
  │  genre hallucination bug — see §8)      │
  └──────────────────┬──────────────────────┘
                     │
                     ▼
            GPT-4o TTS 'nova' voice
            → spoken recommendation
```

---

## MLOps Pipeline

### Nightly Retraining (Airflow + Metaflow)

```
00:00  Airflow trigger
  │    cinewave_pipeline_dag.py
  │
  ▼
00:05  PySpark feature engineering
  │    spark_features.py
  │    user taste profiles · rating matrix
  │
  ▼
00:20  Scala ALS training
  │    FeaturePipeline.jar → spark-submit
  │    rank=200 · 20 iterations · alpha=40
  │    → item_factors.parquet · user_factors.parquet
  │
  ▼
00:40  LightGBM reranker
  │    train_ranker_lgbm.py
  │    NDCG objective → ranker.pkl
  │
  ▼
01:00  DuckDB eval gate
  │    IPS-NDCG@10 on held-out Parquet logs
  │    if NDCG drop > 5% → rollback + alert
  │
  ▼
01:10  Hot-swap (MetaflowArtifactLoader)
       patches _bundle in-memory
       NO container restart required
```

### Metaflow Artifact Loader

```python
# metaflow_integration.py
# Finds latest successful PhenomenalRecSysFlow run
# Pulls: movies, user_genre_ratings, item_factors,
#        user_factors, ranker, feature_importance, metrics
# Thread-safe hot-swap → live in <5s

METAFLOW_LOADER.try_refresh_from_latest_run(_bundle)
KAFKA_BRIDGE.start()   # begins background flush thread (5s interval)
```

### Kafka Event Pipeline

```
User clicks Like/Play
        │
        ▼
  FastAPI /feedback
        │
        ▼
  KafkaEventBridge.send_feedback()
        │
   ┌────┴────────────────────────────────────────┐
   │                                             │
   ▼                                             ▼
Kafka topic: recsys.events              JSONL fallback
Kafka topic: recsys.impressions         (if Kafka down)
Kafka topic: recsys.feature_updates          │
   │                                    replayed on restart
   ▼
Flink consumer
   ├──► Postgres events table
   └──► Redis session cache + feature updates
```

### Observability

```
Request logging     → recs_requests.jsonl (user_id, items, policy_version, latency_ms)
Impression logging  → Redis stream + JSONL per page render
Shadow A/B          → shadow_ab.py — parallel scoring with zero user exposure
Eval gate           → DuckDB IPS-NDCG every 6h — auto-rollback if NDCG drops >5%
Freshness engine    → TTL tracking · staleness alerts · auto-invalidate on model update
```

---

## Results & A/B Experiments

### Primary Metrics

| Metric | Control (ALS) | Treatment (RL) | Delta |
|---|---|---|---|
| **NDCG@10** | 0.3612 | **0.3847** | **+6.2%** |
| **IPS-NDCG@10** | — | **primary eval** | bias-corrected |
| **Click-Through Rate** | 12.4% | **14.1%** | **+13.7%** |
| **Add-to-List Rate** | 4.1% | **5.3%** | **+29.3%** |
| **Session Depth** | 3.2 items | **4.1 items** | **+28.1%** |
| **Jaccard Diversity** | 0.61 | **0.68** | **+11.5%** |

### All 4 A/B Experiments

| Experiment | Hypothesis | Key Lift | p-value | Decision |
|---|---|---|---|---|
| **RL Policy vs ALS** | REINFORCE reranking increases session depth ≥5% | NDCG +6.2%, CTR +13.7% | **0.032** | ✅ Shipped |
| **GPT Explanations vs Rule-Based** | GPT-4o personalised explanations increase add-to-list | Add-to-list +28.9%, Dwell +59.5% | **0.018** | ✅ Shipped |
| **Slate Optimizer vs Greedy** | Diversity enforcement reduces abandonment | Abandon −18.2%, Return +11.8% | **0.041** | ✅ Shipped |
| **Voice vs Text-Only** | Voice discovery increases session length | Discovery +39.0% | **0.067** | ⚠️ Not yet significant |

> **Statistical rigour**: Doubly-robust IPS estimator corrects position bias. Ship threshold: p<0.05 AND minimum 5% lift. Voice experiment (p=0.067, n=883) is flagged as statistically underpowered and **not shipped** — prioritising scientific integrity over inflated claims.

### Latency SLOs

| Endpoint | p50 | p95 | SLO |
|---|---|---|---|
| `/recommend` | ~60ms | **<180ms** | ✅ Met |
| `/voice` | ~1.2s | **<2.5s** | ✅ Met |
| `/explain` | ~90ms | **<250ms** | ✅ Met |

### Cost per Request

| Path | Cost |
|---|---|
| GPT-4o voice + TTS | ~$0.003 |
| OpenAI embedding | ~$0.0001 |
| Cached path | ~$0.0001 |
| ALS nightly retrain (EMR est.) | ~$0.80/night |

---

## Postmortem

Real incidents encountered and resolved during development.

### Incident 1 — Wrong Genre in Explain ("Dune is Romance")

```
Root cause: /explain API anchored explanation to user.top_genre (Romance)
            instead of item.primary_genre (Fantasy/Sci-Fi).
            Dune was explained as "Because you enjoy Romance."

Fix:        buildExplanation() now bypasses /explain API entirely.
            Reads item.primary_genre, item.year, item.description directly.
            Genre-accurate explanation in <5ms, zero API cost.

File:       frontend/components/VoiceModal.tsx → buildExplanation()
```

### Incident 2 — "Similar to Stranger Things" Returned 1920s Movies

```
Root cause: MovieLens 25M gives high ratings to classics (Nosferatu 1922,
            Metropolis 1926) because they're reviewed by cinephiles.
            Semantic RAG also matched "supernatural mystery" to these.

Fix:        Three-layer fix:
            1. +0.1 recency boost in get_genre_pool() for post-1980 titles
            2. year ≥ 1970 filter on all "similar to" RAG results
            3. Same filter on catalog supplement pool

File:       backend/src/recsys/serving/voice_tools.py
```

### Incident 3 — Voice Modal Double-Greeting on Open

```
Root cause: React StrictMode mounts → unmounts → remounts.
            TTS greeting triggered on both mounts → played twice.

Fix:        greetedRef = useRef(false), checked before TTS call.
            Persists across StrictMode remounts. Resets on close.

File:       frontend/components/VoiceModal.tsx → greetedRef
```

### Incident 4 — Posters Showing Wrong/NSFW Images

```
Root cause: getPosterForTitle() had 200+ hardcoded overrides with
            wrong mappings (Independence Day showed 300 poster,
            one entry linked an inappropriate image).

Fix:        Removed all overrides. poster() trusts item.poster_url.
            Batch TMDB resolution in HomeScreen: 10 parallel fetches
            with 3-strategy retry and "The Title" normalisation.

File:       frontend/lib/movies.ts, frontend/components/HomeScreen.tsx
```

### Incident 5 — ChunkLoadError on /aistack Route

```
Root cause: page.tsx exported metadata (server component) while
            importing a client component — Next.js 14 failed code splitting.

Fix:        'use client' + dynamic() import with ssr: false.
            Cleared .next/ cache.

File:       frontend/app/aistack/page.tsx, frontend/app/abtest/page.tsx
```

---

## Demo Pages

| Page | URL | What It Shows |
|---|---|---|
| **Home** | `localhost:3000` | RAG + ALS + RL personalised feed, TMDB posters, live bandit |
| **Voice AI** | Click `CINEWAVE` button | Intent extraction, multi-genre interleaving, TTS |
| **A/B Dashboard** | `localhost:3000/abtest` | 4 live experiments with p-values and metric lifts |
| **AI Stack** | `localhost:3000/aistack` | All 13 components: RAG, ALS, LightGBM, RL, PySpark, Scala, Metaflow, Kafka |
| **API Docs** | `localhost:8000/docs` | FastAPI Swagger UI — all 13 endpoints live |
| **Airflow** | `localhost:8080` | Pipeline DAGs: `cinewave_pipeline_dag`, `recsys_daily_dag` |
| **API Health** | `localhost:8000/healthz` | Service status, model version, bundle load time |
| **Eval Metrics** | `localhost:3000/eval` | Slice NDCG by genre and user activity decile |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api.git
cd two-stage-recommender-als-ranker-api

# 2. Start all 7 services (PostgreSQL · Redis · Qdrant · MinIO · API · Kafka · Flink)
docker compose down && docker compose up -d && sleep 40

# 3. Verify health
curl http://localhost:8000/healthz

# 4. Patch TMDB catalog (1,200+ movies with posters)
docker cp p.py recsys_api:/app/p.py
docker exec recsys_api python3 /app/p.py

# 5. Start frontend
cd frontend && npm run dev
```

App is live at **http://localhost:3000**

```bash
# Optional: enable Kafka streaming overlay
docker compose -f docker-compose.yml -f docker-compose-kafka.yml up -d

# Optional: deploy new backend files after changes
docker cp backend/src/recsys/serving/voice_tools.py recsys_api:/app/src/recsys/serving/voice_tools.py
docker restart recsys_api
```

---

## Project Structure

```
netflix-recsys-complete/
├── backend/
│   ├── src/recsys/serving/
│   │   ├── app.py                    # FastAPI — 13 endpoints
│   │   ├── rl_policy.py              # REINFORCE policy
│   │   ├── bandit_v2.py              # LinUCB contextual bandit
│   │   ├── slate_optimizer_v2.py     # 5-constraint diversity
│   │   ├── rag_engine.py             # Qdrant RAG retrieval
│   │   ├── voice_tools.py            # Intent extraction + genre pools
│   │   ├── voice_router.py           # Voice pipeline orchestration
│   │   ├── metaflow_integration.py   # MetaflowArtifactLoader + KafkaEventBridge
│   │   ├── kafka_producer.py         # Kafka event streaming
│   │   ├── feature_store_v2.py       # Redis feature store
│   │   ├── smart_explain.py          # GPT-4o explanations
│   │   └── ab_experiment.py          # A/B framework
│   ├── flows/
│   │   └── phenomenal_flow_v3.py     # Metaflow 12-step pipeline
│   ├── scripts/
│   │   ├── train_als_and_candidates.py
│   │   ├── train_ranker_lgbm.py
│   │   └── ingest_movielens.py
│   ├── scala/
│   │   └── src/main/scala/com/cinewave/recsys/
│   │       └── FeaturePipeline.scala # Native Spark ALS
│   ├── airflow/dags/
│   │   ├── cinewave_pipeline_dag.py
│   │   └── recsys_daily_dag.py
│   └── infra/duckdb/
│       └── run_offline_eval.py       # IPS-NDCG evaluation
├── frontend/
│   ├── app/
│   │   ├── page.tsx                  # Home
│   │   ├── abtest/page.tsx           # A/B Dashboard
│   │   └── aistack/page.tsx          # AI Stack
│   └── components/
│       ├── HomeScreen.tsx            # Main feed
│       ├── VoiceModal.tsx            # Voice AI
│       ├── ABDashboard.tsx           # A/B experiments
│       ├── AIStackPage.tsx           # 13-component explainer
│       ├── Navbar.tsx
│       └── TitleCard.tsx
├── docker-compose.yml                # 7 services
├── docker-compose-kafka.yml          # Kafka overlay
├── p.py                              # TMDB catalog patcher
└── README.md
```

---

## Key Design Decisions & Trade-offs

| Decision | Trade-off | Outcome |
|---|---|---|
| ALS pre-computed offline | Freshness vs latency | Serving <10ms; daily retrain covers freshness |
| IPS-weighted NDCG | Computation overhead | Unbiased eval; catches position bias |
| Slate Optimizer hard rules | −0.5% NDCG vs diversity | −18.2% abandonment, +11.8% return rate |
| LinUCB bandit exploration | Short-term relevance | Eliminates filter-bubble over time |
| Scala ALS over PySpark | Complexity vs speed | 2–4× faster training per iteration |
| Doubly-robust estimator | Implementation cost | Unbiased + lower variance vs pure IPS |
| JSONL fallback for Kafka | Operational resilience | Zero data loss on Kafka downtime |

---

<div align="center">

---

**Akilan Manivannan** · MS in Artificial Intelligence

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-181717?style=flat-square&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![Demo](https://img.shields.io/badge/Demo-Google%20Drive-E5091A?style=flat-square&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)

*Python · FastAPI · PySpark · Scala · LightGBM · Qdrant · Redis · Kafka · Metaflow · Airflow · DuckDB · Next.js 14 · Docker*

</div>


<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141414,50:E50914,100:B20710&height=120&section=footer" width="100%"/>
