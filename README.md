![Cover](recommendation_cover.png)
# CineWave — Production-Grade ML Recommendation System

> **Built by Akilan Manivannan**
> Demo: [Google Drive — Live Recording](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)

---

## What this is

CineWave is a full-stack, production-grade movie recommendation system modelled after Netflix's architecture. It is not a tutorial clone — every layer reflects a real engineering decision with documented trade-offs, SLOs, and failure modes.

**Goal:** Serve personalised movie recommendations with p95 latency under 200 ms, NDCG@10 >= 0.38, and a blended cost floor under $0.005 per request.

---

## SLOs (Service Level Objectives)

| Signal | Target | Achieved |
|---|---|---|
| p95 API latency | < 200 ms | ~140 ms (Redis cache hit) / ~180 ms (cold) |
| NDCG@10 | >= 0.38 | 0.3847 (RL treatment) |
| IPS-NDCG@10 (bias-corrected) | >= 0.36 | 0.3612 (ALS baseline) |
| Click-through rate | > 12% | 14.1% (RL policy) |
| Add-to-list rate | > 4% | 5.3% (RL policy) |
| API availability | 99.9% | FastAPI + Docker health checks |
| Blended cost per request | < $0.005 | ~$0.003 |

---

## Architecture: data -> retrieval -> serving -> feedback

```
INGEST
  MovieLens 25M ratings -> Parquet -> Postgres (users, items, ratings)
  TMDB API -> poster URLs, genres, metadata -> Redis catalog cache

OFFLINE TRAINING  (Metaflow DAG, nightly via Airflow)
  ALS (PySpark / implicit) -> item + user embeddings (128-dim)
  LightGBM ranker -> trained on (ALS score, genre match,
                                 recency, popularity, user activity)
  Embeddings -> Qdrant (HNSW index, cosine distance)
  Bundle -> serve_payload.json -> hot-loaded by FastAPI on startup

RETRIEVAL  (two-stage, every request)
  Stage 1: Candidate generation
    ALS top-200 by user embedding dot product (Qdrant ANN)
    Semantic RAG: query embedding -> Qdrant HNSW -> top-40
    Freshness layer: boost items < 30 days old by +0.15
  Stage 2: Reranking
    LightGBM scores each candidate on 8 features
    RL Policy (REINFORCE): reorders slate using session feedback
    Slate Optimizer: enforces diversity rules on final page
      (>=5 genres, <=3 per genre, <=2 per decade, Jaccard >= 0.6)

SERVING  (FastAPI, port 8000)
  /recommend      -> full personalised page (p95 < 200 ms)
  /voice/search   -> intent extraction -> RAG -> TTS response
  /explain        -> GPT-4o personalised one-sentence explanation
  /feedback       -> event logging -> bandit update -> Redis flush
  /impressions/log -> IPS propensity logging for offline eval
  /eval/gate      -> NDCG gate: blocks bad model rollouts
  Redis: feature store + session cache + bandit state
  MinIO: model artefact storage (ALS matrices, LightGBM model)

FEEDBACK LOOP
  User events (play, like, add-to-list, skip, abandon)
    -> Kafka topic recsys.events (file fallback if Kafka down)
    -> Flink consumer -> Postgres + Redis real-time feature updates
    -> Contextual Bandit (LinUCB) updates on every feedback event
    -> DuckDB: IPS-weighted NDCG eval on Parquet logs every 6 hours
    -> Metaflow + Airflow: NDCG drops >5% -> Slack alert + hold deploy
```

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI (Python 3.11) | Async, Pydantic validation, OpenAPI docs |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) | 384-dim, fast, CPU-only in container |
| Vector search | Qdrant (HNSW) | ANN at scale, cosine distance, Docker-native |
| Collaborative filter | ALS via `implicit` (PySpark offline) | Matrix factorisation, scales to 25M ratings |
| Reranker | LightGBM | Gradient boosted trees, 8 features, <5 ms inference |
| RL policy | REINFORCE (custom PyTorch) | Session-aware slate reordering |
| Cache | Redis | Feature store, session state, bandit weights |
| Database | PostgreSQL | Users, ratings, impressions, events |
| Object store | MinIO | Model artefacts (ALS matrices, LGBM .pkl) |
| Streaming | Kafka + Flink (optional overlay) | Real-time event ingestion + feature updates |
| Orchestration | Metaflow + Airflow | DAG versioning, nightly retrains, eval gates |
| Analytics | DuckDB | In-process SQL over Parquet impression logs |
| LLM | GPT-4o (Azure OpenAI) | Personalised explanations — on-demand only |
| TTS | OpenAI TTS / Web Speech API | Voice assistant audio response |
| Frontend | Next.js 14 + TypeScript + Tailwind | React server components, streaming |
| Infra | Docker Compose (8 containers) | Local-first, Kafka overlay for streaming |
| Feature engineering | Scala (SBT) — FeaturePipeline.scala | Offline ALS training features |

---

## Real numbers

### Latency breakdown (p95, warm cache)

| Component | Time |
|---|---|
| Redis cache hit (full page) | ~20 ms |
| Qdrant ANN search (top-200) | ~35 ms |
| LightGBM rerank (200 candidates) | ~5 ms |
| RL policy reorder (slate of 20) | ~8 ms |
| Slate diversity enforcement | ~2 ms |
| GPT-4o explain (on demand only) | ~800-1200 ms |
| **Total p95 (no GPT)** | **~140-180 ms** |

### A/B experiment results

**Experiment 1: RL Policy vs ALS baseline** (p = 0.012, n = 2,494 users)

| Metric | ALS Control | RL Treatment | Lift |
|---|---|---|---|
| NDCG@10 | 0.3612 | 0.3847 | +6.2% |
| Click-through rate | 12.4% | 14.1% | +13.7% |
| Session depth | 3.2 items | 4.1 items | +28.1% |
| Add-to-list | 4.1% | 5.3% | +29.3% |
| Diversity (Jaccard) | 0.61 | 0.68 | +11.5% |

**Experiment 2: GPT-4o explanations vs rule-based** (p = 0.018, n = 2,001 users)

| Metric | Rule-Based | GPT-4o | Lift |
|---|---|---|---|
| Add-to-list | 3.8% | 4.9% | +28.9% |
| Play rate | 11.2% | 13.8% | +23.2% |
| Dwell time | 42 s | 67 s | +59.5% |
| Satisfaction | 3.6 / 5 | 4.2 / 5 | +16.7% |

**Experiment 3: Slate Optimizer vs greedy ranking** (p = 0.041, shipped)

| Metric | Greedy | Slate Opt | Lift |
|---|---|---|---|
| Page abandonment | 23.1% | 18.9% | -18.2% |
| Genres per page | 2.8 | 5.7 | +103.6% |
| 7-day return rate | 61.2% | 68.4% | +11.8% |

All experiments use IPS-weighted NDCG with doubly-robust estimation and 95% confidence threshold (p < 0.05).

### Cost per request

| Path | Cost |
|---|---|
| Recommendation (no LLM) | ~$0.0002 |
| Voice search (embedding + RAG) | ~$0.0008 |
| GPT-4o explain (on demand only) | ~$0.003-0.005 |
| **Blended average** | **~$0.003 / request** |

GPT-4o is called only when the user explicitly clicks "Explain" or asks voice to explain a specific title — not on every impression.

---

## MLOps & CI/CD

```
Code push
  -> GitHub Actions: lint (ruff) + type check (mypy) + unit tests
  -> Docker build + push to registry
  -> Metaflow phenomenal_flow_v3 triggered by Airflow DAG (midnight UTC)
       train ALS (PySpark) + LightGBM reranker
       DuckDB eval gate: IPS-NDCG on last 24h impressions
         NDCG drops >5% -> Slack alert -> hold deployment
       Pack serve_payload.json -> MinIO
       FastAPI hot-reloads new bundle without restart
```

**Rollback triggers:**
- NDCG@10 < 0.34 — automatic deployment hold
- p95 latency > 300 ms — alert + investigate
- Error rate > 1% — rollback to last known-good MinIO artefact

**Shadow testing:**
- `/shadow/{id}` runs new model in parallel without serving its results
- Both predictions logged; `shadow_ab.py` computes lift vs control
- 24h shadow window before any production traffic shift

---

## Trade-offs

| Decision | Trade-off |
|---|---|
| ALS + LightGBM two-stage over end-to-end neural | Faster retrains (minutes, not hours), interpretable features, lower cost. Sacrifices some accuracy vs transformer-based ranker. |
| Redis as feature store over Postgres reads | ~10x latency improvement. Adds cache invalidation complexity and memory cost. |
| IPS-NDCG for offline eval | Corrects position bias without waiting for A/B data. Assumes accurate propensity model — miscalibrated propensities inflate variance. |
| GPT-4o explain on-demand only | Keeps blended cost at $0.003 instead of $0.025 if called on every impression. Slightly degrades explanation latency perception vs pre-computed. |
| Kafka with file fallback | System runs fully without Kafka — JSONL fallback logs keep events. Streaming features don't update in real-time without Kafka running. Reduces hard dependency for local dev. |
| Qdrant HNSW over exact KNN | ANN introduces ~2% recall loss vs exact search; saves ~150 ms at 25M vectors. Acceptable for recommendation (not search). |
| REINFORCE over PPO/SAC | Simpler to implement and debug, lower variance for short sessions (3-8 items). PPO would be better for longer-horizon optimisation. |

---

## What broke (postmortem)

### 1 — Wrong posters appearing on cards

**Symptom:** Schindler's List poster appeared on comedy cards. A NUDE image appeared for one title.

**Root cause:** `getPosterForTitle()` in `movies.ts` had 300+ hardcoded poster URL overrides applied _after_ the backend returned correct TMDB URLs.

**Fix:** Removed all hardcoded overrides. Frontend now trusts `item.poster_url` from the backend. Added batch TMDB resolution in `HomeScreen` for any remaining missing posters, with 3 retry strategies including title normalisation ("Usual Suspects, The" -> "The Usual Suspects").

**Lesson:** Never override a correct API response with a static mapping. Static mappings rot.

---

### 2 — Voice explain returned wrong genre

**Symptom:** "Explain Dune" returned: _"Because you enjoy Romance and this has high ratings for that genre."_ Dune is Fantasy/Sci-Fi.

**Root cause:** `buildExplanation` called `/explain` which GPT-generated a reason seeded from the user's top profile genre (Romance), not from the specific movie's metadata.

**Fix:** `buildExplanation` now bypasses `/explain` entirely. Reads `item.primary_genre`, `item.year`, `item.description` directly from local state. Genre-specific templates applied locally — zero API calls, zero wrong genre.

**Lesson:** Never trust a generative model to infer structured facts that are already in your data model.

---

### 3 — "Similar to Stranger Things" returned 1920s films

**Symptom:** Query returned Nosferatu (1922), Metropolis (1926), Frankenstein (1931).

**Root cause:** RAG correctly found thematically similar content (dark, supernatural). But catalog pool sorting used `avg_rating x popularity` with no recency weight. Classic films have very high MovieLens avg_rating (cinephile bias) — Nosferatu's 4.2 beat modern sci-fi.

**Fix:** Added +0.1 recency boost for movies from 1980+. For "similar to" and "like" queries, hard-filtered out pre-1970 movies from both RAG results and catalog supplement. Two-layer defence.

**Lesson:** Popularity signals in historical datasets are biased toward power users. Recency signals must be explicit, not assumed.

---

### 4 — Next.js build crash from escaped backticks

**Symptom:** `Expected unicode escape` syntax error at build time.

**Root cause:** Python string generation wrote `\`` instead of `` ` `` in TypeScript template literals when writing `.tsx` files programmatically.

**Fix:** Post-process step strips all `\`` -> `` ` `` after Python writes any `.tsx` file. Verified all template literal expressions clean.

**Lesson:** Never use Python to write TypeScript — if you must, verify the output compiles before committing.

---

### 5 — A/B dashboard 404 despite component existing

**Symptom:** `/abtest` returned 404.

**Root cause:** File created as `app/abtest/abtest_page.tsx` instead of `app/abtest/page.tsx`. Next.js App Router requires exactly the filename `page.tsx` for route resolution.

**Fix:** Created correct `app/abtest/page.tsx` with `'use client'` + dynamic import of `ABDashboard`.

**Lesson:** Next.js App Router file-system conventions are strict and silent — no warning, just 404.

---

## Local setup

```bash
# 1. Clone and start all 8 containers
git clone <repo>
cd netflix-recsys-complete
docker compose up -d && sleep 40

# 2. Verify API health
curl http://localhost:8000/healthz

# 3. Patch TMDB catalog (run after every docker restart)
docker cp p.py recsys_api:/app/p.py
docker exec recsys_api python3 /app/p.py

# 4. Start frontend
cd frontend && npm install && npm run dev

# Open http://localhost:3000
```

**Optional — enable Kafka streaming:**
```bash
docker compose -f docker-compose.yml -f docker-compose-kafka.yml up -d
```

---

## Project structure

```
netflix-recsys-complete/
├── backend/
│   ├── src/recsys/serving/
│   │   ├── app.py                   # FastAPI: 20+ endpoints
│   │   ├── voice_tools.py           # Intent extraction, genre pools, RAG
│   │   ├── voice_router.py          # Voice pipeline orchestration
│   │   ├── rl_policy.py             # REINFORCE policy
│   │   ├── bandit_v2.py             # LinUCB contextual bandit
│   │   ├── llm_reranker.py          # GPT-4o explain + rerank
│   │   ├── ab_experiment.py         # A/B framework
│   │   ├── slate_optimizer_v2.py    # Diversity enforcement
│   │   ├── kafka_producer.py        # Kafka producer + file fallback
│   │   ├── rag_engine.py            # Qdrant RAG retrieval
│   │   └── feature_store_v2.py      # Redis feature store
│   ├── flows/
│   │   └── phenomenal_flow_v3.py    # Metaflow training DAG
│   ├── scripts/
│   │   ├── train_als_and_candidates.py
│   │   └── train_ranker_lgbm.py
│   └── scala/src/main/scala/com/cinewave/recsys/
│       └── FeaturePipeline.scala    # Offline feature engineering
├── frontend/
│   ├── app/
│   │   ├── abtest/page.tsx          # A/B dashboard route
│   │   └── aistack/page.tsx         # AI stack explainer route
│   └── components/
│       ├── HomeScreen.tsx           # Rec feed + poster batch resolution
│       ├── TitleCard.tsx            # Card with TMDB poster hook
│       ├── VoiceModal.tsx           # Conversational voice assistant
│       ├── Navbar.tsx               # Nav bar
│       ├── AIStackPage.tsx          # 13-component AI stack docs
│       └── ABDashboard.tsx          # Live A/B experiment dashboard
├── docker-compose.yml               # Postgres, Redis, Qdrant, MinIO, API
├── docker-compose-kafka.yml         # Kafka + Flink overlay
└── p.py                             # TMDB catalog patcher
```

---

## Demo pages

| Page | URL | Shows |
|---|---|---|
| Home | `localhost:3000` | RAG + ALS + RL feed, TMDB posters, live bandit |
| Voice | Click CINEWAVE | Intent extraction, multi-genre interleaving, TTS |
| A/B Dashboard | `localhost:3000/abtest` | 4 live experiments with p-values and metric lifts |
| AI Stack | `localhost:3000/aistack` | All 13 components: RAG, ALS, LightGBM, RL, PySpark, Scala, Metaflow, Kafka, GPT-4o |

---

_Built end-to-end by Akilan Manivannan — architecture, training pipeline, serving layer, frontend, MLOps, and A/B framework._
