# Architecture — Netflix-Inspired Recommendation Platform v4

## System overview

A four-plane recommendation system with a FastAPI backend, Next.js frontend,
and a full supporting infrastructure stack (Postgres, Redis, MinIO, Qdrant, Metaflow, Airflow).

```
┌─────────────────────────────────────────────────────────────────┐
│  Plane 1 — Core Recommendation  (<50ms p99, deterministic)      │
│  ALS retrieval → Session intent → Candidate fusion              │
│  → GBM ranker → Diversity reranker → Page assembler             │
│  + Launch-effect detection + Session drift detection             │
├─────────────────────────────────────────────────────────────────┤
│  Plane 2 — Semantic Intelligence  (ALL offline, never in path)  │
│  TMDB hydration + LLM enrichment + MediaFM-proxy embeddings     │
│  + Cosine index + VLM artwork audit                             │
├─────────────────────────────────────────────────────────────────┤
│  Plane 3 — Agentic Eval / Ops  (no autonomous deployment)       │
│  Shadow triage + Drift investigation + Experiment summaries      │
│  Human review gate before every release                          │
├─────────────────────────────────────────────────────────────────┤
│  Plane 4 — GenAI UX  (pre-computed / cached, zero blocking)     │
│  Row titles pre-warmed at startup — NEVER called live            │
│  Mood discovery + Spoiler-safe summaries + SHAP explanations     │
└─────────────────────────────────────────────────────────────────┘
```

---

## What is real and what is a proxy

| Component | What it does | Honest limitation |
|---|---|---|
| ALS retrieval | Implicit-feedback matrix factorisation via `implicit` library | Trained on synthetic ratings (80k interactions, 8% density). Not production scale. |
| GBM ranker | LightGBM pointwise ranker with 8 features | AUC ≈ 0.50 on synthetic data — labels too sparse. Use real watch-completion for production. |
| Session intent | Single GRU cell (hidden=16) trained with cross-entropy on 3000 synthetic session sequences | Not FM-Intent. No watch-time signals. No multi-task. Not production session logs. |
| Two-tower retrieval | 3-layer linear towers (13→48→32→64), trained with in-batch + hard-negative contrastive loss | Small by production standards (D=64, CPU-only). No attention. Static features only. |
| Long-term satisfaction | IPS-weighted logistic regression on 11 features | Proxy: rating≥4 as engagement label. Real LTS requires watch-time, not star ratings. |
| Freshness engine | Per-feature SLA tiers (session=5min, trending=1min, embeddings=24hr), circuit-breaker fallbacks | In-process state, not Redis-backed. Production freshness needs Kafka + per-feature TTL watermarking. |
| Semantic embeddings | OpenAI text-embedding-3-small cosine index | Text+metadata only. Not audio/video. Not a jointly trained user-item space (MediaFM is tri-modal). |
| RAG / semantic retrieval | Cosine search on text embeddings, LLM re-rank over top-15 | "RAG" here = retrieval only, not generation-augmented ranking. Honest in `/architecture` endpoint. |
| Bandit exploration | Cluster-stratified UCB1 with personalised budget | Not a full contextual bandit. Updates are synchronous on feedback events, not online. |
| Page optimizer | Row deduplication, genre balance, impression budget, row-level engagement scoring | Engagement priors hand-specified, not trained on real page-outcome data. |
| Agentic ops | OpenAI tool-calling with structured outputs for triage, drift investigation, experiment summaries | Agent recommends only. Explicit human review required before any deployment action. |
| GenAI UX | Row titles pre-computed at startup and served from cache | If cache misses: rule-based fallback. Zero live OpenAI calls in request path (enforced in code). |

---

## What was fixed in v4

### 1. GenAI-in-request-path contradiction (critical bug)
**Problem:** `GET /ux/row_title/{user_id}` called `personalised_row_title()` which called GPT-4o-mini inline. The claim "zero generative AI in the request path" was false whenever this endpoint ran without a warm cache.

**Fix:**
- `startup_event()` pre-warms row titles for all demo users at API startup.
- `FRESH_STORE` caches each title with a `page_cache` tier (120s TTL).
- `/ux/row_title` returns from cache if fresh, or returns a rule-based fallback immediately if cache is expired.
- Zero OpenAI calls in the live request path — enforced structurally.
- `_MANIFEST["genai_in_request_path"] = False` is included in every `/version` response.
- `/healthz` includes `"genai_in_request_path": false`.
- Frontend Navbar shows a yellow warning if this ever becomes `true`.

### 2. `TOGGLE_SHADOW` dispatch mismatch
**Problem:** `Navbar` dispatched `SET_SHADOW` which didn't exist in the reducer, silently doing nothing.

**Fix:** Changed to `TOGGLE_SHADOW` which is the correct action type.

### 3. Freshness not visible to callers
**Problem:** Freshness SLAs were defined but not surfaced. Callers had no way to know if a feature was stale.

**Fix:**
- `/recommend` response now includes `freshness_watermark` with per-feature age and status.
- `/page` response includes the same watermark.
- `/features/freshness` endpoint exposes full per-feature staleness report with circuit-breaker alerts and launch detector stats.
- Frontend reads the watermark and shows a stale-features badge when any feature is `stale`.
- `/healthz` reports `stale_features` count.

---

## What is still limited (honest)

These are **known limitations**, not bugs. They are documented here and in `/architecture`.

**Scale:** Everything runs on a single Docker Compose host. This is appropriate for a portfolio/research build and a serious dev stack. It is not multi-region production. Moving off Compose requires: Kubernetes or ECS, external secrets management, multi-zone Postgres, Redis Cluster, and a real Kafka backbone for event streaming.

**Session model:** The GRU session encoder is a single cell with hidden_dim=16, trained on 3000 synthetic sessions. Netflix FM-Intent is a large multi-task model trained on billions of real sessions with watch-time and impression labels. The gap is real and documented.

**Two-tower:** Three linear layers, D=64, CPU training in minutes. Production versions use deep transformers, GPU training over billions of user-item pairs, and D=256-512+. The in-batch + hard-negative training is the right pattern; the scale is not.

**Long-term satisfaction:** A weighted heuristic with learned logistic weights. Not a causal reward model. Genuine LTS modeling requires randomised holdouts, counterfactual evaluation, and real watch-completion data — none of which are available from MovieLens-1M ratings alone.

**Streaming layer:** `TRENDING`, `SESSION`, and `LIVE_BOOSTER` are in-process Python objects. Production freshness requires: Kafka for event ingestion, Flink for stream processing, Redis for feature propagation, per-feature TTL enforcement with watermarking, and circuit breakers that degrade gracefully to stale-but-available features.

**Bandit:** The contextual bandit updates synchronously on `/feedback` events. A real online bandit requires an async update loop with propensity tracking, reward estimation, and regret monitoring.

**Artwork trust scores:** The VLM audit produces trust scores (0–1) with a threshold of 0.6 for flagging. This threshold is arbitrary — calibration requires human-labelled ground truth data that does not yet exist.

**MediaFM proxy:** The multimodal embedding fuses text (via OpenAI embeddings) and metadata (19-dim feature vector) in a late-fusion architecture. MediaFM is a tri-modal model trained on audio, video frames, and text at scale. The proxy is directionally correct but not equivalent.

---

## Running the system

```bash
# 1. Copy env template and fill in keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and TMDB_API_KEY

# 2. Patch RAG engine paths and smoke-test imports
bash patch_rag.sh

# 3. Start all core services
docker compose up -d

# 4. Watch API startup (session GRU trains here, row titles pre-warm)
docker compose logs -f api

# 5. Verify
curl http://localhost:8000/healthz
# Expect: {"ok":true, "genai_in_request_path":false, "stale_features":0, ...}

# 6. Run the ML pipeline (optional — generates real trained bundle)
docker compose --profile pipeline up metaflow_runner

# 7. Start frontend
docker compose --profile frontend up frontend
# Open: http://localhost:3000
```

---

## Endpoint reference

| Plane | Method | Path | Description |
|---|---|---|---|
| Core | POST | `/recommend` | ALS+GBM+LTS recs with freshness watermark |
| Core | GET | `/page/{user_id}` | Full page slate with dedup and impression budget |
| Core | GET | `/session/intent/{user_id}` | GRU session intent classification |
| Core | GET | `/trending` | Rolling 5-min trending scores |
| Semantic | GET | `/recommend/rag/{user_id}` | Semantic retrieval + LLM rerank |
| Semantic | GET | `/recommend/two_tower/{user_id}` | Contrastive two-tower retrieval |
| Semantic | GET | `/search/semantic` | Free-text semantic search |
| Semantic | GET | `/similar/{item_id}` | Embedding-based similar titles |
| Semantic | GET | `/multimodal/similar/{item_id}` | Text+metadata fused similarity |
| Semantic | GET | `/vlm/analyse/{item_id}` | VLM poster audit |
| Agentic | POST | `/agent/triage` | Shadow deployment triage (human review required) |
| Agentic | GET | `/agent/drift_investigation` | Drift root-cause investigation |
| Agentic | GET | `/agent/experiment_summary` | Natural language experiment summary |
| GenAI | POST | `/explain` | SHAP-grounded explanation (cached) |
| GenAI | GET | `/ux/row_title/{user_id}` | Personalised row title (pre-computed cache) |
| GenAI | GET | `/ux/mood` | Mood-to-content discovery |
| GenAI | GET | `/ux/summary/{item_id}` | Spoiler-safe summary |
| Eval | POST | `/eval/gate` | CI/CD deployment gate |
| Eval | GET | `/shadow/{user_id}` | Shadow A/B comparison |
| Eval | GET | `/eval/slice_ndcg` | Per-genre NDCG regression detection |
| Freshness | GET | `/features/freshness` | Per-feature staleness report |
| Model | GET | `/model/train_metrics` | Session GRU + two-tower training diagnostics |
| System | GET | `/healthz` | Health + stale feature count |
| System | GET | `/architecture` | Full architecture + honest limitations |
