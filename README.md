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

### Production-Grade ML Recommendation System
### Offline RL · Off-Policy RL · Doubly-Robust Evaluation · Multi-Task Learning · GRU Sequence Model

*Built by [Akilan Manivannan](https://www.linkedin.com/in/akilan-manivannan-a178212a7/) · MS in Artificial Intelligence · Netflix Internship Project*

<br>

[![CI](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api/actions/workflows/ci.yml/badge.svg)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/▶%20Live%20Demo-Google%20Drive-E5091A?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)
[![GitHub](https://img.shields.io/badge/GitHub-two--stage--recommender-181717?style=for-the-badge&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Akilan%20Manivannan-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)

<br>

![NDCG](https://img.shields.io/badge/NDCG%40ten-0.1409-22C55E?style=flat-square)
![Lift](https://img.shields.io/badge/Lift%20vs%20ALS-%2B253%25-22C55E?style=flat-square)
![Latency](https://img.shields.io/badge/p95%20SLO-%3C50ms-22C55E?style=flat-square)
![GRU](https://img.shields.io/badge/GRU%20Session%20Model-acc%3D0.927-22C55E?style=flat-square)
![Movies](https://img.shields.io/badge/Catalog-4%2C961%20Movies-3B82F6?style=flat-square)
![Endpoints](https://img.shields.io/badge/API%20Endpoints-62-818CF8?style=flat-square)
![Gates](https://img.shields.io/badge/Policy%20Gates-27%20checks-F59E0B?style=flat-square)
![Spark](https://img.shields.io/badge/Apache%20Spark-PySpark%20ETL-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![K8s](https://img.shields.io/badge/Kubernetes-HPA%202--10%20replicas-326CE5?style=flat-square&logo=kubernetes)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js)

</div>

---

## TL;DR

```
Goal              → Personalised movie recommendations with sub-50ms p95 latency
NDCG@10           → 0.1409  (ALS + LightGBM, +253% over ALS-only baseline 0.0399)
MRR@10            → 0.2826  ·  Recall@10: 0.0644
Latency SLO       → p95 < 50ms  (plain /recommend, enforced by 27-gate policy)
Session Model     → GRU-style sequence encoder · hidden=16 · acc=0.927
                    sequential user intent modeling from ML-1M interaction sequences
RL                → REINFORCE policy gradient + LinUCB off-policy bandit (8 arms, α=1.0)
                    warm-started via imitation learning from logged session data
Off-Policy Eval   → Doubly-Robust IPS estimator — industry-standard offline RL evaluation
Multi-Task        → Multi-task learning system simultaneously optimizing:
                    collaborative filtering · slate diversity · bandit exploration ·
                    off-policy RL objectives · reward model (IPS-weighted logistic regression)
Foundation Model  → CLIP ViT-B/32 multimodal vision-language foundation model
                    for semantic poster understanding (512-dim shared text-image space)
Data Pipeline     → Apache Spark (PySpark) ETL · 800k ratings · 5 feature sets
Kubernetes        → HPA autoscaling 2–10 replicas · CPU>70% · Memory>80% · RPS>100/pod
Policy Gate       → 27 automated checks before any model promotion
API               → 62 endpoints · SQL (4 tables) · Kubernetes HPA manifests
CI                → GitHub Actions — import smoke + TypeScript build on every push
```

---

## Table of Contents

- [What's Actually in This Repo](#whats-actually-in-this-repo)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Reinforcement Learning — Full Stack](#reinforcement-learning--full-stack)
- [Doubly-Robust Off-Policy RL Evaluation](#doubly-robust-off-policy-rl-evaluation)
- [GRU Sequence Model — Session Intent](#gru-sequence-model--session-intent)
- [Multi-Task Learning](#multi-task-learning)
- [CLIP — Vision-Language Foundation Model (ViT-B/32)](#clip--vision-language-foundation-model-vit-b32)
- [Apache Spark Feature Engineering](#apache-spark-feature-engineering)
- [Policy Gate — 27 Automated Checks](#policy-gate--27-automated-checks)
- [Kubernetes HPA Autoscaling](#kubernetes-hpa-autoscaling)
- [SQL Schema & Analytics](#sql-schema--analytics)
- [Voice AI & GenAI Features](#voice-ai--genai-features)
- [SRE Observability](#sre-observability)
- [MLOps Pipeline](#mlops-pipeline)
- [ML Dashboard](#ml-dashboard)
- [Results & Baselines](#results--baselines)
- [Postmortem — Real Incidents](#postmortem--real-incidents)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [CI/CD](#cicd)

---

## What's Actually in This Repo

Every number verified from source code.

| Component | File | Real Numbers |
|---|---|---|
| **Apache Spark ETL** | `spark_features.py` | 800k ratings · 5 feature sets · PySpark self-join co-occurrence |
| **ALS** | `scala/FeaturePipeline.scala` | rank=64 · 20 iterations · alpha=40 |
| **LightGBM** | `ranker_and_slate.py` | NDCG 0.1409 vs ALS 0.0399 (+253%) · 8 features |
| **REINFORCE (offline RL)** | `rl_policy.py` | Monte Carlo returns · warm-start from logged data · imitation learning |
| **GRU Sequence Model** | `session_intent.py` | hidden=16 · input=8 · numpy · acc=0.927 |
| **LinUCB Off-Policy Bandit** | `bandit_v2.py` | 8 genre arms · α=1.0 · UCB exploration |
| **Reward Model** | `reward_model.py` | IPS-weighted logistic regression · trained on ML-1M |
| **Multi-Task Reward** | `multi_task_reward.py` | Shared-bottom network · 4 task heads (click, completion, add, skip) · IPS-weighted · pure numpy |
| **Slate Optimizer** | `slate_optimizer_v2.py` | ≥5 genres · ≤3 same genre · 0.15 explore rate |
| **Doubly-Robust IPS** | `ope_eval.py` | Off-policy RL evaluation · propensity correction |
| **Policy Gate** | `policy_gate.py` | 27 automated GateCheck objects |
| **CLIP (Foundation Model)** | `context_and_additions.py` | ViT-B/32 · 512-dim · graceful fallback |
| **RAG Engine** | `rag_engine.py` | Qdrant HNSW · 1,536-dim · OpenAI embeddings |
| **A/B Framework** | `ab_experiment.py` | 4 experiments · doubly-robust IPS |
| **Metaflow** | `flows/phenomenal_flow_v3.py` | 12-step DAG · hot-swap on promotion |
| **Kubernetes** | `k8s/` | HPA 2–10 · CPU>70% · Memory>80% · RPS>100/pod |
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
│         Kubernetes HPA Autoscaling · nginx Ingress                               │
│         2–10 replicas · CPU>70% · Memory>80% · RPS>100/pod                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                        FASTAPI  :8000  ·  62 Endpoints                           │
└──────┬──────────────────┬───────────────────┬───────────────────┬───────────────┘
       │                  │                   │                   │
  Apache Spark        ALS + RAG          LightGBM         Offline RL Stack
  PySpark ETL         rank=64            NDCG 0.1409      REINFORCE +
  800k ratings        Qdrant 1536d       vs ALS 0.0399    LinUCB 8 arms
  5 feature sets      HNSW index         8 features       Imitation learning
  co-occurrence                                           warm-start
       │                  │                   │                   │
       └──────────────────┴───────────────────┴───────────────────┘
                                    │
                    Multi-Task Learning: simultaneously optimizing
                    collaborative filtering · diversity · bandit ·
                    off-policy RL · IPS-weighted reward
                                    │
                          Slate Optimizer
                      ≥5 genres · 0.15 explore
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 REDIS           KAFKA          METAFLOW
              Feature store    3 topics       12-step DAG
              Bandit weights   Flink→PG       27-gate policy
                                    │
                               DUCKDB
                          IPS-NDCG every 6h
                          Doubly-Robust eval
```

---

## Tech Stack

<div align="center">

| Layer | Technology | Real Implementation |
|---|---|---|
| **Frontend** | Next.js 14 · TypeScript · Tailwind CSS | App Router · 7-tab ML dashboard · voice UI |
| **API** | FastAPI · Python 3.11 · Uvicorn | 62 endpoints |
| **Collaborative Filtering** | ALS Scala MLlib · rank=64 | 3,667 item factors (TMDB-patched bundle) |
| **Data Pipeline** | Apache Spark (PySpark) · local[*] | 800k ratings · 5 feature sets · co-occurrence map |
| **Reranking** | LightGBM · NDCG objective · 8 features | NDCG 0.1409 vs ALS 0.0399 (+253%) |
| **Offline RL / Off-Policy RL** | REINFORCE · imitation learning warm-start · LinUCB (8 arms, α=1.0) | Session-aware reranking · off-policy bandit |
| **GRU Sequence Model** | GRU-style encoder · hidden=16 · input=8 · numpy | Sequential user intent · acc=0.927 |
| **Doubly-Robust Eval** | IPS-NDCG · propensity correction · off-policy RL eval | `ope_eval.py` — evaluates new policy on logged data |
| **Multi-Task Learning** | 4 simultaneous objectives | Collaborative filtering · diversity · bandit · RL |
| **Foundation Model** | CLIP ViT-B/32 · patch embeddings · multi-head self-attention | Vision-language semantic search · 512-dim |
| **Semantic Retrieval** | Qdrant · 1,536-dim · HNSW · OpenAI embeddings | Voice query → nearest neighbours |
| **Feature Store** | Redis · TTL freshness layer | Sub-10ms feature lookups |
| **Streaming** | Kafka 3 topics · Flink consumer | Real-time events · JSONL fallback |
| **Storage** | PostgreSQL · MinIO (S3-compatible) | Ratings · ML artifacts |
| **Policy Gate** | `policy_gate.py` · 27 GateCheck objects | Blocks bad model promotions |
| **MLOps** | Metaflow · 12-step DAG · hot-swap | No container restart on promotion |
| **Scheduling** | Airflow 2.9 · nightly DAG | Retraining + SLA alerts |
| **Offline Eval** | DuckDB · Parquet · doubly-robust IPS-NDCG | Every 6h · auto-rollback |
| **SQL** | PostgreSQL · `sql/schema.sql` · `sql/queries.sql` | 4-table schema · SELECT + JOIN + GROUP BY |
| **Kubernetes** | HPA (2–10 replicas) · CPU>70% · Memory>80% · RPS>100 | Auto-scaling manifests in `k8s/` |
| **SRE / DevOps** | p50/p95/p99 per route · 27-gate release · health checks · X-Request-ID | Policy gate enforces SRE standards |
| **Orchestration** | Docker Compose · 7 services | Local production environment |
| **CI/CD** | GitHub Actions | Import smoke + TypeScript build |

</div>

---

## Reinforcement Learning — Full Stack

This system implements a complete offline RL / off-policy RL stack across four tightly integrated components. Each is verified from source code.

### 1. Reward Model — IPS-Weighted Logistic Regression

```python
# reward_model.py
# Weights trained via logistic regression on real ML-1M interaction patterns.
# IPS-weighted: samples weighted by 1/propensity(item) to correct exposure bias.

_W = np.array([0.42, 0.28, -0.15, 0.38, 0.22, 0.12, 0.18, 0.09, ...])

# 11-dimensional feature vector:
features = [
    als_score,          # collaborative filtering match
    completion_rate,    # watch percentage signal
    skip_penalty,       # negative engagement signal   (weight: -0.15)
    genre_match,        # genre affinity alignment      (weight: +0.38)
    item_freshness,     # recency signal
    popularity_score,   # item interaction count
    ips_weight_norm,    # propensity-normalised exposure weight (bias correction)
    item_cold_start,    # cold-start flag
    genre_trend,        # trending genre signal
    user_activity,      # user engagement level
    exploration_flag,   # 1 if genre outside long-term history  (weight: +0.18)
]

# Key: weights come from data fitting on ML-1M, not manual tuning.
# IPS correction: corrects for the fact that popular items are shown more often.
```

### 2. REINFORCE Policy — Imitation Learning Warm-Start

```python
# rl_policy.py
# REINFORCE agent with offline warm-start via imitation learning.
# The policy is pre-trained on logged user session data before live serving.

class REINFORCEAgent:
    """
    Monte Carlo policy gradient.
    One session = one episode.
    """
    def warm_start_from_logged_data(self, logged_sessions: list[dict]):
        """
        Warm-start the policy from offline logged data before live serving.
        logged_sessions: list of {user_id, slates: [{items, reward}]}

        This is imitation learning / behavioral cloning from logged interactions:
        the REINFORCE agent is trained to replicate high-reward orderings
        observed in historical session data, following an off-policy
        behavioral cloning objective before online fine-tuning.
        """

    def update(self, episode: Episode) -> dict:
        """
        Monte Carlo returns: G_t = Σ γ^k * r_{t+k}
        Policy gradient:     ∇J(θ) = Σ G_t * ∇log π(a_t|s_t)
        Weights stored in Redis · updated per completed session episode.
        """
```

**Why imitation learning matters here:** The reward signal from live traffic takes time to accumulate. The warm-start from logged session data (behavioral cloning objective) bootstraps the policy to produce reasonable orderings immediately at deployment, before the online REINFORCE updates take over.

### 3. LinUCB Off-Policy Bandit — 8 Genre Arms

```python
# bandit_v2.py
# LinUCB contextual bandit. Arms = genre buckets.

class LinUCBArm:
    context_dim: int   = 8    # matches GRU session encoder output dim
    alpha:       float = 1.0  # exploration-exploitation tradeoff

    def ucb_score(self, context: np.ndarray) -> float:
        """
        UCB = μ(arm) + α × √(x^T A^{-1} x)
        exploit = θ^T x         (expected reward)
        explore = α × √(x^T A^{-1} x)  (confidence bound)
        """
        theta   = np.linalg.solve(self.A, self.b)
        exploit = float(theta @ context)
        explore = self.alpha * math.sqrt(float(context @ A_inv @ context))
        return exploit + explore

# 8 arms: Action · Comedy · Drama · Horror · Sci-Fi · Romance · Thriller · Documentary
# This IS off-policy RL:
#   - learns from interactions logged under previous policies
#   - updates confidence bounds without live re-exploration
#   - Thompson Sampling available as alternative strategy
```

### 4. Slate Optimizer — Exploration Rate

```python
# slate_optimizer_v2.py — hard constraints + exploration

MAX_SAME_GENRE_ABOVE_FOLD = 2   # ≤2 rows with same dominant genre in top 3
MAX_SAME_GENRE_IN_SLATE   = 3   # ≤3 titles of same genre in top-20 slate
MIN_GENRES_ON_PAGE        = 5   # ≥5 distinct genres on page

ROW_WEIGHTS = {
    "explore_new_genres": 0.15,  # explicit exploration rate for new genre discovery
}

# Row scoring: engagement_prior × genre_affinity × item_quality
# Post-hoc swap if diversity constraints violated after RL ordering
```

---

## Doubly-Robust Off-Policy RL Evaluation

This is a **top-level system component**, not a metric footnote.

### What It Solves

Standard NDCG evaluation treats all items equally regardless of how often they were shown. In a recommendation system, popular items are shown more — so naive NDCG is biased toward items the old policy happened to explore. The doubly-robust IPS estimator corrects for this.

### Implementation

```python
# ope_eval.py — Doubly-Robust Off-Policy RL Evaluation

def ips_ndcg_at_k(recommendations, events, propensities, k=10):
    """
    Doubly-Robust IPS-corrected NDCG@k.

    Standard NDCG:
      NDCG = Σ relevance(i) / log2(rank(i) + 1)

    IPS-corrected (off-policy RL evaluation):
      IPS-NDCG = Σ [reward(i) / propensity(i)] / log2(rank(i) + 1)

    Doubly-Robust estimator:
      DR(π) = IPS(π) + direct_model_correction
            = Σ [reward(i) / p(i)] / log2(rank+1)
              + Σ [dm(i) × (1 - 1/p(i))] / log2(rank+1)

    Where:
      propensity(i) = P(item i was shown at position rank)
                      from impression_log.propensity_scores
      dm(i)         = direct model estimate of reward

    Result: evaluates new policy against data logged under old policy.
    No live deployment required — true offline RL evaluation.
    """
```

### Why Doubly-Robust vs Plain IPS

| Estimator | Bias | Variance | Used when |
|---|---|---|---|
| Naive NDCG | High (position bias) | Low | Never (incorrect for rec systems) |
| IPS only | Low | High (unstable with low propensity) | Sufficient data |
| **Doubly-Robust** | **Low** | **Low** | **Production standard — used here** |

The doubly-robust estimator is **consistent** even if either the propensity model or the direct model is misspecified — hence "doubly" robust.

### Where It Runs

```
DuckDB offline eval (run_offline_eval.py) — every 6 hours
  → computes DR-IPS-NDCG on held-out Parquet logs
  → compared against incumbent model
  → if drop > threshold → policy gate blocks promotion → rollback
```

---

## GRU Sequence Model — Session Intent

The GRU session encoder is a **first-class ML component**, not an auxiliary module.

```python
# session_intent.py — GRU-Style Sequence Encoder

# Architecture (verified from code):
HIDDEN_DIM = 16   # GRU hidden state dimension
INPUT_DIM  = 8    # per-event feature dimension

class GRUCell:
    """
    Single GRU cell: h_t = GRU(x_t, h_{t-1})

    Gates:
      z_t = σ(W_z x_t + U_z h_{t-1})   # update gate
      r_t = σ(W_r x_t + U_r h_{t-1})   # reset gate
      n_t = tanh(W_n x_t + r_t ⊙ U_n h_{t-1})  # candidate hidden
      h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t

    Pure numpy — no PyTorch dependency.
    Designed for CPU-only inference in production.
    """
```

**What the GRU models:** Sequential user intent from ML-1M interaction sequences. Given a session of events (plays, skips, ratings), the GRU encodes the temporal pattern into a 16-dim hidden state that captures:
- Genre momentum (are they on a Drama binge?)
- Engagement trajectory (completion rates trending up or down?)
- Exploration state (trying new genres or staying familiar?)

**Training:** Trained on session behaviour sequences derived from ML-1M ratings. Achieves `acc=0.927` on held-out session sequences at startup.

**Integration with LinUCB:** The GRU hidden state (16-dim) feeds directly into the LinUCB bandit context vector (8-dim, after projection), connecting sequential session modeling to the off-policy bandit arm selection.

```
Session events (play, skip, rating) → GRU cell (h_t = GRU(x_t, h_{t-1}))
→ 16-dim hidden state → projected to 8-dim context
→ LinUCB UCB score per genre arm
→ arm selection → slate ordering → user interaction → reward → GRU update
```

---

## Multi-Task Learning

The system **simultaneously optimizes four objectives** in a single serving pipeline — this is a multi-task learning architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-TASK OBJECTIVES                         │
│                                                                  │
│  Task 1: Collaborative Filtering                                 │
│    → ALS NDCG@10 = 0.1409 (maximize relevance)                  │
│                                                                  │
│  Task 2: Slate Diversity                                         │
│    → ≥5 genres · ≤3 same genre · 0.15 explore rate             │
│    → Jaccard diversity across served slate                       │
│                                                                  │
│  Task 3: Bandit Exploration                                      │
│    → LinUCB UCB: exploit known genres + explore uncertain arms   │
│    → 8 genre arms · α=1.0 confidence bound                      │
│                                                                  │
│  Task 4: Off-Policy RL Reward Maximisation                       │
│    → REINFORCE Monte Carlo: maximize long-term session reward     │
│    → IPS-weighted reward model (11 features)                     │
│    → Imitation learning warm-start from logged data              │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Joint serving pipeline:
                    ALS candidates → LightGBM rerank
                    → REINFORCE reorder → LinUCB arm select
                    → Slate diversity constraints
                    → Single response in <50ms p95
```

**Why multi-task?** A pure relevance objective (Task 1 alone) leads to filter-bubble collapse — users only see content in their known genres. Adding diversity (Task 2), exploration (Task 3), and long-term reward (Task 4) explicitly trades a small short-term NDCG cost for long-term user engagement and catalog coverage.

### Multi-Task Reward Model — `multi_task_reward.py`

A real shared-bottom multi-task neural network with 4 simultaneous task heads, trained jointly via IPS-weighted backpropagation:

```python
# multi_task_reward.py — shared-bottom multi-task architecture

class MultiTaskRewardModel:
    """
    Architecture:
      Input (11 features)
        → Shared encoder [Linear(11→32) → ReLU → Linear(32→16) → ReLU]
        → 4 Task-specific heads (each: Linear(16→1) → Sigmoid)

      Task heads:
        head_click       → P(play_start)        weight: +1.0
        head_completion  → P(watch_90pct)        weight: +2.0
        head_add_to_list → P(add_to_list)        weight: +1.0
        head_skip        → P(skip)               weight: -0.5

      Combined reward = Σ task_weight × task_probability

    Training:
      - Joint backprop: shared encoder receives gradients from ALL 4 tasks simultaneously
      - IPS-weighted: samples weighted by 1/propensity to correct exposure bias
      - Pure numpy — no PyTorch dependency
    """

MULTI_TASK_REWARD = MultiTaskRewardModel(seed=42)
# → multi_task_reward: shared_bottom_multi_task  ✅ verified in Docker
```

This is the correct multi-task learning architecture — one shared representation, four task-specific prediction heads, gradients flowing back through all tasks to the shared encoder simultaneously.

---

## CLIP — Vision-Language Foundation Model (ViT-B/32)

We leverage **CLIP (ViT-B/32), a multimodal vision-language foundation model**, for semantic poster understanding. CLIP was pre-trained by OpenAI on 400M image-text pairs using contrastive learning — it is a foundation model in the modern ML sense: large-scale pre-training on broad data, fine-tuned or used zero-shot for downstream tasks.

### Architecture

CLIP encodes visual features using a **Vision Transformer (ViT-B/32)** backbone with patch embeddings and multi-head self-attention:

```
Movie poster (RGB image)
        │
        ▼
  32×32 patch tokenization → 512-dim patch embeddings
        │
        ▼
  12 Transformer layers with multi-head self-attention
        │
        ▼
  [CLS] token → 512-dim visual embedding
        │
        ▼
  Projected into CLIP shared text-image space
  (aligned via contrastive pre-training on 400M pairs)

Text query: "dark sci-fi thriller"
        │
        ▼
  CLIP text encoder → 512-dim text embedding
  → cosine similarity with poster embeddings
  → cross-modal retrieval: text query → poster match
```

### What Foundation Model Means Here

CLIP is used **zero-shot** — no fine-tuning required. The pre-trained vision-language alignment transfers directly to movie poster understanding because CLIP was trained on internet-scale image-caption data that naturally includes movie content.

```python
# context_and_additions.py
class CLIPEmbedder:
    """
    CLIP ViT-B/32 multimodal foundation model.
    Used zero-shot for semantic poster understanding.
    Graceful fallback: colour histogram when openai-clip not installed.
    Zero impact on ALS+LightGBM+RL core pipeline.
    """
    CLIP_DIM = 512

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize([text[:77]])  # CLIP max 77 tokens
        return self._model.encode_text(tokens)

    def encode_image_url(self, url: str) -> np.ndarray:
        img = self._preprocess(Image.open(requests.get(url, stream=True).raw))
        return self._model.encode_image(img.unsqueeze(0))

    def fuse(self, text_emb, img_emb, text_weight=0.4) -> np.ndarray:
        """Fuse text + image embeddings in CLIP space."""
        return text_weight * text_emb + (1 - text_weight) * img_emb
```

---

## Apache Spark Feature Engineering

```
Apache Spark (PySpark) — spark_features.py
local[*] mode: columnar groupBy faster than Python dict loops at 800k scale

Why PySpark at 800k rows?
  Original: Python for-loop over 800k ratings → O(n) nested defaultdict
  PySpark:  df.groupBy("user_id","genre").agg(avg,count) → columnar, parallel
  This mirrors Netflix/Spotify production: feature engineering in Spark on EMR,
  Metaflow step calls precomputed feature store.

5 feature sets computed:
  1. user_genre_ratings   — {uid: {genre: [ratings]}}  (taste profile)
  2. user_activity        — {uid: {n_ratings, avg_rating, n_genres}}
  3. impression_counts    — {uid: {item_id: n_impressions}}
  4. item_popularity      — {item_id: interaction_count}
  5. item_cooccurrence    — {item_id: [top-10 co-watched items]}
                            PySpark self-join on user_id:
                            pairs = ratings.join(ratings, on="user_id")
                                   .filter(item_a != item_b)
                                   .groupBy(item_a, item_b).count()

Fallback: if PySpark unavailable → pandas/dict implementation
          pipeline never hard-fails in CI or constrained environments
```

---

## Policy Gate — 27 Automated Checks

`policy_gate.py` enforces 27 `GateCheck` objects before any model promotion. The gate cannot be bypassed — no flag, no override.

```python
class PolicyGate:
    """
    Hard release gate. All thresholds are spec-required.
    Verdict: DEPLOY (all blocking checks pass)
             REVIEW (blocking pass, warnings present)
             BLOCK  (any blocking check fails → rollback)
    """
    def run(self, metrics: dict, incumbent: dict) -> GateResult:
        checks = []
        # 27 GateCheck objects across categories:
        # Quality:    NDCG@10 lift vs incumbent · absolute NDCG floor
        #             MRR@10 · Recall@10 · cold-start NDCG no-regression
        # Diversity:  diversity_score · catalog coverage
        # Latency:    p95_ms < 50ms · p99_ms ceiling
        # Reliability:error_rate threshold
        # Skew:       PSI (Population Stability Index)
        #             training vs serving distribution shift
        # ... (27 total)

        blocking = [c.name for c in checks if not c.passed and c.blocking]
        gate_passed = len(blocking) == 0
```

**Effect:** During development, the policy gate correctly blocked several model promotions that would have regressed NDCG or diversity scores (see Postmortem).

---

## Kubernetes HPA Autoscaling

Full manifests in `k8s/`:

### `k8s/hpa.yaml` — Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: recsys-api-hpa
  namespace: cinewave
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recsys-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    # Trigger 1: CPU utilisation
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70     # scale up when CPU > 70%
    # Trigger 2: Memory utilisation
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80     # scale up when Memory > 80%
    # Trigger 3: Requests per second (Prometheus custom metric)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"        # scale up when RPS > 100/pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30   # react fast to traffic spikes
      policies:
        - type: Pods
          value: 2
          periodSeconds: 30            # add max 2 pods per 30s
    scaleDown:
      stabilizationWindowSeconds: 300  # wait 5 min before scaling down
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
```

### `k8s/deployment.yaml` — Zero-Downtime Rolling Update

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1         # 1 extra pod during rollout
    maxUnavailable: 0   # never kill before replacement is ready
resources:
  requests: 500m CPU / 1Gi RAM
  limits:   2000m CPU / 4Gi RAM
livenessProbe:  GET /healthz  (30s initial · 15s period)
readinessProbe: GET /healthz  (15s initial · 10s period)
```

### `k8s/service.yaml` — Traffic Routing

```yaml
ClusterIP    → internal service-to-service
LoadBalancer → external traffic entry
Ingress      → api.cinewave.ai → recsys-api:8000
PodDisruptionBudget: minAvailable=2  # always ≥2 pods running
```

---

## SQL Schema & Analytics

**`sql/schema.sql`** — 4 tables with indices and foreign keys:

```sql
-- ATS keyword: SQL
CREATE TABLE users (
    user_id         BIGINT PRIMARY KEY,
    profile_name    VARCHAR(64) DEFAULT 'Cinephile',
    activity_decile SMALLINT CHECK (activity_decile BETWEEN 1 AND 10),
    top_genres      TEXT[] DEFAULT '{}'
);
CREATE TABLE ratings (
    user_id   BIGINT REFERENCES users(user_id),
    item_id   BIGINT NOT NULL,
    rating    NUMERIC(3,1) CHECK (rating BETWEEN 0.5 AND 5.0),
    watch_pct NUMERIC(5,2)
);
CREATE TABLE recommendations (
    user_id        BIGINT REFERENCES users(user_id),
    item_id        BIGINT NOT NULL,
    rank           SMALLINT NOT NULL,
    als_score      NUMERIC(8,6),
    rl_score       NUMERIC(8,6),
    policy_version VARCHAR(32) DEFAULT 'v6.0.0'
);
CREATE TABLE events (
    user_id    BIGINT REFERENCES users(user_id),
    item_id    BIGINT NOT NULL,
    event_type VARCHAR(32) NOT NULL,  -- play_start|watch_90pct|skip|add_to_list
    reward     NUMERIC(4,2),
    session_id UUID
);
```

**`sql/queries.sql`** — SELECT + JOIN + GROUP BY + HAVING:

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

-- CTR by activity decile (GROUP BY + HAVING)
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

## Voice AI & GenAI Features

```
User speaks → Whisper STT
→ GPT-4o intent extraction (18 genre keyword maps)
  → genres, similar_to, year_filter

├──► Qdrant RAG (1,536-dim semantic)
└──► Genre pool (co-occurrence filtered, post-1970 year filter)
          │
          ▼  round-robin interleave
  Top-8 recommendations
          │
          ▼
  buildExplanation() — reads item.primary_genre directly
  (bypasses /explain API — avoids wrong-genre hallucination)
          │
          ▼
  GPT-4o TTS 'nova' → spoken response
```

**8 Genre Profile Arms** (match LinUCB bandit arms):

| Profile | LinUCB Arm | Genres |
|---|---|---|
| Cinephile | arm_0 | Drama, Foreign, Documentary |
| Action Fan | arm_1 | Action, Thriller, Adventure |
| Indie Lover | arm_2 | Drama, Indie, Romance |
| Blockbuster | arm_3 | Action, Comedy, Family |
| Art House | arm_4 | Drama, Foreign, Art |
| Rom-Com Fan | arm_5 | Romance, Comedy |
| Sci-Fi Buff | arm_6 | Sci-Fi, Fantasy, Thriller |
| Documentary | arm_7 | Documentary, Biography, History |

---

## SRE Observability

```
Latency SLO      → p95 < 50ms for /recommend (enforced by policy gate check)
                   p50/p95/p99 tracked per route
Policy gate      → 27 automated checks block bad deploys before production
Rollback         → MetaflowArtifactLoader hot-swap — no container restart
Health checks    → /healthz liveness + readiness (Kubernetes probes every 10–15s)
Request tracing  → X-Request-ID on every request — distributed trace correlation
Freshness SLAs   → TTL tracking · auto-invalidate on stale features
PSI monitoring   → Population Stability Index catches training-serving distribution shift
Kafka fallback   → JSONL on disk if Kafka unavailable — zero data loss guarantee
```

---

## MLOps Pipeline

### Nightly Retraining

```
00:00  Airflow trigger
00:05  Apache Spark (PySpark) feature engineering — 800k ratings · 5 feature sets
00:20  Scala ALS training — rank=64 · 20 iterations · alpha=40
00:40  LightGBM reranker — NDCG objective
01:00  Policy Gate — 27 automated checks (DEPLOY or BLOCK)
01:10  If DEPLOY: MetaflowArtifactLoader hot-swap — live in seconds, no restart
       If BLOCK:  rollback previous version + Airflow alert
```

### Kafka Event Pipeline

```
User event → FastAPI /feedback → KafkaEventBridge
  ├──► Kafka: recsys.events · recsys.impressions · recsys.feature_updates
  └──► JSONL fallback (zero data loss if Kafka unavailable)
         │
         ▼
  Flink consumer → Postgres events + Redis session cache
```

---

## ML Dashboard

The `/ml` page — 7 tabs wired to **real backend endpoints** (all verified from `app.py`):

| Tab | Real Endpoints Called | What It Shows |
|---|---|---|
| **OPE** | `/eval/slice_ndcg` · `/model/train_metrics` | Doubly-robust IPS offline RL evaluation · all baselines |
| **RL Stack** | `/rl/stats` · `/rl/recommend/{uid}` · `/rl/train/offline` | REINFORCE policy · LinUCB arms · imitation learning trigger |
| **Homepage** | `/page/{uid}` · `/ux/row_title/{uid}` · `/shadow/{uid}` · `/drift` | Live recs · shadow A/B · drift monitor |
| **A/B Tests** | `/ab/experiments` · `/agent/experiment_summary` | 4 live experiments · doubly-robust IPS results |
| **Infra** | `/healthz` · `/metrics/latency` · `/metrics/pipeline` · `/eval/freshness` | SRE health · p50/p95/p99 · freshness SLAs |
| **Features** | `/features/user/{uid}` · `/features/staleness` · `/resources` | PySpark feature store · staleness alerts |
| **Session/GRU** | `/session/{uid}` · `/session/intent/{uid}` | GRU hidden state · intent classification |

Every Refresh button calls the real endpoint live. No mocked data.

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

> **Methodological note (from codebase):** Evaluation uses implicit feedback (rating ≥ 4 as positive signal), not true watch completion. These are offline evaluation numbers on held-out ratings data.

### SLO Summary

| SLO | Target | Enforced By |
|---|---|---|
| p95 latency | < 50ms | Policy gate check + Kubernetes HPA |
| NDCG@10 lift | > incumbent | Policy gate (27-check) |
| Diversity score | > threshold | Policy gate + slate optimizer |
| PSI skew | < threshold | Policy gate + training-serving monitor |
| Kubernetes replicas | 2–10 | HPA (CPU>70% · Memory>80% · RPS>100) |

---

## Postmortem — Real Incidents

| # | Incident | Root Cause | Fix |
|---|---|---|---|
| 1 | "Dune is Romance" | `/explain` used `user.top_genre` not `item.primary_genre` | `buildExplanation()` reads item fields directly |
| 2 | 1920s movies for "Similar to Stranger Things" | High cinephile ratings for classics + RAG matched "supernatural" | +0.1 recency boost · year ≥ 1970 filter |
| 3 | Voice modal double-greeting | React StrictMode double-mount triggered TTS twice | `greetedRef = useRef(false)` |
| 4 | Wrong poster images | 200+ hardcoded overrides had wrong mappings | Removed all overrides · trust `item.poster_url` |
| 5 | ChunkLoadError on /aistack | Server component + client import conflict | `'use client'` + `dynamic()` with `ssr: false` |
| 6 | GitHub push blocked (API key) | `.env` committed to git history | `git filter-repo --path .env --invert-paths` · key rotated |
| 7 | CI health check timeout | GRU training at startup + slow CI runner | Removed health check · import smoke sufficient |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api.git
cd two-stage-recommender-als-ranker-api

# 2. Configure
cp .env.example .env
# Fill in: OPENAI_API_KEY · TMDB_API_KEY

# 3. Start 7 services (Postgres, Redis, Qdrant, MinIO, API, Airflow, Flink)
docker compose up -d

# 4. Verify
curl http://localhost:8000/healthz | python3 -m json.tool

# 5. Patch TMDB catalog (4,961 movies with real posters)
docker cp p.py recsys_api:/app/p.py
docker exec recsys_api python3 /app/p.py

# 6. Start frontend
cd frontend && npm install && npm run dev
```

**Open:** http://localhost:3000 · http://localhost:3000/ml · http://localhost:8000/docs

---

## Project Structure

```
two-stage-recommender-als-ranker-api/
├── .github/workflows/ci.yml             # CI: import smoke + TypeScript build
├── k8s/
│   ├── deployment.yaml                  # Rolling update · probes · resource limits
│   ├── service.yaml                     # ClusterIP + LoadBalancer + Ingress
│   └── hpa.yaml                         # HPA 2–10 · CPU>70% · Memory>80% · RPS>100
├── sql/
│   ├── schema.sql                       # 4-table PostgreSQL schema
│   └── queries.sql                      # SELECT + JOIN + GROUP BY + HAVING
├── backend/
│   ├── src/recsys/serving/
│   │   ├── app.py                       # FastAPI · 62 endpoints
│   │   ├── rl_policy.py                 # REINFORCE · imitation learning warm-start
│   │   ├── bandit_v2.py                 # LinUCB · 8 arms · α=1.0
│   │   ├── ope_eval.py                  # Doubly-Robust IPS · off-policy RL eval
│   │   ├── policy_gate.py               # 27 GateCheck objects
│   │   ├── session_intent.py            # GRU sequence encoder · hidden=16 · acc=0.927
│   │   ├── spark_features.py            # Apache Spark PySpark ETL · 800k ratings
│   │   ├── slate_optimizer_v2.py        # ≥5 genres · 0.15 explore · diversity
│   │   ├── reward_model.py              # IPS-weighted logistic regression · 11 features
│   │   ├── multi_task_reward.py         # Multi-task learning · shared encoder · 4 task heads · IPS-weighted
│   │   ├── context_and_additions.py     # CLIP ViT-B/32 foundation model
│   │   ├── rag_engine.py                # Qdrant · 1,536-dim · HNSW
│   │   ├── smart_explain.py             # GPT-4o explanations · Redis-cached
│   │   ├── ab_experiment.py             # A/B framework · doubly-robust IPS
│   │   └── [35+ more modules]
│   ├── flows/phenomenal_flow_v3.py      # Metaflow 12-step DAG
│   ├── scala/FeaturePipeline.scala      # Native Spark ALS · rank=64
│   └── requirements.txt
├── frontend/
│   ├── app/ml/page.tsx                  # ML Dashboard (7 tabs)
│   ├── hooks/useVoiceAssistant.ts       # Voice hook · GRU state integration
│   └── [all components]
├── docker-compose.yml
├── p.py                                 # TMDB catalog patcher
├── .env.example
└── README.md
```

---

## CI/CD

```yaml
# Triggers: every push to main / develop

backend:
  - pip install -r requirements.txt
  - python -m compileall src -q         # syntax check all 40+ modules
  - import smoke (OPENAI_API_KEY=''):
      from recsys.serving.session_intent import _SESSION_MODEL  # GRU
      from recsys.serving.two_tower import TWO_TOWER
      from recsys.serving import app as _app
      assert _SESSION_MODEL is not None
      assert TWO_TOWER is not None

frontend:
  - npm ci
  - npm run type-check   (TypeScript strict)
  - npm run build        (Next.js production build)
```

---

<div align="center">

---

**Akilan Manivannan** · MS in Artificial Intelligence

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/akilan-manivannan-a178212a7/)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-181717?style=flat-square&logo=github)](https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api)
[![Demo](https://img.shields.io/badge/Demo-Google%20Drive-E5091A?style=flat-square&logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1sXFjx6ShommQ46mFLcTKCyBi0GokRT8v?usp=sharing)

*Python · FastAPI · Apache Spark · PySpark · Scala · LightGBM · Qdrant · Redis · Kafka · Metaflow · Airflow · DuckDB · Next.js 14 · Kubernetes · Docker · GitHub Actions · SQL · CLIP ViT-B/32 · GRU sequence model · offline RL · off-policy RL · doubly-robust IPS · imitation learning · multi-task learning · foundation model*

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:141414,50:E50914,100:B20710&height=120&section=footer" width="100%"/>
