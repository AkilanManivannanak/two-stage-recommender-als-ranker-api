# Multi-Agent System Design: Talentra vs CineWave

## Two Independent Implementations, One Pattern

This document compares two multi-agent systems built independently
across different domains using different frameworks — both following
AutoGen-style orchestration principles.

---

## Side-by-Side Comparison

| Dimension | Talentra Copilot | CineWave |
|---|---|---|
| **Framework** | LangGraph (StateGraph) | FastAPI orchestrator |
| **Domain** | HR intelligence | Movie recommendation |
| **Agent count** | 5 nodes | 4 agents |
| **State management** | LangGraph shared state dict | Redis + in-process |
| **Human-in-loop** | Interrupt gates between nodes | Policy gate (27 checks) |
| **Tool use** | FAISS/Chroma RAG tools | ALS/LinUCB/LightGBM tools |
| **Critic/verifier** | Consistency Check node | Policy Gate BLOCK |
| **Rollback** | Stay on rule-based path | Metaflow hot-swap |
| **Latency** | p95 4.81ms | p95 <50ms |
| **Observability** | Prometheus metrics | DuckDB DR-IPS eval |

---

## Agent Architectures

### Talentra — 5-Node LangGraph Pipeline

```
[START]
   │
   ▼
┌─────────────────────┐
│ Intent Classifier   │  Replaces is_evaluation_question() if/else
│ Node                │  LLM classifies: ranking / copilot / evidence
└──────┬──────────────┘
       │
   ┌───┴───────────────┐
   ▼                   ▼
┌──────────┐    ┌──────────────┐
│ Ranking  │    │ Evidence     │
│ Node     │    │ Search Node  │
│          │    │ FAISS+Chroma │
│ Mistral  │    │ IDF-weighted │
│ LoRA+DPO │    │ RAG          │
└──────┬───┘    └──────┬───────┘
       └───────┬────────┘
               ▼
┌──────────────────────┐
│ Answer Synthesis     │
│ Node                 │
│ GPT-4o / Mistral     │
│ Citation formatting  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Consistency Check    │  ← MagenticOne Critic pattern
│ Node                 │  Ensures copilot ≠ contradict eval
│                      │  Human interrupt if inconsistent
└──────────────────────┘
           │
         [END]
```

### CineWave — 4-Agent FastAPI Pipeline

```
[REQUEST]
    │
    ▼
┌─────────────────────┐
│ Retrieval Agent     │  ALS get_candidates(user_id, k=100)
│                     │  RAG semantic_search(query, k=20)
└──────┬──────────────┘
       │ 100 candidates
       ▼
┌─────────────────────┐
│ Reasoning Agent     │  LightGBM rank(candidates, features)
│                     │  MultiTaskReward.score(slate)
│                     │  DR-IPS unbiased reward estimation
└──────┬──────────────┘
       │ top-30 slate
       ▼
┌─────────────────────┐
│ Exploration Agent   │  LinUCB.select_arm(GRU_context)
│                     │  REINFORCE.reorder(slate)
│                     │  Off-policy, no live re-exploration
└──────┬──────────────┘
       │ reordered slate
       ▼
┌─────────────────────┐
│ Critic Agent        │  PolicyGate.evaluate(slate, thresholds)
│ (27 GateChecks)     │  DEPLOY → Metaflow hot-swap
│                     │  BLOCK  → rollback + Airflow alert
└─────────────────────┘
       │
   [RESPONSE]
```

---

## Key Design Differences

### 1. State Management

**Talentra (LangGraph):**
```python
# Shared state dict passed between nodes
class TalentraState(TypedDict):
    query:        str
    intent:       str          # from Intent Classifier
    candidates:   list[dict]   # from Evidence Search
    ranking:      list[str]    # from Ranking node
    answer:       str          # from Answer Synthesis
    consistent:   bool         # from Consistency Check
    human_review: bool         # interrupt flag
```

**CineWave (FastAPI):**
```python
# State passed via function arguments + Redis
# No shared graph state — each agent is stateless per-request
# Long-term state in Redis (bandit weights, GRU session)
```

**Design lesson:** LangGraph explicit state is easier to debug and test.
FastAPI stateless agents are easier to scale horizontally.

---

### 2. Human-in-the-Loop

**Talentra:**
```python
# Interrupt gate — pauses execution for human review
if state["consistent"] == False:
    return Command(goto=END, update={"human_review": True})
    # Human reviews before answer is returned
```

**CineWave:**
```python
# Policy gate — automated gate, no human pause
# Human-in-loop = Airflow alert + manual rollback approval
if gate_result == "BLOCK":
    rollback()
    send_airflow_alert()
    # Engineer reviews within SLA window
```

**Design lesson:** Talentra's interrupt is synchronous (blocks response).
CineWave's gate is asynchronous (alerts engineer post-deployment).
Both implement human oversight — different latency tradeoffs.

---

### 3. Tool Use Pattern

**Talentra (LangChain tools):**
```python
tools = [
    FAISSRetriever(k=10),       # semantic search
    ChromaRetriever(k=5),       # dense retrieval
    RequirementExtractor(),     # JD parsing
    PII_Redactor(),             # privacy
]
# Tools called by LLM via LangChain tool use interface
```

**CineWave (direct function calls):**
```python
# Tools are Python functions, not LLM-callable
candidates = als.get_candidates(user_id, k=100)
slate      = lgbm.rank(candidates, features)
arm        = linucb.select_arm(gru_context)
# Orchestrator decides tool order, not LLM
```

**Design lesson:** LangChain tool use gives the LLM flexibility in
tool selection. Direct calls give the orchestrator more control and
lower latency. CineWave's p95 <50ms SLO requires direct calls.

---

### 4. Critic Agent Design

**Talentra (Consistency Check):**
```
Checks: does copilot answer contradict candidate evaluation?
Trigger: always runs as final node
Action: human interrupt if inconsistent
```

**CineWave (Policy Gate):**
```
Checks: 27 automated GateCheck objects
Trigger: runs before every model promotion
Action: DEPLOY or BLOCK with automatic rollback
```

**Design lesson:** Talentra's critic is content-focused (logical
consistency). CineWave's critic is quality-focused (metric thresholds).
MagenticOne uses both patterns depending on task type.

---

## What We Learned Across Both Systems

**Finding 1 — Explicit state (LangGraph) beats implicit state (FastAPI) for debugging.**
When Talentra's consistency check fired incorrectly, the state dict
showed exactly which node produced the bad output. CineWave incidents
required log tracing across 5 services.

**Finding 2 — LLM-driven tool selection adds latency but gains flexibility.**
Talentra's LangChain tools add ~200ms per query vs CineWave's direct
calls at <10ms. The flexibility is worth it for open-ended HR queries
where tool order is unpredictable.

**Finding 3 — Synchronous vs asynchronous human oversight.**
Talentra's interrupt gate blocks the response until human reviews.
CineWave's policy gate is asynchronous — alerts engineer post-hoc.
The right choice depends on consequence severity: HR decisions warrant
synchronous oversight; recommendation quality does not.

**Finding 4 — AutoGen patterns generalize across domains.**
The same 4-5 agent pattern (retriever, reasoner, explorer, critic,
orchestrator) works for both HR intelligence and recommendation systems.
This suggests AutoGen-style orchestration is a general pattern, not
domain-specific.

---

## Connection to AutoGen / MagenticOne Research

| This Document | AutoGen Paper (Wu et al. 2023) |
|---|---|
| LangGraph nodes | ConversableAgent |
| FastAPI orchestrator | GroupChatManager |
| Consistency Check node | Critic Agent |
| Policy Gate BLOCK | Human proxy interrupt |
| Tool use (FAISS/Chroma) | Tool-use Agent |
| State dict | Shared message history |

The key difference from AutoGen: both Talentra and CineWave use
**deterministic orchestration** — the agent order is fixed by the
pipeline design, not negotiated at runtime via LLM group chat.

This is closer to MagenticOne's approach of assigning fixed roles
to specialized agents under a coordinating Orchestrator, rather than
AutoGen's free-form group conversation.

---

## Reproducibility

**Talentra:**
```bash
git clone https://github.com/AkilanManivannanak/talentra_copilot
pip install -r requirements.txt
python app/services/agent.py  # LangGraph pipeline
```

**CineWave:**
```bash
git clone https://github.com/AkilanManivannanak/two-stage-recommender-als-ranker-api
docker compose up -d
curl http://localhost:8000/recommend -d '{"user_id": 1, "k": 10}'
# Multi-agent pipeline executes on every request
```

---

*Akilan Manivannan · MS in Artificial Intelligence · Long Island University*
*akilan.manivannan@my.liu.edu*
