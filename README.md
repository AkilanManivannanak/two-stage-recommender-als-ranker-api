# CineWave

A two-stage movie recommender: ALS retrieval, LightGBM LambdaRank reranking,
served behind FastAPI with REST and GraphQL, plus a Salesforce Apex integration.

Every number in this file is produced by `backend/scripts/evaluate.py` on a
held-out test split and written to `backend/artifacts/bundle/metrics.json`.
Nothing here is hardcoded. You can reproduce all of it in about two minutes on a
laptop, and `tests/test_quality_regression.py` fails the build if any of it stops
being true.

---

## Reproduce it

```bash
cd backend
pip install -r requirements.txt

python scripts/fetch_movielens.py      # ML-1M, ~24MB, not committed
python scripts/prepare_data.py         # chronological 80/10/10 split
python scripts/train_als.py            # ALS + candidate generation
python scripts/train_ranker.py         # LambdaRank reranker
python scripts/evaluate.py             # all baselines + bootstrap CIs
python scripts/build_catalog.py        # catalogue keyed to the trained factors
python scripts/build_bundle.py         # serving bundle
```

Optional components:

```bash
python scripts/train_session_gru.py    # next-genre GRU
python scripts/build_rl_dataset.py     # offline RL data from logged ratings
```

Then serve:

```bash
PYTHONPATH=src python -m uvicorn recsys.serving.app:app --port 8000
# REST     http://localhost:8000/docs
# GraphQL  http://localhost:8000/graphql
```

---

## Measured results

MovieLens-1M. 1,000,209 ratings, 6,040 users, 3,883 catalogue titles. Split
per-user chronologically 80/10/10 — each user's earliest 80% of interactions are
training, the next 10% validation, the final 10% test. An item counts as
relevant when the user rated it 4.0 or higher.

Evaluated on the **test** split, 5,741 users with at least one relevant item:

| Ranker | NDCG@10 | MRR@10 | Recall@10 | IPS-NDCG@10 | Diversity | Coverage |
|---|---|---|---|---|---|---|
| popularity | 0.0423 | 0.0804 | 0.0402 | 0.0119 | 0.378 | 0.023 |
| co-occurrence | 0.0481 | 0.0845 | 0.0560 | 0.0185 | 0.359 | 0.168 |
| ALS only | 0.0411 | 0.0692 | 0.0514 | 0.0214 | 0.389 | 0.410 |
| **ALS + LambdaRank** | **0.0516** | **0.0957** | **0.0586** | **0.0216** | **0.421** | 0.355 |

Paired bootstrap over users, 2,000 resamples:

| Comparison | Δ NDCG@10 | 95% CI | P(Δ > 0) |
|---|---|---|---|
| ALS+LambdaRank vs ALS only | +0.0105 | [+0.0073, +0.0137] | 1.000 |
| ALS+LambdaRank vs popularity | +0.0093 | [+0.0060, +0.0127] | 1.000 |
| ALS+LambdaRank vs co-occurrence | +0.0035 | [+0.0001, +0.0069] | 0.978 |
| co-occurrence vs ALS only | +0.0070 | [+0.0037, +0.0103] | 1.000 |
| popularity vs ALS only | +0.0012 | [−0.0024, +0.0048] | 0.754 |

**Serving latency**, 600 requests to `/recommend` after warm-up, single
uvicorn process, catalogue in memory. Three runs on the same machine:

```
run 1   p50  8.3ms   p90 20.3ms   p95 21.2ms   p99 53.9ms   max  62.9ms
run 2   p50 15.0ms   p90 31.8ms   p95 32.4ms   p99 70.9ms   max 167.1ms
run 3   p50 21.8ms   p90 31.6ms   p95 32.8ms   p99 84.7ms   max  95.3ms
```

The p95 SLO of 50ms holds across all three. The p99 ceiling of 80ms does not —
run 3 came in at 84.7ms, and the gate flags p99 as a non-blocking warning for
exactly this reason. The spread is machine load, not model behaviour; anyone
quoting a single number from this table should quote the worst one.

Reproduce with `PYTHONPATH=src python scripts/smoke_and_loadtest.py`.

### Four things these numbers say that are worth stating plainly

**ALS retrieval on its own does not beat a popularity list.** 0.0411 against
0.0423, and the confidence interval on that difference spans zero. The
collaborative filtering stage earns its place by producing a *candidate set*
with 0.473 recall@200 and 0.410 catalogue coverage — far more than popularity's
0.023 — not by ranking well. The ranking is the reranker's job.

**The margin over co-occurrence is thin.** +0.0035 with a lower bound of
+0.0001 and p = 0.978. A simple item-item co-occurrence baseline gets most of
the way there. That is worth knowing before anyone deploys this.

**Light users regress.** Broken out by user activity tercile:

| Segment | Users | ALS only | + LambdaRank | Δ |
|---|---|---|---|---|
| light | 1,824 | 0.0548 | 0.0535 | **−0.0013** |
| medium | 1,932 | 0.0345 | 0.0447 | +0.0102 |
| heavy | 1,985 | 0.0348 | 0.0564 | +0.0216 |

The gain is concentrated in users with long histories. Six of the reranker's
fifteen features are user-history statistics, so this is the expected failure
mode, and it is exactly what slice evaluation exists to surface. The policy gate
treats a segment regression worse than −0.02 as a blocking failure.

**Exposure-corrected quality is much lower than raw NDCG.** IPS-NDCG@10 is
0.0216 against a plain 0.0516, weighting each hit by inverse exposure relative
to an average-popularity item. The system leans on head titles more than an
exposure-corrected view rewards. Notably ALS-only scores 0.0214 here — nearly
identical — so most of the reranker's raw-NDCG gain comes from popular items.

---

## Architecture

```
                     ┌─────────────────────────────────────────┐
  MovieLens-1M ─────►│ prepare_data.py                         │
  1,000,209 ratings  │ per-user chronological 80/10/10         │
                     │ train-only feature tables (no leakage)  │
                     └────────────────┬────────────────────────┘
                                      │
                     ┌────────────────▼────────────────────────┐
  STAGE 1            │ implicit ALS  factors=64 reg=0.05       │
  RETRIEVAL          │ alpha=40 iterations=20                  │
                     │ 3,667 item factors · 6,040 user factors │
                     │ top-200 candidates, train-seen masked   │
                     │ recall@200 = 0.473                      │
                     └────────────────┬────────────────────────┘
                                      │
                     ┌────────────────▼────────────────────────┐
  STAGE 2            │ LightGBM LGBMRanker                     │
  RERANK             │ objective=lambdarank  metric=ndcg@10    │
                     │ grouped by user · 15 features           │
                     │ trained on val, reported on test        │
                     └────────────────┬────────────────────────┘
                                      │
                     ┌────────────────▼────────────────────────┐
  SERVE              │ FastAPI · REST + GraphQL · p95 21ms     │
                     │ slate optimiser · exploration slots     │
                     └────────────────┬────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────┐
        ▼                             ▼                         ▼
  Next.js client            Salesforce Apex            policy gate
                            (Agentforce / Flow / LWC)  11 measured checks
```

### Feature definition

The fifteen reranker features live in one file,
`src/recsys/serving/ranker_features.py`, imported by both the training scripts
and the serving path. Training/serving skew in a two-stage recommender almost
always comes from two copies of the feature code drifting; there is one copy.

By gain: `als_score` (9210), `item_pop_log` (4497), `als_rank_norm` (4469),
`genre_share` (2764), `user_cnt_log` (2742), `user_tenure_days` (2510).

### Session GRU

`scripts/train_session_gru.py` trains a single GRU cell with full BPTT on real
ML-1M sessions (30-minute inactivity gap, 11,729 sessions).

The task is **next-genre prediction**: given the first n−1 events of a session,
predict the primary genre of event n. MovieLens carries no session-intent
labels, so intent classification cannot be evaluated on it honestly; next-genre
can, because the label is in the data.

Split by user, 80/20 — no user appears in both halves.

| | Accuracy |
|---|---|
| always predict the most common genre | 0.2907 |
| repeat the previous event's genre | 0.3802 |
| **trained GRU (best epoch by held-out accuracy)** | **0.4485** |

The gradients are checked against central finite differences for all six
parameter tensors in `tests/test_gru_gradients.py`.

The intent taxonomy (binge / discovery / background / …) is retained as an
**unsupervised routing heuristic** over session statistics. It is not trained and
no accuracy is claimed for it.

### Offline RL

`rl_policy.py` implements REINFORCE over slate orderings: Gumbel-max sampling of
permutations, Plackett-Luce log-probability, EMA baseline for variance
reduction, Redis persistence.

`scripts/build_rl_dataset.py` builds the training set from logged interactions —
each reward is the user's own held-out rating (≥4 → 1.0, 3 → 0.3, ≤2 → 0.0), and
items the user never rated are excluded rather than given a default. 982
sessions, 1,299 scored items, reward σ = 0.315.

This is a **warm start**, not a corrected off-policy update: there is no
importance weight, so the gradient is biased by the gap between the logging
policy and the current one. That is acceptable for initialisation and would not
be acceptable as an evaluation, which is what the estimators below are for.

### Off-policy evaluation

`ope_eval.py` provides IPS-NDCG and a doubly-robust estimator:

```
DR = r̂(new) + (1/p)·1[logged == new]·(observed − r̂(logged))
```

Unbiased if either the propensity model or the reward model is correct.
`evaluate_policy(method="dr")` requires a `reward_model` and raises without one
rather than silently computing IPS and labelling the result doubly-robust.

Exposure propensities are estimated from the same logs they correct, so this is
a bias correction, not a randomised experiment.

---

## GraphQL

Mounted at `/graphql`, backed by the same singletons as the REST API — one
recommendation path, not two.

```graphql
query {
  recommendations(userId: 1, k: 5) {
    modelVersion latencyMs diversityScore
    items { itemId title primaryGenre year alsScore rankerScore retrievalSource }
  }
  modelMetrics {
    status ndcgAt10 ndcgAt10AlsOnly ndcgLiftPctVsAls ipsNdcgAt10
    bestBaseline bestBaselineNdcg caveats
    bootstrap { comparison deltaMean ci95Lo ci95Hi pDeltaGt0 }
  }
  policyGate { recommendation gatePassed summary nChecks nInconclusive }
}
```

`modelMetrics.status` is `"measured"` only when a trained bundle is loaded. With
no bundle it returns `"unavailable"` with a remedy — it never returns a default.

`alsScore` and `rankerScore` are separate fields throughout, including across the
Apex boundary, so a collapsed two-stage pipeline is visible rather than hidden.

---

## Salesforce

`salesforce/` is a deployable SFDX project: a Named Credential callout client,
an `@InvocableMethod` for Agentforce and Flow, an `@AuraEnabled` controller with
a Lightning Web Component, and 24 Apex tests against a nine-mode
`HttpCalloutMock` covering every failure branch.

`backend/tests/test_apex_contract.py` parses the GraphQL documents straight out
of the `.cls` source and executes them against the live Python schema, so a
field rename on either side fails in CI rather than in a customer's org.

Deployment steps are in [`salesforce/README.md`](salesforce/README.md).

---

## Policy gate

`policy_gate.py` runs eleven checks before promotion. Every threshold is either
relative to a baseline measured in the same run, or an absolute floor set below
the current measured value so it guards against regression.

```
DEPLOY — All 11 measured checks passed.
  PASS  ndcg_at_10_beats_best_baseline       0.0516 gt 0.0481
  PASS  ndcg_at_10_beats_retrieval_only      0.0516 gt 0.0411
  PASS  ips_ndcg_at_10_floor                 0.0216 gt 0.015
  PASS  candidate_recall_at_200_floor        0.4734 gt 0.4
  PASS  diversity_score_floor                0.4209 gt 0.35
  PASS  catalog_coverage_floor               0.3546 gt 0.25
  PASS  p95_ms_recommend                     18.75  lt 50.0
  PASS  p99_ms_recommend                     70.91  lt 80.0
  PASS  slice_no_regression_light            -0.0013 gte -0.02
  PASS  slice_no_regression_medium           0.0102 gte -0.02
  PASS  slice_no_regression_heavy            0.0216 gte -0.02
```

The gate **fails closed**. An input that was never measured is marked
inconclusive, and an inconclusive critical check blocks the deploy. With fewer
than 100 latency samples the p95 check is inconclusive rather than passing.

---

## Tests

```bash
cd backend && PYTHONPATH=src python -m pytest tests/ -q     # 47 tests
```

| File | What it guards |
|---|---|
| `test_quality_regression.py` | NDCG beats every baseline; CI excludes zero; no segment regresses; no fabricated literals; the reranker actually runs; the gate blocks on unmeasured inputs |
| `test_gru_gradients.py` | BPTT against central finite differences, all six parameter tensors |
| `test_apex_contract.py` | Apex GraphQL literals validate and execute against the live schema |
| `test_core.py` | component smoke tests |

CI trains a model on every run, because a quality gate over a model that does
not exist tests nothing.

---

## Known limitations

- **Offline evaluation only.** No live A/B test has been run. The IPS and
  doubly-robust estimators correct for exposure bias using propensities
  estimated from the same logs; they are not a randomised experiment.
- **The reranker trains on the validation period** (100k interactions), which is
  small. Training it on a held-out slice of the train period would give it more
  data, at the cost of a more complex splitting scheme.
- **Light users regress**, as shown above. Not yet fixed.
- **Poster art is sparse.** The catalogue is now keyed to real MovieLens ids;
  only 218 of 3,883 titles carried over art from the previous TMDB set, because
  that set was modern films and ML-1M ends in 2003. Set `TMDB_API_KEY` and run
  the catalogue enrichment to backfill.
- **Scala `FeaturePipeline.scala`** is not wired into the pipeline. It is
  reference material for how the ALS step would run on Spark at scale; the
  Python path is what executes.
- **Single-process latency.** The p95 above is one uvicorn worker on a laptop
  with the catalogue in memory, not a distributed deployment under production
  load.

---

## Provenance

This README was rewritten on 2026-09-10 after an audit found that the previous
version's headline numbers were not produced by any code in the repository.

The prior README claimed NDCG@10 = 0.1409 and a +253% lift over ALS. Those
values existed only as fallback arguments — `m.get("ndcg_at_10", 0.1409)` — in
three files; no code path computed them, and the repository contained no trained
model and no dataset to compute them from. Also corrected: the reranker was
never executing at request time (mismatched feature vectors and a `predict_proba`
call on a Booster, inside a bare `except`); the catalogue used synthetic ids that
did not match the model's; thirteen of the policy gate's inputs were literals set
just inside their own thresholds, and the p95 check was fed p50; the session GRU
was trained on data generated from its own labels with a broken backward pass and
reported training accuracy; and the offline RL endpoint trained on
`rng.uniform(0.0, 3.0)` rewards drawn independently of the slate.

The previous README is preserved as `README.pre-audit-bak.md`.

The measured result is smaller than the one it replaces. It is also reproducible.
