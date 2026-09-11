"""
graphql_api.py — GraphQL surface over the CineWave serving layer.

Why GraphQL alongside the existing REST API rather than instead of it:

  The REST API is shaped for the web client, which wants one fat /recommend
  response. The consumers added later — a Salesforce Apex callout, an
  Agentforce action, a partner integration — want narrow, specific projections:
  "give me five titles and nothing else", "give me the current model's NDCG and
  whether the gate passed". Over REST that is either a dozen bespoke endpoints
  or a lot of over-fetching across a callout boundary that bills per second.

  It also gives external consumers a typed contract. Apex is statically typed;
  a schema it can introspect is worth more than a JSON blob it has to parse
  defensively.

Mounted at /graphql. The same singletons back both APIs, so there is one
recommendation path, not two.
"""
from __future__ import annotations

import time
from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter


# ── Types ─────────────────────────────────────────────────────────────────────

@strawberry.type
class Movie:
    item_id: int
    title: str
    genres: str
    primary_genre: str
    year: Optional[int]
    avg_rating: float
    vote_count: int
    poster_url: str
    description: str


@strawberry.type
class Recommendation:
    item_id: int
    title: str
    primary_genre: str
    year: Optional[int]
    poster_url: str
    score: float
    als_score: float = strawberry.field(
        description="Stage-1 retrieval score (ALS dot product). Distinct from "
                    "`score`, which is the post-rerank value.")
    ranker_score: float = strawberry.field(
        description="Stage-2 LambdaRank score.")
    retrieval_source: str
    exploration_slot: bool


@strawberry.type
class Slate:
    user_id: int
    k: int
    model_version: str
    latency_ms: float
    diversity_score: float
    items: list[Recommendation]


@strawberry.type
class BaselineMetrics:
    ranker: str
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    ips_ndcg_at_10: float
    diversity_score: float
    coverage: float
    n_users: int


@strawberry.type
class BootstrapInterval:
    comparison: str
    delta_mean: float
    ci95_lo: float
    ci95_hi: float
    p_delta_gt_0: float
    n_users: int


@strawberry.type
class ModelMetrics:
    status: str = strawberry.field(
        description="'measured' when a trained bundle is loaded, 'unavailable' "
                    "otherwise. There are no default values: an unmeasured "
                    "system reports that it is unmeasured.")
    model_version: Optional[str]
    ndcg_at_10: Optional[float]
    ndcg_at_10_als_only: Optional[float]
    ndcg_lift_pct_vs_als: Optional[float]
    mrr_at_10: Optional[float]
    recall_at_10: Optional[float]
    ips_ndcg_at_10: Optional[float]
    diversity_score: Optional[float]
    coverage: Optional[float]
    best_baseline: Optional[str]
    best_baseline_ndcg: Optional[float]
    caveats: list[str]
    baselines: list[BaselineMetrics]
    bootstrap: list[BootstrapInterval]


@strawberry.type
class GateCheckType:
    name: str
    value: Optional[float]
    threshold: Optional[float]
    comparison: str
    passed: bool
    critical: bool
    inconclusive: bool


@strawberry.type
class GateResultType:
    recommendation: str
    gate_passed: bool
    summary: str
    n_checks: int
    n_passed: int
    n_inconclusive: int
    checks: list[GateCheckType]


@strawberry.type
class FeedbackAck:
    accepted: bool
    user_id: int
    item_id: int
    event: str
    message: str


# ── Resolvers ─────────────────────────────────────────────────────────────────

def _app():
    """Late import: the GraphQL module is imported by app.py itself."""
    from recsys.serving import app as A
    return A


def _to_reco(r: dict) -> Recommendation:
    return Recommendation(
        item_id=int(r["item_id"]),
        title=r.get("title", ""),
        primary_genre=r.get("primary_genre", ""),
        year=r.get("year"),
        poster_url=r.get("poster_url", "") or "",
        score=float(r.get("score", 0.0)),
        als_score=float(r.get("als_score", 0.0)),
        ranker_score=float(r.get("ranker_score", 0.0)),
        retrieval_source=r.get("retrieval_source", "unknown"),
        exploration_slot=bool(r.get("exploration_slot", False)),
    )


@strawberry.type
class Query:

    @strawberry.field(description="Personalised slate for a user, through the "
                                  "same two-stage path the REST API uses.")
    def recommendations(self, user_id: int, k: int = 10,
                        session_item_ids: Optional[list[int]] = None) -> Slate:
        A = _app()
        t0 = time.time()
        recs = A._build_recs(int(user_id), k=int(k),
                             session_item_ids=list(session_item_ids or []),
                             request_timestamp=t0)
        ms = (time.time() - t0) * 1000.0
        A._record(ms)
        genres = [r.get("primary_genre", "?") for r in recs]
        return Slate(
            user_id=int(user_id), k=int(k),
            model_version=A._bundle.model_version or "no-bundle-loaded",
            latency_ms=round(ms, 2),
            diversity_score=round(len(set(genres)) / max(len(genres), 1), 4),
            items=[_to_reco(r) for r in recs],
        )

    @strawberry.field(description="A single catalogue title.")
    def movie(self, item_id: int) -> Optional[Movie]:
        m = _app().CATALOG.get(int(item_id))
        if not m:
            return None
        return Movie(
            item_id=int(m.get("item_id", item_id)), title=m.get("title", ""),
            genres=m.get("genres", ""), primary_genre=m.get("primary_genre", ""),
            year=m.get("year"), avg_rating=float(m.get("avg_rating", 0.0) or 0.0),
            vote_count=int(m.get("vote_count", 0) or 0),
            poster_url=m.get("poster_url", "") or "",
            description=m.get("description", "") or "",
        )

    @strawberry.field(description="Catalogue search by title substring, "
                                  "optionally filtered to one genre.")
    def search(self, q: str = "", genre: Optional[str] = None,
               limit: int = 20) -> list[Movie]:
        A = _app()
        ql = q.lower().strip()
        out = []
        for m in A.CATALOG.values():
            if ql and ql not in str(m.get("title", "")).lower():
                continue
            if genre and m.get("primary_genre") != genre:
                continue
            out.append(m)
            if len(out) >= max(1, min(limit, 100)):
                break
        return [Movie(
            item_id=int(m.get("item_id", 0)), title=m.get("title", ""),
            genres=m.get("genres", ""), primary_genre=m.get("primary_genre", ""),
            year=m.get("year"), avg_rating=float(m.get("avg_rating", 0.0) or 0.0),
            vote_count=int(m.get("vote_count", 0) or 0),
            poster_url=m.get("poster_url", "") or "",
            description=m.get("description", "") or "",
        ) for m in out]

    @strawberry.field(description="Measured offline metrics for the loaded "
                                  "model. Never returns a default value.")
    def model_metrics(self) -> ModelMetrics:
        A = _app()
        live = A._live_metrics()
        if live.get("status") != "measured":
            return ModelMetrics(
                status=live.get("status", "unavailable"), model_version=None,
                ndcg_at_10=None, ndcg_at_10_als_only=None,
                ndcg_lift_pct_vs_als=None, mrr_at_10=None, recall_at_10=None,
                ips_ndcg_at_10=None, diversity_score=None, coverage=None,
                best_baseline=None, best_baseline_ndcg=None,
                caveats=[live.get("reason", "")], baselines=[], bootstrap=[])
        b = [BaselineMetrics(
                ranker=name, ndcg_at_10=v["ndcg@10"], mrr_at_10=v["mrr@10"],
                recall_at_10=v["recall@10"], ips_ndcg_at_10=v["ips_ndcg@10"],
                diversity_score=v["diversity_score"], coverage=v["coverage"],
                n_users=v["n_users"])
             for name, v in (A._bundle.baselines or {}).items()]
        boot = [BootstrapInterval(
                    comparison=name, delta_mean=v["delta_mean"],
                    ci95_lo=v["ci95_lo"], ci95_hi=v["ci95_hi"],
                    p_delta_gt_0=v["p_delta_gt_0"], n_users=v["n_users"])
                for name, v in (A._bundle.bootstrap or {}).items()]
        return ModelMetrics(
            status="measured", model_version=live.get("model_version"),
            ndcg_at_10=live.get("ndcg_at_10"),
            ndcg_at_10_als_only=live.get("ndcg_at_10_als_only"),
            ndcg_lift_pct_vs_als=live.get("ndcg_lift_pct_vs_als"),
            mrr_at_10=live.get("mrr_at_10"), recall_at_10=live.get("recall_at_10"),
            ips_ndcg_at_10=live.get("ips_ndcg_at_10"),
            diversity_score=live.get("diversity_score"),
            coverage=live.get("coverage"), best_baseline=live.get("best_baseline"),
            best_baseline_ndcg=live.get("best_baseline_ndcg"),
            caveats=list(live.get("caveats") or []), baselines=b, bootstrap=boot)

    @strawberry.field(description="Run the release gate against the measured "
                                  "metrics and live /recommend latency.")
    def policy_gate(self) -> GateResultType:
        A = _app()
        from recsys.serving.policy_gate import POLICY_GATE
        live = A._live_metrics()
        if live.get("status") != "measured":
            return GateResultType(recommendation="BLOCK", gate_passed=False,
                                  summary=live.get("reason", "no metrics"),
                                  n_checks=0, n_passed=0, n_inconclusive=0,
                                  checks=[])
        gi = dict(A._bundle.metrics)
        gi["slices"] = A._bundle.slices
        res = POLICY_GATE.gate_from_measured_metrics(gi, A._recommend_latency())
        d = res.to_dict()
        return GateResultType(
            recommendation=d["recommendation"], gate_passed=d["gate_passed"],
            summary=d["summary"], n_checks=d["n_checks"], n_passed=d["n_passed"],
            n_inconclusive=d["n_inconclusive"],
            checks=[GateCheckType(**{k: c[k] for k in
                    ("name", "value", "threshold", "comparison", "passed",
                     "critical", "inconclusive")}) for c in d["checks"]])


@strawberry.type
class Mutation:

    @strawberry.mutation(description="Record an interaction. Feeds the reward "
                                     "model and the offline RL training set.")
    def record_feedback(self, user_id: int, item_id: int, event: str,
                        dwell_seconds: float = 0.0) -> FeedbackAck:
        allowed = {"play", "skip", "like", "dislike", "add_to_list", "complete"}
        if event not in allowed:
            return FeedbackAck(accepted=False, user_id=user_id, item_id=item_id,
                               event=event,
                               message=f"event must be one of {sorted(allowed)}")
        A = _app()
        try:
            A._log_interaction(int(user_id), int(item_id), event,
                               float(dwell_seconds))
        except Exception as exc:
            return FeedbackAck(accepted=False, user_id=user_id, item_id=item_id,
                               event=event, message=f"log failed: {exc}")
        return FeedbackAck(accepted=True, user_id=user_id, item_id=item_id,
                           event=event, message="recorded")


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_router = GraphQLRouter(schema, path="/graphql")
