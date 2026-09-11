"""
test_quality_regression.py — tests that fail when the system gets WORSE.

The original suite had seven tests and all of them asserted shapes: a GRU cell
returns a 16-vector, a UCB score is a float, probabilities sum to one. One of
them (test_ddpm_schedule) imported no project code at all — it reimplemented the
noise schedule inline and asserted that numpy's cumprod works.

Shape tests catch typos. None of them would have caught any of the things that
were actually wrong: a reranker that never ran, a gate whose inputs were
literals, metrics served from default arguments, a GRU trained on data generated
from its own labels.

These assert on quality and honesty instead. They require a trained bundle:
    python3 scripts/prepare_data.py && python3 scripts/train_als.py \\
      && python3 scripts/train_ranker.py && python3 scripts/evaluate.py \\
      && python3 scripts/build_bundle.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "bundle"
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    not (BUNDLE / "metrics.json").exists(),
    reason="no trained bundle; run the training pipeline first")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((BUNDLE / "metrics.json").read_text())


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from recsys.serving import app as A
    return TestClient(A.app), A


# ── Quality floors ────────────────────────────────────────────────────────────

def test_system_beats_the_popularity_baseline(metrics):
    """The floor every recommender must clear. A personalised system that loses
    to 'show everyone the most popular titles' has no reason to exist."""
    b = metrics["baselines"]
    assert b["als_lambdarank"]["ndcg@10"] > b["popularity"]["ndcg@10"], (
        f"NDCG@10 {b['als_lambdarank']['ndcg@10']} does not beat popularity "
        f"{b['popularity']['ndcg@10']}")


def test_system_beats_every_measured_baseline(metrics):
    b = metrics["baselines"]
    ours = b["als_lambdarank"]["ndcg@10"]
    losses = {n: v["ndcg@10"] for n, v in b.items()
              if n != "als_lambdarank" and v["ndcg@10"] >= ours}
    assert not losses, f"baselines matching or beating the system: {losses}"


def test_reranking_improves_on_retrieval_alone(metrics):
    """Stage 2 must earn its place. If the ranker does not beat raw ALS order,
    the second stage is latency with no benefit."""
    b = metrics["baselines"]
    assert b["als_lambdarank"]["ndcg@10"] > b["als_only"]["ndcg@10"]


def test_the_improvement_is_statistically_distinguishable(metrics):
    """A point estimate without an interval is not a result."""
    boot = metrics["bootstrap"]["als_lambdarank_vs_als_only"]
    assert boot["ci95_lo"] > 0, (
        f"95% CI on the NDCG delta includes zero: "
        f"[{boot['ci95_lo']}, {boot['ci95_hi']}]")
    assert boot["n_boot"] >= 1000


def test_no_user_segment_regresses_badly(metrics):
    """A global gain that hides a cold-user collapse is a failure, not a win."""
    bad = {k: v["delta"] for k, v in metrics["slices_by_user_activity"].items()
           if v["delta"] < -0.02}
    assert not bad, f"segments regressing more than 2 points of NDCG: {bad}"


def test_retrieval_recall_leaves_the_ranker_something_to_find(metrics):
    """The ranker cannot rank what retrieval never returned."""
    assert metrics["headline"]["candidate_recall_at_200"] > 0.40


def test_slate_is_diverse_enough_to_avoid_a_filter_bubble(metrics):
    assert metrics["baselines"]["als_lambdarank"]["diversity_score"] > 0.35


def test_exposure_corrected_quality_has_not_collapsed(metrics):
    """Plain NDCG can be propped up by head items. IPS-NDCG cannot."""
    assert metrics["headline"]["ips_ndcg_at_10"] > 0.015


# ── Honesty: no invented numbers anywhere ─────────────────────────────────────

FABRICATED = ["0.1409", "0.0399", "0.0362", "0.8124", "0.6923", "0.2826", "0.1637"]


def test_no_fabricated_metric_literals_remain_in_the_source():
    """
    The specific values that used to be hardcoded as fallback defaults. If one
    reappears as an actual NUMBER in the code, someone has reintroduced a metric
    the pipeline never measured.

    Scans numeric tokens rather than raw text, so prose in a docstring
    explaining the history — and a test asserting a value is absent — do not
    trip it. Only a real numeric literal counts.
    """
    import tokenize

    targets = {float(x) for x in FABRICATED}
    offenders = []
    roots = [ROOT / "src", ROOT / "infra", ROOT / "scripts"]
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name.endswith(".pre-audit-bak"):
                continue
            try:
                with tokenize.open(path) as fh:
                    for tok in tokenize.generate_tokens(fh.readline):
                        if tok.type != tokenize.NUMBER:
                            continue
                        try:
                            value = float(tok.string)
                        except ValueError:
                            continue
                        if value in targets:
                            offenders.append(
                                f"{path.relative_to(ROOT)}:{tok.start[0]}: "
                                f"{tok.line.strip()[:90]}")
            except (SyntaxError, UnicodeDecodeError):
                continue

    # A test may legitimately reference a retired value to assert it is gone.
    offenders = [o for o in offenders if "not in" not in o and "!=" not in o]
    assert not offenders, ("fabricated metric literals reintroduced:\n"
                           + "\n".join(offenders))


def test_api_reports_unavailable_rather_than_defaulting(client):
    """
    With no bundle the API must say so. The old code returned
    `m.get("ndcg_at_10", 0.1409)`, so an unmeasured system rendered a dashboard
    indistinguishable from a measured one.
    """
    _, A = client
    saved = A._bundle.metrics
    try:
        A._bundle.metrics = {}
        live = A._live_metrics()
        assert live["status"] == "unavailable"
        assert "ndcg_at_10" not in live
        assert "remedy" in live
    finally:
        A._bundle.metrics = saved


def test_metrics_carry_their_caveats():
    payload = json.loads((BUNDLE / "serve_payload.json").read_text())
    text = " ".join(payload["caveats"]).lower()
    assert "offline" in text and "a/b" in text, \
        "the offline-only caveat must travel with the metrics"


# ── The two-stage pipeline actually runs ──────────────────────────────────────

def test_reranker_runs_on_the_request_path(client):
    """
    The bug this exists for: serving built a differently-shaped feature vector
    and called predict_proba() on a LightGBM Booster inside a bare except, so
    the ranker silently never ran and als_score == ranker_score == score.
    """
    c, _ = client
    items = c.post("/recommend", json={"user_id": 1, "k": 10}).json()["items"]
    assert items, "no recommendations returned"
    assert any(abs(i["als_score"] - i["ranker_score"]) > 1e-9 for i in items), \
        "als_score equals ranker_score for every item: stage 2 is not running"


def test_ranker_failures_are_counted_not_swallowed(client):
    _, A = client
    assert hasattr(A, "RANKER_FAILURES"), \
        "ranker failures must be observable, not silently passed"


def test_recommendations_are_distinct_and_diverse(client):
    c, _ = client
    items = c.post("/recommend", json={"user_id": 7, "k": 10}).json()["items"]
    ids = [i["item_id"] for i in items]
    assert len(set(ids)) == len(ids), f"duplicate items in one slate: {ids}"


def test_catalog_ids_match_the_trained_factors(client):
    """
    The shipped catalogue once used synthetic sequential ids (movieId 1 =
    "The Shawshank Redemption"; real MovieLens movieId 1 is "Toy Story"), so
    every als_score belonged to a different film than the title beside it.
    """
    _, A = client
    import pickle
    factors = pickle.load(open(BUNDLE / "item_factors.pkl", "rb"))
    overlap = set(A.CATALOG) & set(factors)
    assert len(overlap) > 3000, (
        f"only {len(overlap)} catalogue ids have ALS factors; "
        f"the catalogue and the model are keyed differently")
    assert A.CATALOG[1]["title"].lower().startswith("toy story"), \
        f"movieId 1 should be Toy Story, got {A.CATALOG[1]['title']!r}"


# ── The gate cannot pass on unmeasured inputs ─────────────────────────────────

def test_gate_blocks_when_quality_is_below_baseline(metrics):
    from recsys.serving.policy_gate import POLICY_GATE
    bad = dict(metrics["headline"])
    bad["ndcg_at_10"] = bad["best_baseline_ndcg"] - 0.01
    res = POLICY_GATE.gate_from_measured_metrics(
        bad, {"n": 500, "p95_ms": 10.0, "p99_ms": 20.0})
    assert res.recommendation == "BLOCK"
    assert "ndcg_at_10_beats_best_baseline" in res.blocking_checks


def test_gate_blocks_when_latency_was_never_measured(metrics):
    """Fail closed. The old gate substituted p50 for p95, and assumed on-spec
    when there was no data at all."""
    from recsys.serving.policy_gate import POLICY_GATE
    res = POLICY_GATE.gate_from_measured_metrics(dict(metrics["headline"]), None)
    assert res.recommendation == "BLOCK"
    assert "p95_ms_recommend" in res.blocking_checks
    assert any(c.inconclusive for c in res.checks if c.name == "p95_ms_recommend")


def test_gate_blocks_on_too_few_latency_samples(metrics):
    from recsys.serving.policy_gate import POLICY_GATE
    res = POLICY_GATE.gate_from_measured_metrics(
        dict(metrics["headline"]), {"n": 5, "p95_ms": 1.0, "p99_ms": 2.0})
    assert res.recommendation == "BLOCK", \
        "five samples is not a p95 measurement"


def test_gate_passes_only_on_real_measurements(metrics):
    from recsys.serving.policy_gate import POLICY_GATE
    gi = dict(metrics["headline"])
    gi["slices"] = metrics["slices_by_user_activity"]
    res = POLICY_GATE.gate_from_measured_metrics(
        gi, {"n": 600, "p95_ms": 25.0, "p99_ms": 60.0})
    assert res.recommendation in ("DEPLOY", "REVIEW")
    assert res.to_dict()["n_inconclusive"] == 0


def test_gate_has_no_literal_inputs():
    """
    Thirteen of the old gate's twenty-seven inputs were assigned constants
    chosen to sit just inside their own thresholds. Guard against that pattern
    returning to the measured path.
    """
    import inspect
    from recsys.serving import policy_gate as pg
    src = inspect.getsource(pg._measured_gate)
    import re
    assigns = re.findall(r'^\s*m\["[a-z_]+"\]\s*=\s*[0-9.]+', src, re.M)
    assert not assigns, f"literal gate inputs reintroduced: {assigns}"


# ── The trained GRU is honest about what it learned ───────────────────────────

def test_session_gru_beats_its_baselines_on_held_out_users():
    path = BUNDLE / "session_gru_metrics.json"
    if not path.exists():
        pytest.skip("run scripts/train_session_gru.py")
    m = json.loads(path.read_text())
    assert m["val_accuracy"] > m["baseline_majority"], \
        "the GRU does not beat always-guess-the-most-common-genre"
    assert m["val_accuracy"] > m["baseline_persist_genre"], \
        "the GRU does not beat repeating the previous genre"
    assert "by user" in m["split"]


def test_session_gru_reports_held_out_not_training_accuracy():
    path = BUNDLE / "session_gru_metrics.json"
    if not path.exists():
        pytest.skip("run scripts/train_session_gru.py")
    m = json.loads(path.read_text())
    assert m["n_val_sequences"] > 0
    best = [h for h in m["history"] if h["epoch"] == m["best_epoch"]][0]
    assert abs(best["val_acc"] - m["val_accuracy"]) < 1e-6, \
        "reported accuracy must be the held-out value from the best epoch"


# ── Offline RL trains on data, not noise ──────────────────────────────────────

def test_rl_dataset_rewards_come_from_real_ratings():
    path = BUNDLE / "rl_dataset_stats.json"
    if not path.exists():
        pytest.skip("run scripts/build_rl_dataset.py")
    stats = json.loads(path.read_text())
    assert "MovieLens" in stats["source"]
    assert stats["n_sessions"] > 0
    assert stats["reward_std"] > 0, "constant reward carries no learning signal"


def test_rl_endpoint_refuses_to_invent_training_data(client):
    """It must fail loudly with no dataset, never fall back to random rewards."""
    c, _ = client
    import recsys.serving.app as A
    real = A._BUNDLE / "rl_sessions.jsonl"
    tmp = A._BUNDLE / "rl_sessions.jsonl.hidden"
    if not real.exists():
        pytest.skip("run scripts/build_rl_dataset.py")
    real.rename(tmp)
    try:
        body = c.post("/rl/train/offline", json={"n_sessions": 10, "n_epochs": 1}).json()
        assert body["trained"] is False
        assert "remedy" in body
    finally:
        tmp.rename(real)


def test_doubly_robust_refuses_without_a_reward_model():
    """DR was advertised in the README while its function had zero callers.
    Now it runs — and will not quietly degrade to IPS under the DR label."""
    from recsys.serving.ope_eval import CounterfactualEvaluator, LoggedInteraction
    ev = CounterfactualEvaluator()
    ev.log(LoggedInteraction(1, 10, 0, True, True, 1.0, 0.05))
    with pytest.raises(ValueError, match="requires reward_model"):
        ev.evaluate_policy({1: [10]}, {}, method="dr")
    out = ev.evaluate_policy({1: [10]}, {}, method="dr",
                             reward_model=lambda u, i: 0.5)
    assert out["method"] == "dr" and out["n_matched"] == 1
