"""
test_gru_gradients.py — verify BPTT against finite differences.

The previous implementation applied a tanh derivative to a GRU cell, zeroed the
previous hidden state before the update, and touched only one of six parameter
tensors. Nothing caught it, because no test compared the analytic gradient to a
numerical one.

Central differences: df/dp ~= (f(p + eps) - f(p - eps)) / (2 eps), which is
O(eps^2) accurate. In float64 with eps=1e-5 the agreement should be ~1e-8.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.serving.session_gru import (          # noqa: E402
    TrainableGRU, LinearHead, sequence_loss_and_grads, softmax)

INPUT_DIM, HIDDEN_DIM, N_CLASSES = 8, 16, 6
EPS = 1e-5


def _fixture(seq_len: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    gru = TrainableGRU(INPUT_DIM, HIDDEN_DIM, seed=seed)
    head = LinearHead(HIDDEN_DIM, N_CLASSES, seed=seed + 1)
    xs = [rng.normal(0, 1, INPUT_DIM) for _ in range(seq_len)]
    label = int(rng.integers(0, N_CLASSES))
    return gru, head, xs, label


def _loss(gru, head, xs, label) -> float:
    h, _ = gru.forward(xs)
    return -float(np.log(max(softmax(head.logits(h))[label], 1e-12)))


@pytest.mark.parametrize("param", TrainableGRU.PARAMS)
def test_gru_parameter_gradients_match_finite_differences(param):
    gru, head, xs, label = _fixture()
    _, _, grads, _, _ = sequence_loss_and_grads(gru, head, xs, label)

    P = gru.get(param)
    analytic = grads[param]
    assert analytic.shape == P.shape, f"{param}: grad shape {analytic.shape} != {P.shape}"

    rng = np.random.default_rng(7)
    flat = P.reshape(-1)
    idxs = rng.choice(flat.size, size=min(12, flat.size), replace=False)

    for i in idxs:
        original = flat[i]
        flat[i] = original + EPS
        plus = _loss(gru, head, xs, label)
        flat[i] = original - EPS
        minus = _loss(gru, head, xs, label)
        flat[i] = original

        numeric = (plus - minus) / (2 * EPS)
        got = analytic.reshape(-1)[i]
        denom = max(abs(numeric), abs(got), 1e-8)
        assert abs(numeric - got) / denom < 1e-5, (
            f"{param}[{i}]: analytic={got:.10f} numeric={numeric:.10f}")


def test_head_gradients_match_finite_differences():
    gru, head, xs, label = _fixture()
    _, _, _, dW, db = sequence_loss_and_grads(gru, head, xs, label)

    for i in range(N_CLASSES):
        for j in range(0, HIDDEN_DIM, 5):
            original = head.W[i, j]
            head.W[i, j] = original + EPS
            plus = _loss(gru, head, xs, label)
            head.W[i, j] = original - EPS
            minus = _loss(gru, head, xs, label)
            head.W[i, j] = original
            numeric = (plus - minus) / (2 * EPS)
            denom = max(abs(numeric), abs(dW[i, j]), 1e-8)
            assert abs(numeric - dW[i, j]) / denom < 1e-5

    for i in range(N_CLASSES):
        original = head.b[i]
        head.b[i] = original + EPS
        plus = _loss(gru, head, xs, label)
        head.b[i] = original - EPS
        minus = _loss(gru, head, xs, label)
        head.b[i] = original
        numeric = (plus - minus) / (2 * EPS)
        denom = max(abs(numeric), abs(db[i]), 1e-8)
        assert abs(numeric - db[i]) / denom < 1e-5


def test_gradient_actually_flows_through_the_recurrence():
    """
    The specific bug: the old code zeroed h_prev, so earlier timesteps had no
    influence. If the recurrence is intact, perturbing the FIRST event of a
    sequence must change the loss.
    """
    gru, head, xs, label = _fixture(seq_len=6)
    base = _loss(gru, head, xs, label)
    perturbed = [x.copy() for x in xs]
    perturbed[0] = perturbed[0] + 0.5
    assert abs(_loss(gru, head, perturbed, label) - base) > 1e-9, \
        "the first event does not affect the loss: recurrence is disconnected"


def test_longer_sequences_accumulate_gradient():
    """A 1-step and a 10-step sequence must not produce identical gradients."""
    gru, head, _, _ = _fixture()
    rng = np.random.default_rng(3)
    xs = [rng.normal(0, 1, INPUT_DIM) for _ in range(10)]
    _, _, g_short, _, _ = sequence_loss_and_grads(gru, head, xs[:1], 2)
    _, _, g_long, _, _ = sequence_loss_and_grads(gru, head, xs, 2)
    assert not np.allclose(g_short["Wz"], g_long["Wz"]), \
        "sequence length does not change the gradient"


def test_all_six_parameter_tensors_receive_gradient():
    """The old update touched only Wh. All six must move."""
    gru, head, xs, label = _fixture(seq_len=4)
    _, _, grads, _, _ = sequence_loss_and_grads(gru, head, xs, label)
    dead = [p for p in TrainableGRU.PARAMS
            if np.allclose(grads[p], 0.0, atol=1e-12)]
    assert not dead, f"these parameters received no gradient: {dead}"
