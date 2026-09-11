"""
session_gru.py — GRU session encoder with real backpropagation through time.

Replaces the training half of session_intent.py, which had three compounding
problems:

  1. The "training data" came from _generate_training_sessions(), which drew a
     template per intent and used the template index as the label. No MovieLens
     data was involved, so the reported accuracy measured whether the model
     could recover a rule written by hand a few lines above it.
  2. There was no train/validation split. `final_acc` was accuracy on the same
     3,000 sequences the model fit.
  3. The backward pass zeroed h_prev before computing the update, applied a tanh
     derivative to a GRU cell, and touched only Wh. No gradient flowed through
     the recurrence, so the "sequence model" never learned to be sequential.

What changed
------------
Task. MovieLens carries no session-intent labels, and no amount of code makes
one appear, so supervised intent classification cannot be evaluated honestly on
it. The supervised task here is NEXT-GENRE PREDICTION: given the first n-1
events of a real user session, predict the primary genre of event n. That label
exists in the data, which means the accuracy number means something.

Data. Real ML-1M interactions, sessionised on a 30-minute inactivity gap.

Split. By USER, not by row. A user's sessions appear in exactly one of train or
validation, so the model cannot memorise a user in training and be graded on
that same user's other sessions.

Gradients. Full BPTT through the update gate, the reset gate and the candidate,
with the recurrence intact. tests/test_gru_gradients.py checks every parameter
against a central finite-difference estimate.

The intent taxonomy (binge / discovery / ...) is retained in session_intent.py
as an unsupervised routing heuristic. It is not a trained classifier and no
accuracy is claimed for it.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class StepCache:
    """Everything the backward pass needs from one forward step."""
    x:       np.ndarray
    h_prev:  np.ndarray
    z:       np.ndarray
    r:       np.ndarray
    h_hat:   np.ndarray
    xh:      np.ndarray
    xrh:     np.ndarray


class TrainableGRU:
    """
    Single GRU cell with the standard formulation

        z_t = sigmoid(Wz [x_t; h_{t-1}] + bz)          update gate
        r_t = sigmoid(Wr [x_t; h_{t-1}] + br)          reset gate
        n_t = tanh(   Wh [x_t; r_t * h_{t-1}] + bh)    candidate
        h_t = (1 - z_t) * h_{t-1} + z_t * n_t

    and the exact gradients of that formulation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        s = np.sqrt(1.0 / (input_dim + hidden_dim))
        shape = (hidden_dim, input_dim + hidden_dim)
        self.Wz = rng.normal(0, s, shape)
        self.Wr = rng.normal(0, s, shape)
        self.Wh = rng.normal(0, s, shape)
        self.bz = np.zeros(hidden_dim)
        self.br = np.zeros(hidden_dim)
        # A positive candidate bias keeps early updates from collapsing to zero.
        self.bh = np.zeros(hidden_dim)

    # ── Forward ──────────────────────────────────────────────────────────────
    def step(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, StepCache]:
        xh = np.concatenate([x, h_prev])
        z = sigmoid(self.Wz @ xh + self.bz)
        r = sigmoid(self.Wr @ xh + self.br)
        xrh = np.concatenate([x, r * h_prev])
        h_hat = np.tanh(self.Wh @ xrh + self.bh)
        h = (1.0 - z) * h_prev + z * h_hat
        return h, StepCache(x=x, h_prev=h_prev, z=z, r=r, h_hat=h_hat, xh=xh, xrh=xrh)

    def forward(self, xs: list[np.ndarray]) -> tuple[np.ndarray, list[StepCache]]:
        h = np.zeros(self.hidden_dim)
        caches: list[StepCache] = []
        for x in xs:
            h, c = self.step(np.asarray(x, dtype=np.float64), h)
            caches.append(c)
        return h, caches

    def encode(self, xs: list[np.ndarray]) -> np.ndarray:
        return self.forward(xs)[0] if xs else np.zeros(self.hidden_dim)

    # ── Backward ─────────────────────────────────────────────────────────────
    def zero_grads(self) -> dict[str, np.ndarray]:
        return {
            "Wz": np.zeros_like(self.Wz), "bz": np.zeros_like(self.bz),
            "Wr": np.zeros_like(self.Wr), "br": np.zeros_like(self.br),
            "Wh": np.zeros_like(self.Wh), "bh": np.zeros_like(self.bh),
        }

    def backward(self, caches: list[StepCache], dh_last: np.ndarray,
                 grads: Optional[dict] = None) -> dict[str, np.ndarray]:
        """
        BPTT from the final hidden state back through every step.

        dh_last: gradient of the loss with respect to h_T.
        """
        g = grads if grads is not None else self.zero_grads()
        d = self.input_dim
        dh = np.asarray(dh_last, dtype=np.float64).copy()

        for c in reversed(caches):
            # h = (1 - z) * h_prev + z * h_hat
            dz     = dh * (c.h_hat - c.h_prev)
            dh_hat = dh * c.z
            dh_prev = dh * (1.0 - c.z)          # direct carry path

            # candidate: h_hat = tanh(Wh [x; r*h_prev] + bh)
            dpre_h = dh_hat * (1.0 - c.h_hat ** 2)
            g["Wh"] += np.outer(dpre_h, c.xrh)
            g["bh"] += dpre_h
            dxrh = self.Wh.T @ dpre_h
            d_rh = dxrh[d:]                      # d(r * h_prev)
            dr = d_rh * c.h_prev
            dh_prev += d_rh * c.r                # through the reset gate

            # update gate: z = sigmoid(Wz [x; h_prev] + bz)
            dpre_z = dz * c.z * (1.0 - c.z)
            g["Wz"] += np.outer(dpre_z, c.xh)
            g["bz"] += dpre_z
            dh_prev += (self.Wz.T @ dpre_z)[d:]

            # reset gate: r = sigmoid(Wr [x; h_prev] + br)
            dpre_r = dr * c.r * (1.0 - c.r)
            g["Wr"] += np.outer(dpre_r, c.xh)
            g["br"] += dpre_r
            dh_prev += (self.Wr.T @ dpre_r)[d:]

            dh = dh_prev
        return g

    # ── Parameter access, for the gradient check and for persistence ─────────
    PARAMS = ("Wz", "bz", "Wr", "br", "Wh", "bh")

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def state_dict(self) -> dict:
        return {p: getattr(self, p).copy() for p in self.PARAMS} | {
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim}

    @classmethod
    def from_state(cls, sd: dict) -> "TrainableGRU":
        m = cls(sd["input_dim"], sd["hidden_dim"])
        for p in cls.PARAMS:
            setattr(m, p, np.asarray(sd[p], dtype=np.float64))
        return m


class LinearHead:
    """hidden -> class logits."""

    def __init__(self, hidden_dim: int, n_classes: int, seed: int = 99):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (n_classes, hidden_dim))
        self.b = np.zeros(n_classes)

    def logits(self, h: np.ndarray) -> np.ndarray:
        return self.W @ h + self.b

    def state_dict(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}

    @classmethod
    def from_state(cls, sd: dict) -> "LinearHead":
        m = cls(sd["W"].shape[1], sd["W"].shape[0])
        m.W = np.asarray(sd["W"], dtype=np.float64)
        m.b = np.asarray(sd["b"], dtype=np.float64)
        return m


def sequence_loss_and_grads(gru: TrainableGRU, head: LinearHead,
                            xs: list[np.ndarray], label: int):
    """
    Cross-entropy over the final hidden state, with gradients for both modules.

    Returns (loss, correct, gru_grads, dW, db).
    """
    h, caches = gru.forward(xs)
    probs = softmax(head.logits(h))
    loss = -float(np.log(max(probs[label], 1e-12)))
    correct = int(np.argmax(probs) == label)

    dlogits = probs.copy()
    dlogits[label] -= 1.0
    dW = np.outer(dlogits, h)
    db = dlogits
    dh = head.W.T @ dlogits

    gru_grads = gru.backward(caches, dh) if caches else gru.zero_grads()
    return loss, correct, gru_grads, dW, db
