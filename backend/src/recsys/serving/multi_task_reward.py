"""
multi_task_reward.py
====================
Multi-Task Learning reward model for CineWave RecSys.

Simultaneously optimizes 4 objectives in a shared-representation network:

  Task 1 — Click prediction       (binary cross-entropy)
  Task 2 — Completion prediction  (binary cross-entropy on watch_90pct)
  Task 3 — Add-to-list prediction (binary cross-entropy)
  Task 4 — Skip avoidance         (binary cross-entropy, inverted)

Each task shares a common feature encoder (bottom layers) but has its own
task-specific head (top layers). This is the standard Multi-Gate Mixture-of-
Experts (MMoE) / shared-bottom multi-task architecture used in production
recommendation systems.

Why multi-task here?
  - Shared representation: click, completion, add-to-list all depend on the
    same underlying user-item compatibility signal. Joint training improves
    generalisation on sparse labels (add-to-list is rare; completion is noisy).
  - Auxiliary task regularisation: completion prediction regularises the click
    head, preventing it from overfitting to easy positives.
  - Single forward pass: 4 predictions for the price of 1 feature encoding.

Architecture:
  Input (11 features)
    → Shared encoder [Linear(11→32) → ReLU → Linear(32→16) → ReLU]
    → Task heads (each: Linear(16→1) → Sigmoid)
        head_click       → P(play_start)
        head_completion  → P(watch_90pct)
        head_add_to_list → P(add_to_list)
        head_skip        → P(skip)

  Combined reward (for RL policy):
    reward = w_click * P(click)
           + w_completion * P(completion)
           + w_add * P(add_to_list)
           - w_skip * P(skip)

All in pure numpy — no PyTorch dependency.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ── Task weights (matches reward_model.py _W for consistency) ─────────────────
TASK_WEIGHTS = {
    "click":       1.0,   # play_start
    "completion":  2.0,   # watch_90pct — strongest signal
    "add_to_list": 1.0,   # explicit save
    "skip":       -0.5,   # negative engagement
}

# ── Architecture dimensions ───────────────────────────────────────────────────
INPUT_DIM   = 11   # matches reward_model.py 11-feature vector
SHARED_DIM1 = 32   # shared encoder layer 1
SHARED_DIM2 = 16   # shared encoder layer 2 (bottleneck)
N_TASKS     = 4    # click · completion · add_to_list · skip


class MultiTaskRewardModel:
    """
    Multi-task learning reward model.

    Shared-bottom architecture: one encoder, 4 task-specific heads.
    Trained jointly on all 4 objectives with IPS-weighted samples.

    Usage:
        model = MultiTaskRewardModel(seed=42)
        model.fit(samples)   # samples: list of dicts with features + labels
        reward = model.predict(feature_vector)
    """

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        # ── Shared encoder weights ────────────────────────────────────────────
        # Layer 1: INPUT_DIM → SHARED_DIM1
        scale1 = np.sqrt(2.0 / INPUT_DIM)
        self.W1 = rng.normal(0, scale1, (INPUT_DIM, SHARED_DIM1)).astype(np.float32)
        self.b1 = np.zeros(SHARED_DIM1, dtype=np.float32)

        # Layer 2: SHARED_DIM1 → SHARED_DIM2
        scale2 = np.sqrt(2.0 / SHARED_DIM1)
        self.W2 = rng.normal(0, scale2, (SHARED_DIM1, SHARED_DIM2)).astype(np.float32)
        self.b2 = np.zeros(SHARED_DIM2, dtype=np.float32)

        # ── Task-specific heads: SHARED_DIM2 → 1 ─────────────────────────────
        scale_h = np.sqrt(2.0 / SHARED_DIM2)
        self.heads = {
            "click":       (rng.normal(0, scale_h, (SHARED_DIM2,)).astype(np.float32), 0.0),
            "completion":  (rng.normal(0, scale_h, (SHARED_DIM2,)).astype(np.float32), 0.0),
            "add_to_list": (rng.normal(0, scale_h, (SHARED_DIM2,)).astype(np.float32), 0.0),
            "skip":        (rng.normal(0, scale_h, (SHARED_DIM2,)).astype(np.float32), 0.0),
        }

        self._trained   = False
        self._train_loss: dict[str, float] = {}

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Shared encoder: x → 16-dim representation."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        return h2

    def _sigmoid(self, z: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))))

    def _task_predict(self, h: np.ndarray, task: str) -> float:
        """Single task head prediction → probability."""
        w, b = self.heads[task]
        return self._sigmoid(float(h @ w) + b)

    def predict_all(self, x: np.ndarray) -> dict[str, float]:
        """
        Multi-task forward pass.
        Returns per-task probabilities from shared representation.
        """
        h = self._encode(x)
        return {task: self._task_predict(h, task) for task in self.heads}

    def predict(self, x: np.ndarray) -> float:
        """
        Combined reward prediction (scalar).
        Weighted sum of task probabilities — used by RL policy.
        """
        preds = self.predict_all(x)
        return (
            TASK_WEIGHTS["click"]       * preds["click"]
          + TASK_WEIGHTS["completion"]  * preds["completion"]
          + TASK_WEIGHTS["add_to_list"] * preds["add_to_list"]
          + TASK_WEIGHTS["skip"]        * preds["skip"]   # negative weight
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        samples: list[dict],
        lr:       float = 0.01,
        n_epochs: int   = 5,
        ips_clip: float = 10.0,
    ) -> dict[str, float]:
        """
        Joint multi-task training with IPS-weighted binary cross-entropy.

        samples: list of dicts with keys:
          features   : np.ndarray (11-dim)
          click      : int (0/1)  — play_start event
          completion : int (0/1)  — watch_90pct event
          add_to_list: int (0/1)  — add_to_list event
          skip       : int (0/1)  — skip event
          propensity : float      — P(item was shown) for IPS correction

        IPS weighting: each sample weighted by min(1/propensity, ips_clip)
        to correct for exposure bias (popular items shown more often).

        Gradient update: shared encoder receives gradients from ALL 4 tasks
        simultaneously — this is the multi-task learning signal.
        """
        if not samples:
            return {"status": "skipped", "reason": "no samples"}

        task_losses = {t: 0.0 for t in self.heads}

        for _ in range(n_epochs):
            np.random.shuffle(samples)
            for s in samples:
                x          = np.array(s["features"], dtype=np.float32)
                ips_weight = min(1.0 / max(float(s.get("propensity", 0.1)), 1e-6), ips_clip)
                labels     = {
                    "click":       float(s.get("click", 0)),
                    "completion":  float(s.get("completion", 0)),
                    "add_to_list": float(s.get("add_to_list", 0)),
                    "skip":        float(s.get("skip", 0)),
                }

                # ── Forward ──────────────────────────────────────────────────
                h1    = np.maximum(0, x @ self.W1 + self.b1)
                h2    = np.maximum(0, h1 @ self.W2 + self.b2)
                preds = {t: self._task_predict(h2, t) for t in self.heads}

                # ── Per-task gradients at heads ───────────────────────────────
                dh2_total = np.zeros(SHARED_DIM2, dtype=np.float32)

                for task, (w, b) in self.heads.items():
                    y      = labels[task]
                    p      = preds[task]
                    # Binary cross-entropy gradient (IPS-weighted)
                    dL_dp  = ips_weight * (p - y)
                    dp_dz  = p * (1 - p)           # sigmoid derivative
                    dz     = dL_dp * dp_dz

                    # Head gradient
                    dw = dz * h2
                    db = dz

                    # Accumulate gradient into shared encoder
                    dh2_total += dz * w

                    task_losses[task] += abs(dL_dp)

                    # Update head (SGD)
                    self.heads[task] = (w - lr * dw, b - lr * float(db))

                # ── Shared encoder backward (receives all 4 task gradients) ───
                # Layer 2 backward
                dh2_relu = dh2_total * (h2 > 0)
                dW2 = np.outer(h1, dh2_relu)
                db2 = dh2_relu
                dh1 = dh2_relu @ self.W2.T

                # Layer 1 backward
                dh1_relu = dh1 * (h1 > 0)
                dW1 = np.outer(x, dh1_relu)
                db1 = dh1_relu

                # Update shared encoder (SGD)
                self.W2 -= lr * dW2.astype(np.float32)
                self.b2 -= lr * db2.astype(np.float32)
                self.W1 -= lr * dW1.astype(np.float32)
                self.b1 -= lr * db1.astype(np.float32)

        n = max(len(samples) * n_epochs, 1)
        self._train_loss = {t: v / n for t, v in task_losses.items()}
        self._trained    = True
        print(
            f"  [MultiTask] Trained on {len(samples)} samples · "
            f"click_loss={self._train_loss['click']:.4f} · "
            f"completion_loss={self._train_loss['completion']:.4f} · "
            f"add_loss={self._train_loss['add_to_list']:.4f} · "
            f"skip_loss={self._train_loss['skip']:.4f}"
        )
        return {"status": "trained", "n_samples": len(samples), **self._train_loss}

    @property
    def trained(self) -> bool:
        return self._trained

    def summary(self) -> dict:
        return {
            "architecture": "shared_bottom_multi_task",
            "tasks":        list(self.heads.keys()),
            "task_weights": TASK_WEIGHTS,
            "input_dim":    INPUT_DIM,
            "shared_dims":  [SHARED_DIM1, SHARED_DIM2],
            "trained":      self._trained,
            "train_losses": self._train_loss,
            "description":  (
                "Multi-task learning: 4 objectives (click, completion, "
                "add_to_list, skip) jointly trained on shared encoder. "
                "IPS-weighted to correct for exposure bias."
            ),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
MULTI_TASK_REWARD = MultiTaskRewardModel(seed=42)
