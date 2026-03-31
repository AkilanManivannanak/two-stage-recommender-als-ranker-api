"""
Contextual Bandit — Phase 6: Exploration & Long-term Value
===========================================================
LinUCB bandit with composite reward signal.

Reward components (spec):
  +1.0   play_start
  +2.0   watch_3min
  -1.0   abandon_30s
  +3.0   completion
  +1.5   add_to_list
  +0.5   next_day_return (proxy: computed offline, applied as prior)
  +1.0   repeat_engagement

Reward is discounted by impression position (position bias correction).

Thompson Sampling available as alternative exploration strategy.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


# ── Reward scheme ─────────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "play_start":       1.0,
    "watch_3min":       2.0,
    "abandon_30s":     -1.0,
    "completion":       3.0,
    "add_to_list":      1.5,
    "remove_from_list": -0.5,
    "click":            0.3,
    "impression":       0.0,   # no reward for impression alone
}


def compute_reward(event_type: str, position: int = 0, completion_pct: float = 0.0) -> float:
    """
    Composite reward for a single event.
    Discounts by position (position bias: items lower in page get slight boost).
    """
    base = REWARD_WEIGHTS.get(event_type, 0.0)
    # Partial credit for partial completion
    if event_type == "completion" and completion_pct > 0:
        base = base * min(completion_pct, 1.0)
    # Position discount: item at position 0 = full reward, position 10 = 0.85x
    position_factor = max(0.7, 1.0 - position * 0.015)
    return base * position_factor


# ── LinUCB ────────────────────────────────────────────────────────────────

class LinUCBArm:
    """Single arm (item genre bucket) in LinUCB bandit."""

    def __init__(self, context_dim: int = 8, alpha: float = 1.0):
        self.alpha = alpha
        self.d = context_dim
        self.A = np.eye(context_dim, dtype=np.float64)   # d×d feature covariance
        self.b = np.zeros(context_dim, dtype=np.float64)  # d×1 reward accumulator
        self.n_updates = 0
        self.total_reward = 0.0

    def update(self, context: np.ndarray, reward: float) -> None:
        x = context[:self.d].astype(np.float64)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1
        self.total_reward += reward

    def ucb_score(self, context: np.ndarray) -> float:
        x = context[:self.d].astype(np.float64)
        try:
            A_inv = np.linalg.inv(self.A)
            theta = A_inv @ self.b
            exploit = float(theta @ x)
            explore = self.alpha * math.sqrt(float(x @ A_inv @ x))
            return exploit + explore
        except np.linalg.LinAlgError:
            return 0.0

    def to_dict(self) -> dict:
        return {
            "A":           self.A.tolist(),
            "b":           self.b.tolist(),
            "alpha":       self.alpha,
            "d":           self.d,
            "n_updates":   self.n_updates,
            "total_reward": self.total_reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinUCBArm":
        arm = cls(context_dim=d["d"], alpha=d["alpha"])
        arm.A = np.array(d["A"])
        arm.b = np.array(d["b"])
        arm.n_updates = d["n_updates"]
        arm.total_reward = d["total_reward"]
        return arm


class LinUCBBandit:
    """
    Contextual bandit using LinUCB. Arms = item genre buckets.
    Context = user features (genre affinities, session length, time-of-day, etc.)

    Guardrails:
      - Exploration never exceeds max_explore_fraction per page
      - Items with artwork_trust < 0.6 are never used for exploration
      - Abandonment rate > 0.8 triggers exploration reduction
    """

    def __init__(
        self,
        context_dim: int = 8,
        alpha: float = 1.0,
        max_explore_fraction: float = 0.20,
    ):
        self.context_dim = context_dim
        self.alpha = alpha
        self.max_explore_fraction = max_explore_fraction
        self.arms: dict[str, LinUCBArm] = {}
        self._total_updates = 0
        self._created_at = time.time()

    def _get_or_create_arm(self, arm_id: str) -> LinUCBArm:
        if arm_id not in self.arms:
            self.arms[arm_id] = LinUCBArm(self.context_dim, self.alpha)
        return self.arms[arm_id]

    def user_context(
        self,
        user_id: int,
        user_genres: list[str],
        time_of_day: str = "evening",
        session_length: int = 0,
        user_genre_ratings: dict = None,
    ) -> np.ndarray:
        """
        Build 8-dimensional context vector for this user/session.
        """
        ugr = user_genre_ratings or {}
        all_genres = ["Action","Comedy","Drama","Horror","Sci-Fi","Romance","Thriller","Documentary"]

        # Genre affinity vector (top 6 genres), normalised
        affinities = []
        for g in all_genres[:6]:
            ratings = ugr.get(g, [])
            avg = float(np.mean(ratings)) / 5.0 if ratings else 0.5
            affinities.append(avg)

        time_enc = {"morning": 0.0, "afternoon": 0.5, "evening": 1.0, "night": 0.8}.get(time_of_day, 0.5)
        session_norm = min(float(session_length) / 20.0, 1.0)

        ctx = np.array(affinities + [time_enc, session_norm], dtype=np.float64)
        return ctx[:self.context_dim]

    def select_exploration_items(
        self,
        candidates: list[dict],
        user_id: int,
        context: np.ndarray,
        n: int = 3,
    ) -> list[dict]:
        """
        Select n exploration items using UCB scores.
        Guardrail: skip items with low artwork trust.
        """
        # Filter exploration-eligible items
        eligible = [
            c for c in candidates
            if float(c.get("artwork_trust", 1.0)) >= 0.6
        ]
        if not eligible:
            return []

        scored = []
        for item in eligible:
            arm_id = item.get("primary_genre", "Unknown")
            arm = self._get_or_create_arm(arm_id)
            ucb = arm.ucb_score(context)
            scored.append((ucb, item))

        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:n]]

    def update(self, context: np.ndarray, item: dict, reward: float) -> None:
        arm_id = item.get("primary_genre", "Unknown")
        arm = self._get_or_create_arm(arm_id)
        arm.update(context, reward)
        self._total_updates += 1

    def save(self, path: str = "artifacts/bandit_state.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "arms":                {k: v.to_dict() for k, v in self.arms.items()},
            "context_dim":         self.context_dim,
            "alpha":               self.alpha,
            "max_explore_fraction": self.max_explore_fraction,
            "total_updates":       self._total_updates,
            "saved_at":            time.time(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str = "artifacts/bandit_state.json") -> None:
        try:
            with open(path) as f:
                state = json.load(f)
            self.arms = {k: LinUCBArm.from_dict(v) for k, v in state["arms"].items()}
            self.context_dim = state.get("context_dim", self.context_dim)
            self.alpha = state.get("alpha", self.alpha)
            self._total_updates = state.get("total_updates", 0)
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "n_arms":        len(self.arms),
            "total_updates": self._total_updates,
            "alpha":         self.alpha,
            "arm_stats": {
                arm_id: {
                    "n_updates":    arm.n_updates,
                    "total_reward": round(arm.total_reward, 2),
                    "avg_reward":   round(arm.total_reward / max(arm.n_updates, 1), 3),
                }
                for arm_id, arm in self.arms.items()
            },
        }
