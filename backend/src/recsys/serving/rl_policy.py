"""
Policy Gradient RL Agent — REINFORCE for Slate Reranking
=========================================================
Extends the existing LinUCB bandit (bandit_v2.py) with a full policy
gradient agent that learns to rerank slates to maximise long-term reward.

Architecture
------------
LinUCB (bandit_v2.py):
  - Selects which genre ARM to explore/exploit
  - Single-step: pick best arm given context, update on reward

REINFORCE (this module):
  - Takes a slate of K candidates from the ranker
  - Learns a POLICY (probability distribution over orderings)
  - Updates using Monte Carlo returns — credit assignment across
    multiple items in a session, not just the single clicked item
  - Maps to Netflix AWSFT: items that beat the user's baseline value
    get upweighted; items below baseline get downweighted

Why REINFORCE over DQN for slates:
  - Action space is combinatorial (K! orderings) — DQN requires
    discrete actions; policy gradient naturally handles this via
    a scoring network + softmax
  - We don't have a learned environment model — MC returns work
    directly on observed rewards without a simulator
  - Simpler to implement honestly without external RL libraries

State representation (8 features per user-item pair):
  1. collaborative_score    (ALS cosine similarity)
  2. ranker_score           (LGB output)
  3. genre_match            (user genre history ∩ item genre)
  4. item_popularity        (normalised by max)
  5. recency                (1 / (days_since_release + 1))
  6. position_in_candidate  (rank in pre-RL slate, normalised)
  7. user_avg_rating        (normalised to [0,1])
  8. user_n_ratings         (log-normalised)

Policy network: linear scoring (no hidden layers, avoids overfit on
small data) → softmax → sample order via Gumbel-max trick.

Training: REINFORCE with baseline subtraction (advantage = return - baseline)
to reduce variance. Baseline = exponential moving average of returns.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


# ── Hyper-parameters ──────────────────────────────────────────────────────────

LR            = 0.01    # learning rate
GAMMA         = 0.95    # discount factor for multi-step returns
BASELINE_BETA = 0.9     # EMA smoothing for baseline
N_FEATURES    = 8       # state feature dimension
MAX_SLATE     = 20      # maximum slate size


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    item:         dict,
    rank_in_slate: int,
    slate_size:    int,
    user_activity: dict,
) -> np.ndarray:
    """
    Extract 8-dimensional state vector for one item in the slate.
    All features normalised to [0, 1].
    """
    collab   = float(item.get("collaborative_score", item.get("score", 0.5)))
    ranker   = float(item.get("ranker_score", item.get("final_score", 0.5)))
    genre_m  = float(item.get("genre_match", 0.5))
    pop      = float(item.get("popularity", 0.5))
    # Recency: years_since_release → normalise (1980-2025 range)
    year     = int(item.get("year", 2000))
    recency  = 1.0 / max(1.0, 2025 - year + 1)
    recency  = min(recency * 20, 1.0)  # scale to [0,1]
    pos_norm = rank_in_slate / max(slate_size - 1, 1)
    avg_r    = (float(user_activity.get("avg_rating", 3.5)) - 1.0) / 4.0
    n_r      = math.log1p(float(user_activity.get("n_ratings", 50))) / math.log1p(1000)

    return np.array([
        collab, ranker, genre_m, pop, recency,
        1.0 - pos_norm,  # higher = better position
        avg_r, n_r,
    ], dtype=np.float32)


# ── Policy (linear scoring) ───────────────────────────────────────────────────

class LinearPolicy:
    """
    Linear scoring policy: score(s) = W · s + b
    W: (N_FEATURES,) weight vector
    b: scalar bias

    The policy scores each item independently, then applies softmax
    to get a probability distribution over items for sampling.
    This is equivalent to a 1-layer network without activation —
    appropriate for our feature dimension and training set size.
    """

    def __init__(self, n_features: int = N_FEATURES, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Initialise near zero to start close to uniform policy
        self.W = rng.normal(0.0, 0.01, n_features).astype(np.float64)
        self.b = 0.0

    def score(self, features: np.ndarray) -> float:
        return float(np.dot(self.W, features.astype(np.float64)) + self.b)

    def scores(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Score a slate: feature_matrix shape (K, N_FEATURES)."""
        return feature_matrix.astype(np.float64) @ self.W + self.b

    def softmax_probs(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return softmax probability distribution over K items."""
        raw = self.scores(feature_matrix)
        raw = raw - raw.max()  # numerical stability
        exp = np.exp(raw)
        return exp / exp.sum()

    def sample_order(self, feature_matrix: np.ndarray) -> list[int]:
        """
        Sample a permutation of items using Gumbel-max trick.
        Each item gets score + Gumbel(0,1) noise; sort descending.
        This is equivalent to sampling without replacement from the
        softmax distribution.
        """
        raw    = self.scores(feature_matrix)
        gumbel = -np.log(-np.log(np.random.uniform(size=len(raw)) + 1e-10) + 1e-10)
        return list(np.argsort(-(raw + gumbel)))

    def greedy_order(self, feature_matrix: np.ndarray) -> list[int]:
        """Deterministic greedy ordering (for inference)."""
        return list(np.argsort(-self.scores(feature_matrix)))

    def log_prob(self, feature_matrix: np.ndarray, chosen_order: list[int]) -> float:
        """
        Log probability of the chosen ordering under current policy.
        Uses the Plackett-Luce model:
          log P(order) = Σ_i [score(item_i) - log Σ_{j>=i} exp(score(item_j))]
        """
        raw   = self.scores(feature_matrix)
        total = 0.0
        remaining = list(range(len(raw)))
        for idx in chosen_order:
            # log softmax over remaining items
            remaining_scores = raw[remaining]
            log_denom = float(np.log(np.sum(np.exp(remaining_scores - remaining_scores.max())))
                              + remaining_scores.max())
            total += float(raw[idx]) - log_denom
            remaining.remove(idx)
        return total

    def policy_gradient(
        self,
        feature_matrix: np.ndarray,
        chosen_order:   list[int],
        advantage:      float,
        lr:             float = LR,
        max_grad_norm:  float = 1.0,
        l2_lambda:      float = 1e-4,
    ) -> None:
        """
        REINFORCE update with gradient clipping and L2 regularisation:
          ΔW = lr * clip(advantage * ∇log P, max_norm=1.0) - l2 * W

        Gradient clipping (max_norm=1.0) prevents weight explosion when
        advantage is large or features are correlated with reward by chance.
        L2 regularisation keeps weights from drifting far from zero between
        real feedback signals.
        """
        raw   = self.scores(feature_matrix)
        probs = np.exp(raw - raw.max())
        probs /= probs.sum()

        grad_W = np.zeros_like(self.W)
        grad_b = 0.0

        remaining = list(range(len(raw)))
        for idx in chosen_order:
            if not remaining:
                break
            rem_arr    = np.array(remaining)
            rem_scores = raw[rem_arr]
            rem_probs  = np.exp(rem_scores - rem_scores.max())
            rem_probs /= rem_probs.sum()
            exp_f_rem  = rem_probs @ feature_matrix[rem_arr].astype(np.float64)
            grad_W    += feature_matrix[idx].astype(np.float64) - exp_f_rem
            grad_b    += 1.0 - rem_probs.sum()
            remaining.remove(idx)

        # Scale by advantage
        grad_W *= advantage
        grad_b *= advantage

        # Gradient clipping: clip by global L2 norm
        grad_norm = float(np.linalg.norm(grad_W))
        if grad_norm > max_grad_norm:
            grad_W = grad_W * (max_grad_norm / grad_norm)
            grad_b = grad_b * (max_grad_norm / grad_norm)

        # Apply gradient + L2 weight decay
        self.W += lr * grad_W - l2_lambda * self.W
        self.b += lr * grad_b

    def to_dict(self) -> dict:
        return {"W": self.W.tolist(), "b": self.b, "n_features": len(self.W)}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearPolicy":
        obj = cls(n_features=d["n_features"])
        obj.W = np.array(d["W"], dtype=np.float64)
        obj.b = float(d["b"])
        return obj


# ── Episode buffer (one session = one episode) ────────────────────────────────

@dataclass
class Transition:
    """Single step in an episode: a slate presented and a reward observed."""
    feature_matrix: np.ndarray    # (K, N_FEATURES)
    chosen_order:   list[int]     # sampled permutation
    reward:         float         # observed composite reward


@dataclass
class Episode:
    """One user session = one episode."""
    user_id:     int
    transitions: list[Transition] = field(default_factory=list)

    def add(self, fm: np.ndarray, order: list[int], reward: float) -> None:
        self.transitions.append(Transition(fm, order, reward))

    def monte_carlo_returns(self, gamma: float = GAMMA) -> list[float]:
        """
        Compute discounted returns for each step:
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        """
        T       = len(self.transitions)
        returns = [0.0] * T
        G       = 0.0
        for t in reversed(range(T)):
            G          = self.transitions[t].reward + gamma * G
            returns[t] = G
        return returns


# ── REINFORCE Agent ───────────────────────────────────────────────────────────

class REINFORCEAgent:
    """
    REINFORCE policy gradient agent for slate reranking.

    Integration with existing stack:
      1. Ranker (LGB) produces a scored candidate list
      2. REINFORCEAgent.rerank() reorders it using the learned policy
      3. After user engages, REINFORCEAgent.record_reward() logs the reward
      4. At session end, REINFORCEAgent.update() runs the REINFORCE update

    This sits AFTER the LGB ranker and BEFORE the MMR slate optimizer —
    it learns which features predict long-term engagement beyond the
    pointwise LGB score.
    """

    def __init__(self, n_features: int = N_FEATURES, lr: float = LR):
        self.policy      = LinearPolicy(n_features=n_features)
        self.baseline    = 0.0    # EMA of returns for variance reduction
        self.lr          = lr
        self.n_updates   = 0
        self.total_reward= 0.0
        self._episodes: dict[int, Episode] = {}   # user_id -> open episode
        self._history: list[dict] = []            # training log

    # ── Inference ─────────────────────────────────────────────────────────

    def rerank(
        self,
        candidates:    list[dict],
        user_activity: dict,
        explore:       bool = True,
        top_k:         int  = 10,
    ) -> list[dict]:
        """
        Rerank a candidate list using the learned policy.

        explore=True  → sample order via Gumbel-max (training / live serving)
        explore=False → greedy order (evaluation / A/B control arm)

        Returns top_k items in policy-determined order with rl_score appended.
        """
        K = min(len(candidates), MAX_SLATE)
        if K == 0:
            return candidates[:top_k]

        cands = candidates[:K]
        fm    = np.stack([
            extract_features(c, i, K, user_activity)
            for i, c in enumerate(cands)
        ])  # (K, N_FEATURES)

        order = self.policy.sample_order(fm) if explore else self.policy.greedy_order(fm)

        # Annotate with RL score and return top_k
        scores = self.policy.scores(fm)
        result = []
        for rank, idx in enumerate(order[:top_k]):
            item = dict(cands[idx])
            item["rl_score"]    = round(float(scores[idx]), 4)
            item["rl_rank"]     = rank
            item["rl_explore"]  = explore
            result.append(item)

        return result

    # ── Online reward recording ────────────────────────────────────────────

    def start_session(self, user_id: int) -> None:
        """Open a new episode for a user session."""
        self._episodes[user_id] = Episode(user_id=user_id)

    def record_step(
        self,
        user_id:    int,
        candidates: list[dict],
        order:      list[int],
        reward:     float,
        user_activity: dict,
    ) -> None:
        """Record one slate presentation + observed reward into the episode."""
        if user_id not in self._episodes:
            self.start_session(user_id)

        K  = min(len(candidates), MAX_SLATE)
        fm = np.stack([
            extract_features(candidates[i], i, K, user_activity)
            for i in range(K)
        ])
        self._episodes[user_id].add(fm, order[:K], reward)
        self.total_reward += reward

    # ── Training (end of session) ──────────────────────────────────────────

    def end_session(self, user_id: int) -> Optional[dict]:
        """
        Run REINFORCE update at end of user session.
        Returns training stats dict or None if no episode found.
        """
        ep = self._episodes.pop(user_id, None)
        if ep is None or not ep.transitions:
            return None

        returns = ep.monte_carlo_returns(GAMMA)

        # Update baseline (EMA of returns)
        avg_return = float(np.mean(returns))
        self.baseline = BASELINE_BETA * self.baseline + (1 - BASELINE_BETA) * avg_return

        total_loss = 0.0
        for t, (trans, G) in enumerate(zip(ep.transitions, returns)):
            advantage = G - self.baseline   # variance-reduced gradient signal
            # Policy gradient update
            self.policy.policy_gradient(
                trans.feature_matrix,
                trans.chosen_order,
                advantage,
                lr=self.lr,
            )
            log_p     = self.policy.log_prob(trans.feature_matrix, trans.chosen_order)
            total_loss += -advantage * log_p

        self.n_updates += 1
        if hasattr(self, "_redis_client"):
            self.save_to_redis(self._redis_client)
        stats = {
            "user_id":       user_id,
            "n_steps":       len(ep.transitions),
            "avg_return":    round(avg_return, 4),
            "baseline":      round(self.baseline, 4),
            "policy_loss":   round(total_loss / max(len(ep.transitions), 1), 4),
            "n_updates":     self.n_updates,
            "W_norm":        round(float(np.linalg.norm(self.policy.W)), 4),
        }
        self._history.append(stats)
        return stats

    # ── Batch training (offline warm-start from logged data) ──────────────

    def train_offline(
        self,
        logged_sessions: list[dict],
        user_activities: dict,
        n_epochs:        int = 3,
    ) -> dict:
        """
        Warm-start the policy from offline logged data before live serving.
        This is imitation learning (behavioral cloning) from logged interactions:
        the REINFORCE agent learns to replicate high-reward orderings observed
        in historical session data, following an off-policy behavioral cloning
        objective before online fine-tuning begins.

        logged_sessions: list of {user_id, slates: [{items, reward}]}
        user_activities: {user_id: {n_ratings, avg_rating, n_genres}}

        This lets the agent start with a reasonable policy rather than random,
        reducing the exploration cost in live serving.
        """
        total_updates = 0
        total_loss    = 0.0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for session in logged_sessions:
                uid      = session["user_id"]
                activity = user_activities.get(uid, {"n_ratings": 50, "avg_rating": 3.5})
                ep       = Episode(user_id=uid)

                for step in session.get("slates", []):
                    items  = step.get("items", [])
                    reward = float(step.get("reward", 0.0))
                    order  = list(step.get("order", range(len(items))))
                    if not items:
                        continue
                    K  = min(len(items), MAX_SLATE)
                    fm = np.stack([
                        extract_features(items[i], i, K, activity)
                        for i in range(K)
                    ])
                    ep.add(fm, order[:K], reward)

                if not ep.transitions:
                    continue

                returns  = ep.monte_carlo_returns(GAMMA)
                avg_ret  = float(np.mean(returns))
                self.baseline = BASELINE_BETA * self.baseline + (1 - BASELINE_BETA) * avg_ret

                for trans, G in zip(ep.transitions, returns):
                    advantage = G - self.baseline
                    self.policy.policy_gradient(
                        trans.feature_matrix, trans.chosen_order, advantage, lr=self.lr
                    )
                    lp = self.policy.log_prob(trans.feature_matrix, trans.chosen_order)
                    epoch_loss += -advantage * lp
                    total_updates += 1

            total_loss += epoch_loss

        self.n_updates += total_updates
        if hasattr(self, "_redis_client"):
            self.save_to_redis(self._redis_client)
        return {
            "n_epochs":      n_epochs,
            "n_updates":     total_updates,
            "avg_loss":      round(total_loss / max(total_updates, 1), 4),
            "final_baseline":round(self.baseline, 4),
            "W_norm":        round(float(np.linalg.norm(self.policy.W)), 4),
            "top_features":  self._top_features(),
        }

    def _top_features(self) -> list[dict]:
        NAMES = [
            "collaborative_score", "ranker_score", "genre_match",
            "popularity", "recency", "position", "user_avg_rating", "user_n_ratings"
        ]
        pairs = sorted(zip(NAMES, self.policy.W.tolist()), key=lambda x: -abs(x[1]))
        return [{"feature": n, "weight": round(w, 4)} for n, w in pairs]

    # ── Redis persistence ─────────────────────────────────────────────────

    REDIS_KEY = "rl:agent:state"

    def save_to_redis(self, redis_client) -> bool:
        """Persist policy weights to Redis after each update."""
        if redis_client is None:
            return False
        try:
            import json
            redis_client.set(self.REDIS_KEY, json.dumps(self.to_dict()))
            return True
        except Exception:
            return False

    def load_from_redis(self, redis_client) -> bool:
        """Load policy weights from Redis on startup."""
        if redis_client is None:
            return False
        try:
            import json
            raw = redis_client.get(self.REDIS_KEY)
            if not raw:
                return False
            d = json.loads(raw)
            self.policy       = LinearPolicy.from_dict(d["policy"])
            self.baseline     = float(d.get("baseline", 0.0))
            self.n_updates    = int(d.get("n_updates", 0))
            self.total_reward = float(d.get("total_reward", 0.0))
            self._history     = d.get("history", [])
            print(f"  [RL] Loaded policy from Redis: n_updates={self.n_updates} W_norm={float(self.policy.W.__abs__().max()):.4f}")
            return True
        except Exception as e:
            print(f"  [RL] Redis load failed: {e}")
            return False

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "policy":       self.policy.to_dict(),
            "baseline":     self.baseline,
            "n_updates":    self.n_updates,
            "total_reward": self.total_reward,
            "history":      self._history[-50:],   # last 50 updates
        }

    @classmethod
    def from_dict(cls, d: dict) -> "REINFORCEAgent":
        agent = cls()
        agent.policy       = LinearPolicy.from_dict(d["policy"])
        agent.baseline     = float(d.get("baseline", 0.0))
        agent.n_updates    = int(d.get("n_updates", 0))
        agent.total_reward = float(d.get("total_reward", 0.0))
        agent._history     = d.get("history", [])
        return agent

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "REINFORCEAgent":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_updates":       self.n_updates,
            "total_reward":    round(self.total_reward, 2),
            "baseline":        round(self.baseline, 4),
            "W_norm":          round(float(np.linalg.norm(self.policy.W)), 4),
            "top_features":    self._top_features(),
            "open_sessions":   len(self._episodes),
            "algorithm":       "REINFORCE_plackett_luce_advantage",
            "discount_gamma":  GAMMA,
            "lr":              self.lr,
            "n_features":      N_FEATURES,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

RL_AGENT = REINFORCEAgent()
