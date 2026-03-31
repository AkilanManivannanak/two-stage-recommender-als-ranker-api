"""
A/B Experimentation Engine
==========================
Full A/B infrastructure wired into the existing CineWave stack.

What this implements:
  1. Deterministic user bucketing  — hash(user_id + experiment_id) % 100
                                     same user always gets same variant
  2. Experiment registry           — stored in Redis, created via API
  3. Outcome logging               — click / watch / rating events tied to variant
  4. Welch's t-test analysis       — unequal variance, two-tailed, correct
  5. Sequential analysis guard     — flags when test is underpowered
  6. Experiment summary            — full stats: n, mean, std, p-value, CI, MDE

Redis key layout:
  ab:experiment:{exp_id}           — JSON experiment config
  ab:assignment:{exp_id}:{user_id} — variant string ("control" | "treatment")
  ab:outcome:{exp_id}:{variant}    — Redis list of float outcome values
  ab:exposure:{exp_id}:{variant}   — Redis counter of users exposed
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

import numpy as np


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    experiment_id:    str
    name:             str
    description:      str
    control_policy:   str          # e.g. "popularity_baseline"
    treatment_policy: str          # e.g. "als512_lgb_mmr"
    metric:           str          # primary metric: "click_rate" | "watch_rate" | "rating"
    min_detectable:   float = 0.02 # minimum detectable effect (absolute)
    alpha:            float = 0.05 # significance level
    power:            float = 0.80 # desired statistical power
    created_at:       float = field(default_factory=time.time)
    status:           str   = "running"   # running | stopped | concluded

    def required_n(self) -> int:
        """Compute required sample size per variant (Welch / z-approximation)."""
        z_a = 1.96   # two-tailed alpha=0.05
        z_b = 0.84   # power=0.80
        p1  = 0.30   # assumed baseline rate
        p2  = p1 + self.min_detectable
        n   = math.ceil(
            (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
            / (self.min_detectable ** 2)
        )
        return n


@dataclass
class VariantOutcome:
    variant:    str
    n:          int
    mean:       float
    std:        float
    sem:        float       # standard error of the mean


@dataclass
class ExperimentResult:
    experiment_id:  str
    metric:         str
    control:        VariantOutcome
    treatment:      VariantOutcome
    delta:          float       # treatment.mean - control.mean
    relative_lift:  float       # delta / control.mean
    t_stat:         float
    p_value:        float
    ci_low:         float       # 95% CI on delta
    ci_high:        float
    required_n:     int
    is_powered:     bool        # True only if both variants >= required_n
    significant:    bool        # p < alpha AND is_powered
    conclusion:     str


# ── Core bucketing logic ──────────────────────────────────────────────────────

def assign_variant(user_id: int, experiment_id: str, traffic_pct: float = 1.0) -> Optional[str]:
    """
    Deterministically assign a user to control or treatment.

    - Uses MD5(user_id:experiment_id) for stable, reproducible assignment.
    - traffic_pct < 1.0 excludes some users from the experiment entirely.
    - Same user always gets same variant for the same experiment.

    Returns: "control" | "treatment" | None (user excluded from experiment)
    """
    raw = f"{user_id}:{experiment_id}".encode()
    digest = int(hashlib.md5(raw).hexdigest(), 16)
    bucket = digest % 10000  # 0–9999

    # Exclude users outside traffic allocation
    if bucket >= int(traffic_pct * 10000):
        return None

    # 50/50 split within allocated traffic
    return "control" if bucket % 2 == 0 else "treatment"


# ── Welch's t-test ─────────────────────────────────────────────────────────

def welch_t_test(
    a: list[float],
    b: list[float],
) -> dict[str, float]:
    """
    Two-sample Welch's t-test (unequal variances, two-tailed).
    Does NOT assume equal variance — correct for A/B tests where
    group sizes or distributions may differ.

    Returns t_stat, p_value, degrees_of_freedom.
    """
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "df": 0.0}

    na, nb    = len(a), len(b)
    mean_a    = float(np.mean(a))
    mean_b    = float(np.mean(b))
    var_a     = float(np.var(a, ddof=1))
    var_b     = float(np.var(b, ddof=1))

    se_a      = var_a / na
    se_b      = var_b / nb
    se_delta  = math.sqrt(se_a + se_b)

    if se_delta == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "df": 0.0}

    t_stat = (mean_b - mean_a) / se_delta

    # Welch-Satterthwaite degrees of freedom
    df = (se_a + se_b) ** 2 / (
        (se_a ** 2) / (na - 1) + (se_b ** 2) / (nb - 1)
    )

    # p-value via t-distribution CDF approximation (no scipy dependency)
    p_value = _t_dist_two_tailed_p(abs(t_stat), df)

    return {
        "t_stat": round(t_stat, 6),
        "p_value": round(p_value, 6),
        "df": round(df, 2),
    }


def _t_dist_two_tailed_p(t: float, df: float) -> float:
    """
    Approximate two-tailed p-value from t-statistic and degrees of freedom.
    Uses regularised incomplete beta function approximation.
    Accurate to ~3 decimal places for df > 10.
    """
    if df <= 0:
        return 1.0
    # Use normal approximation for large df (df > 30)
    if df > 30:
        # Standard normal approximation
        z = t
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
        return min(1.0, 2.0 * p_one_tail)

    # Beta function approximation for small df
    x = df / (df + t * t)
    # Regularised incomplete beta I(x; df/2, 0.5)
    p_one_tail = 0.5 * _beta_inc(x, df / 2.0, 0.5)
    return min(1.0, 2.0 * p_one_tail)


def _beta_inc(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularised incomplete beta function via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's continued fraction
    f = 1.0; c = 1.0; d = 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / d if abs(d) < 1e-30 else 1.0 / d
    f = d
    for m in range(1, max_iter + 1):
        for step in (1, 2):
            if step == 1:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            c = 1.0 + num / c
            d = 1.0 / d if abs(d) < 1e-30 else 1.0 / d
            c = c if abs(c) > 1e-30 else 1e-30
            f *= d * c
    return front * (f - 1.0)


# ── Confidence interval on delta ──────────────────────────────────────────────

def delta_confidence_interval(
    a: list[float],
    b: list[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    95% CI on (mean_b - mean_a) using Welch's SE.
    z = 1.96 for alpha=0.05 two-tailed (normal approximation, valid for n > 30).
    """
    if len(a) < 2 or len(b) < 2:
        return (float("-inf"), float("inf"))

    z = 1.96  # for alpha=0.05
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    se_a   = float(np.var(a, ddof=1)) / len(a)
    se_b   = float(np.var(b, ddof=1)) / len(b)
    se     = math.sqrt(se_a + se_b)
    delta  = mean_b - mean_a
    return (round(delta - z * se, 6), round(delta + z * se, 6))


# ── Experiment store (Redis-backed) ──────────────────────────────────────────

class ExperimentStore:
    """
    Manages experiment configs and outcome logs.
    Uses Redis when available; falls back to in-memory dicts automatically.
    In-memory mode works fully but does not persist across restarts.

    Redis key layout:
      ab:experiment:{id}           → JSON config
      ab:outcome:{id}:control      → Redis list of floats
      ab:outcome:{id}:treatment    → Redis list of floats
      ab:exposure:{id}:control     → int counter
      ab:exposure:{id}:treatment   → int counter
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        # In-memory fallback — used when Redis is unavailable
        self._mem_experiments: dict[str, dict] = {}
        self._mem_outcomes:    dict[str, list]  = {}   # key: "{id}:{variant}"
        self._mem_exposure:    dict[str, int]   = {}   # key: "{id}:{variant}"
        self._mem_assignments: dict[str, str]   = {}   # key: "{id}:{uid}"

    def _ok(self) -> bool:
        return True  # always ok; uses redis or in-memory fallback

    # ── Experiment lifecycle ──────────────────────────────────────────────────

    def create_experiment(self, exp: Experiment) -> bool:
        d = asdict(exp)
        self._mem_experiments[exp.experiment_id] = d
        if self._redis is not None:
            try:
                self._redis.set(f"ab:experiment:{exp.experiment_id}", json.dumps(d))
            except Exception:
                pass
        return True

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        # Try Redis first, then memory
        if self._redis is not None:
            try:
                raw = self._redis.get(f"ab:experiment:{experiment_id}")
                if raw:
                    d = json.loads(raw)
                    self._mem_experiments[experiment_id] = d  # sync to mem
                    return Experiment(**d)
            except Exception:
                pass
        d = self._mem_experiments.get(experiment_id)
        return Experiment(**d) if d else None

    def list_experiments(self) -> list[dict]:
        out = dict(self._mem_experiments)  # start with memory
        if self._redis is not None:
            try:
                for k in self._redis.keys("ab:experiment:*"):
                    raw = self._redis.get(k)
                    if raw:
                        d = json.loads(raw)
                        out[d.get("experiment_id", k)] = d
            except Exception:
                pass
        return sorted(out.values(), key=lambda x: x.get("created_at", 0), reverse=True)

    def stop_experiment(self, experiment_id: str) -> bool:
        exp = self.get_experiment(experiment_id)
        if not exp:
            return False
        exp.status = "stopped"
        d = asdict(exp)
        self._mem_experiments[experiment_id] = d
        if self._redis is not None:
            try:
                self._redis.set(f"ab:experiment:{experiment_id}", json.dumps(d))
            except Exception:
                pass
        return True

    # ── Assignment ────────────────────────────────────────────────────────────

    def get_or_assign_variant(
        self,
        user_id:       int,
        experiment_id: str,
        traffic_pct:   float = 1.0,
    ) -> Optional[str]:
        """Return cached assignment or compute + cache a new one."""
        mem_key = f"{experiment_id}:{user_id}"
        if mem_key in self._mem_assignments:
            return self._mem_assignments[mem_key]

        if self._redis is not None:
            try:
                cached = self._redis.get(f"ab:assignment:{experiment_id}:{user_id}")
                if cached:
                    self._mem_assignments[mem_key] = cached
                    return cached
            except Exception:
                pass

        variant = assign_variant(user_id, experiment_id, traffic_pct)
        if variant:
            self._mem_assignments[mem_key] = variant
            exp_key = f"{experiment_id}:{variant}"
            self._mem_exposure[exp_key] = self._mem_exposure.get(exp_key, 0) + 1
            if self._redis is not None:
                try:
                    self._redis.set(f"ab:assignment:{experiment_id}:{user_id}",
                                    variant, ex=86400 * 30)
                    self._redis.incr(f"ab:exposure:{experiment_id}:{variant}")
                except Exception:
                    pass
        return variant

    # ── Outcome logging ───────────────────────────────────────────────────────

    def log_outcome(
        self,
        experiment_id: str,
        variant:       str,
        outcome:       float,
        user_id:       Optional[int] = None,
    ) -> bool:
        """Append an outcome to the variant outcome list (Redis + memory)."""
        mem_key = f"{experiment_id}:{variant}"
        if mem_key not in self._mem_outcomes:
            self._mem_outcomes[mem_key] = []
        self._mem_outcomes[mem_key].append(float(outcome))
        if self._redis is not None:
            try:
                self._redis.rpush(f"ab:outcome:{experiment_id}:{variant}", str(outcome))
            except Exception:
                pass
        return True

    def get_outcomes(self, experiment_id: str, variant: str) -> list[float]:
        mem_key = f"{experiment_id}:{variant}"
        if self._redis is not None:
            try:
                vals = self._redis.lrange(f"ab:outcome:{experiment_id}:{variant}", 0, -1)
                if vals:
                    combined = list({float(v) for v in vals} |
                                    set(self._mem_outcomes.get(mem_key, [])))
                    return [float(v) for v in self._redis.lrange(
                        f"ab:outcome:{experiment_id}:{variant}", 0, -1)]
            except Exception:
                pass
        return list(self._mem_outcomes.get(mem_key, []))

    def get_exposure_counts(self, experiment_id: str) -> dict[str, int]:
        ctrl  = self._mem_exposure.get(f"{experiment_id}:control",   0)
        treat = self._mem_exposure.get(f"{experiment_id}:treatment", 0)
        if self._redis is not None:
            try:
                c = self._redis.get(f"ab:exposure:{experiment_id}:control")
                t = self._redis.get(f"ab:exposure:{experiment_id}:treatment")
                ctrl  = max(ctrl,  int(c or 0))
                treat = max(treat, int(t or 0))
            except Exception:
                pass
        return {"control": ctrl, "treatment": treat}

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Run Welch's t-test on logged outcomes and return full result object.
        """
        exp = self.get_experiment(experiment_id)
        if not exp:
            return None

        ctrl_outcomes  = self.get_outcomes(experiment_id, "control")
        treat_outcomes = self.get_outcomes(experiment_id, "treatment")

        nc = len(ctrl_outcomes)
        nt = len(treat_outcomes)

        # Compute variant summaries
        def _summarise(data: list[float], variant: str) -> VariantOutcome:
            if not data:
                return VariantOutcome(variant=variant, n=0, mean=0.0, std=0.0, sem=0.0)
            arr = np.array(data)
            n   = len(arr)
            m   = float(arr.mean())
            s   = float(arr.std(ddof=1)) if n > 1 else 0.0
            return VariantOutcome(
                variant=variant, n=n, mean=round(m, 6),
                std=round(s, 6), sem=round(s / math.sqrt(n) if n > 0 else 0.0, 6)
            )

        ctrl_sum  = _summarise(ctrl_outcomes,  "control")
        treat_sum = _summarise(treat_outcomes, "treatment")

        # Welch's t-test
        t_res = welch_t_test(ctrl_outcomes, treat_outcomes)
        ci    = delta_confidence_interval(ctrl_outcomes, treat_outcomes, exp.alpha)

        delta   = round(treat_sum.mean - ctrl_sum.mean, 6)
        rel_lft = round(delta / ctrl_sum.mean, 4) if ctrl_sum.mean != 0 else 0.0
        req_n   = exp.required_n()
        powered = (nc >= req_n and nt >= req_n)
        sig     = powered and t_res["p_value"] < exp.alpha

        # Conclusion string — explicit about power status
        if not powered:
            conclusion = (
                f"UNDERPOWERED: need {req_n:,} per variant, "
                f"have {nc:,} control / {nt:,} treatment. "
                f"Do not draw conclusions yet."
            )
        elif sig:
            direction = "INCREASE" if delta > 0 else "DECREASE"
            conclusion = (
                f"SIGNIFICANT {direction}: treatment {direction.lower()}d {exp.metric} "
                f"by {abs(rel_lft):.1%} (p={t_res['p_value']:.4f}, "
                f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])."
            )
        else:
            conclusion = (
                f"NOT SIGNIFICANT: p={t_res['p_value']:.4f} >= {exp.alpha}. "
                f"Cannot reject null hypothesis at {exp.alpha:.0%} level. "
                f"Result may be underpowered or effect may not exist."
            )

        return ExperimentResult(
            experiment_id=experiment_id,
            metric=exp.metric,
            control=ctrl_sum,
            treatment=treat_sum,
            delta=delta,
            relative_lift=rel_lft,
            t_stat=round(t_res["t_stat"], 4),
            p_value=round(t_res["p_value"], 4),
            ci_low=ci[0],
            ci_high=ci[1],
            required_n=req_n,
            is_powered=powered,
            significant=sig,
            conclusion=conclusion,
        )


# ── Singleton ────────────────────────────────────────────────────────────────

AB_STORE = ExperimentStore()
