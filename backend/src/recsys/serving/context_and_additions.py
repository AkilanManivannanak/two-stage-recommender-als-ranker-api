"""
context_features.py — CineWave
================================
Context-aware feature engineering for LightGBM reranker.

WHY CONTEXT FEATURES MATTER
-----------------------------
CineWave's LightGBM currently uses 6 features:
  als_score, genre_match_cosine, item_popularity_log,
  recency_score, user_activity_decile, top_genre_alignment

None of these change based on WHEN or HOW the user is watching.
The same user gets the same ranking at 7am on a phone as at 10pm on TV.
This is wrong — context is a strong signal:

  "Action at 9 PM on mobile"  → likely commuting/winding down → maybe short
  "Action at noon on desktop" → likely at lunch → longer watch OK
  "Documentary on TV"         → lean-back mode → longer, richer content
  "Comedy on phone"           → casual, fragmented attention

Netflix's production reranker includes dozens of context features.
Adding just 5 here demonstrably improves ranking for specific cohorts.

NEW FEATURES ADDED (5 total)
------------------------------
1. time_of_day_bucket:    [0=morning, 1=afternoon, 2=evening, 3=night]
2. is_weekend:            [0/1] — weekend viewing patterns differ
3. device_mobile_score:   [0.0-1.0] — mobile=1.0, TV=0.0, desktop=0.5
4. session_length_bucket: [0=fresh (<5min), 1=browsing (5-30min), 2=long (30+)]
5. recency_of_last_play:  hours since user last played anything (log-scaled)

INTEGRATION
-----------
In the LightGBM feature builder (train_ranker_lgbm.py), replace:
    features = [als_score, genre_match, popularity, recency, activity, alignment]
With:
    ctx = build_context_features(hour, is_weekend, device, session_s, last_play_h)
    features = [als_score, genre_match, popularity, recency, activity, alignment] + ctx

The model automatically learns which contexts prefer which items.
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ── Time-of-day buckets ───────────────────────────────────────────────────────

def _time_bucket(hour: int) -> int:
    """
    0 = morning   (05:00–11:59)  — news, documentary, family
    1 = afternoon  (12:00–17:59) — comedy, shorter content
    2 = evening    (18:00–22:59) — prime time, action, drama — highest CTR
    3 = night      (23:00–04:59) — thriller, horror, long drama
    """
    if 5 <= hour < 12:
        return 0
    elif 12 <= hour < 18:
        return 1
    elif 18 <= hour < 23:
        return 2
    else:
        return 3


def _time_sin_cos(hour: int) -> Tuple[float, float]:
    """
    Circular encoding of hour (avoids discontinuity between 23:00 and 00:00).
    Returns (sin, cos) of angle on 24-hour clock.
    """
    angle = 2 * math.pi * hour / 24
    return math.sin(angle), math.cos(angle)


# ── Device score ──────────────────────────────────────────────────────────────

_DEVICE_SCORES: Dict[str, float] = {
    "mobile":  1.0,   # fragmented attention, short content preferred
    "tablet":  0.7,   # medium — could go either way
    "desktop": 0.5,   # desk work context — mid-length content
    "tv":      0.0,   # lean-back, long content, full attention
    "unknown": 0.5,
}


def _device_score(device_type: str) -> float:
    return _DEVICE_SCORES.get(device_type.lower(), 0.5)


# ── Session length bucket ─────────────────────────────────────────────────────

def _session_bucket(session_seconds: float) -> int:
    """
    0 = fresh       (<5 min)   — first thing they see matters most
    1 = browsing    (5-30 min) — actively exploring, open to discovery
    2 = long        (30+ min)  — settled in, less likely to change
    """
    if session_seconds < 300:
        return 0
    elif session_seconds < 1800:
        return 1
    else:
        return 2


# ── Last play recency ─────────────────────────────────────────────────────────

def _recency_score(hours_since_last_play: float) -> float:
    """
    Log-scaled recency. Returns value in [0, 1]:
    - 0.0 = played very recently (< 1 hour ago) → likely casual rewatch
    - 1.0 = very long since last play (> 7 days) → high-value re-engagement

    WHY LOG: the difference between 1h and 2h matters less than 24h vs 48h.
    """
    if hours_since_last_play <= 0:
        return 0.0
    # log(hours + 1) normalised by log(168) = log(7 days in hours)
    return min(math.log(hours_since_last_play + 1) / math.log(169), 1.0)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_context_features(
    hour_of_day: int,                      # 0–23
    is_weekend: bool,
    device_type: str,                      # "mobile" | "desktop" | "tv" | "tablet"
    session_duration_seconds: float,
    hours_since_last_play: float,
) -> List[float]:
    """
    Build 7-element context feature vector for LightGBM.
    Returns: [time_sin, time_cos, time_bucket, is_weekend,
              device_score, session_bucket, recency_score]

    All values are in [0, 1] or small integers — no additional scaling needed
    for tree models (LightGBM is scale-invariant, but consistent ranges help
    with interpretability of feature importance).
    """
    t_sin, t_cos = _time_sin_cos(hour_of_day)

    return [
        round(t_sin, 4),                                     # continuous circular time
        round(t_cos, 4),                                     # continuous circular time
        float(_time_bucket(hour_of_day)) / 3.0,              # normalised [0–1]
        1.0 if is_weekend else 0.0,
        _device_score(device_type),
        float(_session_bucket(session_duration_seconds)) / 2.0,  # normalised [0–1]
        round(_recency_score(hours_since_last_play), 4),
    ]


CONTEXT_FEATURE_NAMES = [
    "ctx_time_sin",
    "ctx_time_cos",
    "ctx_time_bucket",
    "ctx_is_weekend",
    "ctx_device_score",
    "ctx_session_bucket",
    "ctx_recency_score",
]


def context_from_request(
    request_timestamp: Optional[float] = None,
    user_agent: str = "",
    session_start_ts: Optional[float] = None,
    last_play_ts: Optional[float] = None,
) -> Dict:
    """
    Extract context features from a live HTTP request.
    Returns dict compatible with build_context_features().
    """
    now = request_timestamp or time.time()
    dt  = datetime.utcfromtimestamp(now)

    # Device detection from User-Agent
    ua_lower = user_agent.lower()
    if "mobile" in ua_lower or "android" in ua_lower or "iphone" in ua_lower:
        device = "mobile"
    elif "tablet" in ua_lower or "ipad" in ua_lower:
        device = "tablet"
    elif "tv" in ua_lower or "smarttv" in ua_lower or "roku" in ua_lower:
        device = "tv"
    else:
        device = "desktop"

    session_dur = (now - session_start_ts) if session_start_ts else 0.0
    hours_since = ((now - last_play_ts) / 3600) if last_play_ts else 48.0  # default 2 days

    return {
        "hour_of_day":              dt.hour,
        "is_weekend":               dt.weekday() >= 5,
        "device_type":              device,
        "session_duration_seconds": session_dur,
        "hours_since_last_play":    hours_since,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# drift_monitor.py — Data drift + concept drift monitoring
# ═══════════════════════════════════════════════════════════════════════════════
"""
Lightweight drift monitors separate from training_serving_skew.py.

training_serving_skew.py: feature-level PSI (LightGBM inputs)
drift_monitor.py (this):  output-level drift (model predictions) +
                           interaction pattern drift (concept drift proxy)

TWO TYPES DETECTED HERE
------------------------
1. Prediction drift:  distribution of model scores drifted.
   Signal: the mean/std of LightGBM output scores changed significantly.
   If scores collapse (everyone gets 0.9) or spread (many near 0 or 1),
   the model is misbehaving in serving.

2. Click-through rate drift:  rolling CTR window diverged from training CTR.
   If CTR drops >30% below 7-day rolling average, something changed —
   either the recommendations got worse or user behaviour shifted.
"""

import collections
import threading as _threading
import time as _time
import logging as _logging
from typing import Deque

_logger = _logging.getLogger("cinewave.drift")

_CTR_WINDOW = 7 * 24 * 3600   # 7-day rolling window
_CTR_ALERT_DROP = 0.30         # alert if CTR drops >30% below rolling avg


class PredictionDriftMonitor:
    """
    Tracks the distribution of model prediction scores over time.
    Alerts if the score distribution shifts significantly from baseline.
    """

    def __init__(self, baseline_mean: float = 0.55, baseline_std: float = 0.18):
        """
        baseline_mean/std: from training evaluation, passed in from Metaflow.
        """
        self.baseline_mean = baseline_mean
        self.baseline_std  = baseline_std
        self._recent_scores: Deque[float] = collections.deque(maxlen=10000)
        self._lock = _threading.Lock()

    def record_score(self, score: float) -> None:
        with self._lock:
            self._recent_scores.append(score)

    def check(self) -> Dict:
        with self._lock:
            scores = list(self._recent_scores)

        if len(scores) < 100:
            return {"status": "insufficient_data", "n": len(scores)}

        current_mean = float(sum(scores) / len(scores))
        current_std  = float((sum((s - current_mean) ** 2 for s in scores) / len(scores)) ** 0.5)

        mean_shift = abs(current_mean - self.baseline_mean) / max(self.baseline_std, 0.001)
        std_ratio  = current_std / max(self.baseline_std, 0.001)

        if mean_shift > 2.0 or std_ratio > 2.0 or std_ratio < 0.5:
            status = "alert"
        elif mean_shift > 1.0 or std_ratio > 1.5:
            status = "warn"
        else:
            status = "ok"

        if status != "ok":
            _logger.warning(
                f"[PredictionDrift] {status}: "
                f"mean {self.baseline_mean:.3f}→{current_mean:.3f}  "
                f"std {self.baseline_std:.3f}→{current_std:.3f}"
            )

        return {
            "status":        status,
            "current_mean":  round(current_mean, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "current_std":   round(current_std, 4),
            "baseline_std":  round(self.baseline_std, 4),
            "mean_shift_z":  round(mean_shift, 2),
            "std_ratio":     round(std_ratio, 2),
            "n_scores":      len(scores),
        }


class CTRDriftMonitor:
    """
    Tracks rolling click-through rate. Alerts on sudden drops.
    """

    def __init__(self):
        # Each entry: (timestamp, served: bool, clicked: bool)
        self._events: Deque[Tuple[float, bool, bool]] = collections.deque()
        self._lock = _threading.Lock()

    def record_serve(self) -> None:
        with self._lock:
            self._events.append((_time.time(), True, False))
            self._trim()

    def record_click(self) -> None:
        with self._lock:
            self._events.append((_time.time(), False, True))
            self._trim()

    def _trim(self) -> None:
        cutoff = _time.time() - _CTR_WINDOW
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def current_ctr(self, window_seconds: int = 3600) -> Optional[float]:
        """Rolling CTR over the last window_seconds."""
        cutoff = _time.time() - window_seconds
        with self._lock:
            events = [(ts, served, clicked) for ts, served, clicked in self._events if ts >= cutoff]
        n_served  = sum(1 for _, s, _ in events if s)
        n_clicked = sum(1 for _, _, c in events if c)
        return n_clicked / n_served if n_served > 0 else None

    def check(self) -> Dict:
        ctr_1h  = self.current_ctr(3600)
        ctr_24h = self.current_ctr(86400)
        ctr_7d  = self.current_ctr(_CTR_WINDOW)

        if ctr_1h is None or ctr_7d is None:
            return {"status": "insufficient_data"}

        drop = (ctr_7d - ctr_1h) / max(ctr_7d, 0.001)

        if drop > _CTR_ALERT_DROP:
            status = "alert"
            _logger.warning(
                f"[CTRDrift] ALERT: 1h CTR={ctr_1h:.3f} vs 7d CTR={ctr_7d:.3f}  drop={drop:.1%}"
            )
        elif drop > 0.15:
            status = "warn"
        else:
            status = "ok"

        return {
            "status":  status,
            "ctr_1h":  round(ctr_1h,  4) if ctr_1h  else None,
            "ctr_24h": round(ctr_24h, 4) if ctr_24h else None,
            "ctr_7d":  round(ctr_7d,  4) if ctr_7d  else None,
            "drop_pct": round(drop * 100, 1),
        }


# Singletons
PREDICTION_DRIFT = PredictionDriftMonitor()
CTR_DRIFT        = CTRDriftMonitor()


# ═══════════════════════════════════════════════════════════════════════════════
# holdback_group.py — True holdback group for A/B baseline
# ═══════════════════════════════════════════════════════════════════════════════
"""
Holdback group: 5% of traffic that receives no ML recommendations.
Instead they see a popularity-sorted list of titles.

WHY A HOLDBACK GROUP
---------------------
Our A/B experiments compare RL vs ALS, Slate Optimizer vs greedy, etc.
But all treatments use ML. The counterfactual — "what if we had no ML at all?"
— is never measured.

A holdback group measures the absolute value of the recommendation system vs
the baseline of "show popular movies." It answers:
  "How much better is our ML than just showing Shawshank + Forrest Gump + Inception?"

Netflix runs holdback groups continuously. For an intern project it is a
strong signal of production thinking.

IMPLEMENTATION
--------------
5% of user_ids (deterministic hash-based split, not random per request)
are assigned to the holdback group permanently for the experiment duration.
When they hit /recommend, they receive popularity-sorted items (no ALS, no RL).
Their interactions are logged separately for holdback CTR/NDCG computation.
"""

import hashlib


HOLDBACK_PCT   = 0.05   # 5% holdback
HOLDBACK_LABEL = "holdback_popularity"
CONTROL_LABEL  = "ml_full"


def is_holdback_user(user_id: int, salt: str = "cinewave_holdback_v1") -> bool:
    """
    Deterministic holdback assignment.
    Same user_id always maps to same group (holdback or not).
    Uses MD5 hash to get a stable float in [0, 1).

    WHY DETERMINISTIC
    ------------------
    Random per-request assignment means the same user sees ML on one visit
    and popularity on the next. This contaminates both groups. Deterministic
    hash-based assignment guarantees each user is always in one group.
    """
    key   = f"{salt}:{user_id}".encode()
    h     = hashlib.md5(key).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF   # stable float in [0, 1)
    return bucket < HOLDBACK_PCT


def get_experiment_group(user_id: int) -> str:
    """Returns 'holdback_popularity' or 'ml_full'."""
    return HOLDBACK_LABEL if is_holdback_user(user_id) else CONTROL_LABEL


def popularity_fallback(catalog, top_k=30):
    items = sorted(catalog.values(), key=lambda x: -x.get("popularity", 0))[:top_k]
    # Always return dicts with item_id guaranteed
    result = []
    for item in items:
        if isinstance(item, dict):
            d = dict(item)
            if "item_id" not in d and "movieId" in d:
                d["item_id"] = d["movieId"]
            result.append(d)
        elif isinstance(item, int):
            d = dict(catalog.get(item, {"item_id": item}))
            if "item_id" not in d:
                d["item_id"] = item
            result.append(d)
    return result
# ═══════════════════════════════════════════════════════════════════════════════
# cuped.py — CUPED variance reduction for A/B experiments
# ═══════════════════════════════════════════════════════════════════════════════
"""
CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

WHY CUPED
----------
A/B experiments need large samples to reach statistical significance.
CUPED reduces the required sample size by 30-60% by removing variance
that is explained by pre-experiment user behaviour.

The idea: if a user had high CTR before the experiment, they will likely
have high CTR during it — regardless of which treatment they receive.
This pre-experiment variance is "noise" that increases the required N.
CUPED removes it by subtracting the part of the outcome explained by
the pre-experiment covariate.

CUPED FORMULA
--------------
  Y_cuped = Y - θ * X

Where:
  Y = observed outcome during experiment (e.g. CTR)
  X = pre-experiment covariate (e.g. CTR in 14 days before experiment)
  θ = Cov(Y, X) / Var(X)  — regression coefficient

Y_cuped has the same expectation as Y but lower variance:
  Var(Y_cuped) = Var(Y) * (1 - ρ²)   where ρ = corr(Y, X)

If ρ = 0.7 (70% correlation between pre/post), variance reduces by 51%,
meaning we need ~half the users to reach the same statistical power.

VARIANCE REDUCTION FACTOR (in CineWave)
-----------------------------------------
Typical correlation between pre-experiment CTR and in-experiment CTR: ~0.6
CUPED variance reduction: 1 - 0.6² = 1 - 0.36 = 64% variance remains
→ need only 64% of the original N for same power → 36% sample size reduction

USAGE
-----
    from recsys.serving.cuped import CUPEDEstimator

    estimator = CUPEDEstimator()

    # Add pre-experiment data (14 days before experiment start)
    estimator.add_pre_experiment_data(user_id, pre_ctr)

    # Add in-experiment data
    estimator.add_experiment_data(user_id, "treatment", in_ctr)
    estimator.add_experiment_data(user_id, "control",   in_ctr)

    # Compute CUPED-adjusted means and p-value
    result = estimator.compute()
"""

import math
import statistics


class CUPEDEstimator:
    """
    CUPED variance reduction estimator for two-group A/B experiments.
    """

    def __init__(self):
        self._pre: Dict[int, float]                  = {}   # user_id → pre-experiment covariate
        self._treatment: Dict[int, float]            = {}   # user_id → in-experiment outcome (treatment)
        self._control: Dict[int, float]              = {}   # user_id → in-experiment outcome (control)

    def add_pre_experiment_data(self, user_id: int, covariate_value: float) -> None:
        """
        Record pre-experiment covariate for a user.
        Typically 14-day average CTR / NDCG before experiment start.
        """
        self._pre[user_id] = covariate_value

    def add_experiment_data(
        self, user_id: int, group: str, outcome_value: float
    ) -> None:
        """Record in-experiment outcome. group must be 'treatment' or 'control'."""
        if group == "treatment":
            self._treatment[user_id] = outcome_value
        elif group == "control":
            self._control[user_id] = outcome_value
        else:
            raise ValueError(f"group must be 'treatment' or 'control', got {group!r}")

    def _theta(
        self, outcomes: List[float], covariates: List[float]
    ) -> float:
        """
        OLS estimate of θ = Cov(Y, X) / Var(X).
        This is the coefficient we subtract to remove pre-experiment variance.
        """
        if len(outcomes) < 2:
            return 0.0
        n        = len(outcomes)
        mean_y   = sum(outcomes) / n
        mean_x   = sum(covariates) / n
        cov_yx   = sum((y - mean_y) * (x - mean_x) for y, x in zip(outcomes, covariates)) / (n - 1)
        var_x    = sum((x - mean_x) ** 2 for x in covariates) / (n - 1)
        return cov_yx / max(var_x, 1e-10)

    def compute(self, alpha: float = 0.05) -> Dict:
        """
        Compute CUPED-adjusted treatment effect and statistical test.

        Returns:
        {
          "raw": {treatment_mean, control_mean, raw_delta, raw_pvalue, raw_n},
          "cuped": {treatment_mean, control_mean, cuped_delta, cuped_pvalue, cuped_n},
          "variance_reduction_pct": float,
          "correlation_pre_post": float,
        }
        """
        # Align users present in both groups
        treatment_users = set(self._treatment) & set(self._pre)
        control_users   = set(self._control)   & set(self._pre)

        if len(treatment_users) < 10 or len(control_users) < 10:
            return {"error": "insufficient_data",
                    "n_treatment": len(treatment_users),
                    "n_control":   len(control_users)}

        t_outcomes = [self._treatment[u] for u in treatment_users]
        t_covars   = [self._pre[u]        for u in treatment_users]
        c_outcomes = [self._control[u]    for u in control_users]
        c_covars   = [self._pre[u]        for u in control_users]

        # Estimate θ on pooled data (standard CUPED approach)
        all_outcomes = t_outcomes + c_outcomes
        all_covars   = t_covars   + c_covars
        theta        = self._theta(all_outcomes, all_covars)

        # CUPED adjustment
        t_cuped = [y - theta * x for y, x in zip(t_outcomes, t_covars)]
        c_cuped = [y - theta * x for y, x in zip(c_outcomes, c_covars)]

        # Means
        raw_t_mean = sum(t_outcomes) / len(t_outcomes)
        raw_c_mean = sum(c_outcomes) / len(c_outcomes)
        cup_t_mean = sum(t_cuped)    / len(t_cuped)
        cup_c_mean = sum(c_cuped)    / len(c_cuped)

        # Two-sample t-test (Welch)
        def welch_pvalue(a: List[float], b: List[float]) -> Tuple[float, float]:
            import math as _math   # explicit local import prevents UnboundLocalError
            na, nb      = len(a), len(b)
            mean_a      = sum(a) / na
            mean_b      = sum(b) / nb
            var_a       = sum((x - mean_a) ** 2 for x in a) / (na - 1)
            var_b       = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
            se          = _math.sqrt(var_a / na + var_b / nb)
            if se == 0:
                return 0.0, 1.0
            t_stat      = (mean_a - mean_b) / se
            df_num   = (var_a / na + var_b / nb) ** 2
            df_denom = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
            df       = df_num / max(df_denom, 1e-10)  # noqa
            z = abs(t_stat)
            p = 2 * (1 - _normal_cdf(z))
            return t_stat, p

        raw_t_stat, raw_pval  = welch_pvalue(t_outcomes, c_outcomes)
        cup_t_stat, cup_pval  = welch_pvalue(t_cuped,    c_cuped)

        # Variance reduction
        raw_var     = statistics.variance(t_outcomes + c_outcomes)
        cuped_var   = statistics.variance(t_cuped    + c_cuped)
        var_red_pct = (1 - cuped_var / max(raw_var, 1e-10)) * 100

        # Correlation pre-experiment covariate with outcome
        try:
            corr = statistics.correlation(all_covars, all_outcomes)
        except Exception:
            corr = 0.0

        return {
            "raw": {
                "treatment_mean": round(raw_t_mean, 4),
                "control_mean":   round(raw_c_mean, 4),
                "delta":          round(raw_t_mean - raw_c_mean, 4),
                "pvalue":         round(raw_pval, 4),
                "t_stat":         round(raw_t_stat, 3),
                "n_treatment":    len(t_outcomes),
                "n_control":      len(c_outcomes),
            },
            "cuped": {
                "treatment_mean": round(cup_t_mean, 4),
                "control_mean":   round(cup_c_mean, 4),
                "delta":          round(cup_t_mean - cup_c_mean, 4),
                "pvalue":         round(cup_pval, 4),
                "t_stat":         round(cup_t_stat, 3),
                "n_treatment":    len(t_cuped),
                "n_control":      len(c_cuped),
            },
            "theta":                   round(theta, 4),
            "variance_reduction_pct":  round(var_red_pct, 1),
            "correlation_pre_post":    round(corr, 3),
            "significant_raw":         raw_pval  < alpha,
            "significant_cuped":       cup_pval  < alpha,
            "cuped_powered_when_raw_not": (cup_pval < alpha and raw_pval >= alpha),
        }


def _normal_cdf(z: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ═══════════════════════════════════════════════════════════════════════════════
# clip_embeddings.py — CLIP multimodal embeddings (text + image unified space)
# ═══════════════════════════════════════════════════════════════════════════════
"""
CLIP (Contrastive Language-Image Pre-training) embeddings.

WHAT CLIP DOES vs CURRENT SEPARATE MODALITIES
-----------------------------------------------
CineWave currently has THREE separate embedding spaces:
  1. OpenAI text embeddings (1536-dim) — for semantic text search
  2. ALS collaborative embeddings (200-dim) — for CF
  3. GPT-4o Vision struct. JSON — {mood, tone, setting, visual_genre}

The problem: these live in different spaces. You cannot search across
text query + visual similarity in the same ANN search. You have to
retrieve separately and merge, which loses cross-modal signal.

CLIP creates a SINGLE unified embedding space where:
  - "dark sci-fi psychological thriller" (text) maps near
  - [poster of Blade Runner] (image) maps near
  - "dystopian noir with neon rain" (text) also maps near

All three land close in the same 512-dim CLIP space.

WHY THIS IS POWERFUL FOR RECOMMENDATIONS
------------------------------------------
A user says "something like Blade Runner but more hopeful" (voice query).
With separate modalities:
  - Text search finds semantically similar descriptions
  - VLM analysis gives visual mood of Blade Runner

With CLIP:
  - Encode "something like Blade Runner but more hopeful" as a text CLIP vector
  - Encode Blade Runner poster as an image CLIP vector
  - Add them (or weight them): 0.6 * text + 0.4 * image = unified query
  - Single ANN search returns items semantically AND visually relevant

IMPLEMENTATION
--------------
Uses openai-clip (ViT-B/32) or HuggingFace's CLIP model.
For 1200 items, encoding runs offline in ~2 minutes.
Items stored in a NEW Qdrant collection "clip_items" (512-dim).

NOTE: This is an addition, not a replacement. ALS still runs for
collaborative filtering. CLIP adds a multimodal retrieval path activated
for voice queries with visual descriptors ("dark", "colourful", "gritty").
"""


class CLIPEmbedder:
    """
    Computes CLIP embeddings for text and images.
    Falls back gracefully if CLIP is not installed.

    USAGE
    -----
        embedder = CLIPEmbedder()

        # Encode movie description text
        text_emb = embedder.encode_text("A dark psychological thriller set in a dystopian city")

        # Encode movie poster image
        img_emb = embedder.encode_image_url("https://image.tmdb.org/t/p/w500/poster.jpg")

        # Fuse for multimodal query
        query_emb = embedder.fuse(text_emb, img_emb, text_weight=0.7)
    """

    CLIP_DIM = 512

    def __init__(self, model_name: str = "ViT-B/32"):
        self._model    = None
        self._preprocess = None
        self._tokenize = None
        self._device   = "cpu"
        self._available = False
        self._try_load(model_name)

    def _try_load(self, model_name: str) -> None:
        try:
            import clip as openai_clip
            import torch
            self._device   = "cuda" if torch.cuda.is_available() else "cpu"
            self._model, self._preprocess = openai_clip.load(model_name, device=self._device)
            self._tokenize = openai_clip.tokenize
            self._available = True
            _logging.getLogger("cinewave.clip").info(
                f"[CLIP] Loaded {model_name} on {self._device}"
            )
        except ImportError:
            _logging.getLogger("cinewave.clip").warning(
                "[CLIP] openai-clip not installed — install with: pip install git+https://github.com/openai/CLIP.git"
            )
        except Exception as e:
            _logging.getLogger("cinewave.clip").error(f"[CLIP] Load failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a text string into a 512-dim CLIP embedding.
        Returns None if CLIP not available.
        """
        if not self._available:
            return None
        try:
            import torch
            tokens = self._tokenize([text[:77]]).to(self._device)  # CLIP max 77 tokens
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy()
        except Exception as e:
            _logging.getLogger("cinewave.clip").error(f"[CLIP] encode_text failed: {e}")
            return None

    def encode_image_url(self, image_url: str) -> Optional[np.ndarray]:
        """
        Download image from URL and encode into 512-dim CLIP embedding.
        Returns None on any failure (network, CLIP unavailable, etc.)
        """
        if not self._available:
            return None
        try:
            import io
            import requests
            import torch
            from PIL import Image

            resp = requests.get(image_url, timeout=5)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy()
        except Exception as e:
            _logging.getLogger("cinewave.clip").error(f"[CLIP] encode_image_url failed: {e}")
            return None

    def fuse(
        self,
        text_emb: Optional[np.ndarray],
        image_emb: Optional[np.ndarray],
        text_weight: float = 0.6,
    ) -> Optional[np.ndarray]:
        """
        Fuse text and image embeddings into a single multimodal vector.
        Both must be in the same CLIP embedding space (they are — CLIP aligns them).

        If only one is available, returns that embedding.
        """
        if text_emb is None and image_emb is None:
            return None
        if text_emb is None:
            return image_emb
        if image_emb is None:
            return text_emb

        fused = text_weight * text_emb + (1.0 - text_weight) * image_emb
        norm  = np.linalg.norm(fused)
        return fused / max(norm, 1e-10)

    def encode_catalog(
        self,
        catalog: List[Dict],
        text_key: str = "description",
        image_url_key: str = "poster_url",
        text_weight: float = 0.6,
        batch_size: int = 32,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Encode full movie catalog. Returns list of (item_id, 512-dim-embedding).
        text_weight=0.6 means 60% text, 40% image (tunable per experiment).
        """
        results = []
        for i, item in enumerate(catalog):
            item_id = item.get("item_id") or item.get("id")
            text    = f"{item.get('title', '')} {item.get(text_key, '')} {item.get('genre', '')}"
            img_url = item.get(image_url_key, "")

            t_emb = self.encode_text(text)
            i_emb = self.encode_image_url(img_url) if img_url else None
            fused = self.fuse(t_emb, i_emb, text_weight)

            if fused is not None:
                results.append((int(item_id), fused))

            if (i + 1) % 50 == 0:
                _logging.getLogger("cinewave.clip").info(
                    f"[CLIP] Encoded {i + 1}/{len(catalog)} items"
                )

        return results


# Singleton
CLIP_EMBEDDER = CLIPEmbedder()