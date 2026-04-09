"""
training_serving_skew.py — CineWave
=====================================
Detects divergence between the feature distribution seen at training time
vs the feature distribution seen at serving time.

WHY THIS MATTERS
-----------------
The LightGBM reranker is trained on a snapshot of 6 features:
  1. als_score
  2. genre_match_cosine
  3. item_popularity_log
  4. recency_score
  5. user_activity_decile
  6. top_genre_alignment

If any of these features drift at serving time — because user behaviour
changed, the catalog changed, or the way Redis computes them changed —
the model is operating outside its training distribution. NDCG degrades
silently. Without this monitor, the DuckDB eval gate catches the *outcome*
(NDCG drop) but not the *cause* (which feature drifted and by how much).

WHAT WE DETECT
---------------
1. Data drift:     the marginal distribution of a feature changed.
                   Measured with Population Stability Index (PSI).

2. Concept drift:  the relationship between features and the target changed.
                   Proxy: rating velocity (new interactions / hour). If users
                   are rating differently, the model's labels have shifted.

POPULATION STABILITY INDEX (PSI)
---------------------------------
PSI compares two distributions by binning them and computing:
  PSI = Σ (actual% - expected%) × ln(actual% / expected%)

Standard interpretation:
  PSI < 0.10:  No significant change — model stable
  PSI < 0.20:  Moderate change — monitor closely
  PSI >= 0.20: Significant change — retrain recommended

We use 10 equal-frequency bins computed from the training distribution.
Serving distribution is compared to these fixed bins every 6 hours.

ARCHITECTURE
------------
  phenomenal_flow_v3.py (training)
    └─ SkewDetector.record_training_stats(feature_df)
        → writes stats to artifacts/skew/training_stats.json

  FastAPI serving (real-time)
    └─ SkewDetector.record_serving_features(feature_dict)
        → accumulates in-memory buffer (thread-safe)

  Scheduled job every 6 hours (from Airflow or background thread)
    └─ SkewDetector.compute_and_alert()
        → compute PSI per feature
        → if any PSI >= 0.20 → log + alert + optionally trigger retrain

USAGE
-----
    from recsys.serving.training_serving_skew import SkewDetector

    # At training time (in Metaflow flow):
    detector = SkewDetector(stats_path="/app/artifacts/skew/")
    detector.record_training_stats(feature_df)

    # At serving time (in app.py, inside _build_recs()):
    SKEW_DETECTOR.record_serving_features({
        "als_score": als_score,
        "genre_match_cosine": genre_match,
        ...
    })

    # In background scheduler (every 6h):
    report = SKEW_DETECTOR.compute_and_alert()
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cinewave.skew")

# ── PSI implementation ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "als_score",
    "genre_match_cosine",
    "item_popularity_log",
    "recency_score",
    "user_activity_decile",
    "top_genre_alignment",
]

N_BINS       = 10    # equal-frequency bins
PSI_WARN     = 0.10  # PSI threshold: monitor
PSI_ALERT    = 0.20  # PSI threshold: retrain recommended
MIN_SAMPLES  = 200   # minimum serving samples before computing PSI
FLUSH_EVERY  = 6 * 3600  # 6 hours in seconds


def _compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index.

    expected:  reference distribution array (training time)
    actual:    current distribution array (serving time)
    bins:      bin edges computed from expected at training time

    Returns PSI scalar. Higher = more drift.
    """
    # Bin both arrays using training-time bin edges
    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts   = np.histogram(actual,   bins=bins)[0]

    # Convert to proportions (add epsilon to avoid log(0))
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct   = (actual_counts   / len(actual))   + epsilon

    # Normalise proportions
    expected_pct /= expected_pct.sum()
    actual_pct   /= actual_pct.sum()

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _compute_bins(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """
    Compute equal-frequency bin edges from training data.
    Equal-frequency (quantile) bins give each bin roughly the same
    number of samples, which is more robust than equal-width bins for
    skewed distributions (like item popularity, which is power-law).
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # Ensure unique edges (can collapse if many identical values)
    edges = np.unique(edges)
    # Extend last edge slightly to include max value
    edges[-1] = edges[-1] + 1e-6
    return edges


# ── Concept drift proxy ───────────────────────────────────────────────────────

class RatingVelocityMonitor:
    """
    Tracks the number of new user interactions per hour.
    A sudden drop (e.g. >30% below rolling average) signals concept drift:
    users are interacting differently, meaning the model's training labels
    no longer reflect current behaviour.

    This is a lightweight proxy — proper concept drift detection would
    compare the predicted vs actual click-through rate over time,
    but that requires ground-truth labels which arrive with delay.
    """

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self._hourly_counts: List[Tuple[float, int]] = []  # (timestamp, count)
        self._lock = threading.Lock()

    def record_interaction(self) -> None:
        """Call once per user interaction (play, click, feedback)."""
        now = time.time()
        with self._lock:
            # Round to current hour bucket
            hour_bucket = now - (now % 3600)
            if self._hourly_counts and self._hourly_counts[-1][0] == hour_bucket:
                ts, count = self._hourly_counts[-1]
                self._hourly_counts[-1] = (ts, count + 1)
            else:
                self._hourly_counts.append((hour_bucket, 1))
            # Keep only window_hours
            cutoff = now - self.window_hours * 3600
            self._hourly_counts = [
                (ts, c) for ts, c in self._hourly_counts if ts >= cutoff
            ]

    def detect_drift(self, drop_threshold: float = 0.30) -> Dict:
        """
        Returns drift signal if current hour count is >drop_threshold
        below the rolling hourly average.
        """
        with self._lock:
            if len(self._hourly_counts) < 2:
                return {"drift": False, "reason": "insufficient_data"}
            counts = [c for _, c in self._hourly_counts]
            rolling_avg = np.mean(counts[:-1])  # exclude current hour
            current     = counts[-1]
            if rolling_avg > 0 and current < rolling_avg * (1 - drop_threshold):
                drop_pct = (rolling_avg - current) / rolling_avg * 100
                return {
                    "drift":      True,
                    "type":       "rating_velocity",
                    "current":    current,
                    "rolling_avg": float(rolling_avg),
                    "drop_pct":   float(drop_pct),
                    "severity":   "high" if drop_pct > 50 else "medium",
                }
            return {
                "drift":       False,
                "current":     current,
                "rolling_avg": float(rolling_avg),
            }


# ── Main detector ─────────────────────────────────────────────────────────────

class SkewDetector:
    """
    Main class for training-serving skew detection.

    Thread-safe: record_serving_features() is called from FastAPI
    request handlers (concurrent), compute_and_alert() from a background thread.
    """

    def __init__(self, stats_path: str = "/app/artifacts/skew/"):
        self.stats_path = Path(stats_path)
        self.stats_path.mkdir(parents=True, exist_ok=True)
        self._training_stats: Optional[Dict] = None
        self._serving_buffer: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._velocity_monitor = RatingVelocityMonitor()
        self._load_training_stats()

    # ── Training time ───────────────────────────────────────────────────────

    def record_training_stats(
        self,
        feature_arrays: Dict[str, np.ndarray],
    ) -> None:
        """
        Call at training time with the full feature matrix.
        Computes and persists bin edges and summary statistics.

        feature_arrays: {"als_score": np.array([...]), "genre_match_cosine": ..., ...}
        """
        stats: Dict = {
            "recorded_at": datetime.utcnow().isoformat(),
            "n_samples":   len(next(iter(feature_arrays.values()))),
            "features":    {},
        }

        for feature_name, values in feature_arrays.items():
            values = np.array(values, dtype=np.float32)
            bins   = _compute_bins(values)
            stats["features"][feature_name] = {
                "bins":   bins.tolist(),
                "mean":   float(np.mean(values)),
                "std":    float(np.std(values)),
                "p25":    float(np.percentile(values, 25)),
                "p50":    float(np.percentile(values, 50)),
                "p75":    float(np.percentile(values, 75)),
                "min":    float(np.min(values)),
                "max":    float(np.max(values)),
                # Store a sample of training values for PSI reference
                "sample": values[np.random.choice(len(values), min(5000, len(values)), replace=False)].tolist(),
            }

        stats_file = self.stats_path / "training_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f)

        self._training_stats = stats
        logger.info(f"[SkewDetector] Training stats saved — {stats['n_samples']} samples, "
                    f"{len(stats['features'])} features")

    def _load_training_stats(self) -> None:
        """Load persisted training stats from disk if available."""
        stats_file = self.stats_path / "training_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                self._training_stats = json.load(f)
            logger.info(f"[SkewDetector] Loaded training stats from {stats_file}")

    # ── Serving time ────────────────────────────────────────────────────────

    def record_serving_features(
        self,
        features: Dict[str, float],
        is_interaction: bool = False,
    ) -> None:
        """
        Call at serving time for each feature vector served.
        Thread-safe (called from concurrent FastAPI handlers).

        features: {"als_score": 0.72, "genre_match_cosine": 0.45, ...}
        is_interaction: True if this request had a play/click event
        """
        with self._lock:
            for name, value in features.items():
                self._serving_buffer[name].append(float(value))
        if is_interaction:
            self._velocity_monitor.record_interaction()

    # ── Evaluation ──────────────────────────────────────────────────────────

    def compute_psi_report(self) -> Dict:
        """
        Compute PSI for all features using current serving buffer vs
        training distribution. Returns full report dict.
        """
        if self._training_stats is None:
            return {"error": "no_training_stats", "psi_values": {}}

        with self._lock:
            # Take a snapshot and clear the buffer
            serving_snapshot = {k: list(v) for k, v in self._serving_buffer.items()}

        report: Dict = {
            "computed_at":   datetime.utcnow().isoformat(),
            "psi_values":    {},
            "alerts":        [],
            "max_psi":       0.0,
            "status":        "ok",
        }

        for feature_name, feat_stats in self._training_stats["features"].items():
            serving_vals = serving_snapshot.get(feature_name, [])
            if len(serving_vals) < MIN_SAMPLES:
                report["psi_values"][feature_name] = {
                    "psi":    None,
                    "reason": f"insufficient_serving_samples ({len(serving_vals)} < {MIN_SAMPLES})",
                    "status": "skip",
                }
                continue

            training_sample = np.array(feat_stats["sample"])
            serving_arr     = np.array(serving_vals)
            bins            = np.array(feat_stats["bins"])

            psi = _compute_psi(training_sample, serving_arr, bins)

            if psi >= PSI_ALERT:
                status   = "alert"
                severity = "high"
            elif psi >= PSI_WARN:
                status   = "warn"
                severity = "medium"
            else:
                status   = "ok"
                severity = "none"

            report["psi_values"][feature_name] = {
                "psi":             psi,
                "status":          status,
                "severity":        severity,
                "serving_mean":    float(np.mean(serving_arr)),
                "serving_std":     float(np.std(serving_arr)),
                "training_mean":   feat_stats["mean"],
                "training_std":    feat_stats["std"],
                "serving_samples": len(serving_vals),
            }

            if psi > report["max_psi"]:
                report["max_psi"] = psi

            if status in ("alert", "warn"):
                report["alerts"].append({
                    "feature":  feature_name,
                    "psi":      psi,
                    "status":   status,
                    "severity": severity,
                })

        # Concept drift check
        velocity_signal = self._velocity_monitor.detect_drift()
        report["concept_drift"] = velocity_signal

        if report["max_psi"] >= PSI_ALERT or velocity_signal.get("drift"):
            report["status"] = "alert"
            report["recommendation"] = "retrain_recommended"
        elif report["max_psi"] >= PSI_WARN:
            report["status"] = "warn"
            report["recommendation"] = "monitor_closely"

        return report

    def compute_and_alert(self, trigger_retrain_fn=None) -> Dict:
        """
        Run full PSI computation and log/alert results.
        Optionally calls trigger_retrain_fn() if PSI >= PSI_ALERT.

        Called by background scheduler every 6 hours.
        """
        report = self.compute_psi_report()

        if report.get("status") == "alert":
            logger.warning(
                f"[SkewDetector] ALERT: max_psi={report['max_psi']:.3f}  "
                f"alerts={[a['feature'] for a in report.get('alerts', [])]}  "
                f"concept_drift={report.get('concept_drift', {}).get('drift', False)}"
            )
            if trigger_retrain_fn is not None:
                logger.warning("[SkewDetector] Triggering emergency retrain")
                try:
                    trigger_retrain_fn()
                except Exception as e:
                    logger.error(f"[SkewDetector] Retrain trigger failed: {e}")
        elif report.get("status") == "warn":
            logger.info(
                f"[SkewDetector] WARN: max_psi={report['max_psi']:.3f}  "
                f"monitoring closely"
            )
        else:
            logger.info(f"[SkewDetector] OK: max_psi={report['max_psi']:.3f}  no drift detected")

        # Persist report
        report_file = self.stats_path / f"skew_report_{datetime.utcnow().strftime('%Y%m%d_%H')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Clear serving buffer after evaluation
        with self._lock:
            self._serving_buffer.clear()

        return report

    def start_background_checker(self, interval_seconds: int = FLUSH_EVERY) -> None:
        """
        Start a background thread that calls compute_and_alert() every interval_seconds.
        Call once at app startup.
        """
        def _loop():
            while True:
                time.sleep(interval_seconds)
                try:
                    self.compute_and_alert()
                except Exception as e:
                    logger.error(f"[SkewDetector] Background check failed: {e}")

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        logger.info(f"[SkewDetector] Background checker started — interval={interval_seconds}s")


# ── Singleton for use in app.py ───────────────────────────────────────────────

SKEW_DETECTOR = SkewDetector(
    stats_path=os.environ.get("SKEW_STATS_PATH", "/app/artifacts/skew/")
)
