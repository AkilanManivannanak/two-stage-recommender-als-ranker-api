"""
scala_bridge.py — CineWave
===========================
Reads the Parquet outputs produced by the Scala FeaturePipeline and
makes them available to the Python serving layer (app.py, voice_tools.py).

This is the integration point between the Scala/Spark feature engineering
step and the FastAPI recommendation engine.

Data flow:
  Scala FeaturePipeline → Parquet files on disk
  ↓
  scala_bridge.py (this file) loads them at startup
  ↓
  SCALA_FEATURES dict → consumed by _build_recs() in app.py
                      → consumed by RL_AGENT.rerank() in rl_policy.py

Usage:
  from recsys.serving.scala_bridge import SCALA_FEATURES, get_als_score, get_cooc_items
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path configuration ────────────────────────────────────────────────────────

_ARTIFACTS_DIR = Path(os.environ.get("BUNDLE_REF", "/app/artifacts")).parent
_SCALA_FEATURES_DIR = _ARTIFACTS_DIR / "scala_features"


# ── In-memory store ───────────────────────────────────────────────────────────

class ScalaFeatureStore:
    """
    Loads Scala ALS outputs (Parquet) into numpy arrays for fast dot-product
    similarity lookup at serving time.

    Falls back to empty dicts gracefully if the Scala pipeline hasn't run yet.
    """

    def __init__(self):
        self.user_factors:   Dict[int, np.ndarray] = {}   # user_id → [float32 × rank]
        self.item_factors:   Dict[int, np.ndarray] = {}   # item_id → [float32 × rank]
        self.als_top_k:      Dict[int, List[Tuple[int, float]]] = {}  # user_id → [(item_id, score)]
        self.cooccurrence:   Dict[int, List[Tuple[int, float]]] = {}  # item_id → [(item_id, jaccard)]
        self.item_popularity: Dict[int, float] = {}       # item_id → pop_score
        self._loaded = False

    def load(self, features_dir: Path | None = None) -> bool:
        """
        Load all Scala-computed Parquet files.
        Returns True if at least ALS item factors were loaded.
        """
        base = features_dir or _SCALA_FEATURES_DIR
        if not base.exists():
            print(f"  [ScalaBridge] No scala_features dir at {base} — using empty store")
            return False

        ok = False
        ok |= self._load_factors(base / "user_factors", self.user_factors, "user_id")
        ok |= self._load_factors(base / "item_factors", self.item_factors, "item_id")
        self._load_als_predictions(base / "als_predictions")
        self._load_cooccurrence(base / "cooccurrence")
        self._load_item_popularity(base / "item_popularity")
        self._loaded = ok
        if ok:
            print(
                f"  [ScalaBridge] Loaded — "
                f"{len(self.user_factors)} users, "
                f"{len(self.item_factors)} items, "
                f"{sum(len(v) for v in self.als_top_k.values())} ALS predictions"
            )
        return ok

    def _load_factors(self, path: Path, store: dict, id_col: str) -> bool:
        """Load user or item ALS factor Parquet."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df    = table.to_pydict()
            ids   = df[id_col]
            vecs  = df["als_vector"]
            for uid, vec in zip(ids, vecs):
                store[int(uid)] = np.array(vec, dtype=np.float32)
            print(f"  [ScalaBridge] Loaded {len(store)} {id_col} factors from {path.name}")
            return True
        except Exception as e:
            print(f"  [ScalaBridge] Could not load {path}: {e}")
            return False

    def _load_als_predictions(self, path: Path):
        """Load pre-computed top-K ALS recs per user."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df    = table.to_pydict()
            for uid, iid, score in zip(df["user_id"], df["item_id"], df["als_score"]):
                uid, iid = int(uid), int(iid)
                if uid not in self.als_top_k:
                    self.als_top_k[uid] = []
                self.als_top_k[uid].append((iid, float(score)))
        except Exception as e:
            print(f"  [ScalaBridge] als_predictions not loaded: {e}")

    def _load_cooccurrence(self, path: Path):
        """Load item co-occurrence (Jaccard) pairs."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df    = table.to_pydict()
            for a, b, score in zip(df["item_a"], df["item_b"], df["jaccard_score"]):
                a, b = int(a), int(b)
                if a not in self.cooccurrence:
                    self.cooccurrence[a] = []
                if b not in self.cooccurrence:
                    self.cooccurrence[b] = []
                self.cooccurrence[a].append((b, float(score)))
                self.cooccurrence[b].append((a, float(score)))
            # Sort by score descending
            for k in self.cooccurrence:
                self.cooccurrence[k].sort(key=lambda x: -x[1])
        except Exception as e:
            print(f"  [ScalaBridge] cooccurrence not loaded: {e}")

    def _load_item_popularity(self, path: Path):
        """Load per-item popularity scores."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(path))
            df    = table.to_pydict()
            for iid, score in zip(df["item_id"], df["pop_score"]):
                self.item_popularity[int(iid)] = float(score)
        except Exception as e:
            print(f"  [ScalaBridge] item_popularity not loaded: {e}")

    # ── Serving helpers ───────────────────────────────────────────────────────

    def get_als_score(self, user_id: int, item_id: int) -> float:
        """
        Dot-product similarity between user and item ALS vectors.
        Returns 0.5 if either factor is missing (cold-start).
        """
        u = self.user_factors.get(user_id)
        v = self.item_factors.get(item_id)
        if u is None or v is None:
            return 0.5
        score = float(np.dot(u, v))
        # Normalise to [0, 1] range via sigmoid
        return float(1.0 / (1.0 + np.exp(-score)))

    def get_als_top_k(self, user_id: int, k: int = 20) -> List[Tuple[int, float]]:
        """Return pre-computed top-K ALS recs for a user."""
        return self.als_top_k.get(user_id, [])[:k]

    def get_cooc_items(self, item_id: int, k: int = 20) -> List[Tuple[int, float]]:
        """Return top-K co-occurring items (by Jaccard similarity)."""
        return self.cooccurrence.get(item_id, [])[:k]

    def get_popularity(self, item_id: int) -> float:
        """Return Scala-computed popularity score (falls back to 0.5)."""
        return self.item_popularity.get(item_id, 0.5)

    def enrich_candidates(self, candidates: List[dict], user_id: int) -> List[dict]:
        """
        Attach ALS score and Scala popularity to each candidate dict.
        Called by _build_recs() in app.py to add Scala signals to the ranker.
        """
        if not self._loaded:
            return candidates
        for c in candidates:
            iid = c.get("item_id")
            if iid:
                c["als_score"]       = self.get_als_score(user_id, int(iid))
                c["scala_pop_score"] = self.get_popularity(int(iid))
        return candidates

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def stats(self) -> dict:
        return {
            "loaded":            self._loaded,
            "n_user_factors":    len(self.user_factors),
            "n_item_factors":    len(self.item_factors),
            "n_als_predictions": sum(len(v) for v in self.als_top_k.values()),
            "n_cooc_pairs":      sum(len(v) for v in self.cooccurrence.values()) // 2,
            "n_popularity":      len(self.item_popularity),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

SCALA_FEATURES = ScalaFeatureStore()
SCALA_FEATURES.load()  # best-effort at import time; silent if files don't exist yet

# ── Convenience re-exports ────────────────────────────────────────────────────

def get_als_score(user_id: int, item_id: int) -> float:
    return SCALA_FEATURES.get_als_score(user_id, item_id)

def get_cooc_items(item_id: int, k: int = 20) -> List[Tuple[int, float]]:
    return SCALA_FEATURES.get_cooc_items(item_id, k)

def get_popularity(item_id: int) -> float:
    return SCALA_FEATURES.get_popularity(item_id)

def enrich_with_scala(candidates: List[dict], user_id: int) -> List[dict]:
    """Drop-in enrichment step for the candidate generation pipeline."""
    return SCALA_FEATURES.enrich_candidates(candidates, user_id)
