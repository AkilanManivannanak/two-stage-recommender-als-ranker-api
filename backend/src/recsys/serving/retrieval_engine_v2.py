"""
Four-Retriever Fusion Engine — Phase 3
=======================================
Spec budgets:
  collaborative   : top-300  (two-tower / ALS dot-product)
  session-intent  : top-150  (GRU encoder over recent events)
  semantic        : top-150  (Qdrant cosine, MediaFM-inspired embeddings)
  freshness       : top-100  (trending + new launches)
  fused after dedupe: 400-600

Each retriever is independently measured for recall@K.
Fusion = union + dedup + provenance tagging.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RetrievedItem:
    item_id:     int
    score:       float
    source:      str          # "collaborative" | "session" | "semantic" | "freshness"
    metadata:    dict = field(default_factory=dict)


@dataclass
class FusedCandidates:
    items:           list[RetrievedItem]
    sources:         dict              # source → count
    n_deduped:       int = 0
    retrieval_ms:    float = 0.0


class CollaborativeRetriever:
    """
    ALS dot-product retriever. In Phase 2 of roadmap, replace with two-tower.
    Returns top-K items by u_vec · item_mat score.
    """

    def __init__(self, item_factors: dict, item_ids: list):
        self.item_factors = item_factors
        self.item_ids = item_ids
        self._item_mat: Optional[np.ndarray] = None
        self._item_id_list: Optional[list] = None
        self._build_matrix()

    def _build_matrix(self):
        if self.item_factors:
            self._item_id_list = list(self.item_factors.keys())
            try:
                self._item_mat = np.stack(
                    [np.array(self.item_factors[i], dtype=np.float32)
                     for i in self._item_id_list]
                )
            except Exception:
                self._item_mat = None

    def retrieve(self, user_vector: np.ndarray, catalog: dict, k: int = 300) -> list[RetrievedItem]:
        if self._item_mat is None or len(self._item_id_list) == 0:
            return []
        try:
            scores = self._item_mat @ user_vector.astype(np.float32)
            top_idx = np.argsort(-scores)[:k]
            results = []
            for idx in top_idx:
                iid = self._item_id_list[idx]
                item = catalog.get(iid, {})
                results.append(RetrievedItem(
                    item_id=iid,
                    score=float(scores[idx]),
                    source="collaborative",
                    metadata={
                        "title":         item.get("title", ""),
                        "primary_genre": item.get("primary_genre", ""),
                        "popularity":    item.get("popularity", 50),
                        "avg_rating":    item.get("avg_rating", 3.5),
                        "year":          item.get("year", 2000),
                        "runtime_min":   item.get("runtime_min", 100),
                    }
                ))
            return results
        except Exception:
            return []


class SessionRetriever:
    """
    Genre-based session-intent retriever.
    Phase 3 roadmap: replace with GRU encoder outputting item-space vector.
    """

    def __init__(self, catalog: dict):
        self.catalog = catalog
        # Build genre index
        self._genre_index: dict[str, list[int]] = {}
        for iid, item in catalog.items():
            g = item.get("primary_genre", "Unknown")
            self._genre_index.setdefault(g, []).append(iid)

    def retrieve(
        self,
        session_events: list[dict],
        user_genre_ratings: dict,
        k: int = 150,
    ) -> list[RetrievedItem]:
        # Extract intent genres from session events + long-term profile
        session_genres: dict[str, float] = {}

        for ev in session_events[-20:]:
            iid = ev.get("item_id", 0)
            et  = ev.get("event_type", "")
            item = self.catalog.get(iid, {})
            g = item.get("primary_genre", "")
            if g:
                weight = 2.0 if et in ("watch_3min", "completion") else \
                         1.5 if et == "play_start" else \
                         0.5 if et == "click" else -1.0
                session_genres[g] = session_genres.get(g, 0.0) + weight

        # Blend with long-term profile (lower weight)
        for g, ratings in user_genre_ratings.items():
            avg = float(np.mean(ratings)) if ratings else 3.5
            session_genres[g] = session_genres.get(g, 0.0) + (avg - 3.0) * 0.3

        # Rank genres by intent signal
        ranked_genres = sorted(session_genres, key=lambda g: -session_genres[g])[:5]

        candidates = []
        for g in ranked_genres:
            items_in_genre = self._genre_index.get(g, [])
            for iid in items_in_genre:
                item = self.catalog.get(iid, {})
                score = session_genres.get(g, 0.0) * float(item.get("avg_rating", 3.5)) / 5.0
                candidates.append(RetrievedItem(
                    item_id=iid,
                    score=score,
                    source="session",
                    metadata={
                        "title":         item.get("title", ""),
                        "primary_genre": item.get("primary_genre", ""),
                        "intent_genre":  g,
                        "popularity":    item.get("popularity", 50),
                        "avg_rating":    item.get("avg_rating", 3.5),
                    }
                ))

        candidates.sort(key=lambda x: -x.score)
        return candidates[:k]


class SemanticRetriever:
    """
    Cosine-similarity retriever over item embeddings.
    Queries Qdrant if available, falls back to in-process numpy search.
    """

    def __init__(self, item_embeddings: dict, catalog: dict, qdrant_client=None):
        self.item_embeddings = item_embeddings
        self.catalog = catalog
        self._qdrant = qdrant_client
        self._emb_ids: Optional[list] = None
        self._emb_mat: Optional[np.ndarray] = None
        self._build_matrix()

    def _build_matrix(self):
        if self.item_embeddings:
            self._emb_ids = list(self.item_embeddings.keys())
            try:
                self._emb_mat = np.stack(
                    [np.array(self.item_embeddings[i], dtype=np.float32)
                     for i in self._emb_ids]
                )
            except Exception:
                self._emb_mat = None

    def retrieve(self, query_vector: np.ndarray, k: int = 150) -> list[RetrievedItem]:
        # Try Qdrant first
        if self._qdrant:
            try:
                results = self._qdrant.search(
                    collection_name="title_embeddings",
                    query_vector=query_vector.tolist(),
                    limit=k,
                )
                output = []
                for r in results:
                    iid = r.id
                    item = self.catalog.get(iid, {})
                    output.append(RetrievedItem(
                        item_id=iid,
                        score=float(r.score),
                        source="semantic",
                        metadata={
                            "title":         item.get("title", ""),
                            "primary_genre": item.get("primary_genre", ""),
                            "avg_rating":    item.get("avg_rating", 3.5),
                        }
                    ))
                return output
            except Exception:
                pass

        # Numpy fallback
        if self._emb_mat is None:
            return []
        try:
            dim = self._emb_mat.shape[1]
            q = query_vector[:dim] if len(query_vector) >= dim \
                else np.pad(query_vector, (0, dim - len(query_vector)))
            q = q / (np.linalg.norm(q) + 1e-8)
            scores = self._emb_mat @ q
            top_idx = np.argsort(-scores)[:k]
            results = []
            for idx in top_idx:
                iid = self._emb_ids[idx]
                item = self.catalog.get(iid, {})
                results.append(RetrievedItem(
                    item_id=iid,
                    score=float(scores[idx]),
                    source="semantic",
                    metadata={
                        "title":         item.get("title", ""),
                        "primary_genre": item.get("primary_genre", ""),
                        "avg_rating":    item.get("avg_rating", 3.5),
                    }
                ))
            return results
        except Exception:
            return []


class FreshnessRetriever:
    """
    Trending and new-launch retriever. Source: Redis trending scores.
    SLA: 15s. Falls back to popularity ranking if Redis unavailable.
    """

    def __init__(self, catalog: dict, redis_store=None):
        self.catalog = catalog
        self._redis_store = redis_store
        # Build popularity fallback index
        self._by_popularity = sorted(
            catalog.keys(),
            key=lambda iid: -float(catalog[iid].get("popularity", 0)),
        )

    def retrieve(self, k: int = 100) -> list[RetrievedItem]:
        # Try Redis trending
        if self._redis_store:
            try:
                trending = self._redis_store.get_top_trending(k)
                if trending:
                    results = []
                    for iid, t_score in trending:
                        item = self.catalog.get(iid, {})
                        if item:
                            results.append(RetrievedItem(
                                item_id=iid,
                                score=float(t_score),
                                source="freshness",
                                metadata={
                                    "title":         item.get("title", ""),
                                    "primary_genre": item.get("primary_genre", ""),
                                    "trending_score": float(t_score),
                                }
                            ))
                    if results:
                        return results[:k]
            except Exception:
                pass

        # Popularity fallback
        results = []
        for iid in self._by_popularity[:k]:
            item = self.catalog.get(iid, {})
            results.append(RetrievedItem(
                item_id=iid,
                score=float(item.get("popularity", 50)) / 1000.0,
                source="freshness",
                metadata={
                    "title":         item.get("title", ""),
                    "primary_genre": item.get("primary_genre", ""),
                }
            ))
        return results[:k]


class RetrievalEngine:
    """
    Orchestrates all four retrievers, fuses results, and deduplicates.
    Returns FusedCandidates with provenance tags on each item.
    """

    def __init__(self, catalog: dict, item_factors: dict = None,
                 item_embeddings: dict = None, redis_store=None,
                 qdrant_client=None):
        item_factors    = item_factors    or {}
        item_embeddings = item_embeddings or {}
        item_ids        = list(catalog.keys())

        self.collab    = CollaborativeRetriever(item_factors, item_ids)
        self.session   = SessionRetriever(catalog)
        self.semantic  = SemanticRetriever(item_embeddings, catalog, qdrant_client)
        self.freshness = FreshnessRetriever(catalog, redis_store)
        self.catalog   = catalog

    def retrieve(
        self,
        user_id:              int,
        user_vector:          Optional[np.ndarray],
        user_genre_ratings:   dict,
        session_events:       list[dict],
        collab_k:   int = 300,
        session_k:  int = 150,
        semantic_k: int = 150,
        fresh_k:    int = 100,
    ) -> FusedCandidates:
        t0 = time.time()

        # ── Run all four retrievers ──────────────────────────────────────────
        collab_results  = []
        semantic_results = []

        if user_vector is not None:
            collab_results   = self.collab.retrieve(user_vector, self.catalog, collab_k)
            semantic_results = self.semantic.retrieve(user_vector, semantic_k)

        session_results  = self.session.retrieve(session_events, user_genre_ratings, session_k)
        fresh_results    = self.freshness.retrieve(fresh_k)

        # ── Fuse with provenance tracking ────────────────────────────────────
        seen: dict[int, RetrievedItem] = {}
        sources = {"collaborative": 0, "session": 0, "semantic": 0, "freshness": 0}

        for r in collab_results:
            if r.item_id not in seen:
                seen[r.item_id] = r
            sources["collaborative"] += 1

        for r in session_results:
            if r.item_id not in seen:
                seen[r.item_id] = r
            sources["session"] += 1

        for r in semantic_results:
            if r.item_id not in seen:
                seen[r.item_id] = r
            sources["semantic"] += 1

        for r in fresh_results:
            if r.item_id not in seen:
                seen[r.item_id] = r
            sources["freshness"] += 1

        all_items = list(seen.values())
        n_total_before_dedup = (
            len(collab_results) + len(session_results) +
            len(semantic_results) + len(fresh_results)
        )
        n_deduped = n_total_before_dedup - len(all_items)

        return FusedCandidates(
            items=all_items,
            sources=sources,
            n_deduped=n_deduped,
            retrieval_ms=round((time.time() - t0) * 1000, 1),
        )
