"""
Semantic Embeddings Layer  —  Foundation-Model-Inspired Item Towers
====================================================================
Real AI Engineer feature #4.

What it does:
  Implements the "item tower" concept from Netflix's Foundation Model paper.
  Instead of pure collaborative signal (ALS), we represent each item as a
  semantic embedding of its title + genre + description.

  In the full Netflix system:
    User tower   → large transformer on interaction sequences
    Item tower   → text/image encoder on title metadata
    Combined     → dot product in shared embedding space

  Here we implement:
    Item tower   → OpenAI text-embedding-3-small (1536 dims)
    User tower   → centroid of watched item embeddings (same space)
    Retrieval    → cosine similarity in embedding space

Why this matters:
  - Handles ZERO-SHOT cold start: new titles with no ratings get embeddings
    based on their description alone
  - Cross-modal: same space works for text search ("show me something like The Wire")
  - Temporal freshness: new content is immediately available for recommendation

New endpoints this enables:
  POST /search/semantic  — "I want something like Ozark but lighter"
  GET  /similar/{item_id} — semantically similar titles in embedding space
"""
from __future__ import annotations

import json, os
from pathlib import Path
from typing import Any
import numpy as np

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY","")
_CACHE_FILE  = Path("artifacts/item_embeddings.json")
_cache: dict[int, list[float]] = {}


def _embed_text(text: str) -> np.ndarray | None:
    if not _OPENAI_KEY: return None
    try:
        from recsys.serving._http import openai_post
        resp = _utf8_post('/v1/embeddings', json.loads(json.dumps({"model":"text-embedding-3-small","input":text})), _OPENAI_KEY)
        return np.array(resp["data"][0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def load_cache():
    if _CACHE_FILE.exists():
        try:
            raw = json.loads(_CACHE_FILE.read_text())
            _cache.update({int(k): v for k,v in raw.items()})
            print(f"  [Embeddings] Loaded {len(_cache)} cached item embeddings")
        except Exception:
            pass


def save_cache():
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE,"w") as f:
            json.dump({str(k):v for k,v in _cache.items()}, f)
    except Exception:
        pass


def get_item_embedding(item: dict) -> np.ndarray | None:
    mid = int(item.get("item_id", item.get("movieId",0)))
    if mid in _cache:
        return np.array(_cache[mid], dtype=np.float32)
    text = f"{item.get('title','')} {item.get('primary_genre','')} {item.get('description','')}"
    vec  = _embed_text(text)
    if vec is not None:
        _cache[mid] = vec.tolist()
        save_cache()
    return vec


def semantic_search(query: str, catalog: dict[int,dict], top_k: int=10) -> list[dict]:
    """
    Free-text semantic search: "dark crime drama like Ozark"
    Returns items ranked by semantic similarity to the query.
    """
    if not _OPENAI_KEY:
        return [{"error":"OPENAI_API_KEY not set — add it to enable semantic search",
                 "fallback":"Use /catalog/popular for non-semantic results"}]
    query_vec = _embed_text(query)
    if query_vec is None:
        return []

    scores: list[tuple[int,float]] = []
    for mid, item in catalog.items():
        vec = get_item_embedding(item)
        if vec is None: continue
        sim = float(np.dot(query_vec,vec) / (np.linalg.norm(query_vec)*np.linalg.norm(vec)+1e-9))
        scores.append((mid, sim))

    scores.sort(key=lambda x:-x[1])
    results = []
    for mid, sim in scores[:top_k]:
        item = dict(catalog[mid])
        item["semantic_score"] = round(sim, 4)
        item["search_method"]  = "text_embedding_3_small"
        results.append(item)
    return results


def find_similar(item_id: int, catalog: dict[int,dict], top_k: int=10) -> list[dict]:
    """
    Find semantically similar items to a given title.
    Uses cosine similarity in the shared embedding space.
    """
    anchor = catalog.get(item_id)
    if not anchor:
        return []
    anchor_vec = get_item_embedding(anchor)
    if anchor_vec is None:
        # Fallback: genre-based similarity
        genre = anchor.get("primary_genre","")
        return [dict(item) for iid,item in catalog.items()
                if item.get("primary_genre")==genre and iid!=item_id][:top_k]

    scores = []
    for mid, item in catalog.items():
        if mid == item_id: continue
        vec = get_item_embedding(item)
        if vec is None: continue
        sim = float(np.dot(anchor_vec,vec)/(np.linalg.norm(anchor_vec)*np.linalg.norm(vec)+1e-9))
        scores.append((mid,sim))

    scores.sort(key=lambda x:-x[1])
    results = []
    for mid,sim in scores[:top_k]:
        item = dict(catalog[mid])
        item["similarity_score"] = round(sim,4)
        item["similarity_method"] = "semantic_embedding"
        results.append(item)
    return results


load_cache()
